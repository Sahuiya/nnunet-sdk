"""
ONNX inference for nnU-Net.

Uses patch-level ONNX models exported via ``nnUNetPredictor.export_to_jit_and_onnx``.
Preprocessing, sliding-window aggregation, Gaussian blending, TTA mirroring, and
segmentation export reuse the same pipeline as ``nnUNetPredictor``.
"""
import itertools
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            'onnxruntime is required for ONNX inference. Install with: pip install onnxruntime '
            '(GPU: pip install onnxruntime-gpu)'
        ) from e
    return ort


class nnUNetOnnxPredictor(nnUNetPredictor):
    """
    Drop-in predictor that runs the patch forward pass with ONNX Runtime instead of PyTorch.

    Still requires ``plans.json`` and ``dataset.json`` from the trained model folder for
    preprocessing and postprocessing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_sessions: Optional[List] = None
        self.onnx_input_name = 'input'
        self.onnx_output_name = 'logits'
        self._current_session_idx = 0

    @staticmethod
    def configuration_name_from_model_folder(model_training_output_dir: str) -> str:
        folder_name = os.path.basename(model_training_output_dir.rstrip(os.sep))
        parts = folder_name.split('__')
        if len(parts) < 3:
            raise RuntimeError(
                f'Cannot infer configuration name from folder "{folder_name}". '
                f'Expected format Trainer__Plans__configuration. '
                f'Pass configuration_name explicitly to initialize_from_onnx_folder.')
        return parts[-1]

    @staticmethod
    def discover_onnx_model_paths(onnx_folder: str, use_folds: Tuple[Union[int, str], ...]) -> List[str]:
        use_folds = tuple(int(f) if f != 'all' else f for f in use_folds)
        paths: List[str] = []
        for fold in use_folds:
            candidates = [
                join(onnx_folder, f'model_fold_{fold}.onnx'),
                join(onnx_folder, f'model_{fold}.onnx'),
            ]
            found = next((p for p in candidates if isfile(p)), None)
            if found is not None:
                paths.append(found)
        if len(paths) == 0 and isfile(join(onnx_folder, 'model.onnx')):
            if len(use_folds) == 1:
                paths.append(join(onnx_folder, 'model.onnx'))
            else:
                raise RuntimeError(
                    f'Found only model.onnx in {onnx_folder} but {len(use_folds)} folds were requested.')
        if len(paths) == 0:
            raise RuntimeError(
                f'No ONNX model files found in {onnx_folder} for folds {use_folds}. '
                f'Export first with nnUNetv2_export_from_modelfolder or export_to_jit_and_onnx.')
        return paths

    def _onnx_providers(self) -> List[str]:
        if self.device.type == 'cuda':
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def initialize_from_onnx_folder(self,
                                    onnx_folder: str,
                                    model_training_output_dir: str,
                                    use_folds: Tuple[Union[int, str], ...] = (0,),
                                    configuration_name: Optional[str] = None):
        """
        Load ONNX sessions and plans/dataset metadata for full nnU-Net inference.

        Parameters
        ----------
        onnx_folder
            Directory with ``model.onnx`` or ``model_fold_X.onnx`` (from export).
        model_training_output_dir
            Trained model folder containing ``plans.json`` and ``dataset.json``.
        use_folds
            Which exported fold ONNX files to use (ensemble by averaging logits).
        configuration_name
            Plans configuration key (e.g. ``3d_lowres``). Inferred from model folder name if None.
        """
        ort = _import_onnxruntime()

        onnx_folder = os.path.abspath(onnx_folder)
        model_training_output_dir = os.path.abspath(model_training_output_dir)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        meta_path = join(onnx_folder, 'export_metadata.json')
        export_meta = load_json(meta_path) if isfile(meta_path) else {}

        if configuration_name is None:
            configuration_name = export_meta.get('configuration_name')
        if configuration_name is None:
            configuration_name = self.configuration_name_from_model_folder(model_training_output_dir)

        configuration_manager = plans_manager.get_configuration(configuration_name)
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.list_of_parameters = None
        self.network = None
        self.trainer_name = export_meta.get('trainer_name')
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        mirroring_axes = export_meta.get('inference_allowed_mirroring_axes')
        if mirroring_axes is not None:
            self.allowed_mirroring_axes = tuple(mirroring_axes)
        else:
            self.allowed_mirroring_axes = None

        onnx_paths = self.discover_onnx_model_paths(onnx_folder, use_folds)
        providers = self._onnx_providers()
        self.onnx_sessions = []
        for path in onnx_paths:
            sess = ort.InferenceSession(path, providers=providers)
            self.onnx_input_name = sess.get_inputs()[0].name
            self.onnx_output_name = sess.get_outputs()[0].name
            self.onnx_sessions.append(sess)
            print(f'Loaded ONNX: {path} (providers: {sess.get_providers()})')

        if self.verbose:
            print(f'ONNX inference: {len(self.onnx_sessions)} model(s), configuration={configuration_name}')

    def initialize_from_trained_model_folder(self, *args, **kwargs):
        raise RuntimeError(
            'nnUNetOnnxPredictor uses initialize_from_onnx_folder. '
            'Use nnUNetPredictor for PyTorch checkpoints.')

    def _run_onnx_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.onnx_sessions is None:
            raise RuntimeError('Call initialize_from_onnx_folder before predicting.')
        sess = self.onnx_sessions[self._current_session_idx]
        x_np = x.contiguous().float().cpu().numpy()
        logits_np = sess.run([self.onnx_output_name], {self.onnx_input_name: x_np})[0]
        return torch.from_numpy(logits_np).to(device=x.device, dtype=torch.float32)

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self._run_onnx_forward(x)

        if mirror_axes is not None:
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'
            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(
                    self._run_onnx_forward(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        for session_idx in range(len(self.onnx_sessions)):
            self._current_session_idx = session_idx
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.onnx_sessions) > 1:
            prediction /= len(self.onnx_sessions)

        if self.verbose:
            print('Prediction done (ONNX)')
        torch.set_num_threads(n_threads)
        return prediction

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        empty_cache(self.device)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print('step_size:', self.tile_step_size)
                print('mirror_axes:', self.allowed_mirroring_axes if self.use_mirroring else None)

            data, slicer_revert_padding = pad_nd_image(
                input_image, self.configuration_manager.patch_size,
                'constant', {'value': 0}, True, None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(
                        data, slicers, self.perform_everything_on_device)
                except RuntimeError:
                    print(
                        'Prediction on device was unsuccessful, probably due to a lack of memory. '
                        'Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(
                    data, slicers, self.perform_everything_on_device)

            empty_cache(self.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits


def predict_onnx_entry_point_modelfolder():
    """CLI: nnUNetv2_predict_from_onnx_modelfolder"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run nnU-Net inference with exported ONNX models. '
                    'Requires plans.json/dataset.json from the trained model folder and .onnx files from export.')
    parser.add_argument('-i', type=str, required=True, help='Input folder.')
    parser.add_argument('-o', type=str, required=True, help='Output folder.')
    parser.add_argument('-m', type=str, required=True,
                        help='Trained model folder (plans.json, dataset.json).')
    parser.add_argument('--onnx-folder', type=str, required=True,
                        help='Folder containing exported .onnx model(s).')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0,),
                        help='Fold ONNX files to use. Default: (0,)')
    parser.add_argument('-configuration', type=str, required=False, default=None,
                        help='Plans configuration name (e.g. 3d_lowres). Inferred from -m folder name if omitted.')
    parser.add_argument('-step_size', type=float, required=False, default=0.5)
    parser.add_argument('--disable_tta', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_probabilities', action='store_true')
    parser.add_argument('--continue_prediction', '--c', action='store_true')
    parser.add_argument('-npp', type=int, required=False, default=3)
    parser.add_argument('-nps', type=int, required=False, default=3)
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None)
    parser.add_argument('-device', type=str, default='cuda', required=False)
    parser.add_argument('--disable_progress_bar', action='store_true', default=False)

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    from batchgenerators.utilities.file_and_folder_operations import isdir, maybe_mkdir_p
    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda', 'mps'], f'Unsupported device: {args.device}'
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetOnnxPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
    )
    predictor.initialize_from_onnx_folder(
        args.onnx_folder, args.m, use_folds=tuple(args.f), configuration_name=args.configuration)
    predictor.predict_from_files(
        args.i, args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=1, part_id=0)
