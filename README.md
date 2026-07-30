# nnunet-sdk：从数据到分割结果的入门指南

`nnunet-sdk` 用 `seg` 命令封装了 nnU-Net v2 的数据准备、规划、训练、推理和评估流程。你不需要先了解 nnU-Net 的内部目录或环境变量；把任务写进一个 YAML 配置文件后，按本文顺序执行即可。

适用场景：医学图像分割（CT、MR 等），输入与标签为 NIfTI 文件（通常是 `.nii.gz`）。底层训练仍然是 nnU-Net，`seg` 不会改变其模型和训练核心。


## 1. 安装


### 1.1 前置条件

- Linux / Windows / macOS（nnU-Net 官方均支持；本仓库与示例命令以 Linux 为主）
- Python 3.10 或兼容版本
- 训练推荐 NVIDIA GPU + CUDA 匹配的 PyTorch；也可在 CPU / Apple MPS 上跑（更慢）
- 已安装 Git

先根据 [PyTorch 官网](https://pytorch.org/get-started/locally/) 为自己的 CUDA 环境安装 PyTorch。然后在仓库根目录安装本项目：

```bash
# 安装本地 nnU-Net
pip install -e ./nnUNet

# 安装 segkit
pip install -e .

# 如需读取、合并 NIfTI 标签（绝大多数 prepare 场景需要）
pip install -e ".[adapters]"

# 可选：运行项目测试
pip install -e ".[dev]"
```

检查安装：

```bash
seg --version
seg --help
which nnUNetv2_train
```

最后一条应输出 `nnUNetv2_train` 的路径。若没有输出，通常是本地 `nnUNet` 未正确安装或当前 Python 环境不对。


## 2. 写任务配置

每个分割任务使用一个 YAML 文件。它记录数据位置、类别定义、训练方式与推理默认值。YAML 中的相对路径以“该 YAML 文件所在目录”为基准；实际项目建议对 `paths` 使用绝对路径。


### 2.1 初始化命令

以下命令创建一个起始配置：

```bash
seg init --out configs/my_task.yaml --name MyOrgans --id 101
```

- `--out`：配置文件保存位置。
- `--name`：任务名。只能使用适合作为目录名的字符，例如 `MyOrgans`。
- `--id`：数据集编号。它在同一个 `paths.raw` 下必须唯一。

生成后，打开 `configs/my_task.yaml`，将路径、类别和数据来源替换为自己的内容。已有 TotalSegmentator 数据可参考 [configs/organs7_qs300.yaml](configs/organs7_qs300.yaml)。


### 2.2 完整配置文件说明

下面是一个单通道 CT、两类器官的完整可运行示例。注释说明了字段用途；不需要的可选字段可以保留默认值或删除。

```yaml
project:
  name: MY_ORGANS                 # 本次任务的显示名称
  seed: 42                        # 随机种子

paths:
  # nnU-Net 的三个工作目录，以及 segkit 的运行记录目录
  root: /data/nnunet
  raw: /data/nnunet/nnUNet_raw
  preprocessed: /data/nnunet/nnUNet_preprocessed
  results: /data/nnunet/nnUNet_results
  runs: /data/nnunet/runs

dataset:
  id: 101                         # 0--999 内、在 raw 根目录下唯一
  name: MY_ORGANS                 # 会生成 Dataset101_MY_ORGANS
  file_ending: .nii.gz
  channel_names:
    0: CT                         # CT 使用 CT 归一化；MR 常填 MR 或对应模态名
  labels:                         # 背景固定为 0；前景类别从 1 起连续编号
    background: 0
    liver: 1
    spleen: 2
  source:
    # 三种 adapter 之一：nifti_folder / nifti_multilabel_merge /
    # totalsegmentator_folder
    adapter: nifti_folder
    images: /data/source/images   # 原始图像目录
    labels: /data/source/labels   # 与图像同名的多类别标签目录

plan:
  verify_dataset_integrity: true  # 规划前让 nnU-Net 额外检查数据完整性
  no_pp: false                    # false：执行预处理
  configurations: null            # null：让 nnU-Net 自动决定要生成的配置
  # gpu_memory_target: 24         # 可选，单位 GB；改变后须重新 plan
  # overwrite_plans_name: nnUNetPlans_24G

train:
  configuration: 3d_fullres       # 常用三维全分辨率配置
  plans: nnUNetPlans              # 必须与 plan 产生的 plans 名称一致
  fold: 0                         # 先训练第 0 折
  device: cuda                    # GPU 训练；仅调试时可改 cpu
  # 以下训练旋钮全部可选。写任意一个时，自动使用 nnUNetTrainerSegkit。
  # num_epochs: 250
  # loss: dice_ce                 # dice_ce | dice | dice_heavy_ce
  # oversample_fg: 0.33           # 前景 patch 比例；小目标可提高到 1.0
  # mirroring: true               # true | false | only_01

predict:
  input: null                     # 可由命令行 -i 覆盖
  output: null                    # 可由命令行 -o 覆盖
  model_folder: null              # null：按 dataset/train/results 自动寻找模型
  checkpoint: checkpoint_best.pth
  fold: 0
  tta: true                       # 测试时增强；关闭可加快推理
  save_probabilities: false
  device: cuda
  backend: pytorch
  postprocess: null               # null | identity | generic_largest_cc | generic_min_size

eval:
  gt_folder: null                 # 可由命令行 --gt 覆盖
  pred_folder: null               # 可由命令行 --pred 覆盖
  labels: null                    # null：使用 dataset.labels
  chill: true

postprocess:
  enabled: false
  name: identity
  params: {}

bundle:
  include_onnx: false
```


### 2.3 至少要填哪些内容

开始前，至少确认以下内容正确：

1. `paths.raw`、`paths.preprocessed`、`paths.results`、`paths.runs`：四个可写目录。
2. `dataset.id` 与 `dataset.name`：二者会共同决定目录名 `Dataset101_MY_ORGANS`，后续不可随意改名。
3. `dataset.file_ending`、`channel_names`、`labels`：必须和真实数据一致。背景标签必须是 `0`，前景标签必须从 `1` 连续编号。
4. `dataset.source`：选择正确的 adapter 并填写原始数据位置。若你的数据已经是 nnU-Net 官方 raw 布局，不需运行 `seg prepare`，但 YAML 中的 `dataset` 与 `paths` 仍要一致。
5. `train.plans`：默认是 `nnUNetPlans`。若设置过 `plan.gpu_memory_target`，要改为对应的 `nnUNetPlans_XXG`。

不需要手动设置 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results` 环境变量；`seg` 会在执行 plan、train 和 predict 时根据 YAML 注入。


## 3. 数据准备

这一节只处理数据。完成并通过 `seg doctor` 之前，不要开始 `seg plan` 或 `seg train`。


### 3.1 nnU-Net 最终需要的数据布局

无论原始数据长什么样，训练前都必须得到以下目录。`seg prepare` 的作用就是在适用时生成这一布局：

```text
{paths.raw}/
└── Dataset101_MY_ORGANS/
    ├── dataset.json
    ├── imagesTr/
    │   ├── case_001_0000.nii.gz
    │   └── case_002_0000.nii.gz
    ├── labelsTr/
    │   ├── case_001.nii.gz
    │   └── case_002.nii.gz
    └── imagesTs/                 # 可选：测试图像不参与训练
```

- 单通道图像必须命名为 `{病例ID}_0000.nii.gz`。
- 多通道图像使用 `_0000`、`_0001`、……，且每个病例的通道必须齐全。
- 标签文件名没有通道号：`{病例ID}.nii.gz`。
- 图像和标签必须具有相同的空间尺寸、spacing 和方向。
- 标签应为整型，背景为 0，前景编号连续。


### 3.2 选择数据导入方式

**方式 A：已经符合上述布局**

不运行 `seg prepare`。确认目录名与 `dataset.id` / `dataset.name` 一致，再直接执行本节末尾的 `seg doctor`。

**方式 B：每例一张图像、一张已经多类别的标签**

原始数据例如：

```text
/data/source/images/case_001.nii.gz
/data/source/labels/case_001.nii.gz
```

在 YAML 中使用：

```yaml
dataset:
  source:
    adapter: nifti_folder
    images: /data/source/images
    labels: /data/source/labels
```

然后执行：

```bash
seg prepare -c configs/my_task.yaml
```

**方式 C：每个类别是一套独立的二值 mask**

原始数据例如：

```text
/data/source/images/case_001.nii.gz
/data/source/masks_liver/case_001.nii.gz
/data/source/masks_spleen/case_001.nii.gz
```

使用 `nifti_multilabel_merge`。`class_dirs` 的顺序对应标签编号 1、2、……：

```yaml
dataset:
  labels:
    background: 0
    liver: 1
    spleen: 2
  source:
    adapter: nifti_multilabel_merge
    images: /data/source/images
    class_dirs:
      - /data/source/masks_liver
      - /data/source/masks_spleen
```

执行：

```bash
seg prepare -c configs/my_task.yaml
```

**方式 D：TotalSegmentator 病例目录**

每个病例包含 `ct.nii.gz` 和 `segmentations/` 下的多个器官二值 mask 时使用：

```yaml
dataset:
  id: 102
  name: ORGANS7
  labels:
    background: 0
  source:
    adapter: totalsegmentator_folder
    parts_preset: organs_7        # 或 parts: [liver, gallbladder, spleen]
    dataset_path: /data/totalsegmentator
    ct_name: ct.nii.gz
    segmentations_subdir: segmentations
    # train_list: /data/splits/train.txt
    # val_list: /data/splits/val.txt
    # test_list: /data/splits/test.txt
```

执行：

```bash
seg prepare -c configs/my_task.yaml
```

可临时覆盖要合并的结构：

```bash
seg prepare -c configs/my_task.yaml --parts liver,gallbladder,spleen
seg prepare -c configs/my_task.yaml --parts-preset organs_7
```


### 3.3 检查数据

无论使用哪种方式，都运行：

```bash
seg doctor -c configs/my_task.yaml
```

它会检查依赖、路径权限、nnU-Net 命令是否可用，以及已有数据集的图像/标签配对。若 doctor 或后续 plan 报 `Could not find a dataset with the ID ...`，优先检查 `paths.raw` 中是否存在与 YAML 完全同名的 `Dataset{ID:03d}_{name}` 目录。


## 4. 训练与推理

本节是完整命令一览和推荐执行顺序。所有命令均可通过 `seg <命令> --help` 查看完整参数。会调用 nnU-Net 的命令支持 `--dry-run`，可先检查即将执行的操作而不真正运行。


### 4.1 命令一览

```bash
# 生成配置
seg init --out configs/my_task.yaml --name MyOrgans --id 101

# 环境、路径和数据配对检查
seg doctor -c configs/my_task.yaml

# 将非标准原始数据转换为 nnU-Net 数据布局
seg prepare -c configs/my_task.yaml

# 指纹分析、规划网络并预处理
seg plan -c configs/my_task.yaml

# 训练
seg train -c configs/my_task.yaml

# 推理
seg predict -c configs/my_task.yaml -i /data/imagesTs -o /data/predictions

# 评估预测与真值
seg eval -c configs/my_task.yaml --gt /data/labelsTs --pred /data/predictions

# 独立执行后处理：单文件或一个目录
seg postprocess --seg /data/predictions/case_001.nii.gz -o /data/postprocessed \
  --plugin generic_largest_cc
seg postprocess --folder /data/predictions -o /data/postprocessed \
  --plugin generic_largest_cc

# 导出 TorchScript / ONNX 部署产物
seg export -c configs/my_task.yaml -o /data/exported_model

# 打包与解包训练模型
seg pack -c configs/my_task.yaml -o /data/my_model.bundle.zip
seg unpack -b /data/my_model.bundle.zip -d /data/model_install

# 运行预先定义的 Dice 回归基准
seg bench --spec configs/bench_example.json --out runs/bench/report.json
```


### 4.2 执行训练

数据检查通过后，依次运行：

```bash
# 首先只查看计划执行的 nnU-Net 命令
seg plan -c configs/my_task.yaml --dry-run

# 规划并预处理；首次运行可能耗时较长
seg plan -c configs/my_task.yaml

# 首先只检查训练命令
seg train -c configs/my_task.yaml --dry-run

# 开始训练
seg train -c configs/my_task.yaml
```

需要覆盖某次训练的折数或配置时：

```bash
seg train -c configs/my_task.yaml --fold 0 --configuration 3d_fullres
```

训练完成后的权重默认位置：

```text
{paths.results}/Dataset{id:03d}_{name}/
└── {trainer}__{plans}__{configuration}/
    └── fold_0/
        ├── checkpoint_best.pth
        └── checkpoint_final.pth
```


### 4.3 训练参数建议

默认 nnU-Net 配置已经适合大多数任务。需要控制训练预算或小目标行为时，在 `train:` 下添加：

```yaml
train:
  configuration: 3d_fullres
  plans: nnUNetPlans
  fold: 0
  num_epochs: 250
  loss: dice
  oversample_fg: 1.0
  mirroring: false
```

- `num_epochs`：训练轮数。默认 nnU-Net 为 1000。
- `loss`：`dice_ce` 是默认 Dice + Cross Entropy；极小目标出现全背景预测时可尝试 `dice`；`dice_heavy_ce` 是两者折中。
- `oversample_fg`：前景 patch 抽样比例，默认约 0.33；小器官可设为 `1.0`。
- `mirroring`：左右有不同类别（例如左肾、右肾）时设为 `false`，防止左右翻转造成标签语义冲突；`only_01` 是保留部分轴翻转的折中选择。

若提高 `plan.gpu_memory_target`，需重新执行 `seg plan`，并把 `train.plans` 改为生成的名称，例如：

```yaml
plan:
  gpu_memory_target: 24
train:
  plans: nnUNetPlans_24G
```


### 4.4 推理

最常用命令：

```bash
seg predict -c configs/my_task.yaml \
  -i /data/new_images \
  -o /data/predictions
```

输入目录的图像命名也必须符合 nnU-Net 通道规则，例如单通道使用 `patient_001_0000.nii.gz`。不传 `-w` 时，`seg` 根据 YAML 自动寻找训练结果；也可以明确指定模型：

```bash
seg predict -c configs/my_task.yaml \
  -i /data/new_images \
  -o /data/predictions \
  -w /data/nnunet/nnUNet_results/Dataset101_MY_ORGANS/nnUNetTrainer__nnUNetPlans__3d_fullres
```

关闭测试时增强可加快推理：

```bash
seg predict -c configs/my_task.yaml -i /data/new_images -o /data/predictions --disable-tta
```

推理期间逐例后处理：

```bash
seg predict -c configs/my_task.yaml \
  -i /data/new_images \
  -o /data/predictions \
  --postprocess generic_largest_cc
```

此时输出不会覆盖：

```text
/data/predictions/
├── raw/                 # 模型直接输出
└── postprocessed/       # 每个病例完成后立即生成的后处理结果
```

支持的通用后处理插件为 `identity`、`generic_largest_cc` 和 `generic_min_size`。后处理也可按本节 4.1 的 `seg postprocess` 命令在推理完成后单独运行。


### 4.4.1 ONNX 推理流程

`seg predict` 默认用 PyTorch。若要用导出的 ONNX 模型推理，按下面三步即可。

**1）安装 ONNX Runtime**

```bash
# CPU
pip install "nnunet-sdk[onnx]"
# 或 GPU（按官网选择对应包）
# pip install onnxruntime-gpu
```

**2）从训练结果导出 ONNX**

```bash
seg export -c configs/my_task.yaml -o /data/exported_onnx
```

导出目录中至少应有：

```text
/data/exported_onnx/
├── model.onnx                 # 单 fold；多 fold 时为 model_fold_0.onnx 等
└── export_metadata.json
```

注意：ONNX 只包含网络的 patch 前向；完整推理仍需要训练结果目录里的 `plans.json` / `dataset.json`（`seg` 会通过 `-w` 或 YAML 自动解析到该目录）。

**3）用 ONNX 推理**

```bash
seg predict -c configs/my_task.yaml \
  -i /data/new_images \
  -o /data/predictions_onnx \
  --backend onnx \
  --onnx-folder /data/exported_onnx
```

也可写进 YAML，之后不必每次传 `--backend`：

```yaml
predict:
  backend: onnx
  onnx_folder: /data/exported_onnx
```

等价的底层命令是：

```bash
nnUNetv2_predict_from_onnx_modelfolder \
  -i /data/new_images \
  -o /data/predictions_onnx \
  -m /path/to/trained_model_folder \
  --onnx-folder /data/exported_onnx \
  -f 0
```


### 4.5 评估

拥有测试集真值时：

```bash
seg eval -c configs/my_task.yaml \
  --gt /data/labelsTs \
  --pred /data/predictions
```

若推理使用了逐例后处理，请把 `--pred` 指向 `/data/predictions/postprocessed`；评估原始模型输出则指向 `/data/predictions/raw`。

每次 `seg plan`、`seg train`、`seg predict`、`seg eval` 等运行都会在 `paths.runs/<命令>/expN/` 写入配置快照 `args.yaml` 和执行摘要 `summary.json`。


## 5. 模型打包与导出

训练完成后，打包为可分发 zip：

```bash
seg pack -c configs/my_task.yaml -o /data/my_organs.bundle.zip
```

安装并使用该 bundle：

```bash
seg unpack -b /data/my_organs.bundle.zip -d /data/my_organs_model

seg predict \
  -i /data/new_images \
  -o /data/predictions \
  -w /data/my_organs_model/model.bundle
```

导出部署格式：

```bash
seg export -c configs/my_task.yaml -o /data/exported_model
```

也可用 `-w` 导出指定模型目录。成功后，输出目录中包含 `model.pt` 和/或 `model.onnx`（取决于 nnU-Net 导出环境和模型）。


## 6. Python SDK

也可在 Python 程序中调用同一套流程：

```python
from segkit import SegModel

model = SegModel.from_config("configs/my_task.yaml")
model.doctor()
model.prepare()
model.plan()
model.train(fold=0)
model.predict(
    source="/data/new_images",
    predict={"output": "/data/predictions"},
)
model.eval(pred_dir="/data/predictions", gt_dir="/data/labelsTs")
model.pack("/data/my_organs.bundle.zip")
```

从已有训练结果加载：

```python
from segkit import SegModel

model = SegModel.from_result_folder(
    "/data/nnunet/nnUNet_results/Dataset101_MY_ORGANS/...",
    config_path="configs/my_task.yaml",
)
model.predict(source="/data/new_images", predict={"output": "/data/predictions"})
```


## 7. 回归 bench

用于持续验证固定病例上的 Dice 指标。先编写 JSON 规格，格式参考 [configs/bench_example.json](configs/bench_example.json)，再执行：

```bash
seg bench --spec configs/bench_example.json --out runs/bench/report.json
```


## 8. 测试

运行项目测试：

```bash
pytest tests/ -q
```

这些测试不要求 GPU，覆盖配置处理、命令构造、数据 adapter、模型打包和后处理等基础行为。


## 9. 常见问题

**`seg doctor` 提示找不到 nnU-Net 或 `nnUNetv2_train`**

确认正在使用安装项目时的 Python/conda 环境，然后重新执行：

```bash
pip install -e ./nnUNet
pip install -e .
```

**`Could not find a dataset with the ID 101`**

检查 `{paths.raw}/Dataset101_{dataset.name}` 是否存在，特别注意 ID 需要三位补零、任务名大小写以及 YAML 的 `paths.raw`。

**plan 报标签或图像不配对**

检查 `imagesTr/case_001_0000.nii.gz` 是否有匹配的 `labelsTr/case_001.nii.gz`，多通道时还要检查每个病例是否拥有全部 `_000X` 文件。

**训练显存不足**

先保持默认 `nnUNetPlans`；不要仅因为显存很大就盲目提高 `gpu_memory_target`。显存目标改变了 patch 和 batch 设定，需重新 plan，再使用匹配的 `train.plans` 训练。必要时降低目标值或先用默认 plan。

**推理找不到模型**

显式使用 `-w /path/to/模型目录`。自动查找依赖 `dataset.id`、`dataset.name`、`train` 中的 trainer/plans/configuration 与实际训练输出完全一致。

**相对路径位置不对**

YAML 内相对路径是相对于 YAML 文件所在目录，不是当前终端目录。建议生产任务使用绝对路径。
