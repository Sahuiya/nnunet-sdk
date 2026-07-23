# nnU-Net 通用分割流程产品化技术文档

（YOLO / Ultralytics 风格工程形态）

| 项 | 内容 |
|----|------|
| 文档性质 | 设计说明 / 技术规格（不涉及具体业务落地） |
| 适用范围 | 任意医学/体素语义分割任务 |
| 非目标 | 重写 nnU-Net 训练核心；绑定某一器官/病种后处理 |
| 对标对象 | Ultralytics YOLO 的 CLI / 配置 / SDK / runs / 模型包体验 |

---

## 1. 背景与目标

### 1.1 问题

nnU-Net 已提供较强的自适应训练与推理能力，但工程使用形态偏「工具链拼装」：

- 命令分散（`nnUNetv2_plan_and_preprocess` / `train` / `predict` / `evaluate_*` 等）
- 依赖手动环境变量（`nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results`）
- 数据转换、评估、导出、后处理常落在任务脚本中，换任务即复制改造
- 缺少统一产物目录与可脚本消费的摘要格式

### 1.2 目标

在 **不替换 nnU-Net 引擎** 的前提下，增加一层通用「分割操作系统」，使任意分割任务具备接近 YOLO 的使用体验：

```text
seg train  --config configs/task.yaml
seg predict --weights model.bundle --source case.nii.gz
```

以及 SDK：

```python
from segkit import SegModel
m = SegModel("model.bundle")
m.predict("case.nii.gz")
```

### 1.3 成功标准

1. 新人仅需：安装 → 填写一份任务配置 → 执行少数子命令完成 prepare → plan → train → predict → eval
2. 切换任务时，主流程代码不变，只更换配置与可选插件
3. CLI 与 SDK 同源，同一参数可复现同一结果
4. 训练/推理产物目录结构固定，下游可用 `summary.json` 消费

---

## 2. 设计原则

1. **包装引擎，不替换引擎**：训练/预处理/默认推理仍调用 nnU-Net（或等价后端）。
2. **对齐 UX，不对齐任务语义**：学的是 YOLO 的入口/配置/包/runs，不是检测框 API。
3. **任务差异外置**：标签体系、数据源形态、专用后处理通过配置与插件注入。
4. **配置优先，CLI 覆盖**：默认读 YAML；命令行参数覆盖同名字段。
5. **最小可推理单元可打包**：权重 + plans + dataset 元数据可一键分发。

---

## 3. 总体架构

```text
┌──────────────────────────────────────────────┐
│  UX 层：CLI / Python SDK / YAML / runs/      │
├──────────────────────────────────────────────┤
│  编排层：Pipeline / Config / Env / RunStore  │
├──────────────────────────────────────────────┤
│  插件层：DatasetAdapter / Postprocess / Eval │
├──────────────────────────────────────────────┤
│  引擎层：nnU-Net v2（plan/train/predict/...）│
└──────────────────────────────────────────────┘
```

### 3.1 模块职责

| 模块 | 职责 |
|------|------|
| `cli` | 子命令解析、参数覆盖、退出码 |
| `sdk` | `SegModel` 等稳定 Python API |
| `config` | YAML 加载、校验、合并、路径解析 |
| `env` | 根据配置注入 nnU-Net 三路径等环境 |
| `pipeline` | prepare / plan / train / predict / eval / export 编排 |
| `adapters` | 原始数据 → nnU-Net Dataset 约定 |
| `plugins.post` | 可选后处理 |
| `plugins.eval` | 指标计算与报表 |
| `bundle` | 模型打包 / 安装 / 加载 |
| `runs` | 实验目录、args 落盘、summary 写出 |
| `doctor` | 环境与数据完整性检查 |

---

## 4. 用户界面规格

### 4.1 CLI 子命令（建议）

| 子命令 | 作用 | 底层主要能力 |
|--------|------|----------------|
| `seg doctor` | 检查依赖、CUDA、路径、数据集完整性 | 只读检查 |
| `seg init` | 生成任务配置模板与目录骨架 | 写模板 |
| `seg prepare` | 原始数据 → `nnUNet_raw/DatasetXXX_*` | DatasetAdapter |
| `seg plan` | fingerprint + plan + preprocess | `nnUNetv2_plan_and_preprocess` 等 |
| `seg train` | 训练指定 fold/config/trainer | `nnUNetv2_train` |
| `seg predict` | 单例或目录推理 | `nnUNetv2_predict*` |
| `seg val` / `seg eval` | 预测与 GT 评估 | `nnUNetv2_evaluate_*` 或自研指标 |
| `seg export` | TorchScript / ONNX / 其它部署格式 | 引擎导出能力 |
| `seg pack` / `seg unpack` | 模型包创建与安装 | bundle |
| `seg find-best` | 多配置/多 fold 选型（可选） | `find_best_configuration` 等 |

说明：`seg` 仅为示例入口名，可替换为项目品牌命令。

### 4.2 SDK 最小表面

```python
class SegModel:
    @classmethod
    def from_config(cls, path: str) -> "SegModel": ...
    @classmethod
    def from_bundle(cls, path: str) -> "SegModel": ...
    @classmethod
    def from_result_folder(cls, path: str) -> "SegModel": ...

    def prepare(self, **overrides) -> RunResult: ...
    def plan(self, **overrides) -> RunResult: ...
    def train(self, **overrides) -> RunResult: ...
    def predict(self, source: str, **overrides) -> PredictResult: ...
    def eval(self, pred_dir: str, gt_dir: str, **overrides) -> EvalResult: ...
    def export(self, fmt: str, **overrides) -> ExportResult: ...
    def pack(self, out_path: str) -> str: ...
```

约束：CLI 每个子命令最终只调用 SDK/pipeline 函数，禁止维护两套逻辑。

---

## 5. 配置规格（任务无关）

### 5.1 配置文件职责

一份任务配置应能完整描述「这个分割任务如何跑通主流程」，而不嵌入某器官专用几何逻辑。

### 5.2 建议字段结构

```yaml
# configs/example_task.yaml
project:
  name: example_seg
  seed: 42

paths:
  root: /data/segkit                 # 项目或数据根
  raw: ${paths.root}/nnUNet_raw
  preprocessed: ${paths.root}/nnUNet_preprocessed
  results: ${paths.root}/nnUNet_results
  runs: ${paths.root}/runs

dataset:
  id: 101
  name: ExampleOrgans
  file_ending: .nii.gz
  channel_names:
    0: CT
  labels:
    background: 0
    organ_a: 1
    organ_b: 2
  # 原始数据位置（prepare 用）
  source:
    adapter: nifti_folder            # 插件名
    images: /data/raw/images
    labels: /data/raw/labels
    split: { train: 0.8, val: 0.1, test: 0.1 }  # 若 adapter 支持

train:
  configuration: 3d_fullres          # 2d | 3d_fullres | 3d_lowres | 3d_cascade_fullres
  trainer: nnUNetTrainer
  fold: 0                            # 或 all / 0..4
  # 可选覆盖；默认尊重 nnU-Net plans
  # epochs: null

predict:
  checkpoint: checkpoint_best.pth
  backend: pytorch                   # pytorch | onnx
  tta: false
  save_probabilities: false

eval:
  metrics: [dice, hd95]
  labels: null                       # null = 使用 dataset.labels 除 background

export:
  formats: [torchscript, onnx]

postprocess:
  enabled: false
  name: generic_largest_cc           # 插件名；默认关闭以保持任务无关
  params: {}

bundle:
  include_onnx: false
```

### 5.3 配置合并规则

优先级从高到低：

1. CLI / SDK 显式参数
2. 任务 YAML
3. 内置默认值

路径支持相对路径：相对「配置文件所在目录」或 `paths.root` 解析（实现时二选一并写死）。

### 5.4 环境注入

执行 `plan` / `train` / `predict` 前，根据 `paths.*` 设置：

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

用户文档中不再要求手动 `export`。

---

## 6. Dataset Adapter（通用化核心）

### 6.1 为什么需要

nnU-Net 要求严格的 raw 布局与命名；真实任务数据形态多样。Adapter 负责把多样性收敛到引擎约定。

### 6.2 目标输出（引擎侧）

```text
nnUNet_raw/DatasetXXX_Name/
  dataset.json
  imagesTr/  case_XXX_0000.nii.gz
  labelsTr/  case_XXX.nii.gz
  imagesTs/  ...   # 可选
```

### 6.3 最小接口（建议）

```python
class DatasetAdapter(Protocol):
    name: str

    def validate(self, cfg: DatasetConfig) -> list[str]:
        """返回可读错误列表；空列表表示通过。"""

    def convert(self, cfg: DatasetConfig, out_raw_dataset_dir: Path) -> ConvertReport:
        """写出 imagesTr/labelsTr/dataset.json，返回转换报告。"""
```

### 6.4 内置 Adapter 建议集

| 名称 | 输入形态 |
|------|----------|
| `nifti_folder` | 图像/标签两个文件夹，一对一文件名 |
| `nifti_multilabel_merge` | 多通道/多文件类别 mask → 单通道多标签 |
| `msd` | Medical Segmentation Decathlon 布局 |
| `png_slices` | 2D 切片数据集（若需要） |

任务新增数据源时，优先新增 Adapter，而不是改 `prepare` 主流程。

### 6.5 ConvertReport（建议字段）

- 成功/跳过/失败 case 列表
- 标签值集合与映射表
- 输出 `dataset.json` 路径
- 可复现所需的关键统计（间距范围、尺寸范围等，可选）

---

## 7. 流水线阶段定义

### 7.1 prepare

输入：`dataset.source` + adapter  
输出：`DatasetXXX_Name` 于 `nnUNet_raw`  
副作用：写 `runs/prepare/expN/summary.json`

### 7.2 plan

输入：`dataset.id`  
动作：fingerprint → plan experiment → preprocess（可配置跳过某步）  
输出：`nnUNet_preprocessed/...` plans 与预处理缓存

### 7.3 train

输入：dataset id、configuration、fold、trainer  
动作：调用引擎训练  
输出：`nnUNet_results/...`；同时在 `runs/train/expN` 记录 args、日志链接、最终 checkpoint 指针

### 7.4 predict

输入：source（文件或目录）、权重来源（result folder 或 bundle）、predict 配置  
输出：分割结果目录 + `predict_args` + `summary.json`

### 7.5 eval

输入：预测目录 + GT 目录 + 标签列表 + 指标列表  
输出：逐 case / 逐 label 指标表与汇总

### 7.6 export / pack

- `export`：部署格式产物
- `pack`：创建可分发 bundle（见第 9 节）

各阶段均应支持 dry-run（只打印将执行的动作与解析后配置）。

---

## 8. Runs 与产物约定

### 8.1 目录布局

```text
runs/
  prepare/expN/
  plan/expN/
  train/expN/
  predict/expN/
  eval/expN/
  export/expN/
```

`expN` 自增，或允许 `--name` 指定。

### 8.2 每个 run 必备文件

| 文件 | 说明 |
|------|------|
| `args.yaml` | 解析合并后的完整配置快照 |
| `summary.json` | 机器可读摘要（状态、关键路径、指标入口） |
| `logs/` | 可选；或指向引擎日志的软链 |

### 8.3 `summary.json` 最小 schema（示例）

```json
{
  "status": "success",
  "command": "predict",
  "started_at": "...",
  "finished_at": "...",
  "paths": {
    "input": "...",
    "output": "...",
    "weights": "..."
  },
  "metrics": null,
  "artifacts": []
}
```

下游系统只依赖 `summary.json` + 约定输出目录，不解析引擎内部结构。

---

## 9. 模型包（Bundle）规格

### 9.1 目标体验

对标 YOLO 的单权重文件心智：用户拿到一个 bundle 即可 `predict`，无需理解 nnU-Net results 树。

### 9.2 建议包含内容

```text
model.bundle/   # 或 zip
  manifest.json
  dataset.json          # 推理所需标签/通道信息
  plans.json
  fold_X/
    checkpoint_best.pth
  # 可选
  model.onnx
  postprocess.yaml      # 若打包时启用了通用后处理默认项
```

### 9.3 `manifest.json` 建议字段

- 框架版本 / 引擎版本
- dataset id/name、configuration、fold 列表
- 默认 checkpoint 文件名
- 创建时间、git commit（可选）
- 支持的 `file_ending`、模态数

### 9.4 命令语义

- `seg pack`：从 result folder 生成 bundle
- `seg unpack`：安装到本地可 predict 的标准目录
- `seg predict -w bundle`：内部完成定位 plans/checkpoint

---

## 10. 插件体系

### 10.1 后处理插件

主流程默认 **不启用** 任务专用后处理，以保持通用性。

接口建议：

```python
class PostprocessPlugin(Protocol):
    name: str
    def run(self, seg_path: Path, out_dir: Path, params: dict) -> PostprocessReport: ...
```

内置通用插件示例：

- `identity`（透传）
- `generic_largest_cc`（每标签最大连通域）
- `generic_min_size`（按体素数/体积过滤）

器官级、实例重编号、几何坐标系等均视为 **外部插件**，不进入核心包硬编码。

### 10.2 评估插件

默认提供体素分割常用指标（Dice、HD95 等）。允许替换实现，但输出报表 schema 保持稳定。

### 10.3 注册方式

- 入口点 / 装饰器注册表：`register_adapter("nifti_folder")`
- 配置里用字符串名称引用
- 未知名称时 `doctor` / 运行前失败并给出已注册列表

---

## 11. Doctor 与可复现性

`seg doctor` 建议检查：

1. Python / 关键依赖版本
2. CUDA / GPU 可见性（若声称使用 GPU）
3. `paths.*` 可写可读
4. 数据集目录与 `dataset.json` 一致性（Tr 图像与标签配对）
5. 指定 bundle / result folder 是否可加载
6. 可选：磁盘空间粗检

可复现性要求：

- 每次 run 落盘完整 `args.yaml`
- 记录引擎版本与配置 hash
- 文档明确：哪些随机性由引擎控制（增强、多进程等），本层不额外引入隐式全局状态

---

## 12. 与 nnU-Net 官方命令映射

| 本层命令 | 典型映射 |
|----------|----------|
| `prepare` | 无官方等价；本层 Adapter 产出 raw dataset |
| `plan` | `nnUNetv2_plan_and_preprocess`（或拆分的 fingerprint/plan/preprocess） |
| `train` | `nnUNetv2_train` |
| `predict` | `nnUNetv2_predict` / `predict_from_modelfolder` / ONNX 变体 |
| `eval` | `nnUNetv2_evaluate_folder` / `evaluate_simple` 或自研 |
| `export` | 项目/引擎提供的 export 入口 |
| `find-best` | `nnUNetv2_find_best_configuration` 等 |
| `pack` | 可参考官方 model zip 分享，或自定义 bundle |

实现策略：优先 **子进程调用稳定 CLI** 或 **调用 Python API**（二选一，文档中固定一种，避免混用导致环境不一致）。

---

## 13. 非目标与边界

本设计明确不包含：

1. 自研替代 U-Net 搜索/训练循环
2. 检测/关键点等非语义分割任务的一等 API
3. 某垂直临床业务的专用几何后处理（可作为插件存在）
4. 强制 Web UI / 标注平台（可后续扩展）
5. 保证跨引擎（MONAI、其它）第一天兼容——可留 `engine:` 扩展点，但首期只保证 nnU-Net v2

---

## 14. 分阶段交付建议（实现时参考）

| 阶段 | 交付物 | 价值 |
|------|--------|------|
| P0 | config + env 注入 + `doctor` + `predict`/`train`/`plan` CLI | 立刻减少命令与环境摩擦 |
| P1 | `prepare` + 1～2 个内置 Adapter + `runs/` 约定 | 换任务成本下降 |
| P2 | SDK 与 CLI 同源 + `pack`/`unpack` | 接近 YOLO 开箱体验 |
| P3 | eval 报表稳定化 + 通用 post 插件 + golden bench 接口 | 可维护、可 CI |
| P4 | 多后端推理、模板 `init`、可选服务化 | 部署与团队协作增强 |

---

## 15. 验收清单（通用任务视角）

- [ ] 同一套代码，仅改 YAML + 数据路径，可跑通任务 A 与任务 B 的 prepare→plan→train→predict→eval
- [ ] 不手动设置 `nnUNet_*` 环境变量也能训练/推理
- [ ] `predict -w bundle` 在干净环境可运行
- [ ] `runs/*/expN/args.yaml` 与 `summary.json` 始终存在
- [ ] 专用后处理未启用时，核心包无器官/病种硬编码分支
- [ ] CLI 与 SDK 对同一配置产生一致输出路径语义

---

## 16. 术语表

| 术语 | 含义 |
|------|------|
| 引擎 | nnU-Net v2 训练/推理实现 |
| 编排层 | 将引擎能力组织为稳定产品流程的代码 |
| Adapter | 原始数据到 nnU-Net Dataset 的转换插件 |
| Bundle | 可分发的最小可推理模型包 |
| Run | 一次命令执行的产物目录与元数据 |
| 插件 | 可注册、可配置的扩展点实现 |

---

## 17. 实现状态与快速开始（segkit）

本仓库已落地编排层包 **`segkit`**（CLI 入口：`seg`），按文档分阶段实现。

### 17.1 当前能力

| 阶段 | 状态 | 内容 |
|------|------|------|
| P0 | 已实现 | YAML 配置、环境注入、`doctor` / `plan` / `train` / `predict`、`runs/` |
| P1 | 已实现 | `prepare` + `nifti_folder` / `nifti_multilabel_merge` |
| P2 | 已实现 | `SegModel` SDK、`pack` / `unpack` |
| P3 | 已实现 | `eval`、通用 post 插件、`bench` |
| P4 | 未做 | ONNX 一等后端、服务化 |

引擎调用方式：子进程执行 `nnUNetv2_*`（不改训练核心）。

### 17.2 安装

```bash
# 先按原流程安装本地 nnUNet（pip install -e ./nnUNet）与 PyTorch
pip install -e ".[dev]"
# 或仅编排层依赖
pip install -e .
```

依赖：`typer`、`PyYAML`、`rich`。Adapter 合并多标签需要 `nibabel`（`pip install -e ".[adapters]"`）。

### 17.3 常用命令

```bash
seg init --out configs/my_task.yaml --name MyOrgans --id 101
seg doctor --config configs/example_task.yaml

seg prepare --config configs/example_task.yaml
seg plan --config configs/example_task.yaml --dry-run
seg train --config configs/example_task.yaml --dry-run
seg predict -c configs/example_task.yaml -i /path/imagesTs -o /path/pred -w /path/model_folder --dry-run

seg eval -c configs/example_task.yaml --gt /path/gt --pred /path/pred --dry-run
seg postprocess --seg pred.nii.gz --out-dir post_out --plugin identity
seg pack -m /path/model_folder -o model.bundle.zip
seg bench --spec benches/example.json --out runs/bench/report.json
```

每次执行会在 `runs/<command>/expN/` 写入 `args.yaml` 与 `summary.json`。

### 17.4 Python SDK

```python
from segkit import SegModel

m = SegModel.from_config("configs/example_task.yaml")
m.plan(dry_run=True)
m.train(dry_run=True)
m.predict(source="/path/imagesTs", dry_run=True)
```

### 17.5 示例配置

见 [`configs/example_task.yaml`](../configs/example_task.yaml)。相对路径相对于**配置文件所在目录**解析。

### 17.6 测试

```bash
pytest tests/ -q
```

单测覆盖配置合并、官方 CLI argv 形态、Adapter 拷贝与 bundle 打包，不依赖 GPU。

