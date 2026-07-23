# nnunet-sdk

本仓库提供 **segkit**：一层 YOLO 风格的编排工具，把 nnU-Net 的 prepare / plan / train / predict / eval 收成统一 CLI（`seg`）和 Python SDK。  
**不替换 nnU-Net 训练核心**，底层通过子进程调用 `nnUNetv2_`*。本仓库同时 vendored 了带自定义 export CLI 的本地 `nnUNet/`。

设计文档见：[docs/nnunet-yolo-style-pipeline.md](docs/nnunet-yolo-style-pipeline.md)

---

## 1. 安装

在仓库根目录：

```bash
# 1) PyTorch（按你的 CUDA 环境自行安装）
# 2) 本地 nnU-Net
pip install -e ./nnUNet

# 3) segkit
pip install -e .

# 可选：开发测试 / 多标签 Adapter
pip install -e ".[dev]"
pip install -e ".[adapters]"   # 需要 nibabel
```

安装成功后应能运行：

```bash
seg --help
seg --version
```

---

## 2. 写任务配置

```bash
seg init --out configs/my_task.yaml --name MyOrgans --id 101
```

也可直接改示例配置：[configs/example_task.yaml](configs/example_task.yaml)

**重要：** YAML 里的相对路径，都是相对于**配置文件所在目录**解析的。

最少要填：


| 字段                                                         | 含义                 |
| ---------------------------------------------------------- | ------------------ |
| `paths.root` / `raw` / `preprocessed` / `results` / `runs` | nnU-Net 三路径 + 实验目录 |
| `dataset.id` / `name` / `labels`                           | 数据集编号与标签           |
| `dataset.source`                                           | `prepare` 用的原始数据位置 |
| `train.configuration` / `fold` / `trainer`                 | 训练配置               |
| `predict.`*                                                | 推理输入输出、权重目录等       |


无需手动 `export nnUNet_raw=...`：执行 `plan` / `train` / `predict` 时会按配置自动注入。

---

## 3. 数据准备（Gate）

未通过 Gate 前，不要跑 `seg plan` / `seg train`。

### 3.1 目标：官方 raw 布局

`paths.raw` 下必须有与 YAML 一致的 Dataset 目录：


| 项           | 契约                                           | 例子                                       |
| ----------- | -------------------------------------------- | ---------------------------------------- |
| raw 根目录     | YAML `paths.raw` ≡ 官方 `nnUNet_raw`           | `/data/nnunet_data/nnUNet_raw`           |
| Dataset 目录名 | `Dataset{id:03d}_{name}`，与 YAML 完全一致         | `id=101,name=SPINE` → `Dataset101_SPINE` |
| 训练图像        | `{CASE_ID}_{XXXX}{file_ending}`，`XXXX` 四位通道号 | `RibFrac100_0000.nii.gz`                 |
| 训练标签        | `{CASE_ID}{file_ending}`（无通道号）               | `RibFrac100.nii.gz`                      |
| 元数据         | 目录内必须有 `dataset.json`                        | 见下表                                      |
| ID 占用       | 同一 `paths.raw` 下 ID 唯一                       | 已有 `Dataset100`_* 就勿再用 100               |


```text
{paths.raw}/
└── Dataset{ID:03d}_{Name}/
    ├── dataset.json          # 必需
    ├── imagesTr/             # 必需
    ├── labelsTr/             # 必需
    └── imagesTs/             # 可选（训练不用）
```

单通道最小示例：

```text
Dataset101_SPINE/
├── dataset.json
├── imagesTr/
│   ├── caseA_0000.nii.gz
│   └── caseB_0000.nii.gz
└── labelsTr/
    ├── caseA.nii.gz
    └── caseB.nii.gz
```

多通道时，同一 `CASE_ID` 必须齐套：`_0000`、`_0001`、… 与 `dataset.json.channel_names` 一一对应。

### 3.2 `dataset.json` 契约


| 字段                              | 必需  | 约束                                                     |
| ------------------------------- | --- | ------------------------------------------------------ |
| `channel_names`                 | 是   | key 为 `"0"`, `"1"`, …；值为 `"CT"` 时走 CT 归一化，其它多为 z-score |
| `labels`                        | 是   | `name → int`；**背景必须是 0**；前景连续整数                        |
| `numTraining`                   | 是   | 等于 `labelsTr` 中 case 数                                 |
| `file_ending`                   | 是   | 图像与标签同一后缀，如 `.nii.gz`                                  |
| `overwrite_image_reader_writer` | 否   | 如 `"SimpleITKIO"`                                      |


```json
{
  "channel_names": { "0": "CT" },
  "labels": { "background": 0, "foreground": 1 },
  "numTraining": 5,
  "file_ending": ".nii.gz"
}
```

YAML 的 `dataset.labels` / `channel_names` / `file_ending` 应与该文件保持一致（避免 eval/文档漂移）。

### 3.3 数据质量约束（plan 前必须满足）

- 每个 `labelsTr/{CASE_ID}.*` 在 `imagesTr` 中有对应 `{CASE_ID}_0000.*`（多通道则全部 XXXX 齐套）
- 同 case：图像与标签 shape/spacing 一致
- 标签为整型；背景 0；类别连续
- 全数据集通道集合与顺序固定（训练和推理相同）
- 无损格式（`.nii.gz` / `.nrrd` / `.mha` / `.png` 等；不要用 `.jpg`）

### 3.4 如何到位（二选一）


| 现状                       | 动作                                             | 下一步                  |
| ------------------------ | ---------------------------------------------- | -------------------- |
| 已是官方 raw 布局              | 改 YAML：`paths.raw`、`dataset.id`、`dataset.name` | §3.6 验收 → `seg plan` |
| 普通 `images/` + `labels/` | `seg prepare`（§3.5）生成 `DatasetXXX_Name`        | 同上                   |


### 3.5 `seg prepare`：非标准输入 → 官方 raw

**仅当**数据还不是 §3.1 布局时使用。会生成：

```text
{paths.raw}/Dataset{id:03d}_{name}/
```

选哪种 adapter，取决于你的**标签长什么样**。

#### 3.5.1 `nifti_folder`：已经是「一图一标签」

适用：每个病例只有**一个**标签文件，里面已经用整数区分类别（0=背景，1=器官A，2=器官B…）。


| 你现在有的                                        | prepare 之后                   |
| -------------------------------------------- | ---------------------------- |
| `images/caseA.nii.gz`（或 `caseA_0000.nii.gz`） | `imagesTr/caseA_0000.nii.gz` |
| `labels/caseA.nii.gz`                        | `labelsTr/caseA.nii.gz`      |


```yaml
dataset:
  id: 101
  name: MyTask
  file_ending: .nii.gz
  labels: { background: 0, organ: 1 }
  source:
    adapter: nifti_folder
    images: /path/to/images
    labels: /path/to/labels
```

```bash
seg prepare --config configs/my_task.yaml
# 然后做 §3.6 验收
```

#### 3.5.2 `nifti_multilabel_merge`：标签被拆成多个「是/否」文件夹

适用：标注不是一张多标签图，而是**每个类别单独一个文件夹**，文件里只有 0 和 1（有该结构=1，没有=0）。

例如原始数据是：

```text
images/caseA.nii.gz
masks_liver/caseA.nii.gz    # 只有肝的位置是 1
masks_kidney/caseA.nii.gz   # 只有肾的位置是 1
```

nnU-Net 需要合成**一个**标签文件，例如肝=1、肾=2：

```text
labelsTr/caseA.nii.gz       # 背景 0，肝 1，肾 2
```

`nifti_multilabel_merge` 做的就是这件事：按 `class_dirs` 顺序，把多个二值 mask 合并成一张多类别标签，再写入官方目录。  
（读写 NIfTI 需要安装 `nibabel`：`pip install -e ".[adapters]"`）

```yaml
dataset:
  id: 101
  name: MyTask
  file_ending: .nii.gz
  labels:
    background: 0
    liver: 1      # 对应 class_dirs 第 1 个目录
    kidney: 2     # 对应 class_dirs 第 2 个目录
  source:
    adapter: nifti_multilabel_merge
    images: /path/to/images
    class_dirs:
      - /path/to/masks_liver
      - /path/to/masks_kidney
```

若标签本来就是单文件多类别，用 **3.5.1**，不要用 3.5.2。

### 3.6 验收

```bash
seg doctor --config configs/my_task.yaml
```

检查依赖、`nnUNetv2_*` 是否在 PATH、路径可写、以及（若已有 Dataset）图像/标签配对。

常见失败对照：


| 现象                                         | 原因                                | 处理           |
| ------------------------------------------ | --------------------------------- | ------------ |
| `Could not find a dataset with the ID 101` | 无 `Dataset101_*`，或 id/name 与目录不一致 | 建目录或改 YAML   |
| plan 报 missing labels / integrity          | case 对不齐或标签非连续                    | 修命名 / 标签值    |
| 相对路径跑到 `configs/` 下                        | YAML 相对路径相对配置文件目录                 | 改绝对路径或 `../` |


---

## 4. 训练与推理流程

数据 Gate（§3）通过后再跑本节。

### 4.1 推荐命令顺序

```bash
# 1) 规划与预处理（--dry-run 只打印将执行的命令，不真正跑）
seg plan --config configs/my_task.yaml --dry-run
seg plan --config configs/my_task.yaml

# 2) 训练
seg train --config configs/my_task.yaml --dry-run
seg train --config configs/my_task.yaml
#    覆盖 fold / configuration 示例：
seg train --config configs/my_task.yaml --fold 0 --configuration 3d_fullres

# 3) 推理
#    --weights 可省略：自动解析为
#    {paths.results}/Dataset{id}_{name}/{trainer}__{plans}__{configuration}
#    仍可用 -w 显式覆盖
seg predict \
  --config configs/my_task.yaml \
  --source /path/to/imagesTs \
  --output /path/to/preds

#    推理后原地做最大连通域（--output 里直接是处理后的结果）
seg predict -c configs/my_task.yaml -i /path/to/imagesTs -o /path/to/preds \
  --postprocess generic_largest_cc
#    也可在 YAML 写 predict.postprocess: generic_largest_cc

# 4) 导出部署模型（TorchScript .pt + ONNX .onnx；不做分割推理）
#    成功时仅提示「导出成功」及输出目录；产物见该目录下 model.pt / model.onnx
seg export --config configs/my_task.yaml

# 5) 打包权重为可分发 bundle
seg pack --config configs/my_task.yaml --out model.bundle.zip

# 6) 评估（预测结果 vs GT）
seg eval \
  --config configs/my_task.yaml \
  --gt /path/to/labelsTs \
  --pred /path/to/preds

# 7) 后处理（plugins: identity | generic_largest_cc | generic_min_size）
seg postprocess --seg /path/to/preds/case.nii.gz --out-dir post_out --plugin generic_largest_cc
```

每次执行（含 `--dry-run`）还会在 `paths.runs` 下写入实验元数据 `runs/<command>/expN/`：

- `args.yaml`：合并后的配置快照  
- `summary.json`：状态、argv、env、错误摘要

### 4.2 训练产物

`seg train` 本身不另建一套权重目录，而是走 nnU-Net 默认布局，根目录为 YAML 的 `**paths.results**`（即官方 `nnUNet_results`）。

完整路径由配置拼出（`predict` / `export` / `pack` 未显式指定权重时也默认找这里）：

```text
{paths.results}/
└── Dataset{id:03d}_{name}/
    └── {trainer}__{plans}__{configuration}/
        ├── dataset.json
        ├── plans.json
        ├── fold_0/                    # 对应 train.fold；多 fold 则有 fold_1 …
        │   ├── checkpoint_best.pth    # 验证集最优（推理默认常用）
        │   ├── checkpoint_final.pth   # 训练结束时的权重
        │   └── training_log_*.txt     # 训练日志
        └── …
```

例子（与 `configs/example_task.yaml` 一致时）：

```text
…/nnUNet_results/Dataset101_SPINE/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres/fold_0/
```

推理时 `seg predict` 若不传 `-w`，会自动解析到上述 `{trainer}__{plans}__{configuration}` 目录。

### 4.3 命令一览


| 命令                        | 作用                                 |
| ------------------------- | ---------------------------------- |
| `seg doctor`              | 环境与路径检查                            |
| `seg init`                | 生成任务 YAML 模板                       |
| `seg prepare`             | 原始数据 → nnU-Net Dataset 布局          |
| `seg plan`                | `nnUNetv2_plan_and_preprocess`     |
| `seg train`               | `nnUNetv2_train`                   |
| `seg predict`             | 推理；可选 `--postprocess` 对输出目录原地后处理   |
| `seg eval`                | `nnUNetv2_evaluate_simple`         |
| `seg postprocess`         | 通用后处理插件（单文件）                       |
| `seg pack` / `seg unpack` | 模型打包 / 解包                          |
| `seg export`              | `nnUNetv2_export_from_modelfolder` |
| `seg bench`               | 按 JSON 规格跑 Dice 回归                 |


查看子命令帮助：

```bash
seg predict --help
```

---

## 5. 模型打包

```bash
# 将含 fold_* 的训练结果目录打成 zip
seg pack -m /path/to/trained_model_folder -o model.bundle.zip

# 解包
seg unpack -b model.bundle.zip -d /path/to/install_dir

# 用解包后的目录推理
seg predict -i ... -o ... -w /path/to/install_dir/model.bundle
```

---

## 6. Python SDK

```python
from segkit import SegModel

m = SegModel.from_config("configs/my_task.yaml")

m.doctor()
m.prepare()
m.plan(dry_run=True)
m.train(fold=0)          # 通过 overrides 传入嵌套字段时用字典更稳妥
m.predict(source="/data/imagesTs")
m.eval(pred_dir="/data/pred", gt_dir="/data/gt")
m.pack("model.bundle.zip")
```

从已有权重目录加载：

```python
m = SegModel.from_result_folder("/path/to/model_folder", config_path="configs/my_task.yaml")
m.predict(source="/data/imagesTs")
```

---

## 7. 回归 bench（可选）

规格示例见 [configs/bench_example.json](configs/bench_example.json)：

```bash
seg bench --spec configs/bench_example.json --out runs/bench/report.json
```

---

## 8. 测试

```bash
pytest tests/ -q
```

不依赖 GPU；覆盖配置合并、官方 CLI 参数拼装、Adapter、bundle。

---

## 9. 常见问题

**Q: `seg doctor` 报 nnU-Net / torch 找不到？**  
先 `pip install -e ./nnUNet` 并安装 PyTorch，保证 `which nnUNetv2_train` 有输出。

**Q: 路径怎么老跑到 `configs/` 下面？**  
相对路径相对 YAML 所在目录。示例配置用 `root: ..` 指向仓库根；或改成绝对路径。

**Q: 和脊柱后处理 / 坐标导出什么关系？**  
无关。那些是业务插件，不在 `segkit` 主流程里。通用后处理只用 `identity` / `generic_largest_cc` / `generic_min_size`。

**Q: 想先看会执行什么命令？**  
加 `--dry-run`，只打印 argv 并写 `runs/.../summary.json`，不真正训练/推理。

**Q: `Could not find a dataset with the ID 101`？**  
未通过 §3 Gate：`paths.raw` 下没有匹配的 `Dataset101_<name>`。建目录或改 `id`/`name`。

**Q: 相对路径解析到哪？**  
相对 **YAML 所在目录**。生产配置建议 `paths.`* 用绝对路径。