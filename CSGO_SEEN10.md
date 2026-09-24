# X-VLA 接入 CSGO Benchmark v2 Seen-10

本文是 X-VLA 接入 CSGO Benchmark v2 的运行说明。当前只覆盖 Seen-10 定位任务，不包含生成任务，也不包含 CrossMap-4。命令均从本机的 X-VLA 项目根目录执行，例如 `~/task/X-VLA`；不依赖具体用户名。

## 1. 背景、目标和实验范围

CSGO Benchmark v2 用第一视角图像（FPV）、对应地图 radar 和定位指令，要求模型预测相机在地图中的位置和朝向。X-VLA 的接入目标是复用官方数据划分、样本 ID、5D 位姿定义和官方评测器，形成可重复的 Seen-10 定位训练、推理和评测流程。

本项目保留三套可运行实验：

| 实验 | 用途 | 配置 | 输出目录 | 默认 checkpoint |
| --- | --- | --- | --- | --- |
| 初次接入 legacy | 复现工作区最初的 X-VLA 接入行为 | `configs/csgo_seen10.json` | `outputs/csgo_benchmark_v2_seen10/X-VLA/seed_0` | `best` |
| 原 aligned（连接器可训练） | 保留已有 aligned 设置，作为连接器训练策略对照 | `configs/csgo_seen10_xvla_fair.json` | `outputs/csgo_benchmark_v2_seen10/X-VLA-fair/seed_0` | `last` |
| 新 aligned-frozen-vl（连接器冻结） | 视觉—语言连接器冻结，与 `exp32_loc` 对齐 | `configs/csgo_seen10_xvla_fair_frozen_vl.json` | `outputs/csgo_benchmark_v2_seen10/X-VLA-fair-frozen-vl/seed_0` | `last` |

公平对比的主要对象是 UniLIP `exp32_loc` 定位-only 实验；UniLIP `exp32` 的定位结果作为联合生成+定位的次要参考。生成任务、`exp32_gen` 和其他生成模型不在本次接入范围内。三套 X-VLA 实验共享官方 Seen-10 数据、外部 5D 定位目标、测试协议和评测指标；X-VLA 原生的 action head、去噪目标、推理步数和优化稳定性设置作为模型差异保留。

## 2. 数据、地图和输入输出边界

默认数据目录为项目相邻的 `../UniLIP/data/csgo_benchmark_v2`。启动脚本优先使用相邻的 `../csgo_benchmark_v2_eval_general` 官方评测器；该目录不存在时使用仓库自带的 `csgo_benchmark_v2_eval`。定位评测默认使用项目 `.venv`，无需另外创建 UniLIP 环境。数据和样本顺序由官方 manifest 决定，不重新划分数据、不从测试集计算统计量。

Seen-10 使用以下 10 张地图：`cs_agency`、`cs_italy`、`de_ancient`、`de_anubis`、`de_dust2`、`de_inferno`、`de_mirage`、`de_nuke`、`de_overpass`、`de_train`。

固定 split 为：

- `seen_train`：约 50,000 条训练记录；
- `seen_validation`：约 5,000 条验证记录；
- `seen_discrete_test`：20,000 条测试记录，每张地图 2,000 条。

模型输入只有：

- FPV RGB 图像；
- 对应地图 radar；
- localization instruction，其中可以包含地图名称。

模型不接收 GT 坐标、历史位姿、测试集统计量或其他额外状态。外部定位目标是单步 5D pose：`[x, y, z, pitch, yaw]`，`action_horizon=1`。推理文件保存归一化后的 5D 预测，官方评测器负责统一反归一化并计算物理空间指标。

## 3. 三套实验的技术行为

### 初次接入 legacy

legacy 使用 `legacy_reset_dummy` action 路径，保留初次接入时的 5D 到原生 action 宽度的处理方式，训练预算为 50,000 个 optimizer updates。它用于复现已有工作区结果和检查新代码对旧流程的兼容性，不作为当前公平对比的主结果。

legacy 推理默认读取 validation 选择的 `best` checkpoint。旧配置的输出目录和默认参数保持独立，因此不会覆盖 aligned 实验的结果。

### 两种公平对比 aligned 的共同设置

aligned 数据侧先生成 5D pose，再在进入 `XVLA.forward` 前一次性补成 `[5D pose, 15D zeros]` 的 20D action。训练和推理的噪声、`x_t`、预测值以及原生 10 步 refinement 始终使用完整 20D；推理过程中不重复清零后 15D，最终只裁剪前 5D 作为定位结果。

训练继续使用 X-VLA 原生的 clean-action denoising regression。主 loss 只对前 5D 有效定位维度计算并乘以 100，后 15D 仍参与原生 20D action 计算，但不增加直接监督。原生 action encoder 的输入接口保持为 `action20 + zero_proprio20 + time32 = 72D`。这里的 `zero_proprio20` 是为了保留预训练 action encoder 形状而使用的全零槽位，不包含真实 robot state，也不提供额外监督。

aligned 的外部归一化固定为：

```text
x'     = x / 1024
y'     = y / 1024
z'     = (z - z_min_map) / (z_max_map - z_min_map)
pitch' = pitch / (2*pi)
yaw'   = yaw / (2*pi)
```

三套 split 使用同一套按地图冻结的发布标定。X-VLA 原生路径不使用 qnorm、clamp 或额外的 Z epsilon；这保持为 X-VLA 与 UniLIP pi0.5 定位 head 之间的模型差异。

aligned 训练集使用 X-VLA 原生 `ColorJitter(0.2, 0.2, 0.2, 0)`，验证和测试使用确定性预处理，图像尺寸为 224。视觉 encoder 冻结；LLM 和 action expert 的 base 参数冻结并使用 `r=32`、`alpha=64`、`dropout=0.05`、`bias=none` 的 LoRA；action connector、action encoder/decoder、action norm、action head 和 soft prompt 全量训练。视觉—语言 connector 的训练策略是两种 aligned 实验之间的方法差异，见下表；验证与保存时间点另见第 4 节。推理仍使用 X-VLA 原生 10 步，不为了复制 UniLIP 定位 head 而改成其他步数。

legacy 与原 aligned 的关键差异如下；新 aligned-frozen-vl 在此基础上冻结视觉—语言连接器，并将验证与保存间隔改为 4,000 updates：

| 项目 | 初次接入 legacy | 原 aligned（连接器可训练） |
| --- | --- | --- |
| 定位预算 | 50,000 updates，单进程有效 batch 4 | 19,500 updates，有效 batch 128 |
| Action 适配 | 推理每轮只保留前 5D，再把后 15D 补零 | 进入 forward 前补零一次，完整 20D 参与训练加噪和全部推理迭代 |
| 主 loss | 前 5D MSE × 100 | 前 5D MSE × 100 |
| 可训练参数 | 全部参数保留梯度；前 1,000 steps 仅 soft prompt/action head 使用非零 LR，之后全模型训练 | vision encoder 冻结；LLM/action expert 使用 LoRA；connector 和 action 转换层全量训练 |
| 图像增强 | 关闭 | 仅训练集启用原生 ColorJitter |
| 学习率 | 原生 warmup + cosine 设置 | X-VLA 稳定的分阶段常数学习率设置 |
| 保存频率 | 每 10,000 updates，共 5 次 | 每 3,900 updates，共 5 次 |
| 主结果 checkpoint | validation `best` | 训练结束 `last`；`best` 仅作附加结果 |

aligned 对齐的是 `exp32_loc` 的数据、输入信息、5D 目标、定位样本暴露量、有效 batch、optimizer update 数、checkpoint 密度和评测协议。X-VLA 的 20D action head、clean-action denoising objective、原生 10 步求解以及优化器稳定设置仍按模型原生行为保留。它们属于需要在结果表中披露的模型差异。

### 新实验 aligned-frozen-vl：冻结视觉—语言连接器

新实验显式设置 `training.freeze_vision_language_connector=true`，冻结 Florence 视觉特征进入语言 backbone 之前的 `vlm.image_projection` 和 `vlm.image_proj_norm`。这两部分全程 `requires_grad=false`，不进入 optimizer，也不插入 LoRA；训练到第 1,000 个 update 后仍保持冻结。配置未设置该开关时默认为 `false`，原 aligned 行为保持兼容。

| 模块/设置 | 原 aligned | 新 aligned-frozen-vl | UniLIP exp32_loc 对应功能 |
| --- | --- | --- | --- |
| 视觉 encoder | 冻结 | 冻结 | 冻结 |
| 视觉—语言连接器 | 全量可训练，前 1,000 updates 的 LR 为 0 | 全程冻结 | InternVL multimodal projector 冻结 |
| 语言 backbone | base 冻结，LoRA 可训练 | 相同 | base 冻结，LoRA 可训练 |
| Action expert | base 冻结，LoRA 可训练 | 相同 | base 冻结，LoRA 可训练 |
| Action connector / 维度转换层 | 全量可训练 | 相同 | 对应定位连接/转换层可训练 |
| X-VLA soft prompt | 可训练 | 相同 | 不强行对应 |
| 验证与保存 | 每 3,900 updates | 每 4,000 updates，最终 19,500 补做一次 | 维持约 5 次 checkpoint 搜索机会 |
| 数据、增强、action 目标、LR、训练预算、推理与评测 | 本文 aligned 设置 | 相同 | 仅按功能比较；模型原生差异仍保留 |

`transformer.vlm_proj` 和 `transformer.aux_visual_proj` 是连接到 action expert 的投影，属于 action connector，仍可训练。它们不属于本次冻结的视觉—语言连接器。

新实验的 LoRA `modules_to_save` 排除 `vlm.image_proj_norm`，`extra_trainable_parameters` 为空；其余可训练模块沿用原 aligned。训练入口生成的 `parameter_audit.json` 应显示 `vision_language_connector` 的可训练参数数为 0，optimizer 中不应出现这一组。原生模型规模和前向结构保持不变，可训练参数数减少。

新实验从 `pretrained/X-VLA-Pt` 初始化，使用独立输出目录；不能从原 aligned 的 CSGO checkpoint warm-start。`last` 用于主结果，`best` 仅作为 validation-best 附加结果。该实验用于比较连接器冻结策略；结果对比时也应注明验证/保存时间点的差异，其他模型原生差异仍然保留。

## 4. aligned 训练预算和 checkpoint

aligned 的有效定位 batch 为 128 个样本，使用梯度累计适配可见 GPU 数量；总训练量为 19,500 个 optimizer updates，约 50 个 epoch，每个 epoch 390 个 update。每个 epoch 尾部不满有效 batch 的 80 条样本丢弃，不因 GPU 数量改变有效 batch 或总更新数。

使用 AdamW、学习率 `1e-4`、betas `(0.9, 0.95)`、weight decay `0`、gradient clipping `1.0` 和 BF16。前 1,000 个 update 仅训练 soft prompt 和 action head，之后启用对应 aligned 配置中其余可训练模块；新实验的视觉—语言连接器始终冻结。

| 实验 | 验证/保存间隔（optimizer updates） | 实际验证与保存的 step |
| --- | --- | --- |
| 原 aligned | 3,900 | 3,900、7,800、11,700、15,600、19,500 |
| 新 aligned-frozen-vl | 4,000 | 4,000、8,000、12,000、16,000、19,500 |

新实验的 `eval_interval` 和 `save_interval` 均为 4,000；训练结束时即使未满间隔，也会验证并保存 step 19,500。共保留这 5 个 step 目录。`best` 链接到其中 validation 最优的 step；`late` 与兼容现有命令的 `last` 链接到最新已保存的 step，训练完成后均指向 `step_00019500`。这些链接不额外复制 checkpoint。原 aligned 继续使用既有的 `best`/`last` 指针。

论文或主表中的 aligned 主结果使用训练结束的 `last` checkpoint。若额外报告 `best`，必须明确标为 validation-best，不能查看测试集后选择。

## 5. 环境和权重准备

以下准备命令对三套实验共用，每台服务器分别准备一次。Git 同步项目代码后，在新服务器重新创建环境并下载权重；不要跨服务器复制 `.venv`。最小数据集需要另行放置到第 2 节所列目录，包含 JSON、images、radars 和发布标定。数据、权重和训练产物不随 Git 同步。

```bash
cd ~/task/X-VLA
bash scripts/setup_csgo_seen10.sh
bash scripts/download_csgo_checkpoint.sh
```

准备脚本优先复用项目中兼容的隔离 `.venv`；新建环境时自动寻找 Python 3.12、3.11 或 3.10。当前 Conda 环境的 Python 也可以作为候选，不要求系统提供名为 `python3.12` 的命令。如果没有兼容解释器但可以使用 Conda，脚本会在项目 `.cache` 下准备 Python 3.12，再创建 `.venv`，无需 sudo，也不修改 Conda base。已有但损坏或版本不兼容的 `.venv` 会给出诊断，不自动删除。

如需显式选择解释器，可将环境准备命令替换为一行：

```bash
PYTHON_BIN=/path/to/python3.11 bash scripts/setup_csgo_seen10.sh
```

依赖继续使用固定的 PyTorch 2.7.1 / torchvision 0.22.1 CUDA 12.8 组合；Python 版本自动选择不会改变实验配置或 PyTorch 版本。服务器仍需提供兼容的 NVIDIA 驱动。脚本只复用与当前 Python 和平台兼容的本地 wheel，其余依赖按固定版本安装。

权重下载脚本依赖环境准备先成功完成，并需要 `curl`。如果第一条准备命令失败，应先处理该错误，再运行权重下载。

默认目录布局下，第 6 节的命令在两台服务器上完全相同。若数据放在其他位置，在训练、推理和评测命令后都追加相同的 `--data-root /path/to/csgo_benchmark_v2`。外部评测器或 Python 有特殊安排时，可在单条评测命令前指定 `SHARED_EVAL_DIR` 或 `UNILIP_PYTHON`；默认使用项目环境，无需这些变量。

权重下载完成后，原始权重位于 `pretrained/X-VLA-Pt`。三套实验都必须从该原始 X-VLA 权重开始，不能用已经见过 CSGO Benchmark v2 的 checkpoint 继续初始化。

## 6. 直接执行命令

训练入口会拒绝覆盖已有的非空输出目录。以下以 `seed=0` 展示标准命令；当前 legacy 的 `seed_0` 已有完整结果，如需重新训练应改用新的 seed 或显式指定新的 `--output-root`。

### 6.1 初次接入 legacy

脚本默认配置就是 `configs/csgo_seen10.json`，下面三行不需要额外传入配置参数：

```bash
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0
bash scripts/run_csgo_seen10.sh eval --seed 0
```

默认推理是一进程、`batch_size=1`、`num_workers=0`，使用 `best` checkpoint。只在有两张可见 GPU 时，才选择下面的加速推理命令；`batch-size` 是每张 GPU 的 batch：

```bash
./.venv/bin/accelerate launch --multi_gpu --num_processes 2 infer_seen10.py --seed 0 --batch-size 8 --num-workers 4
```

单 GPU 若要使用 Accelerate，应删除 `--multi_gpu` 并把 `--num_processes` 改为 1，同时按显存调整 `--batch-size`。默认推理和加速推理二选一，不要在同一个输出目录中混用两种推理方式。

### 6.2 原 aligned（视觉—语言连接器可训练）

aligned 在每条 legacy 命令上增加显式配置参数：

```bash
bash scripts/run_csgo_seen10.sh train --config configs/csgo_seen10_xvla_fair.json --seed 0
bash scripts/run_csgo_seen10.sh infer --config configs/csgo_seen10_xvla_fair.json --seed 0
bash scripts/run_csgo_seen10.sh eval --config configs/csgo_seen10_xvla_fair.json --seed 0
```

默认推理是一进程、`batch_size=1`、`num_workers=0`，使用 `last` checkpoint。两张可见 GPU 的加速推理命令如下，`batch-size=8` 仍表示每张 GPU 的 batch：

```bash
./.venv/bin/accelerate launch --multi_gpu --num_processes 2 infer_seen10.py --config configs/csgo_seen10_xvla_fair.json --seed 0 --batch-size 8 --num-workers 4
```

单 GPU 若要使用 Accelerate，应删除 `--multi_gpu` 并把 `--num_processes` 改为 1，同时按显存调整 `--batch-size`。默认推理和加速推理二选一；aligned 结果应保持在 `X-VLA-fair/seed_0`，不要将两种推理方式混写到同一输出目录。

### 6.3 新 aligned-frozen-vl（视觉—语言连接器冻结）

使用相同启动入口，只需选择新配置。训练从原始预训练权重开始，输出到 `outputs/csgo_benchmark_v2_seen10/X-VLA-fair-frozen-vl/seed_0`：

```bash
bash scripts/run_csgo_seen10.sh train --config configs/csgo_seen10_xvla_fair_frozen_vl.json --seed 0
bash scripts/run_csgo_seen10.sh infer --config configs/csgo_seen10_xvla_fair_frozen_vl.json --seed 0
bash scripts/run_csgo_seen10.sh eval --config configs/csgo_seen10_xvla_fair_frozen_vl.json --seed 0
```

默认推理使用单进程、`batch_size=1`、`num_workers=0` 和本实验的 `last` checkpoint，不启用额外加速。两张可见 GPU 时可改用以下加速推理命令，每张 GPU 的 batch 为 8：

```bash
./.venv/bin/accelerate launch --multi_gpu --num_processes 2 infer_seen10.py --config configs/csgo_seen10_xvla_fair_frozen_vl.json --seed 0 --batch-size 8 --num-workers 4
```

单 GPU 使用 Accelerate 时删除 `--multi_gpu` 并将 `--num_processes` 改为 1，按显存调整 batch。默认与加速推理二选一，完成后使用上面的同一条 eval 命令。新实验输出目录与 legacy、原 aligned 均独立。

### 6.4 推理与评测口径

当前推理噪声按 batch 设置随机种子，因此改变 batch size 或进程数可能改变具体预测。正式结果必须预先选定一种推理命令，并对所有对比实验保持一致；不能在看到测试结果后切换推理方式。

评测命令只读取对应输出目录中的 `localization/predictions.jsonl`，调用共享官方 evaluator，输出逐地图和 equal-map macro 的 XY、Z、Pitch、Yaw。Seen-10 每张地图固定为 2,000 条，因此这些均值指标的 pooled 结果与 equal-map macro 数值相同。指标越低越好。

## 7. 输出文件和可视化

每个 seed 目录通常包含：

```text
train.log                         训练日志
checkpoints/                      周期 checkpoint、last 和 best 指针；新实验另有 late 别名
localization/predictions.jsonl    测试集 5D 归一化预测
localization/predictions.meta.json 推理配置和样本覆盖信息
metrics/localization/             官方评测结果
visualizations/localization/      每张地图的可视化图片
```

当前 legacy 目录中的 `train_loss_curve.png` 是根据 `train.log` 额外绘制的 loss 曲线；训练入口本身不会自动生成该图片。

可视化每张地图随机选 10 条样本，并在 validation 和正式推理阶段生成：

- 地图位于左侧，10 张 FPV 缩小后在右侧竖直排列；
- 10 个样本使用 10 种固定颜色；地图上的 GT 为实心圆，prediction 为空心圆，并用线连接对应点；
- 每张 FPV 左上角显示对应颜色圆标记；
- 每张 FPV 内部上方居中显示反归一化后的 `gt_xyzhw` 和 `pred_xyzhw`；
- 正式推理的 GT 只用于可视化后处理，不进入模型输入，也不写入标准预测文件。

## 8. 当前结果和执行状态

初次接入 legacy 已有正式结果，目录为 `outputs/csgo_benchmark_v2_seen10/X-VLA/seed_0`：

- 训练已完成 50,000 个 update；
- validation 最优 checkpoint 为 step 30,000；训练结束的 last checkpoint 为 step 50,000；
- `seen_discrete_test` 已完整覆盖 20,000 条样本；
- 该次已有推理使用单进程、`batch_size=4`；
- equal-map macro 指标为：XY `332.9269`，Z `21.5470`，Pitch `3.0978°`，Yaw `90.3268°`，均为越低越好。

这组数值只代表初次接入 legacy，不应作为与 `exp32_loc` 公平比较的 X-VLA 主结果。当前文档要求的新默认推理为 `batch_size=1`；由于推理包含随机初始噪声，重新运行所得数值可能与上述已有 `batch_size=4` 结果不同。

原 aligned 已尝试正式训练，在 step 3,900 完成验证后因 checkpoint 合并时 CPU/GPU 设备不一致而中断；该次未留下可恢复的 step 3,900 权重。保存问题已修复，并通过独立小批量训练、验证、保存和断点恢复检查。这些小批量产物不属于正式实验结果。原 aligned 的 `seed_0` 已有日志等产物，重新训练须指定新的输出目录，并在后续推理/评测中使用同一目录。

新 aligned-frozen-vl 尚未启动训练、推理或评测。验收后可按第 6.3 节命令执行；输出写入独立的 `X-VLA-fair-frozen-vl/seed_0`，保留原 aligned 作为连接器可训练的对照。

本说明只描述正式实验入口和结果解释。正式训练、全量推理和评测由使用者根据验收安排手动启动；smoke 或单 batch 检查不能替代正式 Seen-10 结果。
