# CSGO Benchmark v2 Seen-10 / X-VLA

仅接入 localization；本次按 `RUN_FULL=0` 只执行真实 smoke。数据根目录只读：`/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`。

## 环境与数据验收

项目独立环境 `.venv`：Python 3.12.3、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128，CUDA 12.8；GPU 为 NVIDIA RTX PRO 6000 Blackwell Server Edition（sm_120）。依赖检查、CUDA 小张量和原生模型依赖导入均通过。

真实数据验收通过：Seen-10 固定 10 张地图，train/validation/discrete test 分别为 50,000/5,000/20,000 条；三种 split 的双视图均为 224×224，proprio 全零，test 不暴露 GT action。独立抽查的 train/validation 归一化标签与发布定义一致。验收记录在 `.cache/seen10_data_acceptance.json`。

环境准备和官方权重命令：

```bash
cd /home/jiahao/task/X-VLA
bash scripts/setup_csgo_seen10.sh
bash scripts/download_csgo_checkpoint.sh
```

下载脚本固定官方 `2toINF/X-VLA-Pt` revision `c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`。本次验收环境已存在于 `.venv`，权重位于 `pretrained/X-VLA-Pt`。

## Smoke 实际验收

执行 `bash scripts/run_csgo_seen10.sh smoke --seed 0`，退出码 0。第 1 步训练、验证和原生 checkpoint 保存成功；从 step 1 恢复后完成第 2 步训练、验证和保存。随后生成 10 张地图各 1 条预测，同步 evaluator 成功读取首条 `cs_agency` 预测；smoke 输出标记为非正式，且没有生成正式 summary。

本次隔离 smoke 输出目录：

`/home/jiahao/task/X-VLA/outputs/csgo_benchmark_v2_seen10_smoke/X-VLA/seed_0/run_20260914T180840Z_1213844/`

该次实际选中的 checkpoint：

`/home/jiahao/task/X-VLA/outputs/csgo_benchmark_v2_seen10_smoke/X-VLA/seed_0/run_20260914T180840Z_1213844/checkpoints/step_00000002`

预测及非正式 evaluator 输出分别位于该目录的 `localization/predictions.jsonl` 和 `metrics/localization_smoke.json`。这些 smoke 指标不属于正式 Table 1 结果。

Checkpoint/refill 验收通过：optimizer 更新及原生 RNG 恢复正常；独立副本保留原 3 条并补齐 7 条，重复 infer 后文件不变。正式 eval 拒绝 smoke 目录且没有创建正式 metrics 目录。记录见 `.cache/seen10_checkpoint_acceptance.json`、`.cache/seen10_refill_acceptance.json`。

## 完整运行命令

Shell wrapper：

```bash
cd /home/jiahao/task/X-VLA
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0
bash scripts/run_csgo_seen10.sh eval --seed 0
```

等价的直接 Python/evaluator 命令：

```bash
cd /home/jiahao/task/X-VLA
PYTHON="$PWD/.venv/bin/python"
UNILIP_PYTHON=/home/jiahao/miniconda3/envs/UniLIP/bin/python
DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
OUTPUT_ROOT="$PWD/outputs/csgo_benchmark_v2_seen10/X-VLA/seed_0"

"$PYTHON" train_seen10.py --config configs/csgo_seen10.json --seed 0 \
  --data-root "$DATA_ROOT" --pretrained pretrained/X-VLA-Pt \
  --output-root "$OUTPUT_ROOT"
"$PYTHON" infer_seen10.py --config configs/csgo_seen10.json --seed 0 \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
  --checkpoint "$OUTPUT_ROOT/checkpoints/best"
# 确保 predictions.jsonl 覆盖完整 20,000 条后再评正式指标。
"$UNILIP_PYTHON" csgo_benchmark_v2_eval/run_eval.py localization \
  --pred-root "$OUTPUT_ROOT/localization" --data-root "$DATA_ROOT" \
  --output "$OUTPUT_ROOT/metrics/localization"
```

续训时可将 `train_seen10.py` 的 `--resume` 指向同一 `OUTPUT_ROOT/checkpoints/last` 或具体的 `step_<step>`；`infer` 默认使用 `checkpoints/best`。`last.json` 和 `best.json` 记录实际指向的不可变 step 目录。正式输出根目录为 `outputs/csgo_benchmark_v2_seen10/X-VLA/seed_<seed>/`；预测为 `localization/predictions.jsonl`，指标为 `metrics/localization/`。`scripts/run_csgo_seen10.sh eval` 会先检查 20,000 条预测完整性；直接调用 evaluator 前也必须确认 coverage。

正式训练使用 50,000 条 seen_train，并仅依据 5,000 条 seen_validation 的自由生成验证误差选 checkpoint。50,000 step 默认每 10,000 step 验证并保存一次，共约 5 次；保留最近 5 个周期 checkpoint，并额外保护更老的 best，`last` 始终指向最近一次保存。动作是 horizon=1 的 `[x,y,z,pitch,yaw]`；前视图和 radar 使用原生 224×224 processor，proprio 恒零。smoke 使用独立目录，严禁将其数值填入正式结果表。

训练的每次完整 validation 会在每张地图内按固定随机种子选择 10 条，并写入 `visualizations/validation/step_<8位step>/<map>.png`。正式推理完成 20,000 条预测后写入 `visualizations/localization/<map>.png`；若预测已完整但图片缺失，重复运行 infer 会直接补绘而不重新加载模型。每张图左侧为 radar，右侧为 10 张 FPV 竖列：同一条样本用同一种颜色，GT 是实心圆，prediction 是较大的空心圆。FPV 内部上方显示物理空间的 `gt_xyzhw` 和 `pred_xyzhw`，其中 `xyzhw = x,y,z,pitch,yaw`，角度单位为度。正式推理的 GT 仅在预测文件完整后由发布 split 后处理读取，不进入模型 batch、dataset record 或标准 `predictions.jsonl`。

后续追加独立实验只需将 wrapper 命令中的 `--seed 0` 改为 `--seed 1` 或 `--seed 2`；输出会写入对应的独立 seed 目录。

同步 evaluator 位于 `csgo_benchmark_v2_eval/`，评测使用 `/home/jiahao/miniconda3/envs/UniLIP/bin/python`。正式模型训练、全量推理和正式评测尚未执行。
