# X-VLA 接入 CSGO Benchmark v2 Seen-10

范围：仅 localization，seed=0；RUN_FULL=0 时完成真实数据和预训练 X-VLA 的 smoke，保留全量 train / infer / eval 命令。数据根目录只读。

## 变更文件与接入点

- `csgo_seen10/dataset.py`：根据发布 report、benchmark manifest、split JSON/JSONL 和 calibration 读取固定 Seen-10，不扫描 images，也不重划分。训练/验证才提供 normalized 5DoF 标签；推理 batch 只含身份、图像、任务文本和零 proprio。
- `csgo_seen10/model.py`：复用 XVLAProcessor 的前视图 + radar 两个有效视图、原生 XVLA 和 auto action space。设置 num_actions=1、real_action_dim=5、max_action_dim=20、use_proprio=False；保留原始权重形状，监督并输出前五维绝对位姿。
- `models/action_hub.py`：修正 auto 模式补齐维度在训练和推理之间的不一致；preprocess 先取真实动作维度再补零，防止推理将未监督的 15 维随机噪声作为条件输入，原有机器人动作模式不受影响。
- `configs/csgo_seen10.json`：数据、模型、优化、验证、推理配置。
- `train_seen10.py`：轻量接入原生 Accelerate、train.py 的 AdamW 分组及学习率策略、save_pretrained checkpoint；补充验证集选择、优化器/随机状态恢复，不重写通用训练框架。
- `infer_seen10.py`：加载选定 checkpoint，按 seen_discrete_test 输出标准 predictions.jsonl；支持缺失样本续推理，拒绝覆盖既有结果。
- `scripts/run_csgo_seen10.sh`：train / infer / eval / smoke、--seed 和恢复参数。metric 使用 UniLIP Python 及同步通用 evaluator。
- `requirements-csgo.txt`、环境安装脚本、`CSGO_SEEN10.md`：独立项目环境、实际命令、输出路径和验证结果。

## 核心约束

输入为当前第一视角 RGB、发布 radar 和固定 localization instruction/map name。proprio 始终为零；GT 仅作为训练/验证 action target。XY 除以 1024；split 中弧度制 angle_v/angle_h 除以 2π；Z 使用 (z-z_min)/(z_max-z_min)，范围只读发布 calibration。保持原生 224 图像处理（以发布 processor 为准），不引入 generation 任务。

每个 seed 使用 `outputs/csgo_benchmark_v2_seen10/X-VLA/seed_<seed>/`；完整推理输出 `localization/predictions.jsonl`。smoke 使用独立目录并明确标记非正式结果。验证只用 seen_validation，正式预测覆盖完整 seen_discrete_test。

BUILD_SHARED_EVALUATOR=0：直接整体同步 `/home/jiahao/task/ControlAR/csgo_benchmark_v2_eval` 到本项目，不开发或改写指标。正式运行 `run_eval.py localization`；非正式 smoke 运行 `run_eval.py smoke localization --limit 1`，只读取模型预测首条，不生成正式 summary。

## 实际执行与验收

- 独立 `.venv` 已就绪：Python 3.12.3、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、CUDA 12.8、RTX PRO 6000 sm_120。依赖检查、CUDA 小张量和原生模型依赖导入通过。
- 真实数据验收通过：固定 10 张地图；train/validation/discrete test 分别为 50,000/5,000/20,000 条；双视图、224×224、零 proprio、GT 隔离和归一化标签检查通过。详见 `.cache/seen10_data_acceptance.json`。
- 本次按 `RUN_FULL=0` 只执行真实 smoke：`bash scripts/run_csgo_seen10.sh smoke --seed 0` 退出码为 0。真实预训练模型完成一步训练、保存，再从 step 1 恢复后训练至 step 2；best 选中 step 2。推理生成 10 张地图各一条预测，同步 evaluator 成功评测首条 `cs_agency`；没有生成正式 summary。
- 实际 smoke 目录为 `outputs/csgo_benchmark_v2_seen10_smoke/X-VLA/seed_0/run_20260914T180840Z_1213844/`；实际选中 checkpoint 为该目录下 `checkpoints/step_00000002/`。Smoke 指标不填入 Table 1。
- Checkpoint/refill 验收通过：optimizer 更新、native RNG 恢复正常；独立副本保留原 3 条并补齐 7 条，重复 infer 后文件字节不变；正式 eval 拒绝 smoke 目录且未创建正式 metrics。证据见 `.cache/seen10_checkpoint_acceptance.json` 和 `.cache/seen10_refill_acceptance.json`。
- 按 `RUN_FULL=0`，完整正式训练、20,000 条推理和正式评测未运行。后续追加独立实验只需把 wrapper 中 `--seed 0` 改为 `--seed 1` 或 `--seed 2`。Shell wrapper 与直接 Python/evaluator 命令、正式输出路径见 `CSGO_SEEN10.md`。
- 正式 50,000 step 调整为每 10,000 step 验证并保存，共约 5 次；维护 `last`/`best` 指针，保留最近 5 个周期 checkpoint，并保护更老的 best。
- 训练 validation 和正式 localization 推理均为每张地图固定随机选择 10 条可视化：左侧 radar 绘制同色 GT 实心圆和预测空心圆，右侧竖排 10 张 FPV，并在图内顶部显示物理空间 `gt_xyzhw`/`pred_xyzhw`。正式推理只在预测完成后读取 GT 进行绘图。
