# X-VLA 接入 CSGO Benchmark v2 Seen-10：最终实现方案

本方案只覆盖 Seen-10 localization。公平主实验使用 configs/csgo_seen10_xvla_fair.json，旧 configs/csgo_seen10.json 保留为工作区 main@4aed8865 的 legacy 复现入口。代码、配置、日志和 checkpoint 元数据需要能明确区分 fair 与 legacy；本阶段不启动正式训练、推理或评测。

## 数据与输入边界

- 复用官方 manifest 和固定 split：seen_train、seen_validation、seen_discrete_test；不扫描目录重划分，不使用 CrossMap-4。
- 校验发布 manifest/calibration 的声明哈希，并对 manifest、minimal report、calibration 与 30 个 Seen-10 split 元数据文件生成统一 data-contract SHA256；训练、resume 和推理必须一致。
- 输入只包括 FPV RGB、对应 radar 和 localization instruction/map name；不提供 GT pose、历史位姿、地图坐标或其他额外状态。
- train/validation/test 的外部定位 action 都是单步 5D [x,y,z,pitch,yaw]，action_horizon=1。
- 图像保持 X-VLA 原生 224 输入。train 对 FPV/radar 使用原生 ColorJitter(0.2,0.2,0.2,0)；validation/test 关闭随机增强。

## 维度和 action 适配

数据集只生成官方协议的 5D normalized target。进入 XVLA.forward 前一次性执行 5D + 15D zero -> 20D，再交给原生 Auto action 路径。训练和推理的 randn_like、noise、x_t、prediction 与 10 步 refinement 始终保持 20D；推理每一轮不重置后 15D，最终只返回前 5D。

训练沿用 X-VLA 原生 x0 clean-action denoising objective。主 loss 是 prediction 与 clean target 前 5D 的 MSE，再乘 100；后 15D 不加入直接监督，但保留在 20D action tensor 和预训练 action path 中。原始 action encoder 的 72D 输入继续由 action20 + zero_proprio20 + time32 组成。zero_proprio20 是保留原作者 72D 权重接口的全零槽位，robot state 本身没有进入任务计算。

外部归一化严格为：x/1024、y/1024、(z-z_min_map)/(z_max_map-z_min_map)、pitch/(2*pi)、yaw/(2*pi)。三种 split 共用发布的按地图 calibration；不使用 qnorm、不 clamp、不添加 Z epsilon。qnorm/clamp 作为 X-VLA 原生差异保留为关闭状态，不向 UniLIP 的 pi0.5 定位 head 机械复制。

## 可训练模块和优化

主实验采用功能角色对齐。vision encoder 和 image positional embedding 冻结；LLM base 与 action expert base 冻结，并分别在对应目标层使用 LoRA；视觉到语言 connector、transformer.vlm_proj、transformer.aux_visual_proj、action encoder/decoder/norm 和 soft prompt 全量训练；timestep module、位置 embedding 和未列入上述角色的 base 参数冻结。

LoRA 固定为 r=32、alpha=64、dropout 0.05、bias=none。LLM 目标层为 q_proj,k_proj,v_proj,out_proj,fc1,fc2；action expert 目标层为 attn.qkv,attn.proj,mlp.fc1,mlp.fc2。这统一的是可训练功能角色和 LoRA 配置，不要求不同架构的模块名字或可训练参数总量机械相同。

优化器使用 X-VLA 稳定配置：AdamW、lr=1e-4、betas (0.9,0.95)、weight decay 0、gradient clipping 1.0、BF16、constant schedule。前 1,000 个 optimizer updates 仅让 soft prompt/action heads 使用 1e-4，LoRA/connectors 的 LR 为 0；之后所有 fair trainable groups 使用 1e-4。不做 LR pilot sweep，也不使用测试集选择 LR。

## 训练预算和 checkpoint

- effective localization batch：128；实际 microbatch 和梯度累计由 world size 动态计算。
- optimizer updates：19,500；按 50 epochs、每 epoch 390 updates 组织，尾部丢弃 80 条样本。
- validation/save：每 3,900 updates，一共 5 个周期点 3900,7800,11700,15600,19500。
- checkpoint：保留 5 个周期 checkpoint，另维护 last 和 validation-only best 指针；主论文结果选 last/late，best-val 只能作为明确标注的附加结果。
- 所有训练/推理/评测 sidecar 记录 manifest、split、样本计数、normalization、action mode、20D 适配、state 槽位、推理步数、checkpoint 来源和参数审计。

## 文件级实现边界

- csgo_seen10/dataset.py：读取官方 Seen-10 manifest/split/calibration，输出 5D target、双视图和 zero proprio；实现 train-only ColorJitter 及 deterministic validation/test。
- models/action_hub.py：保留 legacy auto/legacy_reset_dummy，新增 official_auto；官方模式要求调用者先传入 20D，禁止在 XVLA.forward 内把 5D 当作噪声宽度；loss 只返回 valid5×100，最终输出裁成 5D。
- csgo_seen10/model.py：把配置 action contract 传给模型，保留原生 20D action head 与 zero proprio20 结构，保存并校验 fair/legacy mode。
- models/modeling_xvla.py：在进入模型噪声和 VLM 计算前校验 official mode 的 20D 输入，确保 padding 只发生一次。
- train_seen10.py：实现 fair trainable-role mask、精确 LoRA target、角色 LR 分组、world-size independent accumulation、19,500 update budget、5 个 periodic save/eval、last/best 元数据和 initialization hash 审计。
- infer_seen10.py：按 checkpoint 的 action contract 加载模型，fair 模式默认 last，legacy 模式保持 best；固定原生 10 步，最后才输出 5D；补齐预测和可视化时拒绝 contract 冲突。
- scripts/run_csgo_seen10.sh：用户入口默认复现 legacy；传入 --config configs/csgo_seen10_xvla_fair.json 时运行公平 aligned 实验；eval 只调用共享官方 evaluator。

## 验收边界

1. 检查三 split 的 manifest ID、地图数量和样本数，确认 test 不把 GT 放入模型 batch。
2. 检查 5D normalization 与反归一化 round-trip，确认没有 epsilon、clamp、qnorm 或测试集统计。
3. 检查 action target 进入 XVLA.forward 前恰好一次 pad 到 20D；noise、prediction 和 inference loop 全部保持 20D；最终输出才裁 5D。
4. 检查 zero proprio20 为有限全零、无真实 robot state、无额外监督；确认原生 action encoder 72D 形状不变。
5. 检查 vision 冻结、LoRA target/rank/alpha/dropout/bias、connector/head requires-grad、optimizer 实际参数组与每阶段 LR。
6. 检查单 batch forward/backward、有效 batch/累计步数、smoke checkpoint 保存恢复和 metadata；不启动正式 19,500-step 训练。
7. 训练后检查完整 20,000 条 prediction 与官方 evaluator 闭环，报告 pooled/equal-map macro 的 XY、Z、Pitch、Yaw。

## 与既有实现的关系

fair 主实验相对原版 X-VLA 保留其原生 20D action head、x0 denoising objective、10-step inference 和 72D action encoder 接口，仅增加 CSGO 的 5D dataset/evaluator 接口、一次性 5-to-20 padding、zero proprio 槽位说明、Seen-10 预算和可训练角色配置。相对工作区 main@4aed8865 的初始接入，fair 模式消除了训练/推理每轮 reset dummy 的 legacy 维度行为，加入原生 ColorJitter、LoRA/connector 训练策略、严格 128 effective batch/19,500 updates 和 3,900 checkpoint cadence。相对 UniLIP exp32_loc，fair 模式仍使用 X-VLA 原生 20D action path、x0 denoising、10 步推理、原生优化器稳定设置和模型角色映射；共享的是 Seen-10 数据、外部 5D pose、split、样本暴露量、checkpoint 规则和官方评测协议，不强制复制 UniLIP 的 pi0.5 32D flow-matching head 或 10-step implementation。
