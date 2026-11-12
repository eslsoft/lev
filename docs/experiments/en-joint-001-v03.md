# v0.3 持续训练阶段：en-joint-001

本轮目标是交付新的真实研究检查点并验证新增上下文任务，不能以数据构建、提交任务或短程训练完成替代最终35,000词目标。持续目标保持 active，未授权模型发布。

## 数据与初始化

联合包训练6,505请求/8,115判断；development147、calibration118、test159判断。包含en-context-002所有分区与en-general-005中2,784条历史训练回放。1,511条历史非训练记录不进入新包，另外14条训练记录因扩展词族闭包触及历史或新留出被排除。新词义训练仍为17个词，不虚报篇章词汇为审核覆盖。新包保留单词/作者/文档分区规则；35,000词旧测试仍冻结，历史回放的既有prior_train曝光不抹去。

新训练从固定原始Kev初始化，**不是v0.2优化器断点续训**。Kev提交、基础模型revision及初始化revision沿用v0.2固定配置。候选v0.3先跑1个epoch，lr=1e-5、batch=1、accum=8、LoRA rank16、seed42、max_state7552。研究许可限制沿用源包；不用于商业部署。

## 固定顺序与选择规则（读取本轮模型结果之前记录）

1. 32条最长训练state独立smoke，A100-80GB，1epoch。只有全量处理、零截断/拒绝、有限梯度且checkpoint可读取才继续完整训练。失败保留，新尝试新run-id。
2. 用固定v0.3配置训练全部6,505条请求。预计814 optimizer steps；核对真实metrics，不能用预期数代替实测。
3. 在新development和原en-general-005 development比较v0.3与v0.2。新development总准确率必须严格提升、篇章纠错准确率不得下降；原development总准确率下降不超过1个百分点；所有请求必须返回。新语境词义仅8题，单独列出，不声称稳定估计。
4. 若未通过，保存所有结果，继续开发数据/训练迭代；不读取新的calibration/test或35,000词模型成绩来挑候选。
5. 通过后冻结检查点，拟合新calibration温度，固定阈值0.5，进行新test及完整35,000词验收。分别报告所有题、逐词全部已排定项、训练曝光分层、任务缺口；partial不升级为完整词条能力。
6. 本阶段通过仍不是最终英语通用审核能力已达标。所有已打开的测试在后续迭代中注明曝光/诊断用途，不将重复评测称为新盲测。

## 授权与身份

用户已显式授权Modal训练及持续推进。Realmroot当前发现中无Modal服务，使用现有Modal saltbo profile执行本阶段训练、评估与结果下载。仅使用新的实验run-id，既有kev服务不变。

## 原始证据

- 构建：`scripts/build_joint_context.py`、`data/research/en-joint-001/replay-selection.jsonl`、`runs/en-joint-001-build.log`。
- 长样本smoke：`artifacts/en-joint-001-smoke`、`configs/lev-4b-v03-smoke.toml`、run-id `lev-joint001-v03-smoke-01`。
- 完整训练：`artifacts/en-joint-001`、`configs/lev-4b-v03.toml`、run-id `lev-joint001-v03-train-01`（实际状态以Modal证据为准）。
