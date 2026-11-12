# 实验工作流与边界

## 阶段

1. 批准审核规范，建立可溯源标签；冻结 development/calibration/test，保留完整词条和真实错误。
2. 先测原始 Kev-4B 在 development 的表现。若比较 sol，必须使用相同输入、标签、完整词条集合和错误定位口径。当前工程没有 sol API 适配器。
3. 从固定 Kev 检查点继续 LoRA 训练，只传 train.jsonl。输出 checkpoint 目录，附 lev-model.json；状态为 trained-not-validated，不自动称为发布模型。
4. 在 development 调整数据和训练参数；保留全部实验，不拼接不同轮次的最好分数。当前不自动选择最佳检查点，也不搜索超参数。
5. 用 calibration 分区的本地 logits 调用 Kev 自带温度拟合。校准文件绑定模型内容哈希和数据版本，原权重不改写。此校准是同领域不同词族，不能称为“新领域校准”。
6. 冻结模型、温度、阈值与精度配置，再读 test。当前 CLI 不构建远程访问控制；不在 test 上调参是实验纪律，不是文件加密。
7. 独立人工裁定最终错误，填写模型卡。在真实 Vocnet 贡献上旁路验证后，才考虑替换生产审核器。

## 命令和环境

普通 Lev 环境只依赖 Python 标准库。Kev 重型依赖使用现有 checkout 的 uv.lock，通过 `uv run --project "$LEV_KEV_PATH" --frozen --no-dev ...` 执行。第一次实际执行会安装依赖并可能下载权重。serve 额外安装 Kev 的 serve extra。无需把其源代码复制进 Lev。

首版 CUDA 训练采用 bf16 autocast、fp32 冻结基础权重，是上游现有路径，不等同于 4-bit QLoRA。LoRA 节省可训练参数，但并不意味着 8 GB 权重就只需 8 GB 训练显存。4B 训练资源应先用代表性长样本实测；MPS/CPU 使用 fp32，不保证速度。配置 max_state=7552 是训练与 Lev 评测/本地服务共同使用的上下文上限，不是生产保证。Kev 原生 `benchmark --data` 默认只有 384 token，因此 Lev 通过其 LocalPredictor/evaluate_records 接口显式传递上下文，并保留 lev-context.json，不能直接把原生命令当成等价替代。

运行输出目录必须不存在；目录不会复用、清理或覆盖。执行后：

- job.json：命令、上游和本项目源文件哈希、配置、数据版本、检查点标识。
- run.log：上游原始 stdout/stderr。
- status.json：成功或失败以及退出码；中断也记录失败。
- checkpoint/：仅训练产生；无人工伪造或占位权重。
- native/：基准的 rows.json、predictions.jsonl、report.json 等原始证据。
- calibration.json：仅校准产生；绑定检查点，不写入 head.pt。

`lev report` 的分母来自冻结的预期问题，不来自“成功返回的行数”。Kev 基准如记录超长输入拒绝，缺失项在 Lev 报告里仍然保留。如果上游整体失败，则保留失败日志，不生成成功报告。

## 如何读报告

- accuracy_all_requested：判对问题 / 全部请求问题，缺失不当作判对。
- correct_fact_acceptance：正确事实被接受的比例。
- incorrect_fact_detection：错误事实被明确拒绝的比例。
- false_rejections / false_acceptances：已返回结果中的误拒、漏放数。
- brier_completed_only：仅成功返回问题的二元 Brier，必须与 completed/missing 一起看。
- whole_cases：仅 scope=complete；完整正确词条接受率、完整错误词条拒绝率，以及是否真的命中错误。

若含错误的词条因另一个正确项被误拒，incorrect_case_rejection 会计一次拒绝，但 incorrect_case_error_detection 不会计为发现真正错误。缺失结果导致的待处理既不是成功接受，也不是成功拦截。partial 数据永远不计入整词指标。

报告只是点估计。首版没有置信区间、独立专家复核系统或自动发布门槛，不能仅凭小样本高分发布。需要足量独立词族、真实长上下文、困难反例以及完整贡献验收。

## 已验证和未验证

工程本地验收覆盖实际 CLI、导出冻结/改写检测、词族隔离、标签投影、误判/缺失/整词统计和进程失败记录。指定 LEV_KEV_PATH 时，测试执行本地 Kev 源码中的实际 JSON 读取/请求投影函数以及训练参数解析器，不加载模型。

最初的骨架验收不构成 GPU 训练证明。现已在 Modal 完成首轮研究数据实验：真实梯度训练、检查点重新加载、开发集/校准集/测试集评测及温度拟合均已执行，原始证据见 `runs/modal-lev-general-20261021-01`，结果与限制见 `docs/model-card.md`。生产金标准与独立专家验收仍未完成；没有执行模型发布或服务部署。
