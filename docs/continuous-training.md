# 持续训练状态

最终目标保持进行中：英语非生成式审核、35,000词逐词验收、真实模型检查点和完整错误报告。数据包完成或一次实验结束都不是最终目标完成。当前是v0.4开发迭代，研究数据许可限制仍适用。

## 已执行与正在执行

- 长样本训练 `lev-joint001-v03-smoke-01` 已成功：32条请求、4个优化器step，零拒绝/截断，训练耗时56.74秒，峰值GPU分配19,906,699,264 bytes。检查点已下载并按真实训练metrics核验。
- 完整训练 `lev-joint001-v03-train-01` 已在Modal A100-80GB成功完成并下载核验：6,505请求、8,115判断、1epoch，实际814优化器step，零拒绝/截断，耗时4,433.44秒，峰值GPU分配20,044,277,760 bytes。真实检查点位于 `runs/modal-lev-joint001-v03-train-01/result/checkpoint`，核验记录为 `runs/lev-joint001-v03-cycle/training-verification.json`。这只证明训练执行成功，尚不证明准确率提升。
- `scripts/advance_context_cycle.py` 已完成训练下载核验。旧模型的新development评估 `lev-joint001-v02-dev-01` 被Modal抢占中断，失败原件已下载保存；经原因核验后用相同模型、数据和参数启动恢复任务 `lev-joint001-v02-dev-02`。不读取或拼接失败尝试的分数，不重复训练。接着比较候选的新development和旧development，再按已登记规则决定是否校准及读取新test。
- `scripts/advance_vocabulary_evaluation.py` 已运行，只有开发选择和独立校准成功才提交完整词汇评测。最多2个GPU评测任务并行，16个固定分片，无自动重试或挑分。

新development同题比较已经完成（147判断，均零缺失）：v0.2为113/147（76.87%），v0.3为122/147（82.99%）；段落纠错分别107/139和116/139，语境词义均6/8。配对结果是15题改善、6题回退、19题仍错；误拒由22降至10，漏放由12升至15。不能只用总准确率隐去这一权衡，也不能将这些开发成绩称为35,000词成绩。旧development结果为v0.2的483/590（81.86%）与v0.3的468/590（79.32%），下降2.54个百分点，超过允许的1个百分点；v0.3未被选中，新calibration/test和35,000词评测均未打开。

证据：`runs/lev-joint001-v03-cycle/incumbent-recovery-verification.json`核对恢复任务请求（除run-id外）及输入与原任务一致；`candidate-development-report.json`和`development-paired-errors.json`保留候选统计及逐题差异。没有把开发错题复制进训练，也没有更改预先登记的选择规则。

## v0.4 下一迭代

已按[固定协议](experiments/en-joint-001-v04.md)提交`lev-joint001-v04-train-01`。仍从固定原始Kev开始，仅将epoch从1增至2；其余参数和同一干净数据包保持不变。v0.2历史训练含14条扩展词族触及历史留出的记录，因此不直接继承它。v0.3完整训练证据用于复用资源准入，v0.4自身的全部执行和评估仍须重新验证。

运行控制器为`advance_context_cycle.py --version v04`与`advance_vocabulary_evaluation.py --version v04`；日志和阶段记录分别位于`runs/lev-joint001-v04-controller.log`、`runs/lev-joint001-v04-vocabulary-controller.log`和`runs/lev-joint001-v04-cycle/`。预检63项unittest通过，记录在`runs/lev-joint001-v04-preflight-unittest.log`，不是模型成绩。

## 备选训练数据 en-joint-002（尚未训练）

针对v0.3语法回退，按[固定数据方案](experiments/en-joint-002.md)准备了`artifacts/en-joint-002`：保留原联合包全部记录，在原有六个训练construction中添加3,700个BLiMP最小对、7,400判断。新训练共13,905请求／15,515判断；development147、calibration118、test159的原生导出均与en-joint-001逐字节相同。

67,000源pair全部留审计：纳入3,700；留出construction排除30,000；历史或本批已用句子排除844；good/bad相同排除2；固定配额以外32,454。每pair保留正反两句与共同问法，否定问法作明确逻辑取反，不依赖模型标签。来源为CC BY 4.0的BLiMP模板标签，联合包仍受原有研究许可限制；偶然句内词汇不算监督词汇覆盖。

`runs/en-joint-002-source-audit.json`已核对全部新增标签、问句极性、成对关系和旧记录不变；`runs/en-joint-002-tokenization.json`全量四分区真实编码通过、零超限，最长打包2,428 tokens。当前v0.4仍训练en-joint-001，未切换数据、未额外提交训练。待其开发结果出来再决定下一实验，不能把数据准备当效果提升。

## 35,000词验收准备

`artifacts/en-us-vocab-35000-v3-shards-01`完整保留56,639请求、368,056判断、35,000词。16个分片逐字节拼接等于原始test.jsonl；同词所有请求在同一分片。真实固定tokenizer和Kev编码准入已在56,639请求上通过，零超限、零截断，最长状态84 tokens，最长打包1,156 tokens。

合并器拒绝缺片、重复片、混合检查点、温度或推理环境。漏返回问题仍保留在逐词分母，不算通过。分片测试与格式检查不是模型推理证据；此时尚无35,000词模型成绩。

另外已冻结独立语境补充`artifacts/en-us-vocab-context-001`：原35,000词中的1,551词、3,665个WiC来源判断（1,935正例、1,730反例），其余33,449词尚缺这类题。全部已知历史及联合包词族排除，源7,466行的纳入/排除完整留档；625来源文件、全部原始上下文/标签/位置和原生导出核验通过。两分片全量真实分词准入通过，零超限，最长打包214 tokens。研究用途、全部partial，尚无模型评分；不修改当前训练或原35,000词包。

补充协议为[独立语境补充](experiments/en-us-vocab-context-001.md)，证据为`runs/en-us-vocab-context-001-source-family-audit.json`与`runs/en-us-vocab-context-001-tokenization.json`。待开发门槛通过、校准和既定词汇评测完成后，用同一冻结候选另行评测该补充，不据其成绩重新挑选候选。它不能代表全部35,000词的语境能力，也不证明基础模型预训练未见公开WiC。

**最新准入更新：context-001尚未模型评分，当前不得作为独立语境评测运行。** 第一批已由结构化词典支持的词形关系使`coast / union / unionize`的10题与历史已用词族连通；原始上下文和标签仍正确，但先前已知词族闭包不再足够。暂停此补充的评测准入，待剩余词形核对后在新目录重建；不覆盖旧包或旧审计。记录见`runs/en-us-vocab-context-001-admission.json`。这一更新没有在当前en-joint-001训练包发现新分区冲突。

## 词形来源核对（未进入训练）

语境补充修订已完成：`artifacts/en-us-vocab-context-002` 按两批受支持词形关系重新生成，覆盖 1,526 个词、3,619 道题。与 context-001 相比恰好排除审计列出的 46 题（25 词），其余上下文和标签不变，旧版本保留。原始 WiC 标签、上下文、632 项来源依赖和扩展词族隔离核验通过；两个固定分片拼接与导出字节完全一致；全量真实 pinned Kev 分词检查零失败、零截断，最大 packed 长度 214。证据为 `runs/en-us-vocab-context-002-source-audit.json`、`runs/en-us-vocab-context-002-revision-audit.json`、`runs/en-us-vocab-context-002-tokenization.json`。数据准入记录 `runs/en-us-vocab-context-002-admission.json` 仍要求先完成候选选择和独立校准，尚未执行该补充的模型评分。它不覆盖其余 33,474 个目标词的语境判断，也不证明完整词汇掌握。

UniMorph英语原始快照已固定到commit `66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b`，位于`data/raw/en-morphology-001`，保留许可、下载记录和哈希。652,477行涉及31,711个目标词头，但只有16,688个目标词有非原形的单词形式；这些数不等于可信标注覆盖。

原表发现1,989条词典说明误入词形字段的可疑记录，具体反例是`efflux → countable / N;PL`，现有Kaikki原文的复数为`effluxes`。直接把该表加入词族图会产生涉及200训练记录与2测试记录的巨大伪连接，所以没有把这些未经核对的边纳入现行闭包。原始诊断与隔离决定保留在`runs/en-morphology-001-expanded-family-audit.json`和`runs/en-morphology-001-quality-quarantine.json`。没有因此更改或重标正在运行的数据。

该来源整体仅作候选清单，不能直接当金标准。首批1,000个候选词的Kaikki原文已全部获取，位于`data/raw/en-morphology-001/dictionary-check-001`。结构化词性/词形/标签核对完成：995词有中性词形支持，去重后3,452个中性支持关系、41个带限制支持关系、146个说明字段错误、11个未获支持关系，另有1,931个不在本次屈折核对范围内的原形关系。未获支持不自动判为错误；两套提取可能共享底层Wiktionary，不能冒称独立专家裁定。逐条证据及哈希见`runs/en-morphology-001-dictionary-audit-001/`，尚未导出训练金标准。

剩余15,688候选的Kaikki获取已启动，位于`data/raw/en-morphology-001/dictionary-check-002`；进度在`runs/en-morphology-001-dictionary-check-002.log`。两批候选清单互斥，合计16,688词，计划审计见`runs/en-morphology-001-download-coverage-plan.json`。当前只证明下载计划覆盖，不代表第二批完成或全部词形已核验，不能将下载条目数当完整审核词数。

`runs/en-morphology-001-conflict-paths.json`定位两个表面冲突都经由`countable`或`uncountable`说明字段；`runs/en-morphology-001-marker-ablation.json`显示移除五类说明误入词形的行后，新增候选关系没有造成联合包分区冲突。这是错误来源定位，不是对其余词形记录的质量背书；原表仍保持隔离。

`runs/en-morphology-001-corroborated-family-audit-001.json`进一步只使用第一批词典已支持的关系核对：当前联合包零新增分区冲突；995个有支持的词中697个未接触任何已知历史/当前词族，可作为新独立验收候选，298个需要保留曝光说明。第二批完成后仍须重算完整闭包，不能把第一批结果当全量证明。

已用通用审计器复核第一批全部来源哈希，并生成`runs/en-morphology-001-corroborated-family-overlay-001/`。新增2,434个去重词形边后，35,000词中本轮训练目标词族接触更新为2,274个（旧证据下为2,220个），未检测到32,726个；原冻结历史曝光分组不改写，预训练接触仍未知。这是证据扩展，不是训练新增54词或模型准确率。

`scripts/finish_morphology_batch.py --pid 86892`正在等待已确认存活的第二批下载进程。完成后会依次运行逐条词形核对和两批合并词族/曝光审计，输出`runs/en-morphology-001-dictionary-audit-002/`、`runs/en-morphology-001-corroborated-family-audit-002/`和`runs/en-morphology-001-finish-002.json`；日志为`runs/en-morphology-001-finish-002.log`。下载进程消失且无完成记录时显式失败，不把中断当成功、不自动重试，也不生成金标准或改变训练。合并结果仍需审阅，尤其是语境补充准入与当前训练分区是否出现新冲突。

## 搭配来源候选（未进入训练）

第二批词形下载与审计已完成，结果见 `runs/en-morphology-001-finish-002.json`：15,688 个请求中 15,687 个有词典文件，`clothespress` 返回 404，缺失保留。第二批 15,588 词存在受支持的屈折关系；去重后 39,711 个中性支持关系、362 个带限制支持关系、951 个说明字段错误、96 个未获支持关系，未获支持不判假。两批合并后新增 32,334 条受支持的去重词形边，当前联合训练包无新增分区冲突。35,000 词中检测到本轮训练目标词族接触的数量更新为 2,524，未检测到 32,476；这是曝光证据更新，不是词汇掌握率。完整输出在 `runs/en-morphology-001-corroborated-family-audit-002/`，语境补充仍需按新闭包修订；未生成金标准或模型成绩。

ColloCaid 的 Academic Collocation Errors and Other Problems v2 已从官方 Figshare 获取：
https://doi.org/10.6084/m9.figshare.13640624.v2 。原始 ZIP、版本元数据、下载哈希与 CC BY 4.0 归属保存在 `data/raw/en-collocation-001/`。文件长度和官方 MD5 已核对；完整解析的 370 行与工作表行号在 `runs/en-collocation-001-source-audit/source-rows.jsonl`。

资料涉及 76 个节点词，来源分类为 319 条 error、37 条 too informal、11 条 commonly confused、3 条 non-standard。这些是来源分类，不是本项目已经裁定的真假标签。文档明确说明 commonly confused 可能在特定语境正确，non-standard 仅被部分使用者认为错误；too informal 限于学术书面语。即使 error 类也包含 `At the beginning,` 等缺乏完整语境的片段，不能直接转为“所有语境都不成立”的命题。

当前生成金标准数为零，不改变运行中的 en-joint-001。后续必须逐条限定审核命题、补足可核验语境，保留不确定项，再做词族隔离；不得以词典未收录、频率低或更推荐另一表达来制造错误标签。来源核验统计在 `runs/en-collocation-001-source-audit/summary.json`。这份小型学术英语候选资料不代表 35,000 词的通用搭配覆盖。

另查到 Cambridge Error-annotated AN Combinations 资料，但其当前页面要求注册并限制派生产物发布，尚未下载或采用，不影响既有任务继续运行。

## 恢复与证据

v0.4 首次训练 `lev-joint001-v04-train-01` 被 Modal GPU 抢占中断，平台自动再次调用因不可变目录已存在而失败。完整失败产物（14 文件）保存于 `runs/modal-lev-joint001-v04-train-01`；该目录只有训练配置，没有可恢复的权重或优化器状态。平台原始日志见 `runs/lev-joint001-v04-preemption-platform.log`。已登记固定参数恢复 `lev-joint001-v04-train-02`，从原始 pinned Kev 初始化重跑，不是从中断步续训；数据、配置、种子和开发选择门槛不变。登记与证据哈希见 `runs/lev-joint001-v04-cycle/preemption-recovery-01.json`，恢复控制器日志为 `runs/lev-joint001-v04-controller-recovery-01.log`。后续评估读取恢复产物；首轮失败记录不覆盖，不把未完成训练计为成功。

- `runs/lev-joint001-v03-controller.log`：训练/开发/校准控制器日志。
- `runs/lev-joint001-v03-controller-recovery-01.log`、`runs/lev-joint001-v03-vocabulary-controller-recovery-01.log`：抢占后的恢复控制器日志。
- `runs/lev-joint001-v03-cycle/preemption-recovery-01.json`：首次评估失败证据、完整保留路径和固定参数恢复决定。
- `runs/lev-joint001-v03-cycle/`：每阶段开始和完成记录、真实smoke核验、后续开发选择。
- `runs/lev-joint001-v03-vocabulary-controller.log`：词汇评测等待器。
- `runs/lev-joint001-v03-vocabulary/`：后续分片任务、结果与单词报告。
- `runs/en-us-vocab-35000-v3-shards-01-audit.json`：全量原字节审计。
- `runs/en-us-vocab-35000-v3-tokenization.json`：全量真实分词/编码准入。
- `runs/lev-continuous-001-unittest.log`：59项本地测试通过；这些不是模型执行成绩。

恢复时先读取现有阶段记录与Modal状态，不能重复提交已有run-id。控制器失败或开发门槛未过时，持续目标仍未完成：检查失败、改进下一开发迭代，不请求重复授权、不在测试集挑候选。最终模型发布不在本次授权范围。

Modal无Realmroot服务集成；已按身份策略使用现有saltbo profile进行本任务的训练、运行状态读取、容器日志读取和产物下载。
