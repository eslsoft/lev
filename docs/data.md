# 数据与标注约定

## 记录与判断

可执行的格式示例在 `data/examples/format-only.jsonl`。根对象字段：

| 字段 | 含义 |
|---|---|
| id | 全局唯一审核请求 ID |
| family_id | 同语言词族；词形、改写和人工错误变体共享此 ID |
| case_id | 一份完整贡献的具体版本；正确稿与错误变体用不同 ID |
| split | train / development / calibration / test |
| language | 初版仅 en |
| scope | partial 或 complete；局部样本必须是 partial |
| origin | example / generated / curated / mutation |
| rubric_version | 匹配规则文件版本 |
| state | 模型可读的事实、词义上下文及必要外部证据，字符串或 JSON 对象 |
| questions | 问题 ID 到问题对象的映射 |

每个问题包含 `kind`、`claim`、`label`、`review`。当前 v2 的 `kind` 是开放的统计标签，不限制问题类型；`claim` 可以是任意具有明确真假语义的自然语言审核问题。旧 v1 格式保留 sense / form / ipa / syllable / morpheme / membership / relation / distinctness 八类，只为兼容旧数据。claim 指明要判断的准确对象和命题；共享 state 不必为每题重复。导出时使用版本化规则与 claim 构成 instructions。

label 只能是 JSON true、false 或 null，不能使用 1/0 或字符串。review 包含 status、reviewer、rationale、evidence。evidence 每项记录 reference（来源文件、URL、书目等）和 locator（具体页、义项、段落等）。

- draft：未经最终复核；即使有候选标签也不能训练。
- adjudicated：具有明确真值、裁判身份、理由与证据的已裁定样本。
- uncertain：label 必须为 null；保存到候选数据中，不能强制当错误。

`prepare` 不读取模型分数来生成真值。schema 能检查材料完整性，不能证明事实正确或裁判独立。reviewer 必须如实记录；同一 Agent 多次判断不是独立复核。

## 模型输入与标签隔离

Kev 导出文件具有 `state`、`questions`、`_meta`；每题包含 noul 类型、instructions、bool label 和任务 src。Kev 的真实 `api_request` 只投影 state 和问题文本，不发送 label 或 _meta。裁判 rationale、reviewer、evidence 不自动送给模型。

若采用有证据审核，主动把可公开给审核器的证据放到 state；标签的裁定结论和“此项错误”提示不能放进去。当前程序不可能自动发现所有自然语言标签泄露，需人工检查样本。使用语义中性的 ID，避免 ID 包含答案。

## 分区与完整词条

先建立词族清单，再分配四个分区。同 case 的所有请求必须同 scope、词族、分区。相关端点词也可能造成泄露；选择 family_id 时按实际任务聚类，不能仅机械使用词头。

代码会检查词族冲突和规范化后完全相同的 state 跨分区，但不能识别所有语义近重复。去重审查、来源隔离和未见词族评估仍是数据建设工作。

只有独立检查了整份贡献、列齐所有必要审核问题后才能设为 complete。一个 case 可以拆成多个请求，但所有请求都要纳入结果。人工仅挑出一个正确发音问题的记录必须标为 partial。complete 的声明不会自动证明知识无遗漏；应保留贡献原稿和审核项清单作为证据来源。

数据集必须同时有四个非空分区。各类别正反例数量、词族数量、长度和来源分布应人工审查；文件行数不能替代独立词条数量。首轮建议用少量高质量记录跑通，再逐步增加 2,000–5,000 条可信判断。

## 规范仍需裁定

`rubrics/en-v1.json` 是可运行初稿，特意标为 draft。批准前必须补齐：音节采用的具体词典规范、允许的 IPA 变体、语素共时/历时边界、各关系类型定义、词义粒度、古语与地域范围。禁止仅为绕过程序把 draft 改成 approved。

现有 Vocnet 模型输出和旧“人工正确”记录只作为候选材料，不直接导入正式数据。旧裁定曾被推翻，要逐项保存复核理由。资料使用权限由数据来源决定；本项目不自动下载或重新分发词典正文。

## 冻结产物

`prepare` 保存原始标注快照、规则快照、四个 Kev JSONL 及 SHA-256 manifest。后续操作校验文件哈希，并核对导出内容与标注快照一致。不要修改已冻结目录；修订时创建新版本。格式示例只能通过 `--examples` 导出，产物明确标记 example_only，训练命令拒绝它。

## 自由问题与研究数据（v2）

`format_version: 2` 的 rubric 只定义 `rules.default`，所有问题共用判断契约；`kind` 是开放的统计标签，不决定可接收的问题类型。旧 v1 格式和验证仍可使用。

`source_labeled` 表示保留公开来源的原有标签，`source_derived` 表示从明确结构化字段按可检查规则推导的标签；两者都必须提供来源、定位、理由和归属，不代表 Lev 独立人工裁定。模型分数不能作为标签。

显式 `prepare --research` 接受 v2 `status: research` 规范及这些来源标签，冻结产物写入 `research_only: true`，训练模型也保留该标记。这只允许透明的研究实验，不改变生产导出的 `approved` / `adjudicated` 要求；不能通过修改 manifest 标志将研究数据变成生产金标准。原有 `draft` / `uncertain` 和示例数据仍禁止进入训练。

首批研究数据的来源、采样、分区和预先固定实验方案见 `docs/experiments/en-general-002.md`。

## 独立词汇测试包

35,000 词目标使用 `lev-vocabulary-test-v1` test-only 包，详见 [词汇基准](vocabulary-benchmark.md)。它与训练用四分区包分开，不能进入训练入口。单词主指标为全部已排定项通过的词数，仍不能把 partial 样本冒充完整词条审核。
