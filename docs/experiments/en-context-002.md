# en-context-002：真实语境与写作纠错补充

2026-10-21 完成数据构建与本地验收，尚未用于模型训练。冻结包为 `artifacts/en-context-002`，来源与全量排除记录在 `data/research/en-context-002`。这是补充研究包，不替换 v0.2 的历史实验，也不修改冻结的 35,000 词基准。

## 实际交付

| 分区 | 判断／请求 | 篇章纠错判断 | 词义语境判断 | 显式目标单词 | 独立分组 |
|---|---:|---:|---:|---:|---:|
| train | 3,721 | 3,691 | 30 | 17 | 1,637 |
| development | 147 | 139 | 8 | 8 | 78 |
| calibration | 118 | 114 | 4 | 4 | 61 |
| test | 159 | 140 | 19 | 19 | 89 |
| 合计 | **4,145** | **4,084** | **61** | **48** | **1,865** |

每条请求只有一个 Boolean 问题。所有记录均为 `partial`，不能算完整词条能力。显式目标单词来自 WiC；篇章中的词只是上下文，**不能把篇章词频统计冒充单词审核覆盖**。这批数据也没有单独裁定的搭配、词形挑战集，不能声称这些缺口已全部补齐。

## 来源与标签

1. [WiC v1.0](https://pilehvar.github.io/wic/)：原始专家来源的同词异境／同义判断，保留 T/F 标签、词性、两个上下文及目标词索引。标签为 `source_labeled`，不是 Lev 独立人工复核。zip SHA-256 为 `f1a2fb67d903c5b9b1180f1035d4228f7c0254e8f5f868d556235457046bd4b2`。
2. [Cambridge Write & Improve，BEA2019 v2.1](https://www.cl.cam.ac.uk/research/nl/bea2019st/)：使用原始 JSON 篇章与一名标注者的字符级修改。以全部明确修改同时作用后的文本作为“满足该标注标准”的正例；只有原稿确实发生修改时才建立对应负例。保留拼写、标点、语法、用词方面的原始修改，不把这些标签升级为“英语绝对无误”。标签为 `source_derived`。tar SHA-256 为 `d5cbf68cda3da0c3af69dd672614d07287bfe996b87da0c75051d5349d76c666`。
3. 英语 Wiktionary/Kaikki：补下载 56 个候选词的词条，只用于词形及别名隔离，不提供新增判断标签。结构化 `form_of` / `alt_of` 和明确、带词条链接的 “standard spelling of …” 都用于闭包。`honour/honor`、`harbour/harbor`、`neighbour/neighbor` 不被当成陌生词族。

模型输入不包含原始修改列表、原稿／改稿标记、裁判理由和金标签。原稿和改稿的分组与来源定位放在模型不可见元数据中。

**使用范围：仅非商业研究。** WiC 为 CC BY-NC 4.0；W&I 的许可证明确排除产品／服务用途且限制传播。LOCNESS 虽在官方压缩包内，但本包未导出其样本。原始许可和归属见 `data/raw/en-context-001/ATTRIBUTION.md` 及压缩包内原文。没有上传、发布或部署这些数据。

## 隔离与完整候选账目

先完成全量词族、目标词形和重复语境闭包，再分配分区。35,000 词测试目标和历史非训练词族全部保留；新词义样本不能使用这些目标词族。原生 WiC test 不进入开发／校准／训练，原生 dev 不进入训练，原生 train 不冒充新留出。

篇章审核以作者、文档、原稿／改稿规范化重复文本的传递闭包分组。原生 W&I dev 作者及其关联重复文档不进入训练。新 development/calibration/test 在原生 dev 的作者组上确定性分配。**这是新作者／文档留出，不是未见语法构式留出；也不代表篇章里的每个词都未见。**

原始 10,816 个候选单位都有选择记录：

| 来源与处理 | 候选数 |
|---|---:|
| WiC 入选 | 61 |
| WiC 命中冻结词汇或历史留出词族而排除 | 7,405 |
| W&I 入选篇章 | 2,044 |
| W&I 含无法确定替换内容的修改 | 1,088 |
| W&I 与原生留出作者／文档关联 | 115 |
| W&I 缺作者身份 | 53 |
| LOCNESS 未纳入 | 50 |

2,044 篇入选 W&I 文章生成 4,084 条判断，其中 4 篇没有修改，仅保留一个正例；其余保留原稿／改稿成对样本。没有用模型分数制造金标签或挑选样本。

## 为什么词义增量不大

用户要求的 35,000 词目前已经整体冻结为 test-only 基准。严格保留所有这些词族时，WiC 训练部分仅剩 30 个判断、17 个显式目标单词。这是当前协议的真实限制，不能靠复制常用词答案到训练集后继续声称未见词评估。

本次结果主要补强**真实篇章纠错任务数据**，没有达成“35,000 常用词的语境、搭配、词形训练都齐了”。若后续目标是训练时覆盖这 35,000 词，需先明确区分“见过词汇的新语境验收”与“未见词族泛化”，建立相应的新评测协议；旧测试集及其历史接触记录必须保留。本次没有擅自改变该协议。

## 验收与失败记录

- 全量来源审计：4,145 个判断逐项核对原始标签／字符修改；10,816 个候选逐项对账；642 个来源及代码文件核对 SHA-256。独立审计器用逆序替换重建纠错稿，与构建器的顺序拼接实现分开。
- 导出审计：四分区的实际 Kev JSONL 与冻结标注逐条一致；所有词族／作者／重复文本分组无跨分区泄露。
- 真实 Kev 序列化函数检查：固定提交 `1f04cfa28e6a708485b28a9585010b9e436b776e` 的 `load_records` 和 `api_request` 在全部 4,145 条上通过。通过 AST 提取其原函数以避免加载 GPU 栈；这是输入兼容性证据，**不是模型执行**。
- 54 个 unittest 通过，包含 Unicode 分隔符回归和上游格式契约。Ruff 通过。
- 固定 Qwen3.5-4B-Base revision 的真实 tokenizer 和固定 Kev 的 `to_record/materialize/encode` 准入检查在全部 4,145 条上通过：零超限、零截断。最长状态 2,329 tokens、最长打包 2,428 tokens，分别低于 7,552 / 9,216 上限。只下载 tokenizer，没有加载模型权重。临时依赖版本与 Kev lock 一致：transformers 5.17.0、tokenizers 0.23.2、pydantic 2.13.5。
- `en-context-001` 首次导出失败：真实篇章包含 Unicode 段落分隔符，旧 JSONL reader 的 `splitlines()` 将字符串内部误拆成记录。新增回归测试先复现失败，再改为只按 LF 分行。失败数据目录和 `runs/en-context-001-build.log` 均保留；最终另建 `en-context-002`，未覆盖旧版本。

核心验收文件：`runs/en-context-002-source-audit.json`、`runs/en-context-002-kev-contract.json`、`runs/en-context-002-tokenization.json`、`runs/en-context-002-unittest.log`。没有生成任何新模型准确率、检查点或训练成功声明。

## 本地复现

使用已留存的原始字节；输出目录必须不存在：

```sh
uv sync --locked
uv run python scripts/build_context_corpus.py \
  --out data/research/en-context-002-reproduction \
  --artifact artifacts/en-context-002-reproduction
uv run python scripts/audit_context_corpus.py \
  --provenance data/research/en-context-002-reproduction \
  --artifact artifacts/en-context-002-reproduction \
  --out runs/en-context-002-reproduction-audit.json
LEV_KEV_PATH=/Users/saltbo/Develop/oss/kev \
  uv run python -m unittest discover -s tests -v
```

分词准入可独立复现（仅安装临时 tokenizer 依赖，不修改 Kev）：

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run \
  --with transformers==5.17.0 --with tokenizers==0.23.2 --with pydantic==2.13.5 \
  python scripts/check_context_tokenization.py \
  --kev /Users/saltbo/Develop/oss/kev \
  --out runs/en-context-002-tokenization-reproduction.json
```

它是独立补充包，不能直接把旧训练包拼接进去后宣称仍满足本包隔离。任何联合训练包都要重新做包含历史评估的闭包审计。
