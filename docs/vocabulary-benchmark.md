# 美国英语 35,000 词研究测试集

当前交付为 `artifacts/en-us-vocab-35000-v3`。**已建立测试数据并完成来源审计，尚未运行模型，因此没有新的模型准确率。** 旧 v0.2 的 84.87% 不适用于这个基准。

| 覆盖项目 | 数量 |
|---|---:|
| 唯一词头 | **35,000** |
| WordNet 来源义项 | 80,201 |
| Wiktionary 补充义项记录 | 3,217 |
| 判断题 | **368,056** |
| 请求（每个最多 8 题） | 56,639 |
| 每词判断数 | 6–216 |
| 有 CMUdict 发音测试的词 | 26,070 |
| 缺少该发音来源的词 | 8,930 |

义项来自不同词典，不能将两种来源的义项数相加宣称它们全部互不重复。每词的全部已排定题目通过才算该词在本套测试通过；所有记录仍为 partial，不能称为完整词条验证。语境消歧、搭配、语域、构词和词源还没有全面测试。

## 单词怎么选

使用 [Brysbaert 等人的原始美国受试者词汇认知数据](https://osf.io/g4xrt/)，按美国认知指标排序，同分使用 SUBTLEX-US 词频及拼写。认知是人群统计，不是每位美国人都认识该词的证明，也不是词义掌握深度。

选择有 WordNet 或补充 Wiktionary 独立词条证据的 35,000 个唯一词头。同一词的词性、义项、发音变体合并计数，不把上下文中顺带出现的词算进来，也不把仅有屈折关系的新增条目当独立词头凑数。计数单位不是 35,000 个词族，派生词可以分别计词。大写 I 保留代词释义，计数时合并到 i。

WordNet 以实词为主，因此补充了核心功能词和原候选中认知排名前 5,000 的来源缺口。`you`、`the`、`if`、`I`、`with` 等已实际包含。这个版本仍不是原始认知研究最高的 35,000 项；未纳入的项及原因全部保留在 `selection-audit.jsonl`。多词表达、连字符词及大多数大小写专名不在本版计数范围。版本 v1、v2 及发现的问题均保留，均未进行模型评分。

## 问题与真值

测试指定来源收录的词义、词性和发音。词义反例在目标的所有已记录义项中查重，发音反例检查所有合法来源变体；没有将模型判断当金标准。问题明确以指定词典库存为范围：来源未收录不等于自然语言中绝对不存在。

因此这是**有证据的词汇基础研究基准**，不是所有自由英语问法、全部词义用法或美国母语者综合能力的完整验收。不能把对某一词典版本的记忆与普遍语义能力混为一谈。来源标签经过逐项核对，但没有冒称独立专家裁定。

## 既有接触分组

| 目标词族接触 | 词数 |
|---|---:|
| 既有 Lev 训练接触 | 2,984 |
| 仅既有开发／校准／测试接触 | 1,256 |
| 未检测到既有目标词族接触 | 30,760 |

这些分组根据公开词形、派生关系、历史词族与文档联系计算。它们不证明基础模型预训练未见过这些词，也没有把反例所用释义或句中偶然用词认作完全无接触。常用词没有为了追求“全未见”被删去。未来报告分别列出三个分组，不将总成绩冒称纯未见词泛化。

## 文件与复现

- `artifacts/en-us-vocab-35000-v3/words.txt`：35,000 个词，按选择顺序排列。
- `words.jsonl`：每词原始认知评分、来源覆盖、既有接触、题数、请求 ID、缺口。
- `records.jsonl`：完整标签与证据；`test.jsonl`：Kev 输入格式。
- `manifest.json`：全部产物哈希；`source-manifest.json`：原始来源与构建器哈希。
- `runs/en-us-vocab-35000-v3-source-audit.json`：368,056 个来源标签和实际导出逐项核验。
- `runs/en-us-vocab-35000-v3-registry-verification.json`：实际 35,000 词的聚合与漏题检查、核心词检查、下载哈希。

冻结方案见 [v3 协议](experiments/en-us-vocab-35000-v3.md)。这套 test-only 包具有独立格式，不能传给训练用 `lev prepare/train`；没有伪造三个空分区来绕过原有四分区契约。

重新构建必须选新的输出目录：

```sh
uv run python scripts/build_vocabulary_benchmark.py \
  --out artifacts/en-us-vocab-reproduction-001 \
  --version en-us-vocab-35000-v3 \
  --protocol docs/experiments/en-us-vocab-35000-v3.md \
  --wiktionary data/raw/en-us-vocab-35000-v1/wiktionary-supplement

uv run python scripts/audit_vocabulary_benchmark.py \
  --dataset artifacts/en-us-vocab-reproduction-001 \
  --out runs/vocab-reproduction-audit-001.json
```

获得这套数据一次固定模型运行的 Kev 原生 `rows.json` 后，按词评分：

```sh
uv run python scripts/score_vocabulary_benchmark.py \
  --dataset artifacts/en-us-vocab-35000-v3 \
  --rows runs/YOUR_FIXED_EVALUATION/native/rows.json \
  --out runs/YOUR_FIXED_EVALUATION/word-report.json
```

评分按冻结的 35,000 个词做分母；漏题保留为未完成，不算正确。输出逐词结果、按词宏平均、按认知排序每 5,000 词分组、既有接触分组、各项错误及缺口。源码验收 46 项 unittest 通过，ruff 通过；这些不是模型执行证据。

逐词及汇总报告另外保留 `false_acceptances`（错误事实被接受）、`false_rejections`（正确事实被拒绝）和正反标签数量。缺失预测只计入 `missing_judgments`，不能当作正确拒绝，也不能混入误拒。阈值相等时按接受处理，与总准确率一致。新增统计不改变任何冻结题目、标签或模型选择规则。

当前候选训练的目标词族接触另存于 `runs/en-joint-001-vocabulary-exposure/`，不覆盖历史分组：2,220 个词检测到该轮训练目标词族接触，32,780 个未检测到。此计数不包括文章中偶然出现的词，也无法确定基础模型预训练接触；它不是掌握词数或准确率。

完成实际分片评估并合并后，可生成本轮接触分组报告。命令同时核对词表、训练数据与实际检查点内容哈希；尚未有模型结果时不能生成成绩：

```sh
uv run --extra modal python -m lev.exposure_report \
  --report runs/lev-joint001-v03-vocabulary/word-report.json \
  --exposure runs/en-joint-001-vocabulary-exposure \
  --checkpoint runs/modal-lev-joint001-v03-train-01/result/checkpoint \
  --out runs/lev-joint001-v03-vocabulary/candidate-exposure-report.json
```

原始美国认知数据为 CC BY-NC-SA 4.0，词表由其排序产生，本版限定研究；WordNet、Wiktionary 和 CMUdict 保留各自来源许可，不笼统重新许可。未部署、未发布、未用测试答案训练。
