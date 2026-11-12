# en-joint-002：训练语法样本补充候选

本数据版本为可能的后续开发迭代准备，不改变正在训练的v0.4，也不承诺启动新GPU任务。若v0.4通过原开发门槛，先完成其固定评估；若不通过，再依据开发证据登记下一实验。

原`en-joint-001`全部样本和四分区原样保留。仅从已固定BLiMP快照（commit `3e56b06fcabca9b30822fc66435fca6b1aa40bb1`，zip SHA-256 `172a1a714d72e72a89e8384dd531483f3d0f8a9aee3d33dccaf7992d8cecbb73`）补充训练语法最小对。

限定当前训练已有的六类construction：determiner_noun_agreement、npi_licensing、filler_gap_dependency、subject_verb_agreement、argument_structure、irregular_forms。任何历史留出construction不得进入。原生每文件按不含标签的稳定pair身份哈希排序，选择前100个符合要求的新pair。保留所有67,000个源pair的纳入/排除审计，超出配额也写明原因。所有曾在历史或当前任意分区出现的句子、选中pair之间重复的句子以及good/bad完全相同的pair整体排除，不拆对、不静默删除。

每pair的正反句使用同一条固定自然语言问法。问法从两个肯定问题和两个否定问题中由pair身份哈希决定，独立于good/bad标签；否定问题对原始标签作明确逻辑取反。此步骤不使用模型判断。语义仍是BLiMP设定任务中的可接受性，不能将模板生成标签冒称逐句独立专家裁定。

新记录全部`source_labeled`、`partial`、`research_only`。BLiMP为CC BY 4.0；原联合包的非商业研究限制继续适用。句子中的偶然词汇不计为监督词汇覆盖。新包不得用于生产发布。

验收必须逐字节确认development/calibration/test导出与en-joint-001相同，核对全部源标签、问句极性、成对关系、构式分区和句子去重；运行固定Kev编码准入。准备好数据不代表模型执行或效果提升，也不改变此前测试开放纪律。
