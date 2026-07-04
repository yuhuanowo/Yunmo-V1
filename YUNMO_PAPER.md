# Yunmo：面向繁體中文（臺灣）之全功能小型語言模型——MiniMind v3 的增量復現與多教師身份去污染

# Yunmo: A Full-Capability Small Language Model for Traditional Chinese (Taiwan) — An Incremental Reproduction of MiniMind v3 with Multi-Teacher Identity De-contamination

> **文件性質**：本檔為論文手稿草稿（manuscript draft），採 IMRaD 結構，供投稿前撰稿與審閱使用。所有量化結果均經程式驗證（參數實例化、逐筆計數、打包實測、打包後端到端解碼核對）。尚未執行之評測以 *planned* 明確標示，不得預先陳述為結論。底層逐項事實與腳本細節見配套之 `YUNMO_SPEC.md`。
> **狀態**：模型與資料工程已完成並通過驗證；訓練（training）與評測（evaluation）為後續工作。最後更新：2026-07-04。

---

## 摘要（Abstract）

本文提出 **Yunmo v1**，一個面向繁體中文（臺灣用語）的全功能小型語言模型（small language model, SLM）。方法上，本研究將 MiniMind v3 的完整訓練語料以 OpenCC `s2twp`（簡體至臺灣正體之片語級轉換）逐筆 1:1 轉為繁體，併入臺灣原生增量語料（繁體維基百科、臺灣在地指令），並重新訓練一個 24,000 詞的繁體特化分詞器（tokenizer）。模型架構沿用 MiniMind v3（decoder-only Transformer，含 RMSNorm、SwiGLU、旋轉位置編碼 RoPE、分組查詢注意力 GQA 與 QK-Norm），僅刻意調整兩處：層數由 8 增至 24、詞表由 6,400 換為 24,000，總參數量為 **195.4M**。本文的一項關鍵工程貢獻是**開訓前的多教師身份去污染（multi-teacher identity de-contamination）**：由於原語料源自對多個教師模型（minimind、Qwen、DeepSeek-R1、ChatGPT 等）的蒸餾，助手回覆中普遍存在「自稱為該教師模型」之污染，若直接訓練將導致身份混亂。本研究設計一套以句法錨定、並嚴格區分「真實自稱」與「知識討論／角色扮演」的清理流程，將此類自稱改寫為一致的 Yunmo 身份，同時保留世界知識與指令多樣性。經全量複驗，訓練語料中教師自稱由數千筆降至殘留率 0.0003%（SFT）與 0.00066%（pretrain），且經打包後之實際訓練 token 流抽樣（5.12M tokens）驗證為 0 筆教師自稱。最終語料含約 62 億（6.21B）token，規劃於 36 小時硬算力預算（RTX PRO 6000, 96 GB）內完成 pretrain 與 SFT。本文明確界定其貢獻邊界：可靠且既成之優勢為臺灣原生知識、實測較 MiniMind 分詞器少 46.4% 之繁體 token 數、以及一致之品牌身份；「整體能力超越 MiniMind」則為**待評測驗證之假設**，非既成結論。

**Abstract (English).** We present **Yunmo v1**, a full-capability small language model (SLM) specialized for Traditional Chinese as used in Taiwan. We convert the entire MiniMind v3 training corpus to Traditional Chinese via OpenCC `s2twp` on a one-to-one basis, augment it with Taiwan-native data (Traditional Chinese Wikipedia and localized instructions), and retrain a 24,000-token Traditional-Chinese tokenizer. The architecture follows MiniMind v3 (a decoder-only Transformer with RMSNorm, SwiGLU, RoPE, grouped-query attention, and QK-Norm), altering only depth (8→24 layers) and vocabulary (6,400→24,000), yielding **195.4M** parameters. A key engineering contribution is **pre-training multi-teacher identity de-contamination**: because the source corpus is distilled from multiple teacher models (minimind, Qwen, DeepSeek-R1, ChatGPT), assistant turns frequently self-identify as those teachers. We design a syntactically anchored cleaning procedure that rewrites genuine self-identifications to a single consistent Yunmo identity while strictly preserving factual discussion and role-play. After cleaning, residual teacher self-identification falls to 0.0003% (SFT) and 0.00066% (pretrain), and a 5.12M-token sample of the actual packed training stream contains zero teacher self-identifications. We explicitly delimit our claims: the reliable, realized advantages are Taiwan-native knowledge, a measured 46.4% reduction in Traditional-Chinese token count versus the MiniMind tokenizer, and consistent model identity; superiority over MiniMind in general capability remains a *hypothesis to be validated by evaluation*.

---

## 1. 引言（Introduction）

小型語言模型（SLM）以有限算力達成可用之語言能力，於教育、在地化部署與研究復現具重要價值。MiniMind 為此類模型之代表性開源專案，提供從分詞器訓練、預訓練、監督微調（SFT）到偏好對齊（DPO/RL）之完整、可低成本復現之管線。然而，MiniMind 之語料以簡體中文與英文為主，對**繁體中文（臺灣用語）**之支援為結構性缺口：其分詞器未針對繁體優化，語料亦不含臺灣在地知識。

本文的目標並非追求更大規模以取得更強能力，而是在 MiniMind 既有、經驗證之架構與配方上，進行一次**面向繁體臺灣的增量復現（incremental reproduction）**，並解決復現他人蒸餾語料時普遍存在、卻少被討論的**身份污染（identity contamination）**問題。本文之主要貢獻如下：

1. **繁體臺灣特化之全功能 SLM**：以「只增不減」原則將 MiniMind 全語料 1:1 轉繁並注入臺灣增量，保留其全功能（多輪對話、`<think>` 推理、工具呼叫、英文與程式碼），使最終語料為 MiniMind 語料之繁體超集。
2. **原生繁體分詞器**：重訓 24,000 詞分詞器，於繁體臺灣文本上較 MiniMind 之 6,400 詞分詞器**減少 46.4% 之 token 數**（實測）。
3. **多教師身份去污染（本文方法學核心）**：提出一套區分「真實自稱」與「知識討論／角色扮演」之清理流程，將教師自稱一致化為 Yunmo 身份，並經全量與 token 流雙重驗證。
4. **可復現之工程規格與誠實之能力邊界**：完整揭露資料組成、語言分布、已知限制與殘留污染率，並將未經驗證之能力主張明確標示為假設。

本文其餘章節安排如下：第 2 節回顧相關工作；第 3、4 節分述模型架構與分詞器；第 5 節為本文核心，說明資料建構與身份去污染；第 6 節為訓練配置；第 7 節報告打包後之端到端資料驗證；第 8、9 節討論限制與未來工作。

---

## 2. 相關工作（Related Work）

**小型語言模型與深窄結構。** 於固定參數預算下，模型之深度與寬度之取捨影響顯著。MobileLLM（Liu et al., 2024）指出在次十億（sub-billion）參數規模，**深而窄**之結構於常識推理、問答與閱讀理解任務普遍優於淺而寬之結構。本文據此將層數設為 24（MiniMind-3 為 8），並將隱藏維度維持於 768，落於 MiniMind 自身消融所示之有效區間 [512, 1536] 內。

**Transformer 元件。** 本文沿用 MiniMind v3 所採之現代 decoder-only 元件：RMSNorm（Zhang and Sennrich, 2019）、旋轉位置編碼 RoPE（Su et al., 2021）、SwiGLU 前饋（Shazeer, 2020）、分組查詢注意力 GQA（Ainslie et al., 2023），以及對 query/key 於旋轉前施加逐頭正規化之 QK-Norm（見 Qwen 系列）。

**資料受限之縮放。** Chinchilla（Hoffmann et al., 2022）給出計算最優之 token/參數比約 20。於資料受限情形，Muennighoff et al. (2023) 指出對唯一資料重複至約 4 個 epoch 之效益近似於等量新資料，逾此則報酬遞減。本文之曝光估計與 epoch 上限設計即據此。

**身份污染。** 以更強教師模型蒸餾產生指令資料為常見作法，惟其副作用之一為教師之自我認同（self-identity）被一併蒸餾入語料，導致學生模型於被詢問身份時自稱教師。此問題於實務中普遍，卻鮮少被系統性處理與量化。本文將其形式化為一項可量測、可驗證之資料清理任務。

---

## 3. 模型架構（Model Architecture）

Yunmo 之架構完全沿用 MiniMind v3 之模型實作（`model_minimind.py`），僅刻意調整深度與詞表二者。模型不使用獨立之 `config.json`，其超參由 `MiniMindConfig` 之預設值與啟動參數決定。核心配置如表 1。

**表 1：模型配置（Model configuration）**

| 超參 (Hyperparameter) | 值 (Value) | 說明 (Note) |
|---|---|---|
| hidden_size | 768 | 落於有效區間 [512, 1536] |
| num_hidden_layers | **24** | **改動①**：8→24（加深以增容量） |
| num_attention_heads | 8 | |
| num_key_value_heads | 4 | GQA, n_rep = 2 |
| head_dim | 96 | 768 / 8 |
| intermediate_size | 2432 | ⌈768·π/64⌉·64；SwiGLU 標準擴張 |
| vocab_size | **24000** | **改動②**：6400→24000（繁體） |
| max_position_embeddings | 32768 | |
| rope_theta | 1×10⁶ | YaRN 預設關閉 |
| rms_norm_eps | 1×10⁻⁶ | 於 fp32 計算 |
| tie_word_embeddings | True | |
| use_moe | False | 稠密模型 |

**參數量。** 經模型實例化後精確計數，總參數量為 **195.4M**，約為 MiniMind-3（64M）之 3.05 倍。其組成為：詞嵌入（權重共享，24000×768）18.43M（約占 11%），每層 7.375M（注意力 1.770M + 前饋 5.603M），24 層合計非嵌入參數 177.0M。

**繼承自 MiniMind v3 之設計（未更動）。** 模型採 Pre-Norm 殘差結構；RMSNorm 全域 eps 為 1×10⁻⁶（於 fp32 計算後轉回原精度）；注意力為 GQA（8 個 query 頭、4 個 key/value 頭）並對 query/key 施加逐頭 RMSNorm 之 QK-Norm（於 RoPE 之前，value 不正規化）；訓練與 prefill 使用 `scaled_dot_product_attention`（flash/SDPA），增量解碼使用 eager 實作；前饋為無偏置之 SwiGLU；詞嵌入與輸出投影權重共享；損失於模型內部以位移一格之交叉熵計算（`ignore_index = -100`，傳入標籤與輸入 1:1 對齊，位移於模型內完成）。MoE 分支存在但停用（`use_moe = False`）。

**設計理由。** 選擇深而窄（24 層）之依據為 MobileLLM 之結論與 MiniMind 自身之維度消融；隱藏維度 768 使 d_head 不致過小、詞嵌入不致過窄。intermediate 維度 2432 由隱藏維度依 SwiGLU 標準公式導出，非自由超參。採稠密而非 MoE，因 MoE 需 2–4 倍資料以充分訓練專家，而本文之中文預訓練唯一 token 有限（約 2.35B），稠密結構於資料受限下最具效率。

---

## 4. 分詞器（Tokenizer）

本文重新訓練一個繁體特化之分詞器，演算法沿用 MiniMind 之 byte-level BPE（`add_prefix_space = False`、`min_frequency = 2`、以 byte-level alphabet 為初始字元集），詞表大小設為 24,000。特殊 token 共 36 個（ID 0–35），完整對齊 MiniMind v3 之定義（含 `<|im_start|>`、`<|im_end|>`、`<tool_call>`、`<think>` 等），以確保聊天模板、工具呼叫與推理標記之相容。

**訓練語料與規模。** 分詞器訓練語料約 100 MB（128,819 行），以多偏移點抽樣自已轉繁之預訓練與 SFT 語料。訓練於 34 秒內完成，產生 23,708 個合併規則，最終詞表恰為 24,000。

**壓縮效率（實測）。** 於繁體臺灣文本上，本分詞器之壓縮率為每 token 1.89 字元。與 MiniMind 之 6,400 詞分詞器相比，四段代表性繁體文本之合計 token 數為 90 對 168，即**減少 46.4%**（各段減少介於 41% 至 54%）。此為本文可靠且既成之優勢之一。

**設計教訓。** 首次嘗試以約 3 GB 語料訓練，因產生約 5,050 萬個唯一詞導致 BPE 之合併計算無法於合理時間內收斂。診斷發現 MiniMind 官方分詞器僅使用 10,000 行語料，其原理為 BPE 於少量語料即已飽和，過量語料反而有害。據此將語料縮減至 100 MB 並設 `min_frequency = 2`，方得成功。此觀察之啟示為：**分詞器不同於模型，會飽和，語料越多並非越好。**

---

## 5. 資料建構與身份去污染（Data Curation and Identity De-contamination）

### 5.1 核心原則：只增不減

本文之資料原則為：語料與 MiniMind **一致，僅施行簡體至繁體之轉換與臺灣增量，不進行語言過濾、品質過濾、去重、英文下採樣或長樣本丟棄**。此原則源於早期一條過濾管線於轉換時誤棄 753,663 筆（8.9%）英文樣本，經專案決策予以廢除。其邏輯後果為：最終語料為 MiniMind 語料之繁體**超集**，故其能力下限即為「繁體版 MiniMind 之全功能復現」。

此原則有二例外，皆非品質過濾：其一為丟棄「不含任何有效助手回覆」之殘缺記錄（SFT 4,480 筆）；其二為身份去污染，其性質為**改寫（rewrite）而非刪除**，詳見 §5.4。

### 5.2 最終語料組成

**表 2：最終訓練語料（逐筆計數與打包實測）**

| 語料 (Corpus) | 記錄數 (Records) | Token 數 | 打包區塊 (Blocks) |
|---|---|---|---|
| `pretrain_clean.jsonl` | 12,157,237 | 2,353,430,714 (2.35B) | 2,298,272 × 1024 |
| `sft_clean.jsonl` | 5,811,531 | 3,857,469,341 (3.857B) | 1,883,529 × 2048 |
| **合計 (Total, unique)** | | **≈ 6.21B** | |

預訓練語料由兩部分組成：MiniMind 本體之 `pretrain_t2t`（8,468,827 筆，1:1 轉繁）與臺灣增量之繁體維基（3,688,410 段落條目）。SFT 語料由 MiniMind 本體之 `sft_t2t`（5,109,432 筆）、臺灣在地指令 `tw-instruct`（486,580 筆）、R1 蒸餾推理 `distill_r1`（109,999 筆）與 235B 蒸餾推理 `qwen3_235b`（110,000 筆）組成；上述為原始血緣（合計 5,816,011），經清理丟棄 4,480 筆殘缺記錄後為 5,811,531 筆。

### 5.3 轉繁方法與已揭露之限制

轉換使用 OpenCC `s2twp`（片語級之簡體至臺灣正體轉換），每工作行程實例化一次；以字元範圍 U+4E00–U+9FFF 判別中文並僅轉含中文之記錄。為維持科學誠實，本文明確揭露下列限制（供論文 Limitations 引用）：

1. **SFT 語料以英文為主。** `sft_t2t` 經語言抽樣約 86–89% 為英文。因採只增不減，最終 SFT 為多語（英文為主）＋繁中增量，**並非以繁中為主之 SFT**。
2. **轉換閘之字元範圍限制。** U+4E00–U+9FFF 不含 CJK 擴充區，故極少數罕字略過轉換。
3. **s2twp 之已知偏差。** 存在少量一對多誤轉（如「鬱／郁」類）與非技術語境之過度在地化（如「对象→物件」），實測比例可忽略。
4. **`tool_calls` 欄位之簡體殘留（已於清理階段修復）。** 原 MiniMind 管線不轉換工具呼叫欄位，實測有 217,783 筆殘留簡體；本文於清理階段對該欄位之中文字串遞迴施行 s2twp，殘簡歸零（見 §5.4、§7）。

### 5.4 多教師身份去污染

**問題界定。** MiniMind 之 SFT 語料源自對多個教師模型之蒸餾，助手與系統回覆中大量出現「我＝某教師模型」之自稱（如「我是通義千問（Qwen），由阿里雲研發」）。若直接訓練，Yunmo 於被詢問身份時將自稱教師，對一個具品牌之繁體模型不可接受。此問題於預訓練語料亦存在（以扁平文本形式）。本文將此界定為一項可量測之清理任務，並於**開訓前**完成。

**清理策略（策略 C：中性化並注入一致身份）。** 核心設計為嚴格區分「真實自稱」與「非自稱之合法內容」：

- **改寫（rewrite）**：將助手／系統回覆中「第一人稱＋教師／開發者名＋開發動詞」之自稱句，統一替換為單一正式身份句「我是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。」，並保留句首問候與句子其餘內容。經典之兩句式自介（「…語言模型。我的中文名叫通義千問，英文名叫 Qwen。」）於改寫後折疊其重複。
- **保留（preserve，本方法之關鍵）**：教師名之**知識性討論**（「Qwen 是阿里巴巴開發的模型」）、**角色扮演**（「假如我是某公司執行長」「面試者：我叫李明」「主持人 ChatGPT」）、**語氣用法**（「我是說……」）與**事實陳述**（「OpenAI 開發的 ChatGPT」）一律不予更動，以避免破壞世界知識與指令多樣性。
- **偵測方法**：以句法錨定之規則集偵測自稱，涵蓋多種語序、所有格式（「我是阿里雲的 Qwen」）、「基於」式（「我是基於 Qwen 模型」）、經典中英文名式自介，以及 markdown 與全形引號之容錯；同時以排除規則（「假如我」「我認為」「我使用」「扮演」「我是說」等）過濾非自稱語境。規則以人工判讀逾 45 個代表性樣本並輔以對抗性測試迭代收斂，每一輪皆自未修改之原始語料重新執行（非累積式），以杜絕誤刪之累積。
- **特例處理**：對拒答模板「作為由 OpenAI 開發的 AI 語言模型，我無法……」採**子句替換**，僅替換身份子句而保留拒答內容；對 minimind／jingyaogong（非真實世界主題）採全域安全替換；對工具呼叫欄位施行遞迴轉繁。

**清理結果（全量複驗）。** 表 3 報告清理前後之量化結果。

**表 3：身份去污染之量化結果（Quantitative results of de-contamination）**

| 指標 (Metric) | SFT（清理前→後） | Pretrain（清理前→後） |
|---|---|---|
| minimind / jingyaogong 自稱 | 66,984 → **0** | 0 → 0 |
| `tool_calls` 殘留簡體 | 217,783 → **0** | 不適用 |
| 教師「我是 X」自稱 | 數千 → — | 1,074 → 80 |
| 「作為由 X 開發」拒答型 | 192 → 3 | 542 → 0 |
| **殘留合計（廣義偵測）** | 數千 → **16 (0.0003%)** | 1,074 → **80 (0.00066%)** |
| Yunmo 身份注入（次數） | 0 → 22,426 | 0 → 3,173 |

**殘留之誠實揭露。** 清理後之殘留為異質性長尾（罕見措辭，如「由阿里雲提供支援」、「我是使用 OpenAI 資料集訓練的」、「AI 語言模型（Qwen）」括號式），部分並為應予保留之角色扮演（如「銷售員 ChatGPT」）。進一步追逐將提升誤判（false positive）風險並可能損及合法內容，故本文**如實報告殘留率而不宣稱歸零**。此外，預訓練為無監督文本、身份信號較弱，最終模型身份主要由 SFT 之 22,426 次一致注入主導。

---

## 6. 訓練配置（Training Setup）— *planned*

> 本節之配置已鎖定，訓練將於後續執行；以下為將採用之設定，尚無訓練結果。

**算力預算。** 訓練於租用之 RTX PRO 6000（Blackwell 架構，96 GB）進行，硬性上限為 36 小時，於此預算內完成 pretrain 與 SFT。為因應機器於期滿後回收，本文採**時間封頂（time-capped）**策略：以 `--max_minutes` 分配約 900 分鐘予 pretrain、約 1080 分鐘予 SFT，到時存檔並停止，藉此在吞吐未知之情形下確保於期限內完成。

**表 4：鎖定之訓練超參（Locked training hyperparameters）**

| 項目 | Pretrain | SFT |
|---|---|---|
| 資料載入 | packing（EOS 分隔，零填充） | packing + assistant-only loss mask |
| seq_len | 1024 | 2048 |
| batch × 累積 (accum) | 128 × 8 | 64 × 8 |
| 有效 batch | 1,048,576 tokens/step | 1,048,576 tokens/step |
| 峰值學習率 (peak lr) | 3×10⁻⁴ | 5×10⁻⁵ |
| epoch 上限 | 6（實際由時間封頂決定） | 6 |
| 起點 | 隨機初始化 | 承接 pretrain 權重 |

共通設定為 AdamW 最佳化器、bf16 混合精度、梯度裁剪 1.0、餘弦學習率排程（峰值為 lr、下限為 lr/10、無 warmup）、隨機種子 42（含分散式 rank 偏移），並支援斷點續訓。

**序列打包（packing）之理由。** MiniMind 之預訓練樣本多為短問答，其原配方以填充（padding）處理短序列。於 1024 之序列長度下，填充將浪費大量算力於無損失之 padding token（本文早期實測填充浪費達 54–66%）。本文之語料含長維基條目與長推理，採 packing（以 EOS 分隔並串接）可消除填充浪費並完整保留長上下文，使相同浮點運算量可處理更多有效 token；此為於 36 小時內完成訓練之必要條件。SFT 採 assistant-only 之損失遮罩，僅對助手 token 計算損失，並以跨區塊之遮罩完整保留長對話。

**曝光估計。** 於保守（128k tokens/s）至樂觀（213k tokens/s）之吞吐區間，pretrain 之曝光約為 2.9 至 4.9 個 epoch，SFT 約為 2.1 至 3.6 個 epoch，對應之 token/參數比約為 35 至 59，達到或超過 Chinchilla 之計算最優比（約 20）。依 Muennighoff et al. (2023)，此重複程度於資料受限下近似等值於等量之唯一資料。**惟參數是否獲得充分訓練，須經評測驗證，非既成結論。**

**下游對齊（本地執行）。** DPO、RLAIF（CISPO）與 Agent RL 之階段規劃於本地 RTX 4070 Ti（12 GB）執行，不佔用雲端之 36 小時預算。

---

## 7. 打包後之端到端資料驗證（End-to-End Data Verification）

本文強調：資料驗證不應止於管線機制之檢查，更須核對**實際輸入模型之 token**。為此，本文直接解碼打包後之二進位檔（`.bin`）並施行下列核對，全數通過（表 5）。

**表 5：打包後端到端驗證結果（Post-packing verification）**

| 檢查項 (Check) | 方法 (Method) | 結果 (Result) |
|---|---|---|
| 分詞器正確性 | 載入並查詞表 | vocab = 24,000；bos/eos = 1/2 ✓ |
| ids/mask 對齊 | 長度與逐位比對 | 皆 3,857,469,341、對齊 ✓ |
| 繁體臺灣 | 解碼多區塊 | 臺灣地名與正體字 ✓ |
| **教師自稱歸零** | 抽樣 2,500 區塊 / 5.12M tokens，正則掃描 | **0** ✓ |
| Yunmo 身份存在 | 同抽樣 | 702 次 ✓ |
| 注入之系統提示已修 | 解碼遮罩為 0 之片段 | 見繁體 Yunmo 系統提示 ✓ |
| 工具呼叫繁體 | 解碼工具定義 | 「時區名稱」等為繁體 ✓ |
| 聊天結構 | 解碼 | im_start/im_end/think/tool_call 正確 ✓ |
| 損失遮罩正確性 | 解碼遮罩為 0 之片段 | 僅含 system/user 與 assistant 標頭；助手內容全在遮罩為 1 之處 ✓ |

**遮罩分布。** SFT 之損失遮罩為 1（即計算損失之 assistant token）之比例約為 85%。此比例偏高係因推理類資料之 `<think>` 區塊甚長（第 99 百分位約 7,719 token）且全屬助手內容，並經解碼確認 user 與 system 片段確實排除於損失之外，故非缺陷而為語料組成之反映。此結果亦說明本 SFT 語料之 token 利用率高。

**預訓練二進位檔亦經同等驗證**：token 數精確符合 2,353,430,714；抽樣 1,500 區塊之教師自稱為 0。

---

## 8. 限制（Limitations）

1. **能力超越為假設而非結論。** 「Yunmo 整體能力超越 MiniMind」與「參數獲得充分訓練」須經訓練後之評測驗證。依冪律縮放，參數約 3 倍僅使損失下降個位數百分比，非能力之倍增。
2. **SFT 以英文為主。** 如 §5.3 所述，最終 SFT 為多語（英文為主）＋繁中增量，非以繁中為主之 SFT。
3. **殘留污染非零。** 身份去污染之殘留率為 SFT 0.0003%、pretrain 0.00066%，本文如實揭露而不宣稱歸零。
4. **過擬合風險。** 195.4M 參數於約 2.35B 之固定中文預訓練語料上多輪重複，較大模型更易記憶，須以留出集損失與評測早停把關。
5. **對標更強模型不切實際。** 與 Qwen3-0.6B 等模型於中文知識之直接對標並非本文目標，因兩者受同一資料天花板限制且屬不同規模級。

---

## 9. 結論與未來工作（Conclusion and Future Work）

本文完成一個面向繁體中文（臺灣）之全功能小型語言模型 Yunmo v1 之**模型與資料工程**，其確定達成之貢獻為：MiniMind v3 之繁體全功能增量復現、實測減少 46.4% 之繁體分詞 token、臺灣原生知識之注入，以及一套經全量與 token 流雙重驗證之多教師身份去污染方法（訓練語料實測 0 筆教師自稱、殘留率 0.0003%／0.00066%）。

**未來工作（*planned*）**包括：(i) 於 36 小時預算內執行 pretrain 與 SFT；(ii) 以 `lm-evaluation-harness` 於 C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social-IQa 等基準，並另建與訓練集獨立之繁中評測集，量測繁中知識與推理；(iii) 新增**身份一致性評測**，量測模型於身份詢問下答覆 Yunmo 之比率；(iv) 於本地執行 DPO/RLAIF/Agent RL 之下游對齊。上述評測完成前，本文不對能力超越作任何既成之陳述。

---

## 參考文獻（References）

- Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP.
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla).* NeurIPS.
- Liu, Z., et al. (2024). *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* arXiv:2402.14905.
- Muennighoff, N., et al. (2023). *Scaling Data-Constrained Language Models.* NeurIPS.
- Shazeer, N. (2020). *GLU Variants Improve Transformer (SwiGLU).* arXiv:2002.05202.
- Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE).* arXiv:2104.09864.
- Zhang, B., and Sennrich, R. (2019). *Root Mean Square Layer Normalization (RMSNorm).* NeurIPS.
- Gong, J. *MiniMind: Train a Small Language Model from Scratch.* GitHub 開源專案（本文所復現之基準）。
- OpenCC. *Open Chinese Convert.* 開源簡繁轉換工具（`s2twp` 配置）。

---

> **配套文件**：本手稿之底層逐項事實、腳本細節、MiniMind 原碼交叉核對與完整演變史，見同目錄之 `YUNMO_SPEC.md`（技術規格，非論文語體）。二者數字一致，以本手稿為對外語體、以 `YUNMO_SPEC.md` 為工程細節之單一真實來源。
