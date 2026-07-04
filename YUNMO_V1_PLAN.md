# Yunmo v1 訓練計劃書

> ⚠️ **【歷史文件 · 已被取代】** 這是**原始規劃階段**的計劃書。其**資料策略（§5–8：過濾／英文下採樣／去重／丟棄長樣本）已被後來的「只增不減」鐵律整個推翻**，模型規格（166M/20層）亦已改為 **195.4M/24層**。
> **論文與最終事實一律以 [`YUNMO_SPEC.md`](./YUNMO_SPEC.md) 為權威來源。** 本檔僅保留作歷史過程與設計思路參照（§1 定位、§12 MiniMind 權威參照仍具參考價值）。切勿從本檔取用數字。

> **繁體台灣特化 · 全功能小模型 · 基於 MiniMind v3 復現**
> 建立日期：2026-07-03 ｜ 最後校準：2026-07-03（實測後重定位）

---

## 0. 版本定義（重新疊代）

| 版本 | 說明 | 狀態 |
|---|---|---|
| **v0.x** | 舊的 `yunmo_v3` 系列（基於前代 MiniMind、20000 自訂 tokenizer、簡繁混訓）。已封存為歷史。 | 凍結 |
| **v1** | 全新一輪，對齊 **MiniMind v3** 架構與資料格式；重訓 tokenizer；繁中為主；全功能。 | 進行中 |

> 命名原則：舊的一律降為 v0.x 當歷史參照，本輪從 **v1** 重新起算。

---

## 1. 定位與誠實邊界

### 1.1 定位：繁體台灣特化全功能小模型

**價值主張不是「更大所以更強」，而是「MiniMind 做不到的繁體台灣特化」。**

MiniMind 是簡體源模型；它**結構性地**沒有繁體台灣的原生知識與在地語感。我們的護城河就在這裡 —— 不是靠參數多、也不是靠中文知識量碾壓（那個我們沒有，資料規模與它相同）。

### 1.2 三種「贏面」的誠實分級

| 面向 | 會贏嗎 | 性質 |
|---|---|---|
| **繁體/台灣在地知識、語感** | ✅ **結構性穩贏** | MiniMind 資料裡沒有的東西（wiki-zh-tw、tw-instruct 在地化）。這是**護城河**。 |
| **繁體流暢度、tokenize 效率** | ✅ 大概率 | 原生繁體 tokenizer(24000) vs MiniMind 簡體(6400) |
| **全功能**(chat/推理/tool-call/堪用英文code) | ✅ 復現即得 | MiniMind 資料本身已含 → 復現就有 |
| **整體能力超越 MiniMind** | 🟡 **可能小幅、不保證** | 更大模型或許能榨更多，但固定小資料有邊際遞減+過擬合風險；需 eval 證明 |
| **中文知識量碾壓 / 對標 Qwen3-0.6B** | ❌ 不現實 | 同資料天花板；非本專案規模 |

### 1.3 誠實邊界（釘死，不自欺）

- **Scaling law 現實**：參數約 2.5× → loss 僅降 ~5–7%（冪律），非能力加倍；且固定 ~2.2B 小資料下，166M 比 64M 每參數餵更少（同 5ep 下 ~66 vs 172 tok/param），**過擬合風險升高**，很可能需降 epoch/早停。
- **沒有「保證贏」**：任何 ML 結果在訓練+eval 前都不是保證。整體能力的超越是**假設，要驗證**。
- **s2twp 轉換稅**：轉繁有誤轉、新 tokenizer 從頭學 → 是扣分項，靠品質控管抵銷。

**一句話定位**：**穩拿「繁體台灣特化的全功能小 MiniMind」；整體能力力求小幅超越，並用 eval 誠實驗證，不假設。**

---

## 2. 硬體與算力預算

| 資源 | 規格 | 用途 |
|---|---|---|
| 雲端 | RTX PRO 6000 (Blackwell, 96GB) × **36 小時（硬上限）** | **Pretrain + SFT**（吞吐密集階段） |
| 本地 | RTX 4070 Ti **12GB** | **DPO / RLAIF / Agent RL**（生成密集，可跑數天不佔雲端） |

**分工原理**：pretrain/SFT 是 throughput-bound → 用強卡把最貴的階段做滿；RL 是 rollout-bound → 丟本地慢慢跑不佔計時。166M 模型 RL 記憶體需求（policy+ref+優化器）遠小於 12GB，4070 Ti 跑得動。

**算力換算**（有效 ~1.5e14 FLOP/s，前 1hr 實測校正）：36hr ≈ 1.3e5 s × 1.5e14 ≈ **1.9e19 FLOPs**；`token 曝光 = FLOPs / (6 × 參數量)`。
→ 166M 稠密：雲端 pretrain+SFT 共 **~19.5B token 曝光額度**。**實測校準**（§5）：中文 pretrain unique 僅 **~2–3B ≈ MiniMind 的 2.2B**（非先前臆測的 10B）—— 我們與 MiniMind **在同一資料規模**，故該沿用它「**高 epoch**」的餵法，不是少 epoch。~19.5B ≈ 中文核心+TW ~3B × ~4ep + 英文下採樣 ~1.5B×~1.5 + SFT，**與 MiniMind 22B 曝光同量級**，夠。

---

## 3. 模型架構（~166M 稠密，最大化復現 MiniMind v3）

```python
MiniMindConfig(
    hidden_size          = 768,
    num_hidden_layers    = 20,     # 唯一放大：8→20（加深換容量）
    num_attention_heads  = 8,      # 照抄 MiniMind
    num_key_value_heads  = 4,      # 照抄 MiniMind（GQA 2:1；小模型別亂省 KV）
    head_dim             = 96,     # 照抄 MiniMind（768/8）
    vocab_size           = 24000,  # 唯一另一改：繁體 tokenizer（6400→24000）
    intermediate_size    = 2432,   # ceil(768×π/64)×64，SwiGLU（同 MiniMind）
    max_position_embeddings = 32768,
    rope_theta           = 1e6,
    tie_word_embeddings  = True,
    use_moe              = False,
)
```

- **總參數 ≈ 166M**（transformer ~147M + tied embedding 18.4M / 11%，vocab 24000）
- **設計原則：最小改動、最大復現。** 只動兩處 —— **深度(8→20)** 與 **vocab(6400→24000繁體)**；**注意力(heads/kv/head_dim)、FFN、norm 全部照抄 MiniMind**。越少自創變數 → 復現越忠實、風險越低。
- **架構對齊 MiniMind v3（`model_minimind.py`）**：沿用同一份模型碼，自動繼承：
  - **Pre-Norm + RMSNorm**(eps 1e-6)、**SwiGLU**(silu)、**RoPE**(theta 1e6, YaRN 推理可選)、**tie embeddings**、無 bias、flash-attn
  - **QK-Norm**（q/k 每頭 RMSNorm，Qwen3 式）—— MiniMind v3 有，務必保留（穩定深層）
  - shift-by-one CE、`ignore_index=-100`
- **僅兩處刻意差異**：

  | 參數 | MiniMind-3 | Yunmo v1 | 理由 |
  |---|---|---|---|
  | num_hidden_layers | 8 | **20** | 加深換容量（唯一放大維度） |
  | vocab_size | 6400(簡) | **24000(繁)** | 繁體高效 + 英文/code 子詞（護城河） |
  | **注意力/FFN/norm** | — | **完全照抄** | 減少自創變數，穩定復現 |
- **為何深而非寬/MoE**：加深換容量、推理鏈略好；MoE 需 2–4× 資料才訓得起（我們資料稀缺），稠密最穩。
- **自適應/降級（用「曝光量」判斷）**：前 1–2hr 實測吞吐。實測中文 unique 僅 ~2–3B，但 MiniMind 證明 2.2B×多 epoch 足以訓好 64M → 判準是**總曝光=unique×epoch**：目前估 ~16–20B 曝光 → **166M 餵得飽，不需降級**；若曝光 < ~12B 才降至 ~100M。**上限不超過 210M。**

---

## 4. Tokenizer（重訓）

- 用 MiniMind v3 的 `trainer/train_tokenizer.py`（**BPE + ByteLevel**）在**處理後的最終混合語料**上重訓（含 zh/en/code 依比例抽樣，讓 vocab 分配到英文/code 子詞）
- `vocab_size = 24000`（MiniMind 為 6400）：繁中覆蓋佳 + 留位置給英文/code；embedding 占 ~11% 參數
- **特殊 token 必須完整複製 MiniMind v3**（否則 chat template / tool / think 不相容）：
  - pad=`<|endoftext|>`(0)、bos=`<|im_start|>`(1)、eos=`<|im_end|>`(2)、保留 0–20 號
  - 工具/思考：`<tool_call>`(21)`</tool_call>`(22)`<tool_response>`(23)`</tool_response>`(24)`<think>`(25)`</think>`(26)、緩衝 `<|buffer1..9|>`(27–35)
  - **chat template = ChatML**，支援 `tools` 渲染、`tool` role、自適應 `<think>` 注入（`add_generation_prompt` 時）
- 訓練語料 = pretrain 主語料抽樣（不需全量，取代表性 ~2–5GB 即可）

---

## 5. 資料版圖（實測校準版）

### 5.0 實測數據總表（2026-07-03，char-based，MiniMind tokenizer 交叉）⭐

> 方法：每源抽樣 3 萬筆、套 pipeline 同套過濾、量字元長度分布；token 用 char/1.5 估（保守；真值 char/1.05–1.5 視我們 tokenizer 而定，±40%）。**字元為 tokenizer 無關真值；token 絕對值待我們繁體 tokenizer 訓好才定死。**

| 源 | 筆數 | 通過 | ~token估 | token長度 p90/p99 | 註 |
|---|---|---|---|---|---|
| pretrain_t2t | ~7.2M | 99% | **~1.3–2.2B** | 323/471 短 | 99% 中文 |
| pretrain_hq×4 | 1.4M | 100% | ~0.5B | ~320/420 短 | 繁 |
| cntw/distill_pre/qwen_pre | 0.7M | ~98% | ~0.37B | 1400/2600 長 | 推理攤平 |
| **sft_t2t 主線** | 4.9M | **僅14%** | ~0.16B | 860/1270 | **86% 英文!** |
| distill_r1(推理) | 111k | 99% | ~0.18B | **2529/7719 超長** | 有 score |
| qwen235b(推理) | 110k | 93% | ~0.08B | 1497/3370 長 | |
| tw-instruct | 483k | 100% | ~0.13B | 615/1008 | 在地化 |
| sft_512(舊,已丟) | 5.6M | 99% | ~1.36B | 339/507 短 | 重疊 pretrain，§5.3 丟棄 |
| code×多(含dup) | ~1.9M | 100% | ~1.7B | 3500/6300 長 | 英文為主 |
| FineWeb-Edu×3(英) | 2.2M | 100% | ~2.5B | 2290/7800 長 | int_score≥3 |

**池總（去重前，char/1.5 估）**：英文 ~4.5B｜**中文 pretrain ~2.1B**｜sft ~1.9B(含重疊)｜code ~1.7B

### 5.1 五個實測發現（推翻先前假設）⚠️

1. **`sft_t2t` 主線 86% 是英文**（實測 lang[en:25826 zh-cn:4168]）。繁中模型丟英文後**中文 SFT 只剩 ~14%**。**且 `pretrain_t2t`(99%中文) 與 `sft_t2t`(86%英文) 不是同一池** —— 先前「pretrain=sft攤平同池」的假設**在新版不成立**（舊 sft_512 才與 pretrain_t2t 重疊）。
2. **中文 pretrain 僅 ~2–3B，≈ MiniMind 的 2.2B**（非先前臆測的 ~10B）。我們沒有「多 4 倍資料」。
3. **中文才是瓶頸，英文過剩**（英文 ~4.5B ≫ 中文）→ 英文須**大幅下採樣**（比例見 §8.4）。
4. **中文 SFT 稀缺（~0.5–0.8B）** → 靠 `sft_t2t` 中文部分(~0.16B) + distill_r1/qwen235b(~0.26B) + tw-instruct(~0.13B) 湊。**不靠已丟棄的 sft_512**。
5. **SFT 長度雙峰**：指令短(p90~600)、**推理超長(distill_r1 p99≈7700tok)** → 必須設推理長度政策（見 §8.5）。

→ 實測 sft_512↔pretrain_t2t 逐字重疊（sft_512 p99 才 507，與 pretrain 短文重疊）→ **這正是我們「不用舊散檔、只用官方資料」的理由**（§5.3）：避開去重惡夢，而非去建重去重機器。

### 5.2 資料清單與格式

| 檔案 | 格式 (keys) | 語言 | 類型 | 品質信號 | 處置 |
|---|---|---|---|---|---|
| `pretrain_t2t.jsonl` (8.3GB) | `text`（無標記concat） | 簡 | pretrain | MiniMind v3 主線 | 轉繁→**pretrain 主力** |
| `yuhuanstudio/wiki-zh-tw` (待下載) | `title,text` | 繁 | pretrain | 百科、已清洗 | **pretrain 乾淨核心** |
| `cn/distill_r1_110k.jsonl` | `input,content,reasoning_content,score` | 簡 | sft/reason | **有 score 可濾** | 轉繁→依 score 篩→SFT/推理 |
| `cn/qwen3_235b_..._chinese.jsonl` | `messages` | 簡 | sft | 235B 蒸餾 | 轉繁→SFT 候選 |
| `tw/sft/r1_tw_sft.jsonl` | `conversations(role/content)` | 繁 | sft/reason | 含 `<think>` | SFT 推理候選 |
| `tw/sft/qwen_tw_sft.jsonl` | `conversations` | 繁 | sft | 蒸餾 | SFT 候選 |
| `tw/sft/sft_512/1024/2048_cntw.jsonl` (22GB) | `conversations` | 繁+**英** | sft | 舊 MiniMind 全量 | **高度懷疑**，大量與 pretrain_t2t 重疊→去重後殘量再議 |
| `tw/sft/tw_sft.jsonl` | `conversations` | 繁 | sft | = tw-instruct | 與 cntw 重複 |
| `tw/sft/code_sft_tw.jsonl` (2.1GB) | `conversations` | **英** | code | 英文 code Q&A | 僅在要 code 能力時保留少量 |
| `tw-instruct-500k_filtered.jsonl` | `conversations(from/value)+converted_*` | 繁 | sft | 台灣在地、已轉 | SFT 在地化候選 |
| `code.jsonl` (根, 2.3GB) | `instruction,output,tag` | **英** | code | 英文 | 同 code_sft，去重後少量或棄 |
| `tw/pretrain/*` (cntw/code/distill_pre/qwen_pre/pretrain_hq) | `text(<|im_start|>…)` | 繁 | pretrain | **多為上列 sft 的 pretrain 版重複** | 去重來源，勿雙倍計入 |
| `cn/pretrain/*` | 同上簡體版 | 簡 | pretrain | 舊 | 多餘，去重 |

**語言警示**：`code.jsonl`、`code_sft_tw`、部分 `sft_2048` 是**英文**；繁中模型應按語言過濾，英文僅在「刻意要 code 能力」時保留少量。

**轉繁品質注記**：s2twp 有 `鬱/郁` 類一對多誤轉（實測 `濃鬱` 應為 `濃郁`）；`tw-instruct` 的 `converted_string` 已記錄修正。清洗階段加一份修正詞表。

### 5.3 定案資料策略（定位後大幅簡化）⭐

**原則：復現 = 用 MiniMind 官方乾淨資料 + s2twp，不自己亂拼舊散檔。護城河 = TW 原生加值。**

| 用 ✅ | 角色 |
|---|---|
| **MiniMind v3 官方** `pretrain_t2t` / `sft_t2t` / `dpo` / `rlaif` / `agent_rl`（s2twp 轉繁） | **復現核心**（已含 zh+en+code+推理+tool，全功能免費得） |
| **wiki-zh-tw**（你的 regen） | **護城河**：繁體台灣原生知識 |
| **tw-instruct-500k**（在地化） | **護城河**：台灣語感/在地指令 |
| `cn/distill_r1`（score 篩）、`qwen235b` | 補繁中推理 SFT（中文 SFT 稀缺，需補） |

| 丟 ❌ | 原因 |
|---|---|
| FineWeb-Edu | 英文過剩、非定位重點；MiniMind 自帶英文已夠 |
| 舊 v0.x 散檔 `sft_512/1024/2048`、`cn/pretrain/*`、`tw/pretrain/*` 重複版 | **去重惡夢的來源**；用官方新資料即避開 |
| Magicoder / py18k / code.jsonl | MiniMind 自帶 code；除非 eval 顯示 code 不足再精準補 |

→ **不用舊散檔 = 跨集去重問題自動消失**（官方資料 MiniMind 已清過）。去重降級為「僅官方資料池內輕量 exact 去重」即可，MinHash/優先序機器**非必要**。

**待辦**：確認 F:/AI/data 內是否為 MiniMind v3 **官方最新** `pretrain_t2t`/`sft_t2t`（版本對得上），或需重新下載官方版。

---

## 6. 資料品質鐵律（最高優先）

> **預設排除。資料先過關才進訓練，不是「有就吃」。寧可小而乾淨，不要大而髒。**

每一筆資料進入語料前必須通過：

1. **語言純度**：繁體為主；濾除簡體殘留、亂碼；**英文下採樣**（§8.4）
2. **輕量去重**：僅**官方資料池內 exact hash** 去重（因不拼舊散檔，跨集重疊問題消失）。**MinHash-LSH 近似去重 + 優先序保留 = 擱置備用**（若日後真要混多源才啟用）
3. **品質過濾**：長度上下限、重複率、模板化、結構完整；有 `score` 的依 score 篩；推理依 §8.5 長度篩
4. **分池去重（MiniMind 範式）**：pretrain 池與 SFT 池**各自池內去重**、**跨池不去重**（同內容刻意兩池兩用）。僅 **eval 集**須與訓練集獨立。
5. **人工核准**：pipeline 只產出**報告**，不自動進訓練；由人看報告後決定納入

**產出物**：`overlap_report`（各集重疊率）、`quality_report`（各集通過率/被砍原因）、`token_stats`（各層淨 token），供決策。

---

## 7. 資料處理 Pipeline（定位後大幅精簡）

```
MiniMind 官方資料 + wiki + tw-instruct
   → s2twp 轉繁(+修正詞表) → 語言過濾(下採樣英文) + 品質過濾 + 防禦清理
   → 池內輕量 exact 去重(僅官方資料自身, 無需 MinHash/優先序)
   → distill 依 score 篩 + §8.5 推理長度篩 → pretrain & SFT 語料
```

- **去重降級**：因不拼舊 v0.x 散檔，跨集重疊問題消失 → 只需官方資料**池內 exact 去重**，MinHash/優先序機器**擱置備用**。
- streaming 逐行處理；依賴 `opencc`、`tqdm`（`datasketch` 暫不需）。
- 退火排序：**wiki + tw-instruct（TW 護城河）放最後**塑造繁體台灣行為。

---

## 8. 訓練流程（對齊 MiniMind v3；原則對齊、數值實測）

### 8.0 MiniMind v3 權威配方（proven baseline）

來源：`images/dataset.jpg` + 訓練腳本原碼（**非** argparse 快速復現預設，兩者不一致見 §12）。

| 階段 | 資料 | epochs | tokens | seq_len | 起點 |
|---|---|---|---|---|---|
| Pretrain | pretrain_t2t (10GB) | **5** | ~2.2B | 380 | 從頭 |
| SFT | sft_t2t (14GB) | **5** | ~2.2B | **1350** | pretrain |
| DPO | dpo.jsonl | 1 | — | 1024 | full_sft |
| RLAIF(CISPO) | rlaif.jsonl | 1 | — | 768+1024gen | full_sft |
| Agent RL | agent_rl.jsonl | 1 | — | max 2500 | full_sft |

- **共通**：AdamW、bf16、grad_clip 1.0、**LR schedule** `lr*(0.1+0.45*(1+cos(π·step/total)))` → **峰值=lr、底=lr/10、無 warmup**。
- **關鍵洞察**：MiniMind-3 是 **「小 unique(pretrain ~2.2B) × 高 epoch(5)」** 路線 —— 不靠資料量堆，靠品質 + 多輪重複。（註：dataset.jpg 的 pretrain/sft 皆標 2.2B，但**我們實測自己的 sft_t2t 86% 是英文**，故繁中 SFT 遠小於 pretrain，兩者非同池；勿假設 MiniMind 兩池同內容。）

### 8.1 我們的路線 + 「不輸原版」下限機制（實測校準後）⚠️

**實測翻轉先前臆測**：原以為有 ~10B unique 中文 → 想少 epoch。**實測後：中文 pretrain 僅 ~2–3B ≈ MiniMind 的 2.2B**。→ **我們與 MiniMind 同規模，就該沿用它的高 epoch（3–5，eval 驅動），不是少 epoch。** 這反而讓路線更清楚：**「復現 MiniMind（繁中版）」= 用同量中文資料 + 同 epoch 數，穩拿下限。**（註：「不輸」靠的是「最壞回退=復現」的**下限邏輯**，非「保證贏」；整體超越仍須 eval 驗證，見 §1.3。）

**理論**（資料受限縮放律，Muennighoff 2023）：unique token 價值 > 重複，但重複 ≤~5 epoch 前近似等值。MiniMind 的 5ep 正在此甜蜜區 → 我們照跑安全。

**保證機制（定位後簡化）**：
1. **復現脊柱 = 下限**：MiniMind 官方 pretrain_t2t（s2twp）照它 recipe 跑 → **下限 = 繁中版 MiniMind（穩拿）**。英文/code 靠 MiniMind 資料自帶，不另加。
2. **TW 原生加值 = 護城河**：wiki-zh-tw + tw-instruct 併入，epoch 與核心同量級 → 這是 MiniMind **結構性沒有**的內容，穩定差異化。

   | 內容 | 角色 | unique(估) | epoch |
   |---|---|---|---|
   | MiniMind pretrain_t2t(s2twp) | 復現核心(自帶en/code) | ~2.2B | **見下方過擬合警示** |
   | wiki-zh-tw + tw-instruct | TW 護城河 | ~0.8–1B | 同核心 |
   | 高分蒸餾(distill_r1/qwen235b) | 補繁中推理 | ~0.3B | 併入 |
3. **eval 逐增量把關**：每加一批對照 MiniMind v3 baseline，**好留差棄**（§9）。最壞回退「純復現核心 = 繁中 MiniMind」。

> ⚠️ **過擬合警示（不再無腦 5ep）**：166M 在固定 ~3B 中文資料上跑 5ep = ~90 tok/param、且大模型更會背。**epoch 數改為 eval 驅動**：從 3ep 起，看 held-out loss/eval，過擬合就早停；不硬套 MiniMind 的 5（那是給 64M 的）。**這是誠實版對「加大模型」風險的具體對策。**

### 8.2 雲端 recipe（36hr，RTX PRO 6000）

算力換算：36hr × ~1.5e14 FLOP/s ≈ 1.9e19 FLOPs；**166M** 下 `tokens ≈ FLOPs/(6×N)` = 1.9e19/(6×1.66e8) → **~19.5B token 曝光額度**（前 1hr 實測校正）。分配：

| 階段 | 資料 | epoch | seq_len | 載入 | 峰值 lr | 預算 |
|---|---|---|---|---|---|---|
| Pretrain | MiniMind `pretrain_t2t`(s2twp) + wiki + tw-instruct | **3 起, eval 驅動** | **1024** | **packing** | **3e-4**(hr1 微調) | ~24–28hr |
| SFT | `sft_t2t` 中文部分 + 高分蒸餾 + tw-instruct（§8.5 篩推理） | **3 起, eval 驅動** | **2048**† | padding+**bucketing** | 1e-5 | ~8–10hr |

- **pretrain 用 packing + 1024**（改 MiniMind 原生 padding 載入器）：MiniMind pretrain 是 padding+短資料故用 380；我們池含 wiki 長文/推理長鏈，380/512 會截斷。packing（EOS 分隔 concat 填滿）→ **零 padding 浪費 + 保長上下文**，同 FLOP 處理更多真實 token。**建議加 block-diagonal mask** 避免跨文件注意力污染（或至少 EOS 分隔）。
- †SFT seq_len **2048 padding + length-bucketing**（§8.5）：SFT 一對話一序列、逐 assistant 遮罩；2048 完整保留推理 ~p80（勝 MiniMind 1350），bucketing 解決短指令 padding 浪費，丟棄 >2048 樣本（不截斷）。確切上限待真 tokenizer 量出 token 分布後微調。
- 峰值 lr **不抄** MiniMind（它圖 1e-4 / argparse 5e-4 自相矛盾，見 §12）；166M + batch 較大 → 起手 3e-4，前 1hr 看 loss 微調。
- **若 36hr 不足**：先砍 SFT epoch，再砍英文曝光；**中文核心 + TW 護城河的 epoch 不動**（保下限）。

### 8.3 本地 recipe（4070 Ti 12GB，不佔計時）

沿用 MiniMind 腳本預設（已驗證）：

| 階段 | 資料 | 關鍵超參 | 備註 |
|---|---|---|---|
| DPO | `dpo.jsonl`(轉繁) | lr 4e-8、β 0.15、seq 1024 | 偏好對齊；凍結 ref |
| RLAIF **CISPO** | `rlaif.jsonl` | lr 3e-7、num_gen 6、β 0.1 | **需 InternLM2-1.8B-Reward**（bf16 ~3.6GB，12GB 吃緊但可） |
| Agent RL | `agent_rl*.jsonl` | lr 3e-7、max_total 2500、thinking_ratio 0.1 | tool-use，可選 |

> 注：MiniMind RL 預設 loss_type = **CISPO**（非 GRPO）；reward 用外部 InternLM2-1.8B-Reward 模型，需另下載並放專案同層目錄。

### 8.4 語言比例（定位後：繁中優先，英文/code 靠 MiniMind 自帶）
- **來源不外掛**：英文/code **只用 MiniMind 資料本身自帶的**（`sft_t2t` 實測 86% 英文、pretrain 也含英文/code），不加 FineWeb/Magicoder。
- **關鍵動作：下採樣 MiniMind 的英文**。MiniMind SFT 86% 英文，直接用會英文喧賓奪主 → **繁中特化需把英文下採樣**，讓整體回到「繁中為主(~70%+)、英文/code 堪用(各~10-15%)」。
- **繁中優先靠 TW 加值頂上**：wiki + tw-instruct 把中文（尤其繁體台灣）比重與品質拉起來。
- **中文內部**：繁 ~100%（簡體 s2twp 轉繁後併入）；SFT/RL 中文回覆一律繁體。

### 8.5 SFT 推理長度政策（實測逼出）⚠️

實測：SFT 呈**雙峰**。指令短（tw-instruct p90~615tok），但**推理超長**（distill_r1 **p99≈7719tok、max≈26000tok**，qwen235b p99≈3370）。code 亦長（p90~3500）。

- **問題**：SFT seq_len 無法蓋推理 p99（要 seq~7000 荒謬且貴）。1024 會把長推理**攔腰截斷**，學到殘缺推理鏈反而有害。
- **定案政策**：**SFT seq_len = 2048 + length-bucketing**，配合 **score 篩選 + 丟棄 token > 2048 樣本（不截斷，寧缺勿殘）**。
  - 2048 能完整保留推理 ~p80（比 MiniMind 1350 保住更多完整推理鏈 = 贏過原版的真槓桿）
  - length-bucketing（按長度分桶批次）解決短指令 padding 浪費
  - **上限 2048 不再往上**：166M 小模型難駕馭 >2000-token 長推理，邊際遞減
- **但書**：確切 2048 上限待真 tokenizer 量出 token 長度分布後微調（char 估 ±40%）。

---

## 9. 評估（兼作 §8.1 的「好留差棄」把關）

- **對照組**：MiniMind v3 官方權重（下限脊柱）+ yunmo_v1 各階段 checkpoint
- **客觀**：lm-evaluation-harness，沿用 MiniMind 7 項（C-Eval / CMMLU / ARC-Easy / PIQA / OpenBookQA / HellaSwag / Social-IQa），比對數概率法。**繁中另切獨立 eval 集**（不進訓練），量繁中知識/推理。
- **功能**：`eval_llm.py` 繁中對話 + `eval_toolcall.py` tool-call；推理看 `<think>` 品質
- **把關規則（強制）**：每加一批增量資料或多一層 epoch，跑上述對照；**若未優於 baseline 即回退該批**。最壞情況收斂到「僅核心 = 復現 MiniMind」，確保不輸原版。

---

## 10. 風險與決策點

| 風險 | 對策 |
|---|---|
| **中文 SFT 稀缺(~0.5–0.8B)** | 靠 sft_t2t中文部分 + distill + tw-instruct；SFT epoch 拉高；必要時合成補繁中指令 |
| 中文 pretrain 僅 ~2–3B | **不降級**（MiniMind 2.2B×多 epoch 已證足夠）；中文核心多 epoch |
| 推理資料超長(p99≈7700) | §8.5：seq 2048 + score 篩 + 丟過長，寧缺勿殘 |
| 英文過剩、稀釋中文 | **下採樣 MiniMind 自帶英文**（不外掛 FineWeb）；比例見 §8.4 |
| s2twp 誤轉 | 修正詞表；抽樣人工檢查 |
| token 絕對值不定 | **先訓 tokenizer 再重量一次**才定 epoch/seq_len |
| 36hr 不夠 | RL 全下放本地；先砍英文曝光，中文核心+TW 護城河 epoch 不動 |

---

## 11. 里程碑

1. [x] 下載新來源 + 親自抽樣檢查 + **char-based 實測**（§5.0）
2. [~] Wiki 生成（繁中，含 2 bug 修正）— 進行中
3. [ ] 資料 pipeline：轉繁 + 去重 + 報告（含 wiki）
4. [ ] **重訓 tokenizer**（vocab 24000，特殊 token 對齊 MiniMind）← **解鎖 token 絕對值**
5. [ ] **用真 tokenizer 重量一次** → 定死 epoch/seq_len/推理上限（§8.1/§8.2/§8.5）
6. [ ] **人工核准**語料（依報告）
7. [ ] 雲端 pretrain（3ep 起、eval 驅動；中文核心+TW 護城河 + 英文下採樣）
8. [ ] 雲端 SFT（seq 2048 + score 篩 + 丟過長推理）
9. [ ] 本地 DPO /（選）RLAIF-CISPO / Agent RL
10. [ ] Eval 對照 MiniMind v3（好留差棄）

---

## 12. 附錄：MiniMind v3 權威參照（避免被舊圖/預設誤導）

> **原則：MiniMind 的圖、argparse、README 三者已漂移，數值不可盲抄。以下為原碼交叉核對後的權威值。對齊「架構與方法」，數值一律實測定案。**

### 12.1 架構權威值（`model/model_minimind.py`）
minimind-3(64M)：hidden 768 / layers 8 / q_heads 8 / **kv_heads 4** / **head_dim 96** / vocab **6400** / intermediate 2432 / rope_theta 1e6 / max_pos 32768 / tie True / **QK-Norm(q,k 每頭 RMSNorm)** / RMSNorm eps 1e-6 / SwiGLU。moe 版：4 專家 top-1、無 shared expert、aux_coef 5e-4。

### 12.2 已知內部不一致（12 處，勿踩雷）
| # | 項目 | argparse | 圖/README | 採信 |
|---|---|---|---|---|
| 1 | pretrain lr | 5e-4 | 圖 1e-4 | 實測定 |
| 2 | pretrain/SFT epochs | 2 | dataset.jpg **5**(full) | full=5、Zero=2 |
| 3 | pretrain seq_len(mini) | 340 | README 768 / 圖 380 | full=380 一致 |
| 4 | SFT seq_len(full) | 768 | dataset.jpg **1350** | full=1350 |
| 5 | DPO lr | 4e-8 | help「≤5e-8」 | 4e-8 |
| 6 | DPO beta | argparse 0.15 | code 0.1(dead) | 0.15 實跑 |
| 7 | rlaif 大小 | 24MB | prose 20MB | — |
| 8 | moe 標註 | 198M-A64M / 198M / 0.2B-A0.06B | 同一模型 |
| 9 | RMSNorm eps | class 1e-5(未用) | config 1e-6 | 1e-6 |
| 10 | 蒸餾 seq_len | 340 vs 資料建議768 | 實測 |
| 11 | Zero 時間 | 2h vs 1ep 2.31h×2 | 估算粗略 |
| 12 | max_pos vs tokenizer max_len | 32768 vs 131072 vs YaRN orig 2048 | 32768 |

### 12.3 資料與方法要點
- dataset.jpg 標 pretrain_t2t 與 sft_t2t 皆 ~2.2B，但**我們實測自己手上的 sft_t2t 86% 是英文**（§5.1）→ **勿假設兩者同內容池**；MiniMind 是多語 SFT。
- MiniMind pretrain=padding（非 packing），故用短 seq 380（我們改 packing+1024）。
- SFT loss 只算 assistant 區間；20% 隨機注入 system prompt；~80% 移除空 think 標籤。
- RL 資料由 SFT 篩（末個 assistant 留空供 rollout）；reward=InternLM2-1.8B-Reward。
- Eval：lm-evaluation-harness，7 項（C-Eval/CMMLU/ARC-e/PIQA/OpenBookQA/HellaSwag/Social-IQa），比對數概率非自由生成。
