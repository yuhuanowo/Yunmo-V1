# Yunmo v1 — 權威技術規格（論文事實骨幹 · 完整版）

> **單一真實來源（Single Source of Truth）**。本檔為「實際做了什麼」的完整定稿；所有數字均經程式驗證（實例化數參數、逐筆計數、打包實測、**打包後端到端解碼核對**），並與完整聊天紀錄、所有腳本與稽核報告交叉比對。
> 論文的 Model / Data / Training / Limitations 章節請直接以本檔為準。舊的 `YUNMO_V1_PLAN.md`（計劃書）與 `YUNMO_DATA_STATUS.md`（資料盤點）已標為歷史，**其資料策略（過濾/下採樣/去重）已被本檔「只增不減」定案取代，切勿引用其數字**。
> 定位：繁體台灣特化 · 全功能小語言模型 · 基於 MiniMind v3 增量復現。最後校準 **2026-07-04**（含開訓前身份污染大清理與打包後驗證）。

---

## 0. 摘要（一段話）

Yunmo v1 是一個**繁體台灣特化的全功能小語言模型**，方法上是 **MiniMind v3 的增量復現**：把 MiniMind 的全部訓練資料以 OpenCC `s2twp` **1:1 轉為繁體台灣用語（不丟任何一筆）**，併入台灣原生增量（繁體維基、台灣在地指令），重訓一個 24000 詞的繁體 tokenizer，並在開訓前對 SFT 語料做**多老師身份污染清理**（將自稱 Qwen/DeepSeek/ChatGPT/minimind 等的回覆改寫為一致的 Yunmo 身份）。模型在 MiniMind v3 架構上**僅加深（8→24 層）與換詞表（6400→24000）**，達 **195.4M 參數**，於 **36 小時硬算力預算**（RTX PRO 6000, Blackwell 96GB）內完成 pretrain + SFT。價值主張**不是「更大所以更強」**，而是「MiniMind 結構性做不到的繁體台灣在地化 + 全功能 + 一致品牌身份」。整體能力超越 MiniMind 是**待驗證假設**，非既成結論。

---

## 1. 定位與貢獻（含誠實邊界）

| 貢獻 | 性質 | 證據等級 |
|---|---|---|
| 繁體台灣在地知識/語感 | MiniMind 資料結構性所無（wiki-zh-tw、tw-instruct） | 結構性差異（穩） |
| 原生繁體 tokenizer | vocab 24000 vs MiniMind 6400；繁體 token 效率 | **已實測：少 46.4% token** |
| 一致品牌身份（Yunmo） | 清除多老師身份污染 + 注入一致自我認同 | **已實測：token 流 0 老師自稱** |
| 全功能復現 | chat / 推理`<think>` / tool-call / 英文·code；1:1 轉繁全保留 | 復現即得 |
| 較大規模 | 195.4M（24 層，3.05× MiniMind-3 64M） | 已驗證 |
| **整體能力超越 MiniMind** | **假設，須 eval 驗證** | ⚠️ 未證 |
| 對標 Qwen3-0.6B / 中文知識碾壓 | **不現實**（同資料天花板、不同規模級） | ⚠️ 明確否定 |

**誠實邊界（論文 Limitations 直接用）：**
- 「整體超越 MiniMind」「tok/param 餵得飽」為**待訓後 eval 驗證的假設**，非結論。Scaling law 下參數 ~3× 僅使 loss 降個位數百分比（冪律），非能力倍增。
- **最壞回退 = 繁體版 MiniMind 全功能復現**（因最終資料是 MiniMind 資料 1:1 轉繁的**超集**）——這是「不輸原版」的下限邏輯，非「保證贏」。
- 可靠的既成優勢：**TW 原生知識**（結構性）+ **實測 46% tokenizer 效率** + **一致 Yunmo 身份**（無多老師污染）。
- 過擬合風險：195.4M 在固定 ~2.35B 中文 pretrain 上多輪重複，大模型更易記憶 → 靠 held-out loss / eval 早停把關。
- 身份清理殘留率 **0.0003%**（誠實報告，非宣稱歸零；見 §4.7）。

---

## 2. 模型架構（195.4M，深而窄）

架構完全沿用 MiniMind v3 的 `model/model_minimind.py`（同一份模型碼），**僅刻意改動兩處**：深度與詞表。無獨立 `config.json`／`generation_config.json`，架構由 `MiniMindConfig` 預設 + 啟動器參數決定。

```python
MiniMindConfig(
    hidden_size          = 768,
    num_hidden_layers    = 24,      # 改動①：8→24（加深換容量）
    num_attention_heads  = 8,
    num_key_value_heads  = 4,       # GQA，n_rep=2
    head_dim             = 96,      # 768/8
    intermediate_size    = 2432,    # ceil(768·π/64)·64 = 38·64；SwiGLU（與 MiniMind 同公式，非自由旋鈕）
    vocab_size           = 24000,   # 改動②：6400(簡)→24000(繁)；init 時由 len(tokenizer) 同步
    max_position_embeddings = 32768,
    rope_theta           = 1e6,
    rms_norm_eps         = 1e-6,
    tie_word_embeddings  = True,
    use_moe              = False,
)
```

**參數量（實例化數出，非估算）：**

| 組成 | 參數 |
|---|---|
| 詞嵌入（tied，24000×768） | 18.43M（占 ~11%） |
| 每層 = attn 1.770M + ffn 5.603M | 7.375M |
| 24 層合計（非嵌入） | 177.0M |
| **總計** | **195.4M**（3.05× MiniMind-3 64M） |

**繼承自 MiniMind v3（逐項，未改動）：**
- **Pre-Norm 殘差**：`h = x + Attn(RMSNorm(x)); h = h + MLP(RMSNorm(h))`。
- **RMSNorm**：全域 eps **1e-6**（於 fp32 計算再轉回 dtype）。（RMSNorm class 預設 1e-5 但每處實例化都傳 1e-6。）
- **注意力 = GQA + QK-Norm**：q_heads 8 / kv_heads 4（n_rep 2）、head_dim 96、投影無 bias。**QK-Norm = 對 q、k 每頭在 head_dim 軸做 RMSNorm（eps 1e-6），且在 RoPE 之前**；**v 不正規化**。（Qwen3 式，專為穩定較深的堆疊保留。）
- **注意力核**：訓練/prefill 用 `F.scaled_dot_product_attention`（flash/SDPA）；增量解碼走 eager（`qk/√head_dim` + `triu(1)` 因果遮罩，softmax 於 fp32）。
- **RoPE**：theta 1e6、rotate_half 式、max_pos 32768；**YaRN 預設關**（啟用則 factor 16 / original_max_pos 2048）。
- **MLP = SwiGLU**：`down(silu(gate(x))·up(x))`，無 bias。
- **tied embeddings**：`lm_head.weight` 與 `embed_tokens.weight` 共享。
- **損失於模型內部計算**：shift-by-one CE，`x=logits[:,:-1]`、`y=labels[:,1:]`，`ignore_index=-100`；傳入 labels 與 input_ids **1:1 對齊未預移位**，模型負責移位。輸出 `MoeCausalLMOutputWithPast(loss, aux_loss, logits, ...)`。
- **MoE（存在但未用，`use_moe=False`）**：如啟用為 4 專家 top-1、無 shared expert、Switch 式負載均衡 aux_loss、`router_aux_loss_coef=5e-4`。本專案 aux_loss=0。

**設計理由（供論文 Design 章）：**
- **為何深而非寬（24 層）**：據 MiniMind README 引用之 **MobileLLM（arXiv 2402.14905）**——固定參數下**深度比寬度重要**，125M/350M 級 30–42 層窄堆疊於常識/QA/閱讀勝 ~12 層胖模型。MiniMind 自身 d_model/n_layers 實驗給界：**d_model < 512 傷（嵌入太窄、d_head 太小）、> 1536 則加寬優於加深** → **768 落在 [512,1536] 甜蜜帶**。先例：`minimind2`(104M) 已是 768/16 深窄；minimind-3 退回 8 層是「3 美元 2 小時」訓練速度的行銷取捨，非品質最優。24 層為 96GB/36h 算力確認後所選（MobileLLM 最優約 30–42 層，24 為保守值）。
- **intermediate 2432**：`ceil(768·π/64)·64`，標準 SwiGLU ~π×(≈3.17×) 擴張，使 3 矩陣 SwiGLU 與 4× 香草 FFN 參數等價；hidden 固定後即固定，非自由旋鈕。
- **為何稠密非 MoE**：MoE 需 2–4× 資料訓專家；本專案資料受限（中文 pretrain unique ~2.35B）→ 稠密最資料高效。
- **自適應降級規則（未觸發）**：以「總曝光 = unique×epoch」判斷；僅當曝光 < ~12B 才降至 ~100M；硬上限 210M。最終曝光足夠，195.4M 成立。

---

## 3. Tokenizer（重訓，繁體特化）

- **演算法**：BPE + ByteLevel（`add_prefix_space=False`），沿用 MiniMind `train_tokenizer.py` 方法；`vocab_size=24000`、**`min_frequency=2`**、`initial_alphabet=ByteLevel.alphabet()`。
- **特殊 token — 36 個（IDs 0–35，完整對齊 MiniMind v3，否則 chat template/tool/think 不相容）**：`<|endoftext|>`(0, pad/unk)、`<|im_start|>`(1, bos)、`<|im_end|>`(2, eos)、保留 3–20（object_ref/box/quad/vision/audio/tts 等）；`<tool_call>`(21)`</tool_call>`(22)`<tool_response>`(23)`</tool_response>`(24)`<think>`(25)`</think>`(26)；緩衝 `<|buffer1..9|>`(27–35)。後處理僅把 0–20 這 21 個重註冊為 special，21–35 標為非 special。
- **tokenizer_config**：`add_bos_token=False`、`add_eos_token=False`、`add_prefix_space=False`、`model_max_length=32768`、`clean_up_tokenization_spaces=False`、`PreTrainedTokenizerFast`；完整 MiniMind ChatML `chat_template` 逐字複製。
- **實際位置**：`model/`（`AutoTokenizer.from_pretrained("./model")`）；打包與訓練皆載此，**打包後解碼實測 vocab=24000、bos/eos=1/2**。
- **訓練語料**：~**100MB（128,819 行）**，多點偏移抽樣自已轉繁的 `yunmo/{pretrain,sft}.jsonl`（24 偏移點/檔、每行截 3000 字、丟 <5 字碎片；pretrain 70MB + sft 30MB）。
- **訓練實測**：**34 秒**完成（tokenize 2,150,949 words、compute-merges 0→**23,708**）；最終 vocab 恰 24000、merges.txt 23,709。
- **壓縮率（實測，字元/token）**：繁中 **1.89**、英文 4.58、code 2.29、混合 2.68。
- **對 MiniMind 6400 tokenizer（繁體台灣文本，token 越少越好）**：4 句合計 ours=90 vs MiniMind=168 → **少 46.4% token**（單句省 +54/+41/+45/+44%）。round-trip 一致、特殊 token ID 正確、`apply_chat_template` 通過 → 生產就緒。
- **被放棄的首次嘗試（BPE 爆炸事件）**：首版餵 ~3.96M 行/**3GB** → **5050 萬個唯一詞** → BPE compute-merges 停滯（僅 count-pairs 就 ~53 分，推估數天）。根因診斷：**MiniMind 官方 tokenizer 只用 10,000 行**（`train_tokenizer.py:15 if i>=10000: break`）——「BPE 早飽和，更多資料反而錯」。修法：語料砍到 100MB（~13× MiniMind）+ `min_frequency=2`。**論文洞見：tokenizer 不同於模型，會飽和，資料越多 ≠ 越好。**
- **不因污染清理重訓**：身份污染在資料內容而非詞表；清理後重打包即可，tokenizer 不變（見 §4.7）。

**Chat template（ChatML）行為要點：**
- system/tools：有 tools 則渲染 `# Tools` 區塊（各工具 `tojson` 於 `<tools></tools>`）+ 呼叫格式說明；無 tools 但首訊為 system 則 `<|im_start|>system\n{content}<|im_end|>\n`。
- assistant：**一律渲染 thinking 區塊** `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}`；reasoning 取自 `reasoning_content` 或由 content 依 `</think>` 拆出；有 tool_calls 則附 `<tool_call>\n{...}\n</tool_call>`。
- tool role：連續 tool 訊息包成單一 user turn，各裹 `<tool_response>...</tool_response>`。
- 生成提示（`add_generation_prompt`）：附 `<|im_start|>assistant\n` 後，開思考則 `<think>\n`，否則空 `<think>\n\n</think>\n\n`。

---

## 4. 資料（鐵律：只增不減 + 開訓前污染清理）

> **鐵律**：資料與 MiniMind **一模一樣，只做簡→繁 + 台灣增量，不丟任何一筆**。無語言過濾、無品質過濾、無去重、無英文下採樣、無長樣本丟棄。
> **唯二例外（皆非品質過濾）**：① 丟棄「無任何有效 assistant 回覆」的殘缺記錄（4,480 筆）；② **身份污染清理為「改寫」非「刪除」**（見 §4.7）。
> **起源（關鍵轉折，使用者原話）**：早期過濾管線在轉換時丟了 **753,663 筆英文（8.9%）**，使用者否決：**「保留啊 我們應該資料跟 minimind 一模一樣 只是增加 而不應該有任何減少」**。此指令回溯性廢除了整條過濾/下採樣/去重管線（見附錄 B）。

### 4.1 最終訓練資料（逐筆計數 + 打包實測，非抽樣）

| 檔案 | 記錄數 | 打包 token | 打包區塊 | 磁碟（bin） |
|---|---|---|---|---|
| `pretrain_clean.jsonl`† | **12,157,237** | **2,353,430,714**（2.35B） | 2,298,272 × 1024 | 4.71 GB |
| `sft_clean.jsonl`※ | **5,811,531** | **3,857,469,341**（3.857B） | 1,883,529 × 2048 | 11.6 GB（ids 7.71 + mask 3.86） |
| 合計 unique | | **~6.21B** | | ~16.3 GB |

> ※ **SFT 由污染清理後的 `sft_clean.jsonl` 打包**（見 §4.7）。原始 `sft.jsonl` 5,816,011 筆 → 丟 4,480 筆「無任何有效 assistant 回覆」殘缺記錄 → **5,811,531**；身份自稱句改寫為 Yunmo、tool_calls 補轉繁。token 較原始打包 3.875B 少 0.44%（長老師自介句換成較短 Yunmo 句）。
> † **pretrain 亦經身份清理並重打包**（見 §4.7）：原始 `pretrain.jsonl` 全掃發現 1,074 筆老師「我是 X」自稱 + 542 筆 refusal 型（0 殘簡、0 minimind），改寫為 Yunmo（不刪記錄），輸出 `pretrain_clean.jsonl`；token 由 2,353,402,018 微增至 2,353,430,714（+0.001%，注入 3,173 句 Yunmo 所致）。

### 4.2 血緣組成（只增不減）

**pretrain（12,157,237；清理不涉及）**
| 來源 | 記錄數 | 血緣 |
|---|---|---|
| pretrain_t2t（s2twp 1:1 轉繁） | 8,468,827 | **MiniMind 本體**，零丟失 |
| wiki-zh-tw（繁體維基） | 3,688,410 | **台灣增量**（§4.5） |

**sft（原始 5,816,011 → 清理後 5,811,531）**
| 來源 | 記錄數 | 血緣 |
|---|---|---|
| sft_t2t（s2twp 1:1；tool-call/英文全留） | 5,109,432 | **MiniMind 本體**，零丟失 |
| tw-instruct（在地化，已有繁體版） | 486,580 | 台灣增量 |
| distill_r1（R1 推理，含 reasoning_content） | 109,999 | MiniMind 本體（用 tw/ 現成繁體版） |
| qwen3_235b（235B 蒸餾推理） | 110,000 | MiniMind 本體（用 tw/ 現成繁體版） |

> 上表為**原始血緣**（合計 5,816,011）。身份清理丟 4,480 殘缺、其餘全數保留並就地改寫 → **5,811,531**。

### 4.3 ⚠️ 重要誠實揭露（論文 Data/Limitations 必寫）

1. **SFT 語料以英文為主**：sft_t2t 5.1M 筆實測 **~86–89% 英文**（語言抽樣 en:25826 / zh-cn:4168；另一 pass en 17821 / zh-cn 2176 / zh-tw 3）。因只增不減全保留 → **最終 SFT 是多語（英文為主）+ 繁中增量**，**並非「繁中為主的 SFT」**。打包後解碼抽樣亦見大量英文/code（如 Julia、CSS）。這是全功能復現 MiniMind 多語 SFT 的直接結果，須誠實陳述、勿誤稱繁中 SFT。
2. **「只增不減」在 SFT 路徑有例外（畸形/不完整丟棄，非品質過濾）**：
   - `clean_sft.py`：丟 **4,480 筆**「無任何非空 assistant 回覆」之殘缺記錄（純 user 提問無答案者）。
   - `convert_sftt2t.py`：`if not conv: return None` — 丟無 `conversations` 之記錄。
   - `assemble` `norm_r1`：`if not inp or not ans: return None`（distill_r1 因此 110,000→**109,999**）；`norm_messages`：`if not msgs: return None`。
   - `pack_sft_batch`：`except Exception: continue` — **靜默丟棄任何套 chat template/tokenize 出錯的對話**。
   - **pretrain 路徑則為乾淨 1:1**（僅跳空行/壞 JSON）。
3. ~~**tool_calls / tools 欄位不轉繁**→ 工具參數內若含簡體會殘留。~~ **【v1 已修復，見 §4.7】** 原 MiniMind 管線不轉 tool_calls，實測 **217,783 筆**工具參數殘留簡體；清理階段對 tool_calls/tools 的中文字串做遞迴 s2twp → **殘簡歸零**（打包後解碼實見 tool 定義為繁體，如「時區名稱」「將文本翻譯成目標語言」）。
4. **HAN 轉換閘只涵蓋 U+4E00–9FFF**（不含 CJK Ext-A/B、相容表意字）→ 極少數罕字略過轉換。
5. **SFT 打包時 20% 機率注入 system prompt** → 原 MiniMind 的 10 條 pool **4 條含 minimind 身份、5 條為簡體**，對繁體品牌模型為污染源。**【v1 已修復】** `dataset/lm_dataset.py` 的 `SYSTEM_PROMPTS` 已全數改為繁體、身份改為 Yunmo/中性（見 §4.7），重打包時生效——**打包後解碼實證**注入的正是「你是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。」等繁體 prompt；另 80% 機率移除空 `<think>\n\n</think>\n\n` 區塊（不變）。
6. **身份清理殘留 0.0003%**（16 筆異質性長尾）——不宣稱歸零（見 §4.7 C）。

### 4.4 轉繁方法與品質

- **OpenCC `s2twp`**（簡→繁台灣化用語，phrase-level），每 worker 實例化一次；HAN regex `[一-鿿]` 判別，只轉含中文之記錄。
- **1:1 轉換**：`conv_line` 僅於 `not line`（空行）或 `json.loads` 例外（壞行）回傳 None；其餘全留（含純英文/code/缺 text）。
- **繁體品質（content/reasoning 欄位，抽樣 3000 筆/檔，s2t 交叉檢測）**：偵測 0.13% 差異，逐一檢視**全為台灣標準字 vs 古典異體**（群/唇/念 台標 vs s2t 偏好 羣/脣/唸），**非殘留簡體**；真實殘簡 < 0.05%，與 MiniMind 原資料同級。**唯 `tool_calls` 欄位原本未轉、殘簡 217,783 筆，已於清理階段補轉（§4.3 #3、§4.7）。**
- **已知 s2twp 缺陷（實測 negligible）**：一對多誤轉（`濃鬱`應為`濃郁`，鬱/郁 類）；非技術語境過度在地化（对象→物件、类型→型別，實測 162k 字中物件 2/型別 15）。

### 4.5 wiki-zh-tw（台灣增量，使用者自有 `WikiZH_Dataset` 專案）

- **來源 dump**：`zhwiki-20260601-pages-articles.xml.bz2`（官方 3.10 GB，2026-06-01）。
- **管線**：下載 → `wiki_parser.py` 解析 → `md_converter.py` XML→Markdown → `md_to_json.py` Markdown→JSON（OpenCC `s2twp` + pangu 中英空格）。
- **產出**：**1,548,140 篇文章 → 3,688,410 段落條目**（每筆為一段落，非整篇），6 個 `wiki_pretrain_part1..6.json`；格式 `{"title":"主題 - 章節","text":...}`；infobox/表格/圖連結/分類皆剝除。
- **修正的 4 個 bug（供資料品質章）**：① OpenCC 每檔重載瓶頸（1.5M 檔 2h→7h，改 per-process 快取）；② `{{%}}/{{val}}/{{pct}}` 數值模板被通用 `{{}}` 剝除致 `14%`→`14`（先還原再剝、標點清理不刪 `%`）；③ `wiki_parser.py` 反斜線殘留 `\1\2`（污染 46% markdown 檔，如 `螺（学名：\1\2Ellobium）`，修為正確 backref）；④ part 檔名雙副檔名 bug。
- **記錄數 contradiction**：本次建置（2026-06-01 dump）= **3,688,410**；README 舊表列 3,607,037（2026-01-02 舊 HF 發布）。**v1 一律用 3,688,410。**

### 4.6 資料長度分布（實測，供 seq_len 決策佐證）

- pretrain_t2t：p50 269 / p90 564 / p95 651 / p99 815 / max 1459 tok（vocab6400 量）→ **512 會截半，故用 ≥1024**。
- SFT 呈**雙峰**：指令短（tw-instruct p90≈615），推理超長（**distill_r1 p99≈7719、max≈26000**；qwen235b p99≈3370）；code p90≈3500。打包時巨型記錄（實測有 89,499、377,025 token 者）**不截斷、跨多塊保留**。

### 4.7 身份污染稽核與清理（策略 C：中性化 + 注入一致 Yunmo 身份）

> **動機**：MiniMind SFT 語料源自多老師蒸餾（minimind/DeepSeek-R1/通義千問-Qwen/ChatGPT-OpenAI 等），大量 assistant 回覆**自稱為這些老師模型**。若直接訓練，Yunmo 會產生**身份混亂**（問「你是誰」時自稱 Qwen/DeepSeek）。這是繁體品牌小模型不可接受的污染。GPU 訓練前的大範圍稽核發現此問題，於**開訓前**修復（腳本 `data/script/clean_sft.py`，複製於 repo `scripts/clean_sft.py`）。**Yunmo 正式身份句：「我是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。」**

**A. 稽核發現（原始 `sft.jsonl` 5,816,011 筆全掃）**
| 污染類型 | 量級 |
|---|---|
| minimind 身份自稱 | 63,139 |
| jingyaogong（MiniMind 作者）身份 | 3,845 |
| Qwen/通義千問 自稱 | 數千（多語序） |
| DeepSeek-R1/深度求索 自稱 | 數百 |
| ChatGPT/OpenAI/GPT 「我是…」直接自稱 | ~30 |
| 「作為由 OpenAI 開發的 AI…」refusal 模板 | 192 |
| tool_calls/tools 參數殘留簡體 | 217,783 |
| 無任何有效 assistant 回覆的殘缺記錄 | 4,480 |

**B. 清理原則（策略 C）—— 只中性化「真·自稱」，絕不動討論/事實/角色扮演**
- **改寫**：assistant/system 中「我＝某老師模型」的自稱句 → 統一 canonical Yunmo 身份句（保留句首問候與句子其餘內容）。經典兩句自介（「…語言模型。我的中文名叫通義千問，英文名叫 Qwen。」）改寫後折疊重複。
- **保留（關鍵，使用者明確要求「不要無腦取代」）**：老師名的**知識性討論**（「Qwen 是阿里巴巴開發的模型」）、**角色扮演**（「如果我是阿里巴巴 CEO」「我叫馬雲」「我是主持人 ChatGPT」「面試者：我叫李明」）、**語氣詞**（「我是說…」）、**事實**（「OpenAI 開發的 ChatGPT」）、**技術討論**（「setHidden 方法並非阿里雲團隊特有」）一律**不動**——避免破壞世界知識與指令多樣性。
- **偵測法**：以「第一人稱繫詞（我是／我，X，／我由…／中文名叫…）＋ 開發者/模型名 ＋ 開發動詞（研發/開發/訓練/建立/推出…）」錨定自稱，涵蓋多語序、possessive（我是阿里雲的 Qwen）、基於（我是基於 Qwen 模型）、markdown/全形引號容錯（我的中文名是**通義千問**）；並排除 `如果我/我認為/我使用/扮演/我是說/…` 等討論語境。規則以**人工判讀 45+ 代表樣本 + 對抗性測試迭代收斂**（每輪從未修改的原始檔重跑）。
- **特例**：refusal 模板「作為由 OpenAI 開發的 AI 語言模型，我無法…」採**子句替換**（只換身份子句為「作為 Yunmo（…）」，**保留拒答內容**）；minimind/jingyaogong 因非真實世界主題採全域安全替換；tool_calls/tools 中文字串遞迴 s2twp 補轉繁。

**C. 清理後複驗（`sft_clean.jsonl` 5,811,531 筆全掃）**
| 指標 | 清理前 → 後 |
|---|---|
| minimind / jingyaogong | 66,984 → **0** |
| tool_calls 殘簡 | 217,783 → **0** |
| 作為-錨定 refusal 自稱 | 192 → **3** |
| 我-錨定真·自稱殘留（廣義偵測，排除角色扮演/討論） | 數千 → **13** |
| **合計殘留** | 數千 → **16（0.0003%）** |
| Yunmo 身份注入（訊息數） | 0 → **22,426** |

- **殘留 16 筆**為異質性長尾（罕見措辭：「由阿里雲**提供**支援」、「我是使用 OpenAI 資料集**訓練**的」、「AI 語言模型**（Qwen）**」括號式等）；再追則 FP 風險上升，**故誠實報告殘留率 0.0003%，不宣稱歸零**。
- **可復現性**：清理**每次從未修改的原始 `sft.jsonl` 重跑**（`SRC=sft.jsonl` → `OUT=sft_clean.jsonl` 全覆蓋），非累積式；體積僅縮 0.047%（15.70→15.69 GB），證明為外科手術式改寫而非大量刪除。原始 `sft.jsonl` 修改時間停在建置日、從未被動。
- **pretrain 亦需清理（開訓前驗證修正之前的錯誤假設）**：初判「pretrain 僅知識提及非自稱」經全掃**證偽**——實有 1,074 筆「我是 ChatGPT／文心一言…」自稱 + 542 筆 refusal 型（0 殘簡、0 minimind）。以同一 `clean_content` 邏輯清理（`clean_pretrain.py`，複用 SFT 身份規則），改寫為 Yunmo（不刪記錄）→ **老師自稱 1,074→80、refusal 542→0、注入 Yunmo 3,173**。殘留 80（0.00066%）為異質長尾與角色扮演（如「銷售員 ChatGPT」），如實揭露不宣稱歸零；pretrain 為無監督文本、身份信號弱，最終身份由 SFT 主導。
- **tokenizer 不需重訓**：污染在資料內容非詞表，重打包即可。

### 4.8 打包後端到端驗證（水質檢查，非僅管線機制）

> 教訓：早期只驗「水管」（管線機制）未驗「水質」（內容衛生），才漏掉污染。此節為**直接解碼實際訓練用的 `.bin`**（GPU 真正吃的 token）之核對，均已通過。

| 檢查項 | 方法 | 結果 |
|---|---|---|
| tokenizer 正確 | 載入 `./model` | vocab **24000**、bos/eos **1/2** ✓ |
| ids/mask 對齊 | 長度比對 | 皆 **3,857,469,341**、逐位對齊 ✓ |
| 繁體台灣 | 解碼開頭/中/末塊 | 臺北/忠孝東路/西門町/101 大樓/綻放 ✓ |
| **老師自稱歸零** | 取樣 2,500 塊 / 5.12M token，regex 掃「我是…Qwen/DeepSeek/minimind…」 | **0** ✓✓ |
| Yunmo 身份存在 | 同取樣 | **702 次** ✓ |
| 注入 prompt 已修 | 解碼 mask=0 段 | 見「你是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。」等繁體 ✓ |
| tool_calls 繁體 | 解碼 tool 定義 | 「時區名稱」「將文本翻譯成目標語言」✓ |
| chat 結構 | 解碼 | `<|im_start|>`/`<|im_end|>`/`<think>`/`<tool_call>` 正確 ✓ |
| **loss-mask 正確** | 解碼 mask=0 段 | 僅含 system/user/`<|im_start|>assistant\n` 表頭；assistant 內容全在 mask=1 ✓ |
| mask=1 佔比 | 全 20M + 取樣 5.12M | **~85%**（合理：推理 `<think>` 超長皆屬 assistant；非 bug） |

---

## 5. 訓練（雲端 pretrain+SFT；本地 DPO/RL）

### 5.1 鎖定超參

| | Pretrain | SFT |
|---|---|---|
| 資料載入 | **packing**（EOS 分隔 concat，零 padding） | **packing + loss-mask**（僅 assistant 段算 loss，長對話跨塊保留） |
| seq_len | 1024 | 2048 |
| batch × accum | 128 × 8 | 64 × 8 |
| 有效 batch | 1,048,576 token/step | 1,048,576 token/step |
| 峰值 lr | 3e-4 | 5e-5 |
| epochs 上限 | 6（實際由時間封頂決定） | 6 |
| 時間封頂（`--max_minutes`） | ~900 分（~15h） | ~1080 分（~18h） |
| 起點 | 從頭（from_weight=none） | 接 pretrain 權重 |

**共通**：AdamW、bf16、grad_clip 1.0、cos schedule `get_lr = lr·(0.1+0.45·(1+cos(π·step/total)))`（峰值=lr、底=lr/10、**無 warmup**）、seed 42（含分散式 rank 偏移）、`--from_resume` 斷點續訓（存 fp16 CPU state_dict + 分離的 resume bundle）。

### 5.2 打包內部機制（reproducibility）

- **pretrain**：每 doc `[bos] + tokenize(text) + [eos]` → 連續 uint16 串流 `yunmo_pretrain_packed.bin`；`PackedPretrainDataset` 切 1024 塊，`labels = input_ids.clone()`（全 token 訓練，模型內部移位）。
- **SFT**：`pre_processing_chat`（20% 注入 system prompt，**已修為繁體 + Yunmo/中性 10 條**）→ `apply_chat_template(tools=...)` → `post_processing_chat`（80% 移空 think）→ tokenize → **assistant-only loss mask**。mask 演算法掃描 `<|im_start|>assistant\n`(bos_id) 起、至 `<|im_end|>\n`(eos_id) 止，**mask=1 涵蓋 assistant 內容含結尾 `<|im_end|>\n`、排除 `<|im_start|>assistant\n` 表頭與所有 user/system**（§4.8 已解碼實證）。輸出 `yunmo_sft_ids.bin`(uint16) + `yunmo_sft_mask.bin`(uint8)；`PackedSFTDataset` 以 `labels[~mask]=-100` 只監督 assistant。**輸入為 `sft_clean.jsonl`（污染清理後）。**
- **重打包範圍**：污染清理後**兩階段皆重打包**——SFT（`--stage sft --sft_in …/sft_clean.jsonl`）與 pretrain（`--stage pretrain --pretrain_in …/pretrain_clean.jsonl`）。

### 5.3 算力預算與曝光

- **雲端**：RTX PRO 6000（Blackwell，96GB），**36 小時硬上限**，做 pretrain+SFT。CUDA 12.8+（sm_120；cu126 → no kernel image）。
- **時間封頂機制**：到時存檔並停止 → 保證 36h 內完成，且吞吐未知也能填滿時間（前 1hr benchmark 後再分配 900/1080）。取代早期「eval 驅動 epoch」。
- **算力換算**：有效 ~1.5e14 FLOP/s × 1.3e5 s ≈ **1.9e19 FLOPs**；`token = FLOPs/(6N)`。
- **曝光（依吞吐，時間封頂自適應）**：

| 吞吐 | pretrain（2.35B/ep） | SFT（3.857B/ep） |
|---|---|---|
| 保守 128k tok/s | ~2.9 ep | ~2.1 ep |
| 樂觀 213k tok/s | ~4.9 ep | ~3.6 ep |

  **tok/param ≈ 35–59**（近/超 Chinchilla 最優 20）→ 餵得飽（*待 eval 驗證*）。重複 ≤~5 epoch 於資料受限縮放律（**Muennighoff 2023**）近似等值於等量 unique。
- **本地**：RTX 4070 Ti（12GB）做 DPO/RLAIF/Agent RL（rollout-bound，不佔雲端計時）。

**設計理由：**
- **packing**：MiniMind pretrain 為短 Q&A（avg <400 tok）padding，故用 seq 380；1024-padding 下 300-tok 樣本浪費 ~70% 算力於 pad（pad 無 loss）。我方池含 wiki 長文/長推理，380/512 會截斷。packing → 零 padding 浪費 + 保長上下文，同 FLOP 處理更多真 token（實測早期 padding 浪費 54–66%）。無 packing 則 36h 不夠（單 pretrain 2ep ~23–30h）。
- **SFT seq 2048**：覆蓋推理 ~p80（勝 MiniMind 1350），為勝過原版推理的真槓桿；packing+跨塊 loss-mask 完整保留長對話（非早期 bucketing+丟棄）。SFT 打包後 mask=1 佔 ~85%（推理 `<think>` 超長皆屬 assistant），token 利用率高。

### 5.4 監控與模型保全（wandb）

- **指標**：loss / logits_loss / aux_loss / **grad_norm**（發散預警）/ learning_rate / **tok_per_sec** / progress / tokens_seen / elapsed_min；GPU 使用率/顯存/功耗/溫度（自動）。
- **Config**：全超參 + 參數量 + vocab + **git_sha** + 資料塊數/token（機器回收後可完整重現）。
- **模型保全**：雲端機器用後回收 → 中途 checkpoint 每 90 分（`CKPT_MIN`）上傳 wandb artifact，**時間制（非步數）→ 儲存可預測**，只留最近 2 版（`CKPT_KEEP`，~2×408MB），prune 於 **daemon 執行緒 → 不阻塞訓練**；階段結束自動上傳 `pretrain_768` / `full_sft_768`。
- **環境釘版**：`transformers==4.57.6`、`trl==0.13.0`、**`huggingface_hub[hf_transfer]<1.0`**（hub 1.x 破 transformers 4.57 import）。repo `github.com/yuhuanowo/minimind`(master)；packed bin 經私有 HF dataset `yuhuanowo/yunmo-v1-packed` 中轉（**~16.3 GB**）。**污染清理後 pretrain 與 SFT bin 均已更新，3 個 bin（pretrain + sft ids/mask）+ meta 皆須重傳。**

---

## 6. 下游對齊（本地 4070 Ti，不佔 36h）

沿用 MiniMind 已驗證腳本預設：
| 階段 | 資料 | 關鍵超參 | 備註 |
|---|---|---|---|
| DPO | `dpo.jsonl`(s2twp) | lr **4e-8**、β **0.15**、seq 1024 | 凍結 ref |
| RLAIF **CISPO** | `rlaif.jsonl` | lr **3e-7**、num_generations **6**、β **0.1** | **需外部 reward `InternLM2-1.8B-Reward`**（bf16 ~3.6GB，12GB 吃緊可跑）；MiniMind RL 預設 loss_type=**CISPO** 非 GRPO |
| Agent RL | `agent_rl*.jsonl` | lr 3e-7、max_total 2500、thinking_ratio 0.1 | tool-use，可選 |

RL 資料由 SFT 資料衍生（末個 assistant 留空供 rollout）。**若 RL 資料另行打包，須同樣走 §4.7 身份清理**（目前 clean_sft.py 針對 sft.jsonl；DPO/RL 檔如含老師自稱應比照處理）。

---

## 7. 評估（規劃，訓後執行）

- **對照**：MiniMind v3 官方權重（下限脊柱）+ Yunmo v1 各階段 checkpoint。
- **客觀**：`lm-evaluation-harness`，MiniMind 7 項——**C-Eval / CMMLU / ARC-Easy / PIQA / OpenBookQA / HellaSwag / Social-IQa**，對數概率法（非自由生成）；另切**繁中獨立 eval 集**（不進訓練）量繁中知識/推理。**唯 eval 集須與訓練集獨立**。
- **功能**：繁中對話（`eval_llm.py`）+ tool-call（`eval_toolcall.py`）+ `<think>` 推理品質。
- **身份 eval（新增，因污染清理）**：問「你是誰／你叫什麼／你是哪家公司開發」應答 Yunmo/YuhuanStudio，**不得**自稱 Qwen/DeepSeek/ChatGPT/minimind；量測身份一致率。

---

## 8. 復現性總表

| 項目 | 值 |
|---|---|
| Repo / branch | github.com/yuhuanowo/minimind / master |
| 資料中轉 | HF dataset `yuhuanowo/yunmo-v1-packed`（private，~16.3 GB） |
| seed | 42（+ rank 偏移） |
| 每 run 記錄 | git_sha + 全超參（wandb config） |
| tokenizer | `./model`，24000 vocab / 23,708 merges / 36 special（解碼實證） |
| 清理腳本 | `scripts/clean_sft.py`（= `data/script/clean_sft.py`）；輸入原始 `sft.jsonl`、輸出 `sft_clean.jsonl` |
| 打包 meta | `yunmo_pack_meta.json`（pretrain + sft 各 file/tokens/seq_len/blocks/source） |
| 打包指令（SFT 重打包） | `python scripts/pack_yunmo_data.py --stage sft --sft_in F:/AI/data/yunmo/sft_clean.jsonl --workers 16` |

---

## 附錄 A. MiniMind v3 權威參照（原碼交叉核對）

> MiniMind 的圖、argparse、README 三者數值已漂移，不可盲抄；以下為 `model_minimind.py` 原碼權威值。

**架構權威值（minimind-3, 64M）**：hidden 768 / layers 8 / q_heads 8 / kv_heads 4 / head_dim 96 / vocab 6400 / intermediate 2432 / rope_theta 1e6 / max_pos 32768 / tie True / QK-Norm / RMSNorm eps 1e-6 / SwiGLU。MoE 版：4 專家 top-1、無 shared expert、aux_coef 5e-4。

**權威配方（proven baseline，供對照）**：Pretrain 5ep/seq380/padding；SFT 5ep/seq1350/padding；DPO/RLAIF(CISPO)/AgentRL 各 1ep。共通 AdamW/bf16/grad_clip1.0/同 cos schedule。SFT loss 僅算 assistant；20% 注入 system prompt；~80% 移空 think。

**已知 12 處內部不一致（擇要）**：pretrain lr argparse 5e-4 vs 圖 1e-4；full epochs 5(dataset.jpg) vs argparse 2；SFT seq_len full 1350 vs argparse 768；DPO β 0.15 vs code 0.1(dead)；RMSNorm eps config 1e-6 vs class 1e-5(未用)；max_pos 32768 vs tokenizer 131072 vs YaRN orig 2048；MoE 標註 198M-A64M / 198M / 0.2B-A0.06B 混用。**Yunmo 一律以原碼權威值 + 自身實測為準。**

**MiniMind 原資料本身即含多老師身份污染**（minimind/Qwen/DeepSeek 自稱），為蒸餾語料通病；Yunmo 的 §4.7 清理即針對此。論文可將此列為「復現他人蒸餾資料時的身份一致性處理」貢獻。

---

## 附錄 B. 演變史與被放棄的方案（供論文 Design/Ablation 敘事）

| 主題 | 早期（計劃書，已放棄） | 最終（實際） | 觸發原因 |
|---|---|---|---|
| 模型 | 166M/20層（另有 32K vocab、12層/kv2 變體構想） | **195.4M/24層** | 96GB/36h 確認後選 24；先估 185M，實測 195.4M |
| 資料策略 | 過濾 + 英文下採樣 + MinHash-LSH 跨集去重 + 丟長推理 + 棄舊散檔 | **只增不減（1:1 全留）** | 使用者「保留啊…不應有任何減少」；dropEN 753,663 事件 |
| token 數 | char 估 ~2–3B → 抽樣估 4.27B/5.85B | **打包實測 2.35B/3.857B** | 抽樣被巨型離群記錄（89k–377k tok）灌爆均值 |
| 身份污染 | 未意識（沿用 MiniMind 資料） | **開訓前大清理（策略 C）** | 開訓前大範圍稽核發現多老師自稱、tool_calls 殘簡 |
| pretrain=SFT 同池假設 | 假設同內容 | **證偽**：pretrain_t2t 99%中文、sft_t2t 86%英文，非同池 | 實測語言分布 |
| epoch | 「少 epoch」（誤以為 10B unique） | 高 epoch → **時間封頂自適應** | 實測中文僅 ~2.35B ≈ MiniMind 2.2B |
| SFT seq/載入 | 1350→1024，padding+bucketing+丟>2048 | **2048 + packing + 跨塊 loss-mask（不丟）** | 只增不減 + packing |
| SFT lr | 1e-5 | **5e-5** | — |
| tokenizer | 3GB 語料（OOM，5050 萬唯一詞） | **100MB（34 秒成功）** | BPE 早飽和，MiniMind 只用 10k 行 |
| 去重 | 跨集去重（stage2 抽樣測重疊率 26.8%） | **不去重**（只增不減下無意義） | 鐵律 |
| DPO/RL | 早期規劃 | 不變（本地 4070Ti） | — |

**被放棄的資料元件**：FineWeb-Edu（英文過剩）、Magicoder / py18k / code.jsonl 額外 code、舊 v0.x 散檔 sft_512/1024/2048、cn/tw pretrain 重複版。**注意**：`stage1_stats.json` / `measure_char.txt` / `measure_real.txt` / `stage2_overlap_report.txt` 等實測報告皆屬**此被放棄的過濾管線**（含 dropEN/去重/品質過濾），僅為「促成決策的前期資料分析（EDA）」，**不代表最終資料**——最終資料走 1:1 轉換 + §4.7 身份清理。

---

## 附錄 C. 一句話誠實聲明（貼進論文）

> Yunmo v1 **確定達成**：繁體台灣特化的全功能小 MiniMind 復現、**實測 46% 更高的繁體 tokenizer 效率**、MiniMind 結構性所無的台灣原生知識注入、**清除多老師身份污染並注入一致 Yunmo 身份（token 流實測 0 老師自稱、殘留 0.0003%）**。**尚待 eval 驗證（不可預先聲稱為結論）**：整體能力對 MiniMind 的超越、195.4M 於此資料量下的訓練充分度。
