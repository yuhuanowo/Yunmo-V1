# Yunmo v1 — 權威技術規格（論文事實骨幹）

> **單一真實來源（Single Source of Truth）**。本檔全部數字皆為「實際做了什麼」的定稿值，均經程式驗證（實例化數參數、逐筆計數、打包實測、抽樣核對）。
> 論文的 Model / Data / Training 章節請直接以本檔為準。舊的 `YUNMO_V1_PLAN.md`（原始計劃書）與 `YUNMO_DATA_STATUS.md`（資料盤點）已標為歷史，其資料策略已被本檔的「只增不減」定案取代。
>
> 定位：**繁體台灣特化 · 全功能小語言模型 · 基於 MiniMind v3 增量復現**。最後校準 2026-07-04。

---

## 1. 定位與貢獻（誠實邊界）

**價值主張不是「更大所以更強」，而是「MiniMind 結構性做不到的繁體台灣特化 + 全功能復現，於較大規模」。**

MiniMind v3 是簡體源模型，結構上沒有繁體台灣的原生知識與在地語感。Yunmo v1 的貢獻：

| 貢獻 | 性質 |
|---|---|
| **繁體台灣在地知識 / 語感** | MiniMind 資料所無（wiki-zh-tw、tw-instruct 在地化增量）——核心差異化 |
| **原生繁體 tokenizer** | vocab 24000 繁體特化 vs MiniMind 簡體 6400；繁體壓縮率大幅提升（§3） |
| **全功能復現** | chat / 推理(`<think>`) / tool-call / 英文·code —— MiniMind 資料本身已含，1:1 轉繁全數保留 |
| **較大規模** | 195.4M（24 層，3.05× MiniMind-3 64M）|

**誠實邊界（不自欺，供論文 Limitations）：**
- 整體能力「超越 MiniMind」是**假設，須 eval 驗證**，非保證。Scaling law 下參數 ~3× 僅使 loss 降個位數百分比（冪律），非能力倍增。
- s2twp 簡→繁轉換有少量誤轉（如 `濃鬱`→應為`濃郁` 類一對多），為扣分項，靠抽樣核對控管。
- 最壞回退＝「繁體版 MiniMind 全功能復現」（因方法上是 MiniMind 資料 1:1 轉繁的超集）。

---

## 2. 模型架構（195.4M，深而窄）

架構完全沿用 MiniMind v3 的 `model/model_minimind.py`，**僅刻意改動兩處**：深度與詞表。

```python
MiniMindConfig(
    hidden_size          = 768,
    num_hidden_layers    = 24,      # 改動①：8→24（加深換容量）
    num_attention_heads  = 8,
    num_key_value_heads  = 4,       # GQA 2:1
    head_dim             = 96,      # 768/8
    intermediate_size    = 2432,    # ceil(768·π/64)·64，SwiGLU（與 MiniMind 同公式）
    vocab_size           = 24000,   # 改動②：6400(簡)→24000(繁)
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
| 詞嵌入（tied，24000×768） | 18.43M |
| 每層 = attn 1.770M + ffn 5.603M | 7.375M |
| 24 層合計 | 177.0M |
| **總計** | **195.4M**（非嵌入 177.0M；3.05× MiniMind-3） |

**繼承自 MiniMind v3（未改動）：** Pre-Norm + RMSNorm(eps 1e-6)、SwiGLU(silu)、RoPE(theta 1e6)、**QK-Norm**（q/k 每頭 RMSNorm，Qwen3 式，穩定深層）、tied embeddings、無 bias、flash-attention(SDPA)、shift-by-one CE（模型內部，ignore_index=-100）。

**設計理由（深而窄）：** 固定參數下加深優於加寬（MobileLLM, arXiv 2402.14905；MiniMind2 768/16 亦此路線）。768/24 落在 d_model 512–1536 的合理帶內。MoE 需 2–4× 資料方訓得起，資料受限故用稠密。**最小改動原則**：只動深度與詞表兩個變數 → 復現忠實、風險最低。

---

## 3. Tokenizer（重訓，繁體特化）

- **演算法**：BPE + ByteLevel（沿用 MiniMind `train_tokenizer.py` 方法），`vocab_size = 24000`，`min_frequency = 2`。
- **訓練語料**：從已轉繁的 `yunmo/{pretrain,sft}.jsonl` 多點抽樣 ~100MB 代表性語料（含 zh/en/code）。
- **特殊 token（36 個，完整對齊 MiniMind v3，否則 chat template / tool / think 不相容）**：
  `<|endoftext|>`(0, pad)、`<|im_start|>`(1, bos)、`<|im_end|>`(2, eos)、`<tool_call>`(21)…`</think>`(26)、緩衝 `<|buffer1..9|>`(27–35)。ChatML 模板，支援 `tools` 渲染、`tool` role、自適應 `<think>` 注入。
- **壓縮率（實測，char/token）**：繁中 **1.89**、英文 4.58、code 2.29、混合 2.68。於繁體台灣文本較 MiniMind 簡體 tokenizer **少用約 46% token**。

---

## 4. 資料（鐵律：只增不減）

> **鐵律**：資料與 MiniMind **一模一樣，只做簡→繁 + 台灣增量，不丟任何一筆**。無語言過濾、無品質過濾、無去重、無英文下採樣。

### 4.1 最終訓練資料（逐筆計數，非抽樣）

| 檔案 | 筆數 | 打包後 token | 打包後區塊 |
|---|---|---|---|
| `pretrain.jsonl` | **12,157,237** | **2,353,402,018**（2.35B） | 2,298,244 × 1024 |
| `sft.jsonl` | **5,816,011** | **3,874,644,587**（3.87B） | 1,891,916 × 2048 |
| 合計 unique | | **~6.22B** | |

### 4.2 血緣組成（只增不減）

**pretrain（12,157,237）**
| 來源 | 筆數 | 血緣 |
|---|---|---|
| pretrain_t2t（s2twp 1:1 轉繁） | 8,468,827 | **MiniMind 本體**，零丟失 |
| wiki-zh-tw（繁體維基） | 3,688,410 | **台灣增量** |

**sft（5,816,011）**
| 來源 | 筆數 | 血緣 |
|---|---|---|
| sft_t2t（s2twp 1:1，tool-call/英文全留） | 5,109,432 | **MiniMind 本體**，零丟失 |
| tw-instruct（在地化） | 486,580 | 台灣增量 |
| distill_r1（R1 推理，含 reasoning_content） | 109,999 | MiniMind 本體（用 tw/ 現成繁體版） |
| qwen3_235b（蒸餾推理） | 110,000 | MiniMind 本體（用 tw/ 現成繁體版） |

### 4.3 轉繁方法與品質

- **s2twp**（OpenCC，簡體→繁體台灣化用語，phrase-level）。以 HAN regex `[一-鿿]` 判別，僅轉中文字，英文/code/結構原樣保留。
- **1:1 轉換**：只跳過空行/不可解析行；保留 tool_calls / reasoning_content 結構，只轉其中中文內容。
- **繁體品質（抽樣 3000 筆/檔，s2t 交叉檢測）**：偵測到 0.13% 差異，但逐一檢視全為「台灣標準字 vs 古典異體」（如 群/唇/念 我方用台標，s2t 偏好 羣/脣/唸），**非殘留簡體**；真實殘簡遠低於 0.05%，與 MiniMind 原資料同級。

---

## 5. 訓練（雲端 pretrain+SFT；本地 DPO/RL）

### 5.1 鎖定超參

| | Pretrain | SFT |
|---|---|---|
| 資料載入 | **packing**（EOS 分隔 concat 填滿，零 padding） | **packing + loss-mask**（僅 assistant 段算 loss，長對話跨塊保留） |
| seq_len | 1024 | 2048 |
| batch × accum | 128 × 8 | 64 × 8 |
| 有效 batch | 1,048,576 token/step | 1,048,576 token/step |
| 峰值 lr | 3e-4 | 5e-5 |
| 起點 | 從頭（from_weight=none） | 接 pretrain 權重 |

**共通**：AdamW、bf16、grad_clip 1.0、cos schedule `lr·(0.1+0.45·(1+cos(π·step/total)))`（峰值=lr、底=lr/10、**無 warmup**）、seed 42、`--from_resume` 斷點續訓。

### 5.2 算力預算與曝光

- **雲端**：RTX PRO 6000（Blackwell，96GB），**36 小時硬上限**，做 pretrain + SFT。CUDA 12.8+（Blackwell sm_120）。
- **時間封頂機制**（`--max_minutes`）：到時存檔並停止 → 保證 36h 內完成，且吞吐未知也能填滿可用時間。分配 pretrain 900 分、SFT 1080 分（前 1hr benchmark 後可調）。
- **曝光量（依實測吞吐，時間封頂自適應）**：

| 吞吐 | pretrain（2.35B/ep） | SFT（3.87B/ep） |
|---|---|---|
| 保守 128k tok/s | ~2.9 epoch | ~2.1 epoch |
| 樂觀 213k tok/s | ~4.9 epoch | ~3.6 epoch |

  tok/param ≈ 35–59（近/超 Chinchilla 最優 20 → 餵得飽，非欠訓）。多輪重複 ≤~5 epoch 於資料受限縮放律（Muennighoff 2023）近似等值於等量 unique。
- **本地**：RTX 4070 Ti（12GB）做 DPO / RLAIF(CISPO) / Agent RL（rollout-bound，不佔雲端計時）。

### 5.3 監控與模型保全（wandb）

- **指標**：loss / logits_loss / aux_loss / **grad_norm**（發散預警） / learning_rate / **tok_per_sec** / progress / tokens_seen / elapsed_min；GPU 使用率/顯存/功耗（自動）。
- **Config**：全超參 + 參數量 + vocab + **git_sha** + 資料塊數/token（機器回收後可完整重現）。
- **模型保全**：中途 checkpoint 每 90 分上傳 wandb artifact（時間制 → 儲存可預測；只留最近 2 版 → 硬上限；prune 於 daemon 執行緒 → 不阻塞訓練）；階段結束自動上傳最終 `pretrain_768` / `full_sft_768`。防雲端機器回收造成模型遺失。

---

## 6. 復現性

- 程式碼、tokenizer、config 全部在 repo（`github.com/yuhuanowo/minimind`，branch master）；packed bin 經 HuggingFace dataset 中轉。
- 每次 wandb run 記錄 git_sha + 完整超參 → 可精確重現。
- 隨機種子 42（含分散式 rank 偏移）。

---

## 7. 評估（規劃，待訓後執行）

- **對照**：MiniMind v3 官方權重（下限脊柱）+ Yunmo v1 各階段 checkpoint。
- **客觀**：lm-evaluation-harness，MiniMind 7 項（C-Eval / CMMLU / ARC-Easy / PIQA / OpenBookQA / HellaSwag / Social-IQa），對數概率法；另切**繁中獨立 eval 集**（不進訓練）量繁中知識/推理。
- **功能**：繁中對話 + tool-call + `<think>` 推理品質。

---

## 附錄 A. MiniMind v3 權威參照（原碼交叉核對）

> MiniMind 的圖、argparse、README 三者數值已漂移，不可盲抄。以下為 `model/model_minimind.py` 原碼交叉核對後的權威值。

**架構權威值（minimind-3, 64M）**：hidden 768 / layers 8 / q_heads 8 / kv_heads 4 / head_dim 96 / vocab 6400 / intermediate 2432 / rope_theta 1e6 / max_pos 32768 / tie True / QK-Norm(q,k 每頭 RMSNorm) / RMSNorm eps 1e-6 / SwiGLU。

**已知內部不一致（擇要）**：pretrain lr argparse 5e-4 vs 圖 1e-4；full epochs 5 vs argparse 2；SFT seq_len full 1350 vs argparse 768；RMSNorm eps config 1e-6 vs class 1e-5(未用)；max_pos 32768 vs tokenizer 131072。**Yunmo 一律以原碼權威值 + 自身實測為準。**

**MiniMind v3 權威配方（proven baseline，供對照）**：Pretrain 5 epoch / seq 380 / padding；SFT 5 epoch / seq 1350 / padding；DPO 1 epoch；RLAIF(CISPO) + Agent RL 各 1 epoch。共通 AdamW / bf16 / grad_clip 1.0 / 同 cos schedule。**Yunmo 差異**：pretrain 改 packing+1024、SFT 改 packing+2048、epoch 改時間封頂自適應。
