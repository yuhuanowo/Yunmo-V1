# Yunmo v1 資料總整理

> 更新：2026-07-03 · 定位：繁體台灣特化 · 全功能小模型 · 基於 MiniMind v3 增量復現
> 鐵律：**資料與 MiniMind 一模一樣，只簡→繁 + 台灣增量，不丟任何一筆。**

---

## 1. 最終訓練資料（正式檔 —— 訓練只用這兩個）

| 檔案 | 大小 | 筆數 | 格式 |
|---|---|---|---|
| `F:/AI/data/yunmo/pretrain.jsonl` | 10.2 GB | **12,157,237** | `{"text": ...}` |
| `F:/AI/data/yunmo/sft.jsonl` | 14.6 GB | **5,816,011** | `{"conversations":[{role,content,reasoning_content?}]}` |

### pretrain.jsonl 組成
| 來源 | 筆數 | 血緣 |
|---|---|---|
| pretrain_t2t（繁 1:1） | 8,468,827 | **MiniMind 本體**，只簡→繁 |
| wiki（繁體維基） | 3,688,410 | **台灣增量**（WikiZH_Dataset 今日產出） |

### sft.jsonl 組成
| 來源 | 筆數 | 血緣 |
|---|---|---|
| sft_t2t（繁 1:1，含 tool-call/英文） | 5,109,432 | **MiniMind 本體**，只簡→繁、英文全留 |
| tw-instruct（在地化） | 486,580 | 台灣增量 |
| distill_r1（R1 推理） | 109,999 | MiniMind 本體（cn/，用 tw/ 現成繁體版） |
| qwen3_235b（蒸餾推理） | 110,000 | MiniMind 本體（cn/，用 tw/ 現成繁體版） |

---

## 2. 資料血緣圖（一模一樣，只增不減）

```
MiniMind v3 本體（簡體）          Yunmo v1（繁體）
─────────────────────           ─────────────────────
pretrain_t2t.jsonl    ──s2twp──▶ pretrain_t2t_zhtw   ┐
                                 + wiki（台灣新增）    ├─▶ pretrain.jsonl
sft_t2t.jsonl         ──s2twp──▶ _sftt2t_zhtw         ┐
cn/distill_r1_110k    ──(tw/已有繁體版)──────────────  ├─▶ sft.jsonl
cn/qwen3_235b         ──(tw/已有繁體版)──────────────  │
                                 + tw-instruct（新增） ┘
```
- **轉繁**：opencc `s2twp`（簡體→繁體台灣化用語）；英文/code 靠 HAN 檢測原樣過
- **零丟失**：只跳空行/壞行；無語言過濾、無品質過濾、無去重

---

## 3. 資料夾地圖

| 路徑 | 內容 | 狀態 |
|---|---|---|
| `data/yunmo/pretrain.jsonl` | 最終 pretrain | ✅ 正式 |
| `data/yunmo/sft.jsonl` | 最終 SFT | ✅ 正式 |
| `data/yunmo/pretrain_t2t_zhtw.jsonl` | pretrain_t2t 1:1 轉繁（已併入上方） | 🟡 中間，可留作重組源 |
| `data/yunmo/_sftt2t_zhtw.jsonl` | sft_t2t 1:1 轉繁（已併入上方） | 🟡 中間，可留作重組源 |
| `data/pretrain_t2t.jsonl` / `sft_t2t.jsonl` | MiniMind 官方原檔（簡體） | 📦 原料，保留 |
| `data/tw/` | 舊 v0.x 繁體版元件 | 📦 用了 distill/qwen/tw_sft，其餘閒置 |
| `data/cn/` | MiniMind distill/qwen 簡體原檔 | 📦 原料 |
| `data/_yunmo/norm/pretrain_t2t.jsonl` | **廢**：舊 stage1 砍英文版(768萬) | ❌ 刪 |
| `data/_yunmo/tokenizer_corpus.txt` | **廢**：舊 tokenizer 抽樣法 | ❌ 刪 |
| `data/_yunmo/reports/` | log + 測量報告 | 🟡 log 可清，報告可留 |
| `WikiZH_Dataset/output/tw/wiki_pretrain_part*.json` | wiki 繁體原始輸出(6 part) | 📦 已併入 pretrain，保留 |

---

## 4. 腳本清單（`data/script/`）

| 腳本 | 作用 | 狀態 |
|---|---|---|
| `convert_pretrain_t2t.py` | pretrain_t2t 1:1 轉繁 | ✅ 用 |
| `convert_sftt2t.py` | sft_t2t 1:1 轉繁 | ✅ 用 |
| `assemble_yunmo.py` | 組裝 pretrain/sft 最終檔 | ✅ 用 |
| `train_yunmo_tokenizer.py` | 訓 tokenizer(vocab 24000) | ⏳ 下一步（需改抽樣源） |
| `build_tokenizer_corpus.py` | 舊 tokenizer 抽樣（邊讀邊轉） | 🟡 待改：改從 yunmo/ 已轉繁資料抽 |
| `yunmo_dataprep.py` | 舊 stage1 精細管線(含過濾) | 🟡 廢棄流程，留 helper 供參考 |

---

## 5. 本輪定案決策

1. **定位**：繁體台灣特化，非「靠加大規模超越 MiniMind」（誠實版）
2. **鐵律**：資料與 MiniMind 一模一樣，只轉繁 + 台灣增量，**零丟失**
3. **tw/ 現成繁體版**：distill_r1 / qwen235b / tw-instruct 直接用 `tw/`，免重轉
4. **只需轉繁的新主資料**：pretrain_t2t、sft_t2t（`tw/` 沒有繁體版）
5. **sft_t2t 必加**：含 tool-call，是全功能復現本體，英文全留
6. **wiki**：台灣增量，369萬筆繁體併入 pretrain
7. **config**：**~195.4M**（hidden 768 / **layers 24** / heads 8 / kv 4 / head_dim 96 / vocab 24000）—— 對 MiniMind 只改深度 8→24、vocab 6400→24000（深而窄，MobileLLM/MiniMind2 依據）
8. **丟棄**：舊 v0.x 散檔（sft_512/1024/2048、tw/pretrain/*、FineWeb、Magicoder…）

---

## 6. 下一步

1. **清廢檔**（見 §4，省 ~10GB+）
2. **改 tokenizer 抽樣源** → 從 `yunmo/pretrain.jsonl` + `sft.jsonl` 抽（已繁體，不必再轉）
3. **訓 tokenizer**（vocab 24000）→ 驗繁體壓縮率 >1.3
4. **用真 tokenizer 重量 token** → 定死 pretrain/SFT 的 epoch、seq_len
5. **開訓**：pretrain(1024/packing) → SFT(2048/bucketing) → DPO/RLAIF（照 MiniMind recipe）

---

## 7. 待精算（tokenizer 訓好後）

- pretrain 實際 token 數：**~4.27B**（1216萬筆 × 均 351 tok；wiki 長條目拉高均值）
- SFT 實際 token 數：**~5.85B**（582萬筆 × 均 1007 tok）；共 unique ~10.1B
- SFT 實際 token 數、英文佔比、reasoning 長度分佈
- 依 MiniMind recipe 定 epoch（資料受限 scaling law：unique > repeats，≤~5ep 近等效）
