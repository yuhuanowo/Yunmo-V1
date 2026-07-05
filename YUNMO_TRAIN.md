# Yunmo v1 訓練報告

繁體台灣特化 · 基於 MiniMind v3 增量復現 · ~195M · RTX PRO 6000（~28h，pretrain + SFT）

## 模型

hidden 768 / layers 24 / heads 8 / kv 4 / head_dim 96 / intermediate 2432 / vocab 24000，
QK-Norm · RMSNorm(1e-6) · SwiGLU · RoPE(θ=1e6) · GQA · tied-emb · 無 bias —— 共 **195.4M**（Qwen3-style）。

## 資料（packing，零 pad）

| 階段 | 檔案 | token | 塊數 |
|---|---|---|---|
| Pretrain | `yunmo_pretrain_packed.bin` | 2.353B | 229.83 萬 |
| SFT | `yunmo_sft_ids.bin` + `mask.bin`（loss-mask，長對話跨塊保留） | 3.857B | 188.35 萬 |

語料 = MiniMind v3 全量以 OpenCC `s2twp` 轉繁 + 台灣在地增量；開訓前做多教師身份去污染（`scripts/clean_pretrain.py`、`scripts/clean_sft.py`，行為驗證見 [`YUNMO_EVAL.md`](./YUNMO_EVAL.md)）。

## 訓練設定

| | Pretrain | SFT |
|---|---|---|
| seq_len | 1024 | 2048 |
| batch / accum（effective ≈ 1M tok/step） | **48 / 21** | **16 / 32** |
| 峰值 lr | 3e-4 | 5e-5 |
| max_minutes | 860（~14h） | 700（~12h） |
| 實際訓練量 | ~1.65 epoch / ~3.9B tok（Chinchilla 匹配） | ~1 epoch |

共通：AdamW · bf16 · grad_clip 1.0 · cos schedule（底 = lr/10、無 warmup）· 斷點續訓。
吞吐 ~75k tok/s，**bandwidth-bound**（瓶頸在大詞表 logits 讀寫，非算力、非 attention）；顯存 ≈ 24 + 1.0×batch GB。

## 復現

```bash
# 1) 打包（本地，用 repo model/ 的 tokenizer；輸入為去污染後的 *_clean.jsonl）
python scripts/pack_yunmo_data.py --stage both \
  --pretrain_in pretrain_clean.jsonl --sft_in sft_clean.jsonl \
  --out_dir ./dataset --workers 16

# 2) 訓練（PRO 6000/Blackwell 需 cu128 torch；一鍵跑完兩階段，時間封頂保證停）
PRETRAIN_MIN=860 SFT_MIN=700 \
  EXTRA_PRE="--batch_size 48 --accumulation_steps 21" \
  EXTRA_SFT="--batch_size 16 --accumulation_steps 32" \
  bash scripts/train_yunmo.sh
# → out/pretrain_768.pth、out/full_sft_768.pth
```
