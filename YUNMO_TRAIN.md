# Yunmo v1 訓練手冊（PRO 6000 · 36h · pretrain+SFT）

繁體台灣特化 · 基於 MiniMind v3 增量復現 · ~195M

## 鎖定設定

| | Pretrain | SFT |
|---|---|---|
| 資料 | yunmo_pretrain_packed.bin (~4.27B tok) | yunmo_sft_ids.bin + mask.bin (~5.85B tok) |
| 載入 | **packing**（零 pad） | **packing + loss-mask**（零 pad，長對話跨塊保留） |
| seq_len | 1024 | 2048 |
| batch / accum | 128 / 8 | 64 / 8 |
| 峰值 lr | 3e-4 | 5e-5 |
| epochs 上限 | 6（由 --max_minutes 時間封頂決定實際量） | 6 |
| 時間封頂 | ~900 min | ~1080 min |

模型：hidden 768 / **layers 24** / heads 8 / kv 4 / vocab 24000 / QK-Norm / tied-emb（~195M）
共通：AdamW · bf16 · grad_clip 1.0 · cos schedule(峰=lr/底=lr/10/無warmup) · from_resume 斷點續訓
分工：**pretrain+SFT 在 PRO 6000**；DPO/RL 之後本地 4070Ti。

---

## Step 1｜本地預打包（在有資料的機器）

把 `F:/AI/data/yunmo/{pretrain,sft}.jsonl` 打包成 token 二進位（零 pad）。用 repo model/ 的 tokenizer。

```bash
python scripts/pack_yunmo_data.py --stage both \
  --pretrain_in F:/AI/data/yunmo/pretrain.jsonl \
  --sft_in      F:/AI/data/yunmo/sft.jsonl \
  --out_dir     ./dataset --workers 16
```

產出 `dataset/`：`yunmo_pretrain_packed.bin`、`yunmo_sft_ids.bin`、`yunmo_sft_mask.bin`、`yunmo_pack_meta.json`。

## Step 2｜雲端環境（RTX PRO 6000 = Blackwell，需 CUDA 12.8+）

```bash
git clone https://github.com/yuhuanowo/minimind.git && cd minimind
# Blackwell(sm_120) 需 cu128 torch（cu126 會 no-kernel-image）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6" "trl==0.13.0" datasets accelerate tiktoken jinja2 \
            einops rich psutil ujson jsonlines nltk numpy
# 上傳 Step1 的 3 個 .bin 到 minimind/dataset/
python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

## Step 3｜一鍵訓練（Pretrain → SFT 全跑完）

```bash
export WANDB_API_KEY=你的key          # wandb 監控（在 wandb.ai 看即時 loss/lr）
bash scripts/train_yunmo.sh            # 自動：pretrain(~15h) → SFT(~18h)，時間封頂保證停
```
- 一支腳本跑完兩階段；`--from_resume 1` 斷點續訓保護（中斷可重跑同指令續訓）
- wandb project = `yunmo-v1`（pretrain/SFT 兩個 run）
- 時間分配可調：`PRETRAIN_MIN=840 SFT_MIN=1140 bash scripts/train_yunmo.sh`
- 加大 batch 吃滿 VRAM：`EXTRA_PRE="--batch_size 192" EXTRA_SFT="--batch_size 96" bash scripts/train_yunmo.sh`

**前 1hr benchmark**：看 wandb loss 曲線 + `nvidia-smi`。VRAM 有餘→加大 batch；loss 發散→降 lr；量測吞吐後調 `PRETRAIN_MIN/SFT_MIN` 分配，確保 36h 內做完。

**產出（本地 + wandb 雙保存）**：
- 本地：`out/pretrain_768.pth`、`out/full_sft_768.pth`
- **wandb artifact**：每階段結束（或時間封頂）自動上傳最終模型到 wandb（project `yunmo-v1`，artifact `pretrain_768` / `full_sft_768`）。**雲端機器會回收，靠這個把模型帶回來**。
- 事後下載：`wandb artifact get yunmo-v1/full_sft_768:latest` 或在 wandb.ai 網頁下載。

## 時間預算（36h）

| 階段 | 建議 --max_minutes | 時數 |
|---|---|---|
| Setup（環境+上傳bin） | — | ~2h |
| Pretrain | 900 | ~15h |
| SFT | 1080 | ~18h |
| Buffer | — | ~1h |

> 前 1hr benchmark 實際吞吐後，可調整兩者 max_minutes 分配（吞吐高→多給、確保 36h 內做完）。

## 之後（本地 4070Ti，不佔 36h）

DPO / RLAIF / GRPO：用 `out/full_sft_768.pth`，見 trainer/train_dpo.py 等。
