# Yunmo v1 訓練手冊（PRO 6000 · 36h · pretrain+SFT）

繁體台灣特化 · 基於 MiniMind v3 增量復現 · ~195M

## 鎖定設定

| | Pretrain | SFT |
|---|---|---|
| 資料 | yunmo_pretrain_packed.bin (2.353B tok / 229.83萬塊) | yunmo_sft_ids.bin + mask.bin (3.857B tok / 188.35萬塊) |
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

產出 `dataset/`：`yunmo_pretrain_packed.bin`、`yunmo_sft_ids.bin`、`yunmo_sft_mask.bin`、`yunmo_pack_meta.json`（共 ~16.3GB）。

> **注意（v1 定案流程）**：打包前先跑身份去污染 `python scripts/clean_sft.py` 與 `python scripts/clean_pretrain.py`（策略：多老師自稱→Yunmo、嚴格區分真實自稱與知識討論／角色扮演；行為驗證見 `YUNMO_EVAL.md` 二），再以 `--pretrain_in …/pretrain_clean.jsonl --sft_in …/sft_clean.jsonl` 打包。上方指令為原始流程，實際 v1 bin 由 `*_clean.jsonl` 產出。

### Step 1.5｜透過 HuggingFace 中轉 bin（大檔上傳最穩，斷點續傳）

本地上傳（Windows PowerShell）：
```powershell
pip install "huggingface_hub[hf_transfer]<1.0"   # 釘 <1.0：hub 1.x 會與 transformers 4.57 衝突
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"      # 加速大檔
hf auth login                              # 貼 write token（huggingface.co/settings/tokens）
hf repo create yunmo-v1-packed --repo-type dataset --private
cd F:\AI\minimind\dataset
hf upload yuhuanstudio/yunmo-v1-packed yunmo_pretrain_packed.bin --repo-type dataset
hf upload yuhuanstudio/yunmo-v1-packed yunmo_sft_ids.bin        --repo-type dataset
hf upload yuhuanstudio/yunmo-v1-packed yunmo_sft_mask.bin       --repo-type dataset
hf upload yuhuanstudio/yunmo-v1-packed yunmo_pack_meta.json     --repo-type dataset
```
> 中斷就重跑同一行 —— Xet 會跳過已傳分塊、只補剩下的。

## Step 2｜雲端環境（RTX PRO 6000 = Blackwell，需 CUDA 12.8+）

```bash
git clone https://github.com/yuhuanowo/minimind.git && cd minimind
# Blackwell(sm_120) 需 cu128 torch（cu126 會 no-kernel-image）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6" "trl==0.13.0" datasets accelerate tiktoken jinja2 \
            einops rich psutil ujson jsonlines nltk numpy wandb "huggingface_hub[hf_transfer]<1.0"
# ⚠ huggingface_hub 必須 <1.0，否則 transformers 4.57.6 import 會掛（ImportError）
# 從 HuggingFace 拉回 Step1.5 上傳的 bin 到 minimind/dataset/
export HF_HUB_ENABLE_HF_TRANSFER=1
hf auth login          # 貼 token（read 即可）
hf download yuhuanstudio/yunmo-v1-packed --repo-type dataset --local-dir dataset
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

### wandb 監控內容（機器回收後仍完整可查）

| 類別 | 內容 |
|---|---|
| 純量曲線 | loss / logits_loss / aux_loss / **grad_norm**（發散預警）/ learning_rate / **tok_per_sec**（吞吐）/ progress / tokens_seen / elapsed_min |
| 系統 | GPU 使用率 / 顯存 / 功耗 / 溫度（wandb 自動抓） |
| Config | 全部超參 + 模型參數量 + vocab + **git_sha**（可重現）+ 資料塊數/token |
| **中途模型** | 每 `CKPT_MIN` 分（預設 90）上傳一次 checkpoint artifact，只留最近 `CKPT_KEEP` 版（預設 2）→ **崩潰/回收也救得回** |
| **最終模型** | 階段結束自動上傳 `pretrain_768` / `full_sft_768` |

**防爆保護**：中途上傳用「**時間間隔**」（非步數）→ 上傳次數與儲存量可預測；`CKPT_KEEP` 硬限只留最近 N 版（~2×408MB）。prune 在 daemon 執行緒跑，**卡住也不影響訓練**。
- 想更省：`CKPT_MIN=180`（少一半上傳）；想全關中途上傳：`CKPT_MIN=0`（只留結束上傳）。

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

---

## 效能排查實錄與 v2 路線（2026-07-04 · PRO 6000 Blackwell 實測）

> 這章是血淚。下次重租機器**先讀結論**，別再走一遍這 3 小時彎路。

### 定案配置（28h）

| | Pretrain | SFT |
|---|---|---|
| batch / accum | **48 / 21**（effective 1008 seq ≈ 1M tok） | **16 / 32**（effective 512 seq ≈ 1M tok） |
| max_minutes | **860**（~1.65 epoch / ~3.9B tok，正 Chinchilla） | **700**（~1 epoch / ~3.9B tok） |

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tmux new -s yunmo          # 背景訓練，斷線不死；Ctrl-b d 脫離，tmux attach -t yunmo 重連
PRETRAIN_MIN=860 SFT_MIN=700 \
  EXTRA_PRE="--batch_size 48 --accumulation_steps 21" \
  EXTRA_SFT="--batch_size 16 --accumulation_steps 32" \
  bash scripts/train_yunmo.sh
```
- 吞吐 ~75k tok/s，1 epoch ≈ 520 min。loss 從 ~10.1（≈ln 24000）開始降為正常。

### OOM 排查結論（關鍵：別再誤判 flash）

- **症狀**：batch 128/64 都 OOM（~95GB / 89.5GB）。
- **誤判過程**：以為 SDPA 在 head_dim=96/Blackwell 上退回 math backend、materialize `[B,H,T,T]` scores 撐爆。→ 試全域 `enable_math_sdp(False)`（無效，見 trainer 內已加的設定）、升 torch 2.12（無效）。
- **探針證實**（`scripts/probe_flash.py`，batch 2 真模型 fwd+bwd）：
  `default 4.358GB` vs `強制 flash 4.387GB` → **0% 差異**。
  → **default 本就走高效 backend，scores 從未 materialize，flash 完全不是瓶頸。**（`with sdpa_kernel` 在真實模型上也不會炸，那個 cudnn `__exit__` 錯誤只是探針用 `__enter__()` 沒配對 `with` 在行程結束時的假警報。）
- **真兇**：每 sample **~1.4GB** = 逐層 activations + **vocab 24000 的 logits/CE**（大詞表 + 小 hidden 的固有成本）。
  驗算：batch 48 ≈ 固定 1.56 + 48×1.4 + Adam ≈ **~75GB** ✓ 對上實測。
- **batch sizing 公式**（實測 32→57GB、64→89.5GB）：`VRAM(GB) ≈ 24 + 1.0×batch`。想填 VRAM 用 batch 48（75GB 安全）；56=88GB 太緊**且無加速**。

### 為何「加大 batch / flash / 加大模型」都不該做

- **bandwidth-bound**（nvitop：GPU-util 100% / MBW 84% / Tensor Core 低）：瓶頸是大詞表 logits 讀寫 + 逐元素運算（RMSNorm 的 `x.float()` 升 fp32、殘差、SwiGLU），**非算力、非 attention**。
- **加大 micro-batch**：effective batch 用 accum 鎖住 → 不影響收斂；bandwidth-bound → 不影響吞吐 → 純虧記憶體 + OOM 風險。
- **flash**：已在用，無空間。
- **加大模型**：只有 2.35B pretrain token，195M 已是 Chinchilla 匹配；500M 需 ~10B token → 現在加大只會訓練不足。**資料才是綁定瓶頸。**

### 1.65 epoch 是物理極限，不是沒調好

MiniMind 用 6 epoch 是因為它資料小（重複刷）+ 更小模型 + Linux flash 齊全。Yunmo 更大 + bandwidth-bound → 每 epoch 慢，28h 給 1.65 epoch。對 195M 而言 3.9B token 是 Chinchilla 甜蜜點，**不是將就**。

### v2 真正的效能槓桿（照優先級）

1. **fused / chunked cross-entropy（Liger-Kernel 式）** ← 對症「大詞表 logits」，同時省記憶體+頻寬，是這種「大詞表小模型」的正解。預期 10–30% 吞吐（只治 logits 那段，非 2×）。**必須先驗數值正確性再上**（錯的 CE 會白訓一整輪）。
2. **torch.compile** 融合逐元素運算、降記憶體流量（低-中風險）。
3. **更多/更好資料** → 撐得起更大模型後才談加大規模。

> **v2 第一刀是 fused CE，不是 flash、不是加大 batch、不是加大模型。**
