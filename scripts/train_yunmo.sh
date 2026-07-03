#!/usr/bin/env bash
# Yunmo v1 一鍵訓練：Pretrain → SFT（PRO 6000 · 時間封頂 · wandb）
# 用法：
#   export WANDB_API_KEY=你的key        # wandb 監控（強烈建議）
#   bash scripts/train_yunmo.sh          # 用預設時間預算
#   PRETRAIN_MIN=840 SFT_MIN=1140 bash scripts/train_yunmo.sh   # 自訂分配（benchmark 後）
set -e

PRETRAIN_MIN=${PRETRAIN_MIN:-900}          # pretrain 時間上限（分）
SFT_MIN=${SFT_MIN:-1080}                    # SFT 時間上限（分）
WANDB_PROJECT=${WANDB_PROJECT:-yunmo-v1}
EXTRA_PRE=${EXTRA_PRE:-}                    # 額外參數，如 EXTRA_PRE="--batch_size 192"
EXTRA_SFT=${EXTRA_SFT:-}

cd "$(dirname "$0")/../trainer"

# --- 前置檢查 ---
if [ -z "$WANDB_API_KEY" ]; then
  echo "⚠  未設 WANDB_API_KEY，wandb 會要求互動登入或轉離線。建議先 export WANDB_API_KEY=xxxx"
fi
for f in ../dataset/yunmo_pretrain_packed.bin ../dataset/yunmo_sft_ids.bin ../dataset/yunmo_sft_mask.bin; do
  [ -f "$f" ] || { echo "✗ 缺 $f —— 請先本地跑 pack_yunmo_data.py 並上傳 bin"; exit 1; }
done
python -c "import torch;assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0),'| bf16:',torch.cuda.is_bf16_supported())"

echo ""
echo "========== [1/2] PRETRAIN  (上限 ${PRETRAIN_MIN} 分) =========="
python train_pretrain.py \
  --max_minutes "$PRETRAIN_MIN" --from_resume 1 \
  --use_wandb --wandb_project "$WANDB_PROJECT" $EXTRA_PRE

echo ""
echo "========== [2/2] SFT  (上限 ${SFT_MIN} 分, 接 pretrain 權重) =========="
python train_full_sft.py \
  --max_minutes "$SFT_MIN" --from_weight pretrain --from_resume 1 \
  --use_wandb --wandb_project "$WANDB_PROJECT" $EXTRA_SFT

echo ""
echo "========== ✅ 完成 =========="
echo "  pretrain → out/pretrain_768.pth"
echo "  SFT      → out/full_sft_768.pth"
echo "  之後 DPO/RL 在本地 4070Ti 用 full_sft_768.pth"
