"""
雲端安全探針：在真實 195M 模型上比較 default vs 強制 flash 的峰值記憶體。
用 batch 2（~5GB），可與訓練 job 並存（不建優化器、不落盤）。
用法：在第二個終端 `python scripts/probe_flash.py`
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

torch.manual_seed(0)
dev = "cuda"
B, T = 2, 1024
VOCAB = 24000

cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=24,
                     num_attention_heads=8, num_key_value_heads=4,
                     vocab_size=VOCAB, max_position_embeddings=T, use_moe=False)
model = MiniMindForCausalLM(cfg).to(dev).train()
n = sum(p.numel() for p in model.parameters())
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
print(f"model {n/1e6:.1f}M | probe B={B} T={T} vocab={VOCAB}\n")

ids = torch.randint(0, VOCAB, (B, T), device=dev)
labels = torch.randint(0, VOCAB, (B, T), device=dev)
autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)

def measure(mode):
    for p in model.parameters():
        p.grad = None
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    err = ""
    try:
        with autocast:
            if mode == "default":
                out = model(ids, labels=labels)
            else:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    out = model(ids, labels=labels)
            loss = out.loss + out.aux_loss
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
    except Exception as e:
        peak = -1; err = f"{type(e).__name__}: {str(e)[:120]}"
    return peak, err

res = {}
for m in ["default", "flash"]:
    peak, err = measure(m)
    res[m] = peak
    print(f"  {m:8s}: {peak:.3f} GB" if peak >= 0 else f"  {m:8s}: FAIL  {err}")

print()
d, fl = res.get("default", -1), res.get("flash", -1)
if d > 0 and fl > 0:
    save = d - fl
    print(f"batch {B} 省下 {save:.3f} GB ({save/max(d,1e-9)*100:.0f}%)")
    # 線性外推到 batch 48
    per = save / B
    print(f"→ 外推 batch 48：約省 {per*48:.1f} GB")
    if save / max(d, 1e-9) > 0.15:
        print("→ flash 明顯省記憶體：scores 是元兇，值得改模型強制 flash")
    else:
        print("→ flash 省不多：記憶體瓶頸在別處（logits/CE），改 flash 幫助有限")
elif fl < 0:
    print("→ 強制 flash 在真實張量上失敗（見上方錯誤）：這條路走不通，維持現狀")
