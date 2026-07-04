import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PackedPretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def prune_ckpt_artifacts(wandb, art_name, keep=2):
    """只保留最近 keep 個中途 checkpoint 版本 → wandb 儲存有硬上限。
    丟到 daemon 執行緒執行：即使 wandb.Api() 連線卡住（如 offline）也絕不阻塞訓練。"""
    import threading

    def _job():
        try:
            api = wandb.Api()
            full = f"{wandb.run.entity}/{wandb.run.project}/{art_name}"
            vers = list(api.artifacts("model-checkpoint", full))
            vers.sort(key=lambda a: int(''.join(filter(str.isdigit, a.version)) or 0))
            for a in (vers[:-keep] if keep > 0 else []):
                try:
                    a.delete(delete_aliases=True)
                except Exception:
                    pass
        except Exception:
            pass
    threading.Thread(target=_job, daemon=True).start()


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    global LAST_CKPT_UPLOAD
    start_time = time.time()
    last_step = start_step
    grad_norm = 0.0
    tokens_seen = 0
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        tokens_seen += input_ids.numel()
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                gn = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                tok_s = tokens_seen / max(spend_time, 1e-9)
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss,
                           "learning_rate": current_lr, "grad_norm": gn, "tok_per_sec": tok_s,
                           "tokens_seen": tokens_seen, "progress": step / max(iters, 1),
                           "elapsed_min": (time.time() - TRAIN_START) / 60, "epoch": epoch + 1, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # ===== Yunmo: 定期上傳中途 checkpoint（時間間隔→儲存可預測 + 只留最近N版→硬上限）=====
        if (wandb is not None and args.wandb_ckpt_minutes > 0 and is_main_process()
                and (time.time() - LAST_CKPT_UPLOAD) / 60 >= args.wandb_ckpt_minutes):
            LAST_CKPT_UPLOAD = time.time()
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
            art = wandb.Artifact(f'{args.save_weight}_{lm_config.hidden_size}_ckpt', type='model-checkpoint',
                                 metadata={'epoch': epoch + 1, 'step': step})
            art.add_file(ckp)
            wandb.log_artifact(art)
            model.train()
            Logger(f'☁ 中途 checkpoint 已上傳 wandb (epoch {epoch + 1}, step {step})')
            prune_ckpt_artifacts(wandb, f'{args.save_weight}_{lm_config.hidden_size}_ckpt', keep=args.wandb_ckpt_keep)

        # ===== Yunmo 時間封頂：到時存檔並停止（保證 36h 內做完）=====
        if args.max_minutes > 0 and (time.time() - TRAIN_START) / 60 >= args.max_minutes:
            if is_main_process():
                moe_suffix = '_moe' if lm_config.use_moe else ''
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)
                sd = raw_model.state_dict()
                torch.save({k: v.half().cpu() for k, v in sd.items()}, ckp)
                lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
                del sd
                Logger(f'⏰ 達時間上限 {args.max_minutes} 分，已存檔停止 (epoch {epoch + 1}, step {step})')
            del input_ids, labels, res, loss
            return True

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数上限（實際由 --max_minutes 時間封頂決定）")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size（Yunmo/PRO6000 96GB；VRAM 有餘可上调）")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="初始学习率（Yunmo 峰值，前1hr看loss微调）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=24, type=int, help="隐藏层数量（Yunmo 深而窄）")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="packing 塊長度（須≤打包時 --pretrain_seq）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/yunmo_pretrain_packed.bin", help="预训练 packed token 二进位（pack_yunmo_data.py 产生）")
    parser.add_argument("--max_minutes", type=int, default=0, help="時間封頂(分鐘)，到時存檔停止；0=不限。雲端建議 pretrain ~930")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--wandb_ckpt_minutes", type=int, default=0, help="每N分鐘上傳中途checkpoint到wandb（0=僅結束上傳；防機器回收，時間制→儲存可預測）")
    parser.add_argument("--wandb_ckpt_keep", type=int, default=2, help="wandb 只保留最近N個中途checkpoint版本（硬上限，防儲存爆）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    # 强制高效 SDPA 后端：关掉 math backend，避免其 materialize O(T^2) 的 attention scores 撑爆 VRAM。
    # 背景：head_dim=96 + 部分新卡上，SDPA 预设 dispatcher 会退回 math backend（batch 32 就吃 57GB）；
    # flash / mem-efficient 实测可用且是 O(T) 显存，禁用 math 后即强制走高效路径。cudnn backend 在 torch 2.12 有小 bug，一并关闭。
    if device_type == "cuda":
        for _fn, _on in [("enable_flash_sdp", True), ("enable_mem_efficient_sdp", True),
                         ("enable_math_sdp", False), ("enable_cudnn_sdp", False)]:
            if hasattr(torch.backends.cuda, _fn):
                getattr(torch.backends.cuda, _fn)(_on)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PackedPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # ===== Yunmo: 記錄完整超參/模型/資料/git 到 wandb config（機器回收後仍可查、可重現）=====
    if args.use_wandb and is_main_process() and wandb is not None:
        try:
            import subprocess
            _git = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                           cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
        except Exception:
            _git = 'unknown'
        _np = sum(p.numel() for p in model.parameters())
        wandb.config.update({**vars(args), 'git_sha': _git, 'vocab_size': lm_config.vocab_size,
                             'intermediate_size': lm_config.intermediate_size, 'num_params_M': round(_np / 1e6, 2),
                             'num_blocks': len(train_ds), 'approx_tokens': len(train_ds) * args.max_seq_len,
                             'stage': 'pretrain'}, allow_val_change=True)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    TRAIN_START = time.time()   # Yunmo 時間封頂基準
    LAST_CKPT_UPLOAD = time.time()   # Yunmo 中途 checkpoint 上傳計時基準
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            stop = train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            stop = train_epoch(epoch, loader, len(loader), 0, wandb)
        if stop:   # 時間封頂觸發，跳出
            break

    # ========== 8.5 上傳最終模型到 wandb artifact（雲端機器會回收，務必保存）==========
    if args.use_wandb and is_main_process() and wandb is not None:
        moe_suffix = '_moe' if lm_config.use_moe else ''
        final_ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(final_ckp):
            art = wandb.Artifact(f'{args.save_weight}_{lm_config.hidden_size}', type='model',
                                 metadata={'layers': lm_config.num_hidden_layers, 'hidden': lm_config.hidden_size, 'vocab': lm_config.vocab_size})
            art.add_file(final_ckp)
            wandb.log_artifact(art)
            Logger(f'✅ 已上傳最終模型到 wandb artifact: {final_ckp}')
        wandb.finish()

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()