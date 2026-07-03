#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地預打包 Yunmo 訓練資料 → token 二進位（零 pad packing）。

用我們 repo model/ 的 tokenizer（vocab 24000）：
  pretrain：每篇 [bos]+tokenize(text)+[eos] 拼成連續 uint16 流 → pretrain_packed.bin
  sft     ：每對話 chat_template→tokenize→ids + assistant loss-mask，拼成連續流
            → sft_ids.bin(uint16) + sft_mask.bin(uint8)

訓練時 PackedPretrainDataset / PackedSFTDataset memmap 讀、切塊，長對話自然跨塊保留（不丟不截）。
多進程加速；本地跑一次，上傳 bin 到雲端。

用法：
  python scripts/pack_yunmo_data.py --stage both \
      --pretrain_in /path/pretrain.jsonl --sft_in /path/sft.jsonl \
      --out_dir ./dataset --workers 16
"""
import os, sys, json, argparse
import numpy as np
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TOK_PATH = os.path.join(REPO, "model")
BATCH = 1000

_tok = _bos = _eos = _abos = _aeos = None

def _init():
    global _tok, _bos, _eos, _abos, _aeos
    from transformers import AutoTokenizer
    from dataset.lm_dataset import pre_processing_chat, post_processing_chat  # noqa: 觸發可 import
    _tok = AutoTokenizer.from_pretrained(TOK_PATH)
    _bos, _eos = _tok.bos_token_id, _tok.eos_token_id
    _abos = _tok(f'{_tok.bos_token}assistant\n', add_special_tokens=False).input_ids
    _aeos = _tok(f'{_tok.eos_token}\n', add_special_tokens=False).input_ids

# ---------------- pretrain ----------------
def pack_pretrain_batch(texts):
    out = []
    for t in texts:
        if not t:
            continue
        ids = _tok(str(t), add_special_tokens=False).input_ids
        out.append(_bos); out.extend(ids); out.append(_eos)
    return np.array(out, dtype=np.uint16).tobytes()

# ---------------- sft ----------------
def _chat_prompt(conv):
    messages, tools = [], None
    for m in conv:
        m = dict(m)
        if m.get("role") == "system" and m.get("tools"):
            tools = json.loads(m["tools"]) if isinstance(m["tools"], str) else m["tools"]
        if m.get("tool_calls") and isinstance(m["tool_calls"], str):
            m["tool_calls"] = json.loads(m["tool_calls"])
        messages.append(m)
    return _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

def _mask(ids):
    mask = [0] * len(ids); nb, ne = len(_abos), len(_aeos); i = 0
    while i < len(ids):
        if ids[i:i + nb] == _abos:
            start = i + nb; end = start
            while end < len(ids):
                if ids[end:end + ne] == _aeos:
                    break
                end += 1
            for j in range(start, min(end + ne, len(ids))):
                mask[j] = 1
            i = end + ne if end < len(ids) else len(ids)
        else:
            i += 1
    return mask

def pack_sft_batch(convs):
    from dataset.lm_dataset import pre_processing_chat, post_processing_chat
    ids_out, mask_out = [], []
    for conv in convs:
        try:
            conv = pre_processing_chat(conv)
            prompt = post_processing_chat(_chat_prompt(conv))
            ids = _tok(prompt).input_ids
            m = _mask(ids)
        except Exception:
            continue
        ids_out.extend(ids); mask_out.extend(m)
    return (np.array(ids_out, dtype=np.uint16).tobytes(),
            np.array(mask_out, dtype=np.uint8).tobytes())

# ---------------- streaming ----------------
def stream(path, key):
    buf = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            v = o.get(key)
            if v:
                buf.append(v)
                if len(buf) >= BATCH:
                    yield buf; buf = []
    if buf:
        yield buf

def run_pretrain(inp, out_dir, workers, seq_len):
    out = os.path.join(out_dir, "yunmo_pretrain_packed.bin")
    total = 0
    with open(out, "wb") as w, Pool(workers, initializer=_init) as p:
        for i, b in enumerate(p.imap(pack_pretrain_batch, stream(inp, "text"), chunksize=1)):
            w.write(b); total += len(b) // 2
            if i % 200 == 0:
                print(f"  pretrain: {total/1e6:.0f}M tok", flush=True)
    print(f"[pretrain] {total:,} tok → {total//seq_len:,} 塊(seq={seq_len}) → {out}", flush=True)
    return {"file": os.path.basename(out), "tokens": total, "seq_len": seq_len, "blocks": total // seq_len}

def run_sft(inp, out_dir, workers, seq_len):
    out_i = os.path.join(out_dir, "yunmo_sft_ids.bin")
    out_m = os.path.join(out_dir, "yunmo_sft_mask.bin")
    total = 0
    with open(out_i, "wb") as wi, open(out_m, "wb") as wm, Pool(workers, initializer=_init) as p:
        for i, (ib, mb) in enumerate(p.imap(pack_sft_batch, stream(inp, "conversations"), chunksize=1)):
            wi.write(ib); wm.write(mb); total += len(ib) // 2
            if i % 100 == 0:
                print(f"  sft: {total/1e6:.0f}M tok", flush=True)
    print(f"[sft] {total:,} tok → {total//seq_len:,} 塊(seq={seq_len}) → {out_i} + {out_m}", flush=True)
    return {"ids_file": os.path.basename(out_i), "mask_file": os.path.basename(out_m),
            "tokens": total, "seq_len": seq_len, "blocks": total // seq_len}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "sft", "both"], default="both")
    ap.add_argument("--pretrain_in", default="F:/AI/data/yunmo/pretrain.jsonl")
    ap.add_argument("--sft_in", default="F:/AI/data/yunmo/sft.jsonl")
    ap.add_argument("--out_dir", default=os.path.join(REPO, "dataset"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pretrain_seq", type=int, default=1024)
    ap.add_argument("--sft_seq", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    meta = {}
    if a.stage in ("pretrain", "both"):
        meta["pretrain"] = run_pretrain(a.pretrain_in, a.out_dir, a.workers, a.pretrain_seq)
    if a.stage in ("sft", "both"):
        meta["sft"] = run_sft(a.sft_in, a.out_dir, a.workers, a.sft_seq)
    with open(os.path.join(a.out_dir, "yunmo_pack_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("完成 →", json.dumps(meta, ensure_ascii=False), flush=True)
