#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Yunmo pretrain 身份污染清理（與 SFT 策略C 一致，複用同一身份規則）。
輸入 yunmo/pretrain.jsonl {"text":...} → 輸出 yunmo/pretrain_clean.jsonl（不覆蓋原檔）。

pretrain 為無監督扁平文本，已是繁體（0 殘簡）；此處僅做身份自稱句改寫
（「我是 ChatGPT/文心一言/…由 OpenAI 訓練的模型」→ canonical Yunmo），
討論/事實/角色扮演一律保留（沿用 clean_sft 的 IDENT/EXCLUDE/ZUOWEI 規則）。
"""
import json, sys
from multiprocessing import Pool

sys.path.insert(0, "script")
from clean_sft import clean_content  # 複用身份改寫（含 ZUOWEI/dedup/global）

SRC = "F:/AI/data/yunmo/pretrain.jsonl"
OUT = "F:/AI/data/yunmo/pretrain_clean.jsonl"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
BATCH = 3000

def proc(line):
    line = line.strip()
    if not line:
        return ("skip", None)
    try:
        o = json.loads(line)
    except Exception:
        return ("skip", None)
    t = o.get("text")
    if isinstance(t, str) and t:
        # 以 assistant 模式套用身份改寫（pretrain 文本內含模型自稱）
        o["text"] = clean_content(t, "assistant")
    return ("keep", json.dumps(o, ensure_ascii=False))

def main():
    kept = skipped = 0
    with open(SRC, encoding="utf-8", errors="replace") as f, \
         open(OUT, "w", encoding="utf-8") as w, \
         Pool(WORKERS) as pool:
        for tag, res in pool.imap(proc, f, chunksize=BATCH):
            if tag == "keep":
                w.write(res + "\n"); kept += 1
            else:
                skipped += 1
            n = kept + skipped
            if n % 1000000 == 0:
                print(f"  {n:,} (keep {kept:,})", flush=True)
    print(f"完成：保留 {kept:,}，跳過 {skipped:,} → {OUT}", flush=True)

if __name__ == "__main__":
    main()
