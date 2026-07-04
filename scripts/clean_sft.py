#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Yunmo SFT 污染清理（策略C：身份中性化 + 注入一致 Yunmo 身份）。
輸入 yunmo/sft.jsonl → 輸出 yunmo/sft_clean.jsonl（不覆蓋原檔）。

清理項目：
  1. 身份污染（僅 assistant/system 的「自稱」句改寫為 canonical Yunmo；討論/角色扮演/事實保留）：
     - minimind / jingyaogong 全域→Yunmo / YuhuanStudio（非真實世界主題，安全）
     - Qwen/通義千問/DeepSeek-R1/文心一言 等「自名」，或「我+阿里雲/通義實驗室/達摩院/
       深度求索/jingyaogong+研發/開發+模型/助手」自稱句 → 整句換成 Yunmo 身份句
     - 排除：如果我/我叫X/角色扮演/我認為·我使用/純討論（避免誤改事實內容與世界知識）
  2. tool_calls / tools 內殘留簡體 → s2twp 補轉
  3. 丟棄「無任何非空 assistant 回覆」的殘缺記錄
"""
import json, re, sys
from multiprocessing import Pool

SRC = "F:/AI/data/yunmo/sft.jsonl"
OUT = "F:/AI/data/yunmo/sft_clean.jsonl"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BATCH = 3000

YUNMO_I = "我是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。"
YUNMO_U = "你是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。"

NAME = r"(通[義义]千[問问]|Qwen|DeepSeek[-\s]?R1|DeepSeek|文心一言|ChatGPT|GPT-?[0-9](?:\.[0-9])?)"
COMP = r"(阿里[雲云]|阿里巴巴[雲云]?|通[義义]實驗室|達摩院|深度求索|jingyaogong|OpenAI)"
# 第一人稱「自稱」偵測（多語序）
_MODEL = r"(語言模型|大模型|AI模型|人工智慧助手|智慧助手|AI助手|助手|程式|自然語言處理模型|模型|人工智慧|AI)"
Q = r"[「『\"“”‘’'*_\s]{0,3}"  # 名字前的引號/markdown/空白容錯
_IDENT = (
    r"(我是|我叫|我就是|叫我|我的名字是|我的名字叫|我，|我（|我\()[\s*「『\"'（(【\[]{0,4}" + NAME
    + r"|" + NAME + r"(（我）|，即我|，我，)"
    + r"|我(是|，|，是|（|實際上是|其實是|本質是|本質上是|作為)[^。！？\n]{0,20}" + COMP + r"[^。！？\n]{0,18}(研發|開發|訓練|設計|維護|建立|創建|創造|打造|構建|研製|推出|問世)的?[^。！？\n]{0,12}(" + _MODEL + r"|" + NAME + r")"
    + r"|我(是|，|，是|（|實際上是|其實是|作為)[^。！？\n]{0,22}" + _MODEL + r"[^。！？\n]{0,18}(由|，由)[^。！？\n]{0,10}" + COMP + r"[^。！？\n]{0,18}(研發|開發|訓練|維護|建立|創建|創造|打造|構建|研製|進行訓練)"
    + r"|我(由|是由|實際上是由|其實是由)[^。！？\n]{0,18}" + COMP + r"[^。！？\n]{0,8}(開發|研發)"
    + r"|我是由[^。！？\n]{0,8}" + COMP + r"[^。！？\n]{0,4}(建立|創建|創造|打造|構建|研製)的?[。，、！？\n]"
    + r"|(而是|乃是)[^。！？\n]{0,12}(由)?[^。！？\n]{0,10}" + COMP + r"[^。！？\n]{0,12}(開發|研發)的?[^。！？\n]{0,10}(" + _MODEL + r"|" + NAME + r")"
    + r"|我是[^。！？\n]{0,6}" + COMP + r"[^。！？\n]{0,4}的[^。！？\n]{0,10}(語言模型|大型語言模型|大規模語言模型|模型|智慧助手|助手|AI|" + NAME + r")"
    + r"|我是基於" + Q + NAME
    + r"|(我的)?中文名(字)?[叫是為稱]{0,2}" + Q + NAME
    + r"|(我的)?英文名(字)?[叫是為稱]{0,2}" + Q + r"(Qwen|ChatGPT|GPT-?[0-9])"
    + r"|我是[^。！？\n]{0,4}版的?" + Q + NAME
    + r"|我代表的是阿里[雲云]"
)
IDENT = re.compile(_IDENT, re.I)
# 第二人稱「人設」偵測（system prompt：你是X）
YOUID = re.compile(r"(你是|您是|你叫|你的名字是|你的名字叫)\s*" + NAME
                   + r"|(你是|您是)[^。！？\n]{0,20}" + COMP + r"[^。！？\n]{0,12}(研發|開發)的?[^。！？\n]{0,10}(模型|助手|AI|語言模型)", re.I)
EXCLUDE = re.compile(r"如果我|假設我|假如我|我叫馬雲|扮演|網路名人|著名人物|我認為|我覺得|我使用|我用的|我是說|我想說|我要說|我的意思是|我聽說")
# 「作為由X開發的模型，我…」子句：只換身份子句、保留後續（拒答等）內容
ZUOWEI = re.compile(
    r"作為(一個|一款|一種|一名)?(由)?[^，。！？\n]{0,5}" + COMP + r"[^，。！？\n]{0,10}(開發|研發|訓練|打造|創建|建立|推出)的?[^，。！？\n]{0,8}(語言模型|大規模語言模型|大模型|模型|智慧助手|助手|人工智慧|AI)"
    + r"|作為[^，。！？\n]{0,4}" + COMP + r"的[^，。！？\n]{0,6}(語言模型|大模型|模型|智慧助手|助手|人工智慧|AI)", re.I)
ZUOWEI_REPL = "作為 Yunmo（由 YuhuanStudio 開發的繁體中文小型語言模型）"
re_mini = re.compile(r"mini\s*mind", re.I)
re_jing = re.compile(r"jingyaogong", re.I)
GREET = re.compile(r"\s*(您好[！!]|你好[！!，,]?|嗨[！!]|哈囉[！!]|大家好[！!，,]?)")
SENT = re.compile(r"[^。！？\n]*[。！？\n]|[^。！？\n]+")
HAN = re.compile(r"[一-鿿]")

_cc = None
def cc():
    global _cc
    if _cc is None:
        import opencc
        _cc = opencc.OpenCC('s2twp')
    return _cc

def clean_content(t, role):
    if not t:
        return t
    # 1) 自稱句改寫（僅 assistant/system；user 提問保留）
    if role in ("assistant", "system"):
        # 1a) 先中性化「作為由X開發的模型」子句（保留句子其餘內容，如拒答）
        t = ZUOWEI.sub(ZUOWEI_REPL, t)
        parts = []
        for s in SENT.findall(t):
            g = GREET.match(s)
            gr = g.group(1) if g else ""
            if "我" in s and IDENT.search(s) and not EXCLUDE.search(s):
                parts.append(gr + YUNMO_I)
            elif role == "system" and YOUID.search(s) and not EXCLUDE.search(s):
                parts.append(gr + YUNMO_U)
            else:
                parts.append(s)
        t = "".join(parts)
        # 折疊連續重複的 Yunmo 身份句（經典兩句自介改寫後會重複）
        t = re.sub(r"(我是 Yunmo，一個由 YuhuanStudio 開發的繁體中文小型語言模型。)(\s*\1)+", r"\1", t)
    # 2) 殘留品牌詞全域安全替換（非真實世界主題）
    t = re_mini.sub("Yunmo", t)
    t = re_jing.sub("YuhuanStudio", t)
    return t

def conv_tool(obj):
    if isinstance(obj, str):
        return cc().convert(obj) if HAN.search(obj) else obj
    if isinstance(obj, list):
        return [conv_tool(x) for x in obj]
    if isinstance(obj, dict):
        return {k: conv_tool(v) for k, v in obj.items()}
    return obj

def proc(line):
    line = line.strip()
    if not line: return ("skip", None)
    try: o = json.loads(line)
    except: return ("skip", None)
    conv = o.get("conversations")
    if not conv: return ("skip", None)
    if not any(m.get("role") == "assistant" and (m.get("content") or "").strip() for m in conv):
        return ("drop", None)
    for m in conv:
        role = m.get("role", "")
        c = m.get("content")
        if isinstance(c, str): m["content"] = clean_content(c, role)
        rc = m.get("reasoning_content")
        if isinstance(rc, str): m["reasoning_content"] = clean_content(rc, role)
        for key in ("tool_calls", "tools"):
            if m.get(key): m[key] = conv_tool(m[key])
    return ("keep", json.dumps(o, ensure_ascii=False))

def main():
    kept = dropped = skipped = 0
    with open(SRC, encoding="utf-8", errors="replace") as f, \
         open(OUT, "w", encoding="utf-8") as w, \
         Pool(WORKERS) as pool:
        for tag, res in pool.imap(proc, f, chunksize=BATCH):
            if tag == "keep": w.write(res + "\n"); kept += 1
            elif tag == "drop": dropped += 1
            else: skipped += 1
            n = kept + dropped + skipped
            if n % 500000 == 0: print(f"  {n:,} (keep {kept:,} / drop {dropped:,})", flush=True)
    print(f"完成：保留 {kept:,}，丟棄殘缺 {dropped:,}，跳過 {skipped:,} → {OUT}", flush=True)

if __name__ == "__main__":
    main()
