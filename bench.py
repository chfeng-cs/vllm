#!/usr/bin/env python3
"""
Fixed-interval benchmark with unique KV per request.
Compare with/without --eager-prefetch in start_vllm.sh.
"""

import http.client
import json
import os
import re
import sys
import time
import threading

HOST           = "localhost"
PORT           = 8000
REPEATS        = 5
FIRST_INTERVAL = 2.4   # gap before req 2 (seconds)
INTERVAL       = 2.4   # gap between warm requests (seconds)
MAX_TOKENS     = 100
SUFFIX         = "What is 1+1? Answer briefly."
VLLM_LOG       = "/tmp/vllm.log"
LMCACHE_LOG    = "/tmp/lmcache.log"
TIMELINE_OUT   = "/tmp/bench_timeline.json"

SHARED_BODY = "You are a helpful AI assistant. " * 1250  # ~10k tokens


def make_prompts(n: int) -> list[str]:
    return [f"[Request-{i:04d}] " + SHARED_BODY + SUFFIX for i in range(n)]


def get_model_name(host, port):
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/v1/models")
        return json.loads(conn.getresponse().read())["data"][0]["id"]
    except Exception:
        return "unknown"


def wait_for_server(host, port, timeout=120):
    print(f"Waiting for vLLM at {host}:{port} ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            conn.getresponse()
            print(" ready!")
            return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    print(" timeout!")
    return False


def send_streaming_request(host, port, model, prompt, max_tokens):
    payload = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    t_start = time.perf_counter()
    t_first = None
    prompt_tokens = completion_tokens = 0
    completion_id = ""
    buf = b""

    try:
        conn = http.client.HTTPConnection(host, port, timeout=300)
        conn.request("POST", "/v1/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return {"ok": False, "error": f"HTTP {resp.status}: {resp.read()[:200]}"}

        while True:
            chunk = resp.read(512)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if not completion_id:
                    completion_id = event.get("id", "")
                choices = event.get("choices", [])
                if choices and t_first is None and choices[0].get("text", ""):
                    t_first = time.perf_counter()
                if usage := event.get("usage"):
                    prompt_tokens    = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        t_end = time.perf_counter()
        return {
            "ok": True,
            "ttft":           (t_first - t_start) if t_first else (t_end - t_start),
            "decode_time":    (t_end - t_first)   if t_first else 0,
            "total_time":     t_end - t_start,
            "prompt_tokens":  prompt_tokens,
            "completion_tokens": completion_tokens,
            "t_submitted":    t_start,
            "completion_id":  completion_id,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_kv_timing_stats(vllm_log, offset: int = 0) -> list[dict]:
    stats = []
    try:
        with open(vllm_log) as f:
            f.seek(offset)
            for line in f:
                if "[KV_TIMING]" not in line:
                    continue
                m_req = re.search(r"req=(\S+)", line)
                m_enq = re.search(r"(?:enqueue|on_new_request)→kv_ready: ([\d.]+) ms", line)
                if m_req:
                    stats.append({
                        "req_id": m_req.group(1),
                        "enqueue_to_ready_ms": float(m_enq.group(1)) if m_enq else None,
                    })
    except FileNotFoundError:
        pass
    return stats


def fetch_prefetch_ms(lmcache_log, offset: int = 0) -> dict[str, float]:
    result = {}
    try:
        with open(lmcache_log) as f:
            f.seek(offset)
            for line in f:
                if "Prefetch request completed" not in line:
                    continue
                m_ms  = re.search(r"in ([\d.]+) ms", line)
                m_req = re.search(r"external_request_id=(\S+?)[,)]", line)
                if m_ms and m_req:
                    result[m_req.group(1)[:8]] = float(m_ms.group(1))
    except FileNotFoundError:
        pass
    return result


def main():
    if not wait_for_server(HOST, PORT):
        sys.exit(1)

    model   = get_model_name(HOST, PORT)
    prompts = make_prompts(REPEATS)
    labels  = [f"req-{i:04d}" for i in range(REPEATS)]
    submit_times = [0.0] + [FIRST_INTERVAL + (i - 1) * INTERVAL for i in range(1, REPEATS)]

    print(f"\nModel: {model}  requests={REPEATS}  max_tokens={MAX_TOKENS}\n")

    vllm_log_offset    = os.path.getsize(VLLM_LOG)    if os.path.exists(VLLM_LOG)    else 0
    lmcache_log_offset = os.path.getsize(LMCACHE_LOG) if os.path.exists(LMCACHE_LOG) else 0

    results = [None] * REPEATS
    t_wall_start = time.perf_counter()

    def run(idx):
        delay = t_wall_start + submit_times[idx] - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        print(f"  [t={time.perf_counter()-t_wall_start:5.1f}s] Submitting Req {idx+1}: {labels[idx]}")
        results[idx] = send_streaming_request(HOST, PORT, model, prompts[idx], MAX_TOKENS)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(REPEATS)]
    for t in threads: t.start()
    for t in threads: t.join()

    t_wall_total = time.perf_counter() - t_wall_start

    kv_stats    = fetch_kv_timing_stats(VLLM_LOG, vllm_log_offset)
    disk_by_req = fetch_prefetch_ms(LMCACHE_LOG, lmcache_log_offset)

    kv_by_id: dict[str, dict] = {}
    for s in kv_stats:
        enq  = s["enqueue_to_ready_ms"]
        disk = disk_by_req.get(s["req_id"])
        kv_by_id[s["req_id"]] = {
            "enq":   enq,
            "disk":  disk,
            "qwait": (enq - disk) if (enq is not None and disk is not None) else None,
        }

    _none = {"enq": None, "disk": None, "qwait": None}
    has_kv = bool(kv_by_id)
    sep = "=" * (86 + (34 if has_kv else 0))

    print("\n" + sep)
    hdr = (f"  {'#':>2}  {'T_sub':>6}  {'TTFT':>8}  {'Decode':>8}  "
           f"{'Total':>8}  {'PromptTok':>9}  {'OutputTok':>9}  {'Label':<10}")
    if has_kv:
        hdr += f"  {'enq→ready':>10}  {'disk_ms':>7}  {'QueueWait':>9}"
    print(hdr)
    print(sep)

    all_kv: list[dict] = []
    for i, (r, label) in enumerate(zip(results, labels)):
        if not r["ok"]:
            print(f"  {i+1:>2}  ERROR: {r['error']}")
            all_kv.append(_none)
            continue
        kv = kv_by_id.get(r["completion_id"][:8], _none)
        all_kv.append(kv)
        t_sub = r["t_submitted"] - t_wall_start
        row = (f"  {i+1:>2}  {t_sub:>5.1f}s  {r['ttft']:>7.3f}s  "
               f"{r['decode_time']:>7.3f}s  {r['total_time']:>7.3f}s  "
               f"{r['prompt_tokens']:>9}  {r['completion_tokens']:>9}  {label:<10}")
        if has_kv:
            fmt = lambda v, u: f"{v:.0f}{u}" if v is not None else "N/A"
            row += f"  {fmt(kv['enq'],'ms'):>10}  {fmt(kv['disk'],'ms'):>7}  {fmt(kv['qwait'],'ms'):>9}"
        print(row)

    valid = [r for r in results if r["ok"]]
    ttfts = [r["ttft"] for r in valid]
    print(f"\nSummary  (wall={t_wall_total:.1f}s)")
    if ttfts:
        print(f"  TTFT:       avg={sum(ttfts)/len(ttfts):.3f}s  min={min(ttfts):.3f}s  max={max(ttfts):.3f}s")
    if has_kv:
        avg = lambda vals: f"{sum(vals)/len(vals):.0f} ms" if vals else "N/A"
        print(f"  enq→ready:  avg={avg([kv['enq']   for kv in all_kv if kv['enq']   is not None])}")
        print(f"  disk_ms:    avg={avg([kv['disk']  for kv in all_kv if kv['disk']  is not None])}")
        print(f"  QueueWait:  avg={avg([kv['qwait'] for kv in all_kv if kv['qwait'] is not None])}")

    # Export compact timeline (JSON Lines, one record per line, append mode).
    # All t_* fields are seconds relative to t_wall_start; *_ms fields are milliseconds.
    # Load with: import json; rows = [json.loads(l) for l in open("/tmp/bench_timeline.json")]
    timeline = []
    for i, (r, label, kv) in enumerate(zip(results, labels, all_kv)):
        if not r["ok"]:
            continue
        t_sub   = round(r["t_submitted"] - t_wall_start, 4)
        t_first = round(t_sub + r["ttft"], 4)
        t_end   = round(t_sub + r["total_time"], 4)
        timeline.append({
            "idx":   i,
            "label": label,
            # phase anchors (seconds from wall start, usable as x-axis directly)
            "ttft_start":   t_sub,
            "ttft_end":     t_first,
            "ttft_s":       round(r["ttft"], 4),
            "decode_start": t_first,
            "decode_end":   t_end,
            "decode_s":     round(r["decode_time"], 4),
            "total_start":  t_sub,
            "total_end":    t_end,
            "total_s":      round(r["total_time"], 4),
            # token counts
            "prompt_tokens":     r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            # KV timing (milliseconds, null if unavailable)
            "enq_ms":    round(kv["enq"],   1) if kv["enq"]   is not None else None,
            "disk_ms":   round(kv["disk"],  1) if kv["disk"]  is not None else None,
            "qwait_ms":  round(kv["qwait"], 1) if kv["qwait"] is not None else None,
        })
    with open(TIMELINE_OUT, "a") as f:
        f.write(json.dumps(timeline, separators=(",", ":")) + "\n")
    print(f"\nTimeline data → {TIMELINE_OUT}  (+1 run, {len(timeline)} requests)")


if __name__ == "__main__":
    main()
