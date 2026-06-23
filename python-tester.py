#!/usr/bin/env python3
"""
Ollama + Kubernetes RAG Diagnostics
Pod: ollama-669d46f776-jllqm
"""

import subprocess
import json
import time
import sys
import urllib.request
import urllib.error
from datetime import datetime

POD = "ollama-777dd96f8b-7wx45"
KUBECONFIG = "/Users/adambenltaifa/.kube/ai-adam-config"
OLLAMA_PORT = 11434
MODEL = "qwen3:8b"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✘{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}→{RESET}  {msg}")
def section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")

# ── kubectl helpers ───────────────────────────────────────────────────────────
def kubectl(*args, stdin=None) -> tuple[str, str, int]:
    """Run a kubectl command, return (stdout, stderr, returncode)."""
    cmd = ["kubectl", "--kubeconfig", KUBECONFIG, *args]
    result = subprocess.run(cmd, capture_output=True, text=True, input=stdin)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def kubectl_exec(*args) -> tuple[str, str, int]:
    return kubectl("exec", POD, "--", *args)

def port_forward_request(path: str, data: dict | None = None, timeout: int = 60) -> dict | None:
    """
    Open a one-shot kubectl port-forward and hit the Ollama HTTP API.
    Returns parsed JSON or None on error.
    """
    import threading, socket, os, signal

    # pick a free local port
    with socket.socket() as s:
        s.bind(("", 0))
        local_port = s.getsockname()[1]

    pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", POD, f"{local_port}:{OLLAMA_PORT}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)   # let the tunnel stabilise

    url = f"http://localhost:{local_port}{path}"
    try:
        body = json.dumps(data).encode() if data else None
        req  = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        warn(f"HTTP error on {path}: {e}")
        return None
    finally:
        pf_proc.terminate()
        pf_proc.wait()


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 1 — kubectl connectivity
# ═════════════════════════════════════════════════════════════════════════════
def check_kubectl():
    section("1 · kubectl connectivity")
    out, err, rc = kubectl("get", "pod", POD, "--no-headers",
                           "-o", "custom-columns=STATUS:.status.phase,READY:.status.containerStatuses[0].ready")
    if rc != 0:
        fail(f"Cannot reach pod: {err}")
        sys.exit(1)
    ok(f"Pod reachable  →  {out}")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Resource limits (GPU request)
# ═════════════════════════════════════════════════════════════════════════════
def check_resources():
    section("2 · Pod resource limits / GPU request")
    out, err, rc = kubectl(
        "get", "pod", POD,
        "-o", "jsonpath={.spec.containers[0].resources}"
    )
    if rc != 0 or not out:
        warn("Could not read resource spec"); return

    try:
        res = json.loads(out)
    except json.JSONDecodeError:
        info(f"Raw output: {out}"); return

    limits   = res.get("limits",   {})
    requests = res.get("requests", {})

    gpu_limit   = limits.get("nvidia.com/gpu")
    gpu_request = requests.get("nvidia.com/gpu")
    cpu_limit   = limits.get("cpu", "not set")
    mem_limit   = limits.get("memory", "not set")

    if gpu_limit:
        ok(f"GPU limit set    → nvidia.com/gpu: {gpu_limit}")
    else:
        fail("No GPU limit set  → Ollama may fall back to CPU!")

    if gpu_request:
        ok(f"GPU request set  → nvidia.com/gpu: {gpu_request}")
    else:
        warn("No GPU request   → scheduler won't guarantee GPU placement")

    info(f"CPU limit: {cpu_limit}  |  Memory limit: {mem_limit}")
    print(f"\n  Full spec: {json.dumps(res, indent=4)}")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 3 — nvidia-smi inside the pod
# ═════════════════════════════════════════════════════════════════════════════
def check_nvidia_smi():
    section("3 · GPU status (nvidia-smi inside pod)")
    out, err, rc = kubectl_exec("nvidia-smi",
                                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits")
    if rc != 0:
        fail(f"nvidia-smi failed: {err}")
        warn("Ollama is very likely running on CPU only!")
        return

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, mem_used, mem_free, mem_total, util = parts
        ok(f"GPU: {name}")
        info(f"VRAM used/free/total: {mem_used} / {mem_free} / {mem_total} MiB")
        info(f"GPU utilisation: {util}%")

        mem_used_i = int(mem_used)
        mem_free_i = int(mem_free)
        # qwen3:8b needs ~5500 MiB
        if mem_free_i < 5500:
            fail(f"Only {mem_free_i} MiB VRAM free — qwen3:8b needs ~5500 MiB. "
                 "Model will offload layers to CPU (slow)!")
        elif mem_used_i > 5000:
            warn(f"{mem_used_i} MiB already in use — possible contention with another process")
        else:
            ok(f"{mem_free_i} MiB free — enough for qwen3:8b")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Ollama logs (GPU vs CPU offload)
# ═════════════════════════════════════════════════════════════════════════════
def check_ollama_logs():
    section("4 · Ollama logs — GPU/CPU offload detection")
    out, err, rc = kubectl("logs", POD, "--tail=200")
    if rc != 0:
        warn(f"Could not fetch logs: {err}"); return

    keywords = {
        "offload": RED,
        "cpu":     YELLOW,
        "cuda":    GREEN,
        "GPU":     GREEN,
        "layers":  CYAN,
        "error":   RED,
        "warning": YELLOW,
    }

    hits = []
    for line in out.splitlines():
        lower = line.lower()
        for kw in keywords:
            if kw in lower:
                hits.append((kw, line))
                break

    if not hits:
        info("No GPU/CPU keywords found in last 200 log lines")
    else:
        for kw, line in hits:
            colour = keywords[kw]
            print(f"  {colour}{line}{RESET}")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Ollama API reachable + model loaded
# ═════════════════════════════════════════════════════════════════════════════
def check_ollama_api():
    section("5 · Ollama API — model availability")
    info("Opening port-forward to Ollama …")
    data = port_forward_request("/api/tags")
    if data is None:
        fail("Could not reach Ollama API"); return

    models = [m["name"] for m in data.get("models", [])]
    if not models:
        warn("No models found in Ollama — is qwen3:8b pulled?")
        return

    ok(f"Models loaded: {', '.join(models)}")
    if MODEL in models or any(MODEL in m for m in models):
        ok(f"{MODEL} is present")
    else:
        fail(f"{MODEL} NOT found — run: kubectl exec {POD} -- ollama pull {MODEL}")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 6 — Timed inference with full Ollama metrics
# ═════════════════════════════════════════════════════════════════════════════
def check_inference_timing():
    section("6 · Timed inference — Ollama perf metrics")
    prompt = "Reply with exactly one sentence: what is vector search?"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": 512,
            "num_predict": 64,
        }
    }

    info("Sending test prompt via port-forward (this may take a while) …")
    t0 = time.perf_counter()
    data = port_forward_request("/api/generate", data=payload, timeout=180)
    wall = time.perf_counter() - t0

    if data is None:
        fail("Inference request failed or timed out"); return

    def ns(key):
        return data.get(key, 0) / 1e9

    load_s        = ns("load_duration")
    prompt_eval_s = ns("prompt_eval_duration")
    eval_s        = ns("eval_duration")
    eval_count    = data.get("eval_count", 0)
    tps           = eval_count / eval_s if eval_s > 0 else 0

    print(f"\n  {'Metric':<28} {'Value':>12}")
    print(f"  {'─'*42}")
    print(f"  {'Wall clock':.<28} {wall:>10.2f}s")
    print(f"  {'Model load (cold load)':.<28} {load_s:>10.2f}s")
    print(f"  {'Prompt eval':.<28} {prompt_eval_s:>10.2f}s")
    print(f"  {'Token generation':.<28} {eval_s:>10.2f}s")
    print(f"  {'Tokens generated':.<28} {eval_count:>12}")
    print(f"  {'Tokens / second':.<28} {tps:>10.1f}")

    # Diagnosis
    print()
    if load_s > 5:
        fail(f"load_duration={load_s:.1f}s — model is being evicted from VRAM between calls. "
             "Set OLLAMA_KEEP_ALIVE env var.")
    elif load_s > 1:
        warn(f"load_duration={load_s:.1f}s — slightly slow model load")
    else:
        ok(f"load_duration={load_s:.2f}s — model stays warm ✓")

    if tps < 5:
        fail(f"{tps:.1f} tok/s — extremely slow, likely CPU fallback or heavy GPU contention")
    elif tps < 15:
        warn(f"{tps:.1f} tok/s — below expected for qwen3:8b on GPU (~20-40 tok/s)")
    else:
        ok(f"{tps:.1f} tok/s — GPU inference looks healthy")

    if prompt_eval_s > 3:
        warn(f"prompt_eval_duration={prompt_eval_s:.1f}s — even a short prompt is slow. "
             "Suspect CPU prefill or large KV cache pressure.")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Prompt token counter (RAG context size)
# ═════════════════════════════════════════════════════════════════════════════
def check_token_count():
    section("7 · RAG prompt token count")
    sample = (
        "Context:\n"
        "Kubernetes is an open-source container orchestration platform. "
        "It automates the deployment, scaling, and management of containerised applications. "
        "Qdrant is a vector similarity search engine. "
        "It provides a production-ready service with a convenient API to store, search, "
        "and manage points (vectors with payloads).\n\n"
        "Question: How do I deploy Qdrant on Kubernetes?\n"
        "Answer:"
    )

    data = port_forward_request("/api/tokenize",
                                data={"model": MODEL, "content": sample})
    if data is None:
        warn("Could not tokenize sample prompt"); return

    count = len(data.get("tokens", []))
    ok(f"Sample RAG prompt → {count} tokens")
    if count > 2000:
        fail("Prompt is very large — trim your context chunks to top 3-5 by score threshold")
    elif count > 1000:
        warn("Prompt is moderately large — consider score_threshold=0.75 in Qdrant search")
    else:
        ok("Token count is fine for qwen3:8b context window")


# ═════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Node-level GPU sharing
# ═════════════════════════════════════════════════════════════════════════════
def check_node_gpu_sharing():
    section("8 · Node GPU allocation — sharing with other pods")
    node_out, _, rc = kubectl("get", "pod", POD,
                              "-o", "jsonpath={.spec.nodeName}")
    if rc != 0 or not node_out:
        warn("Could not determine node name"); return

    node = node_out.strip()
    info(f"Ollama is on node: {node}")

    # List all pods on same node requesting GPU
    out, _, rc = kubectl(
        "get", "pods", "--all-namespaces",
        "--field-selector", f"spec.nodeName={node}",
        "-o", "jsonpath={range .items[*]}{.metadata.namespace}/{.metadata.name}"
               " limits={.spec.containers[0].resources.limits}\\n{end}"
    )
    if rc != 0:
        warn("Could not list pods on node"); return

    gpu_pods = [line for line in out.splitlines() if "nvidia.com/gpu" in line]
    if not gpu_pods:
        ok("No other pods have GPU limits on this node")
    else:
        warn(f"{len(gpu_pods)} pod(s) with GPU limits on the same node:")
        for p in gpu_pods:
            print(f"    {YELLOW}{p}{RESET}")
        if len(gpu_pods) > 1:
            fail("GPU contention detected — multiple pods sharing the GPU. "
                 "This is your most likely bottleneck!")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def summary():
    section("Summary")
    print(f"""
  Paste the output of this script and share it for deeper analysis.

  Quick fixes to try right now:
  {GREEN}1.{RESET} Set OLLAMA_KEEP_ALIVE=10m in your Ollama deployment env
  {GREEN}2.{RESET} Ensure nvidia.com/gpu: "1" in pod limits AND requests
  {GREEN}3.{RESET} Check VRAM free — qwen3:8b needs ~5.5 GB
  {GREEN}4.{RESET} Limit Qdrant results: score_threshold=0.75, top_k=5
  {GREEN}5.{RESET} Reuse httpx.AsyncClient (connection pool) in your RAG agent
    """)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  Ollama RAG Diagnostics  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}  Pod: {POD}{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    check_kubectl()
    check_resources()
    check_nvidia_smi()
    check_ollama_logs()
    check_ollama_api()
    check_inference_timing()
    check_token_count()
    check_node_gpu_sharing()
    summary()