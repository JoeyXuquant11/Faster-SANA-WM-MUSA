"""SANA-WM Stage-1 bidirectional FSDP2 training bench on 4x MTT S5000.

Run from the workspace root with Python. GPU runs acquire /tmp/sana_wm_gpu.lock themselves, one at a time;
do NOT wrap bench.py commands in flock (that would deadlock).

  python3 bench/bench.py check   --tree <dir> --out <dir> [--config <file>] [--quick | --skip-timing | --n1]
        full verdict (gates 0-3) -> <out>/check.json
        --quick: gates 0 and 2 (exact + bitwise) only, no timing, no trajectories
        --skip-timing: gates 0, 2, 3 (no gate 1)
        --n1: gates 0 and 3 (statistical trajectory only, 1 seed against frozen 3-sigma bound)
  python3 bench/bench.py time    --tree <dir> --out <dir> [--config <file>]
        one timing run of the tree (2 windows of 5 iterations after 3 warmup iterations) -> <out>/time.json
  python3 bench/bench.py profile --tree <dir> --out <dir> [--config <file>] [--iters N]
        torch.profiler (CPU + PrivateUse1) over N iterations (default 2) after warmup ->
        <out>/profile.json, <out>/profile.md, <out>/timing_profile.trace.json,
        <out>/kernel_util.md, <out>/kernel_util.json
  python3 bench/bench.py baseline --out <dir>          (formulator) reference noise floor check
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent


def _expand(value):
    if isinstance(value, str) and "$" in value:
        out = os.path.expandvars(value)
        if "$" in out:
            raise SystemExit(f"{value}: set ASSETS (and any other referenced env var) before running the bench")
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


CFG = _expand(json.loads((BENCH / "bench.json").read_text()))
sys.path.insert(0, str(BENCH))
import recipe  # noqa: E402

LOCK = Path(CFG["gpu_lock"])
PY = Path(CFG["python"])
TORCHRUN = Path(CFG.get("torchrun") or (PY.parent / "torchrun"))
PYUSER_SITE = CFG.get("pyuser_site", "")
TIERS = ["N0", "N1", "N2", "N3", "N4", "N5"]
SCRATCH = Path(CFG.get("scratch", "/tmp"))
TIMEOUT = CFG["timeouts_s"]
RESERVED = set(CFG["reserved_env"])
SIGNATURES = [(re.compile(x["regex"]), x["reason"]) for x in CFG.get("invalid_run_signatures", [])]
COST = {"gpu_seconds": 0.0, "gpu_runs": 0}
TREE_SKIP = {".git", "__pycache__", "wandb", "outputs", "output", "figures"}


def path(key):
    p = Path(_expand(key))
    return p if p.is_absolute() else (ROOT / p).resolve()


def log(msg):
    print(f"[bench {datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluator_sha256():
    files = sorted(p for p in (ROOT / "numerics").rglob("*.py"))
    files += sorted(
        p
        for p in BENCH.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".yaml") and "ref" not in p.parts
    )
    if (BENCH / "records").exists():
        files += sorted(p for p in (BENCH / "records").rglob("*.json"))
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------- GPU runs


class GpuLock:
    def __init__(self, what, timeout=8 * 3600):
        self.what, self.timeout = what, timeout

    def __enter__(self):
        self.f = open(LOCK, "a+")
        t0, said = time.time(), False
        while True:
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not said:
                    log(f"waiting for {LOCK} ({self.what}); if this command is wrapped in flock, unwrap it")
                    said = True
                if time.time() - t0 > self.timeout:
                    raise SystemExit(f"could not get {LOCK} within {self.timeout}s for {self.what}")
                time.sleep(5)
        self.t0 = time.time()
        log(f"GPU lock held: {self.what}")
        return self

    def __exit__(self, *_):
        COST["gpu_seconds"] += (time.time() - self.t0) * CFG["world_size"]
        COST["gpu_runs"] += 1
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def cost():
    return {
        "gpu_hours": round(COST["gpu_seconds"] / 3600, 3),
        "gpu_runs": COST["gpu_runs"],
        "method": f'GPU lock hold time x {CFG["world_size"]} cards (startup, warmup and failed runs included)',
    }


def scratch_dir(prefix):
    free = shutil.disk_usage(SCRATCH).free / 2**30
    need = CFG.get("min_scratch_free_gib", 50)
    if free < need:
        raise SystemExit(f"{SCRATCH} has only {free:.1f} GiB free (< {need}); clean up before measuring")
    return tempfile.TemporaryDirectory(prefix=prefix, dir=SCRATCH)


def gpu_state():
    """Query visible MTT S5000 GPUs via mthreads-gmi (-L for UUIDs, -cf for util and memory)."""
    uuids = {}
    q_l = subprocess.run(["mthreads-gmi", "-L"], capture_output=True, text=True, timeout=60)
    for line in q_l.stdout.splitlines():
        m = re.search(r"GPU\s+(\d+)\s*:.*UUID\s*:\s*([0-9a-fA-F-]+)", line)
        if m:
            uuids[int(m.group(1))] = m.group(2)

    q_cf = subprocess.run(["mthreads-gmi", "-cf"], capture_output=True, text=True, timeout=60)
    out = []
    for line in q_cf.stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+.+?\|[^|]+\|\s*(\d+)%\s+(\d+)MiB\(\d+MiB\)", line)
        if m:
            idx = int(m.group(1))
            util = int(m.group(2))
            mem = int(m.group(3))
            out.append({"index": idx, "uuid": uuids.get(idx, f"gpu{idx}"), "memory_mib": mem, "util": util})
    return out


def wait_idle(what, patience=600):
    t0 = time.time()
    while True:
        state = gpu_state()
        if len(state) < CFG["world_size"]:
            return state, f"expected {CFG['world_size']} visible MTT S5000 GPUs, found {len(state)}: {state}"
        busy = [g for g in state if g["memory_mib"] > 1024 or g["util"] > 5]
        if not busy:
            return state, None
        if time.time() - t0 > patience:
            return state, f"GPUs busy for {patience}s before {what}: {busy}"
        time.sleep(10)


def _tagged(tag):
    me, pids = os.getuid(), []
    mark = f"SANA_BENCH_RUN={tag}".encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid == me and mark in (d / "environ").read_bytes().split(b"\0"):
                pids.append(int(d.name))
        except OSError:
            continue
    return pids


def _kill_tagged(tag):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _tagged(tag)
        if not pids:
            return 0
        for p in pids:
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        time.sleep(3)
    return len(_tagged(tag))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_env(tmp, cand_env=None, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k not in RESERVED and not k.startswith("SANA_BENCH")}
    env.update(cand_env or {})
    caches = Path(tmp) / "caches"
    for v in ("inductor", "triton", "tilelang", "tvm", "mcc", "cuda", "xdg", "mpl", "home"):
        (caches / v).mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "TORCHINDUCTOR_CACHE_DIR": str(caches / "inductor"),
            "TRITON_CACHE_DIR": str(caches / "triton"),
            "TILELANG_CACHE_DIR": str(caches / "tilelang"),
            "TVM_CACHE_DIR": str(caches / "tvm"),
            "MCC_CACHE_DIR": str(caches / "mcc"),
            "CUDA_CACHE_PATH": str(caches / "cuda"),
            "XDG_CACHE_HOME": str(caches / "xdg"),
            "MPLCONFIGDIR": str(caches / "mpl"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "PYTHONNOUSERSITE": "1",
            "SANA_LOCAL_MODEL_ROOT": CFG["model_root"],
            "SANA_GEMMA2_PATH": CFG["gemma2_path"],
            "MKL_NUM_THREADS": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "GDN_DISABLE_COMPILE": "1",
            "DISABLE_XFORMERS": "1",
            "PATH": f"{PY.parent}:/usr/local/bin:/usr/bin:/bin",
        }
    )
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("PYTORCH_MUSA_ALLOC_CONF", "expandable_segments:True")
    pp = [str(p) for p in (pythonpath or [])]
    if PYUSER_SITE:
        pp.append(PYUSER_SITE)
    env.pop("PYTHONPATH", None)
    if pp:
        env["PYTHONPATH"] = ":".join(pp)
    return env


def run_gpu(cmd, cwd, env, log_file, what, timeout):
    tag = uuid.uuid4().hex[:12]
    with GpuLock(what):
        before, problem = wait_idle(what)
        if problem:
            return {"status": "INVALID", "reason": problem, "returncode": None, "gpus_before": before}
        env = dict(env, SANA_BENCH_RUN=tag)
        t0 = time.time()
        status = None
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc, status = None, "TIMEOUT"
        wall = time.time() - t0
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        left = _kill_tagged(tag)
        after, _ = wait_idle(f"{what} (after)", patience=60)
    res = {
        "status": status or "ok",
        "returncode": rc,
        "wall_s": round(wall, 1),
        "gpus_before": before,
        "gpus_after": after,
    }
    if status == "TIMEOUT":
        res["reason"] = f"{what} exceeded {timeout}s; process group killed; see {Path(log_file).name}"
    if left:
        res["warning"] = f"{left} processes of the run survived SIGKILL"
    if any(g["memory_mib"] > 1024 for g in after or []):
        res["warning"] = (res.get("warning", "") + " GPU memory still in use 60s after the run").strip()
    return res


def signatures(log_file):
    hits, text = [], Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
    for pattern, reason in SIGNATURES:
        m = pattern.search(text)
        if m:
            line = text[text.rfind("\n", 0, m.start()) + 1 : text.find("\n", m.end())]
            hits.append({"reason": reason, "line": line.strip()[:300]})
    return hits


def launch(argv, port=None):
    return [
        str(TORCHRUN),
        "--nnodes=1",
        f'--nproc-per-node={CFG["world_size"]}',
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(port or free_port()),
        *argv,
    ]


# ---------------------------------------------------------------- inputs & sources


def tree_digest(tree):
    h = hashlib.sha256()
    tree = Path(tree)
    for f in sorted(tree.rglob("*")):
        rel = f.relative_to(tree)
        if any(part in TREE_SKIP for part in rel.parts) or f.is_symlink() or not f.is_file():
            continue
        h.update(str(rel).encode() + b"\0" + sha256_file(f).encode() + b"\n")
    return h.hexdigest()


def file_fingerprint(p, kind):
    if kind == "sha256":
        return sha256_file(p)
    st = Path(p).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def verify_inputs(when):
    spec = json.loads((ROOT / "task" / "formulation.json").read_text())["inputs"]
    t0, bad = time.time(), {}
    for rel, want in spec.items():
        p = path(rel)
        got = file_fingerprint(p, want["kind"]) if p.exists() else None
        if got != want["value"]:
            bad[rel] = {"want": want["value"], "got": got}
    log(f"inputs {when}: {len(spec) - len(bad)}/{len(spec)} match the formulation ({time.time() - t0:.1f}s)")
    return {"when": when, "ok": not bad, "mismatches": bad, "checked": len(spec), "seconds": round(time.time() - t0, 1)}


def freeze_inputs(paths, big=400 << 20):
    out = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(
            f for f in root.rglob("*") if f.is_file() and not f.name.endswith(".lock")
        )
        for f in files:
            kind = "stat" if f.stat().st_size > big else "sha256"
            out[str(f)] = {"kind": kind, "value": file_fingerprint(f, kind)}
    return out


def check_reference():
    ref = path(CFG["reference"])
    git = ["git", "-c", "safe.directory=*", "-C", str(ref)]
    dirty = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if dirty or head != CFG["reference_commit"]:
        raise SystemExit(f"reference tree is not the frozen upstream commit (head {head}, dirty: {bool(dirty)})")
    return ref


def materialize(src, dest):
    def ignore(directory, names):
        return [n for n in names if n in TREE_SKIP or n.endswith(".pyc")]

    shutil.copytree(src, dest, ignore=ignore, symlinks=False)
    for d in [Path(dest), *(p for p in Path(dest).rglob("*") if p.is_dir())]:
        d.chmod(d.stat().st_mode | 0o700)
    return Path(dest)


def candidate_environment(config_file):
    if not config_file:
        return {}
    data = json.loads(Path(config_file).read_text())
    extra = sorted(set(data) - {"environment"})
    if extra:
        raise SystemExit(f'{config_file}: only an "environment" mapping is accepted, found {extra}')
    env = {str(k): str(v) for k, v in (data.get("environment") or {}).items()}
    bad = sorted(k for k in env if k in RESERVED or k.startswith("SANA_BENCH"))
    if bad:
        raise SystemExit(f"{config_file}: {bad} are managed by the bench and cannot be candidate switches")
    return env


def versions():
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "torch_musa", "mate", "tilelang", "diffusers", "transformers", "accelerate"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


FINGERPRINT_KEYS = ("argv", "math_mode", "gpus", "compile_cache", "instrumentation", "versions", "timing_plan")


def fingerprint_diff(a, b):
    if a is None or b is None:
        return ["missing fingerprint"]
    return [k for k in FINGERPRINT_KEYS if a.get(k) != b.get(k)]


# ---------------------------------------------------------------- gate 0: coverage


def setup_fingerprint(tree):
    """AST of the __main__ block and main(cfg) with the per-step `for batch in train_dataloader:` body blanked."""
    module = ast.parse((Path(tree) / recipe.ENTRY).read_text())
    parts, batch_loops = {}, 0
    for node in module.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            parts["__main__"] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.FunctionDef) and node.name == "main":
            fn = copy.deepcopy(node)
            for inner in ast.walk(fn):
                if (
                    isinstance(inner, ast.For)
                    and isinstance(inner.target, ast.Name)
                    and inner.target.id == "batch"
                    and isinstance(inner.iter, ast.Name)
                    and inner.iter.id == "train_dataloader"
                ):
                    inner.body = [ast.Pass()]
                    batch_loops += 1
            parts["main"] = ast.dump(fn, include_attributes=False)
    return parts, batch_loops


def setup_guard(ref, tree):
    want, _ = setup_fingerprint(ref)
    try:
        got, loops = setup_fingerprint(tree)
    except (OSError, SyntaxError) as exc:
        return [f"{recipe.ENTRY} unreadable: {exc}"]
    problems = []
    if loops != 1:
        problems.append(f"expected exactly one `for batch in train_dataloader:` loop in main(), found {loops}")
    for name in ("__main__", "main"):
        if got.get(name) != want.get(name):
            problems.append(
                f"{name} differs from the reference" + (" outside the per-step loop body" if name == "main" else "")
            )
    return problems


LOADER_PROBE = r"""
import hashlib, json, os, random, sys
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
from dataclasses import asdict
import pyrallis, torch
sys.path.insert(0, ".")
from diffusion.data.builder import build_dataset
from diffusion.utils.camctrl_config import SanaVideoCamCtrlConfig

cfg_path = sys.argv[1]
random.seed(0)
torch.manual_seed(0)
config = pyrallis.parse(config_class=SanaVideoCamCtrlConfig, args=["--config_path", cfg_path])
if config.model.aspect_ratio_type is not None:
    config.data.aspect_ratio_type = config.model.aspect_ratio_type
ds = build_dataset(
    asdict(config.data),
    resolution=config.data.image_size,
    max_length=config.text_encoder.model_max_length,
    config=config,
    caption_proportion=config.data.caption_proportion,
    sort_dataset=config.data.sort_dataset,
    vae_downsample_rate=config.vae.vae_stride[-1],
    num_frames=config.data.num_frames,
)

def dig(obj):
    h = hashlib.sha256()
    def walk(x):
        if isinstance(x, torch.Tensor):
            h.update(b"T" + str(x.dtype).encode() + str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b"D")
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"L")
            for item in x:
                walk(item)
        else:
            h.update(b"V" + repr(x).encode())
    walk(obj)
    return h.hexdigest()

n = len(ds)
idxs = sorted({0, 1, 2, 3, n // 2, n - 1})
items = [dig(ds[i]) for i in idxs]
g = torch.Generator().manual_seed(0)
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, drop_last=True, generator=g, num_workers=0)
it = iter(dl)
batches = [dig(next(it)) for _ in range(3)]
print(json.dumps({"length": n, "items": items, "batches": batches}))
"""


def loader_digest(tree):
    with scratch_dir("sana-ld-") as tmp:
        cfg_path = recipe.write_config(
            Path(tmp) / "cfg",
            path(CFG["base_yaml"]),
            path(CFG["fixture_dir"]),
            path(CFG["model_root"]),
            seed=recipe.DEFAULT_SEED,
            num_workers=0,
        )
        env = base_env(tmp)
        r = subprocess.run(
            [str(PY), "-c", LOADER_PROBE, str(cfg_path)],
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    if r.returncode:
        return None, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def coverage(tree):
    ref_src = check_reference()
    problems = setup_guard(ref_src, tree)
    if problems:
        return "ERROR", {
            "passed": False,
            "stage": "setup_guard",
            "problems": problems,
            "note": (
                "the numerics adapter runs main() setup and the per-step loop body verbatim; "
                "move changes into helper functions (_build_fsdp2_plugin, _encode_prompts, model modules, optimizer) "
                "or into the `for batch in train_dataloader:` body"
            ),
        }
    frozen = json.loads((ROOT / "task" / "loader_digest.json").read_text())
    got, err = loader_digest(tree)
    if got != frozen:
        return "INCORRECT", {
            "passed": False,
            "stage": "loader_check",
            "error": err,
            "note": "the dataset yields different items or first batches than the reference on the frozen fixture",
        }
    return None


# ---------------------------------------------------------------- gate 2: bitwise records


def adapter_kwargs(tree, scratch):
    return {
        "repo": str(tree),
        "base_yaml": str(path(CFG["base_yaml"])),
        "fixture_dir": str(path(CFG["fixture_dir"])),
        "model_root": str(path(CFG["model_root"])),
        "scratch": str(scratch),
    }


def record_cases(tree, rec, cases, logs_dir, cand_env, label, losses_only=False, sketch=True):
    n = CFG["numerics"]
    for case in cases:
        with scratch_dir("sana-r-") as tmp:
            cases_file = Path(tmp) / "case.json"
            cases_file.write_text(json.dumps([case]))
            env = base_env(tmp, cand_env, pythonpath=[ROOT])
            cmd = launch(
                [
                    "-m",
                    "numerics",
                    "record",
                    "--adapter",
                    n["adapter"],
                    "--adapter-kwargs",
                    json.dumps(adapter_kwargs(tree, Path(tmp) / "train")),
                    "--cases",
                    str(cases_file),
                    *(
                        ["--losses-only"]
                        if losses_only
                        else (["--sketch-dim", str(n["sketch_dim"])] if sketch else [])
                    ),
                    "--math-mode",
                    CFG["math_mode"],
                    "--deterministic",
                    "--out",
                    str(rec),
                ]
            )
            log_file = Path(logs_dir) / f'{label}_{case["name"]}.log'
            r = run_gpu(
                cmd,
                ROOT,
                env,
                log_file,
                f'{label}: case {case["name"]}',
                TIMEOUT["trajectory" if losses_only else "numerics"],
            )
        if r["status"] != "ok" or r["returncode"]:
            return {
                "status": "INVALID" if r["status"] == "INVALID" else "ERROR",
                "reason": r.get("reason") or f'record exited {r["returncode"]}; see {log_file.name}',
            }
    return None


def missing_tensor_failures(report):
    out = []
    for err in report.get("errors") or []:
        if isinstance(err, dict) and err.get("missing") and not err.get("unexpected"):
            out.append(
                {
                    "where": err["where"],
                    "check": "missing_tensors",
                    "note": "the candidate produced none of these tensors (it did not do that work)",
                    "missing": err["missing"][:20],
                }
            )
    return out


def run_numerics(tree, out, cand_env):
    n = CFG["numerics"]
    rec = out / "records"
    shutil.rmtree(rec, ignore_errors=True)
    cases = json.loads(path(n["cases"]).read_text())
    failed = record_cases(tree, rec, cases, out, cand_env, "record")
    if failed:
        return failed
    report = out / "compare.json"
    env = base_env(out, pythonpath=[ROOT])
    r = subprocess.run(
        [
            str(PY),
            "-m",
            "numerics",
            "compare",
            "--ref",
            str(path(n["ref"])),
            "--cand",
            str(rec),
            "--thresholds",
            str(path(n["thresholds"])),
            *[a for x in n.get("exclude", []) for a in ("--exclude", f'{x["regex"]}={x["reason"]}')],
            "--out",
            str(report),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    (out / "compare.log").write_text(r.stdout + r.stderr)
    shutil.rmtree(out / "caches", ignore_errors=True)
    if not report.exists():
        return {"status": "ERROR", "reason": f"compare exited {r.returncode}; see compare.log"}
    data = json.loads(report.read_text())
    if not n.get("keep_records"):
        shutil.rmtree(rec, ignore_errors=True)
    data["exact_failures"] = (data.get("exact_failures") or []) + missing_tensor_failures(data)
    summary = {
        k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in data.items() if k not in ("status", "measured_tier")
    }
    bitwise = (
        data.get("checked") == data.get("bitwise_identical")
        and not data.get("dtype_changes")
        and not data.get("precision_mode_changes")
        and data.get("bitwise_trusted", False)
    )
    return {
        "status": data.get("status"),
        "measured_tier": data.get("measured_tier"),
        "report": str(report),
        "errors": len(data.get("errors") or []),
        "exact_failures": data.get("exact_failures") or [],
        "bitwise": bitwise,
        "summary": summary,
    }


# ---------------------------------------------------------------- gate 3: trajectories (1 seed, 3-sigma bound)


def traj_case(spec, seed):
    return dict(spec["case"], name=f"traj_s{seed}", seed=seed)


def trajectories(tree, out, cand_env):
    tj = CFG["trajectory"]
    spec = json.loads(path(tj["spec"]).read_text())
    frozen_path = path(tj["frozen"])
    frozen = json.loads(frozen_path.read_text())
    seeds = frozen["reference_seeds"][: int(tj.get("seeds", 1))]
    sigma_k = float(tj.get("sigma_k", 3.0))
    rec = out / "trajectory"
    shutil.rmtree(rec, ignore_errors=True)
    failed = record_cases(tree, rec, [traj_case(spec, s) for s in seeds], out, cand_env, "traj", losses_only=True)
    if failed:
        return {"verdict": "error", **failed}
    report_path = out / "traj_report.json"
    env = base_env(out, pythonpath=[ROOT])
    subprocess.run(
        [
            str(PY),
            "-m",
            "numerics",
            "traj-compare",
            "--frozen",
            str(frozen_path),
            "--cand",
            str(rec),
            "--sigma-k",
            str(sigma_k),
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    shutil.rmtree(out / "caches", ignore_errors=True)
    if not report_path.exists():
        return {"verdict": "error", "reason": "traj-compare produced no report"}
    report = json.loads(report_path.read_text())
    report["report"] = str(report_path)
    return report


# ---------------------------------------------------------------- gate 1: timing


def time_tree(tree, out, label, cand_env=None, profile=None, profile_iters=2):
    t = CFG["timing"]
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f"timing_{label}"
    for f in out.glob(f"timing_{label}.r*.json"):
        f.unlink()
    with scratch_dir("sana-t-") as tmp:
        train_args = recipe.argv(
            Path(tmp) / "train",
            path(CFG["base_yaml"]),
            path(CFG["fixture_dir"]),
            path(CFG["model_root"]),
        )
        hook_args = [
            "--warmup",
            str(t["warmup"]),
            "--window",
            str(t["window"]),
            "--repeats",
            str(t["repeats"]),
            *(["--profile", profile, "--profile-iters", str(profile_iters)] if profile else []),
        ]
        argv = launch([str(BENCH / "timing.py"), "--out", str(prefix), *hook_args, "--", *train_args])
        stable_argv = [
            f'torchrun --nproc-per-node={CFG["world_size"]}',
            "timing.py",
            *hook_args,
            "--",
            *[x.replace(str(tmp), "<scratch>") for x in train_args],
        ]
        env = base_env(tmp, cand_env)
        log_file = out / f"timing_{label}.log"
        what = f'timing {label}{f" ({profile})" if profile else ""}'
        r = run_gpu(argv, tree, env, log_file, what, TIMEOUT["profile" if profile else "timing"])
    plan = [t["warmup"], t["window"], t["repeats"]] if not profile else [t["warmup"], profile_iters, 1]
    fingerprint = {
        "argv": stable_argv,
        "candidate_environment": cand_env or {},
        "math_mode": CFG["math_mode"],
        "gpus": [(g["index"], g["uuid"]) for g in r.get("gpus_before") or []],
        "compile_cache": "cold (fresh per run)",
        "instrumentation": profile or "none",
        "timing_plan": plan,
        "versions": versions(),
    }
    run = {
        "run_status": r["status"],
        "wall_s": r.get("wall_s"),
        "fingerprint": fingerprint,
        "log": str(log_file),
        **({"warning": r["warning"]} if r.get("warning") else {}),
    }
    if r["status"] != "ok":
        return {"status": r["status"], "reason": r["reason"], **run}
    ranks = []
    for rank in range(CFG["world_size"]):
        f = Path(f"{prefix}.r{rank}.json")
        ranks.append(json.loads(f.read_text()) if f.exists() else None)
    if r["returncode"] or any(x is None or x["status"] != "ok" for x in ranks):
        errors = [x["error"] for x in ranks if x and x.get("error")]
        return {"status": "ERROR", "reason": f'timing exited {r["returncode"]}; {errors[:1]}; see {log_file.name}', **run}
    hits = signatures(log_file)
    if hits:
        return {
            "status": "INVALID",
            "reason": "invalid-run signature in the log: " + hits[0]["reason"],
            "signatures": hits,
            **run,
        }
    return {**summarize_timing(ranks, r["wall_s"], profile, profile_iters), **run, "_ranks": ranks}


def summarize_timing(ranks, wall, profile=None, profile_iters=2):
    t = CFG["timing"]
    repeats = 1 if profile else t["repeats"]
    size = profile_iters if profile else t["window"]
    per_window = {}
    for x in ranks:
        for w in x["windows"]:
            prev = per_window.get(w["k"])
            if prev is None or w["seconds"] > prev["seconds"]:
                per_window[w["k"]] = w
    problems = []
    ws = [per_window[k] for k in range(repeats) if k in per_window]
    if len(ws) != repeats:
        problems.append(f"{len(ws)} of {repeats} windows")
    for w in ws:
        if w.get("compiles_in_window"):
            problems.append(f'window {w["k"]}: compiled {w["compiles_in_window"]} graphs inside the timing window')
    secs = [w["seconds"] for w in ws]
    total = sum(secs) if secs else None
    frozen = None
    ref_file = ROOT / "task" / "timing_ref.json"
    if not profile and ref_file.exists():
        frozen = json.loads(ref_file.read_text())
        n_check = t["warmup"] + t["window"] * t["repeats"]
        for x in ranks:
            want = (frozen.get("rng_fingerprints") or {}).get(str(x["rank"]), [])[:n_check]
            if want and x["rng_fingerprints"][:n_check] != want:
                k = next((i for i, (a, b) in enumerate(zip(x["rng_fingerprints"], want)) if a != b), None)
                problems.append(
                    f'rank {x["rank"]}: random-number stream differs from the reference from iteration {k} on'
                )
                break
    res = {
        "status": "ok" if not problems and secs else "ERROR",
        "problems": problems,
        "window_s": [round(s, 3) for s in secs],
        "total_s": round(total, 3) if total else None,
        "iterations_timed": size * repeats,
        "s_per_iteration": round(total / (size * len(secs)), 4) if secs else None,
        "peak_memory_allocated_gib": round(max(x["peak_memory_allocated_bytes"] for x in ranks) / 2**30, 2),
        "peak_memory_reserved_gib": round(max(x["peak_memory_reserved_bytes"] for x in ranks) / 2**30, 2),
        "startup_to_first_iteration_s": round(max(x["startup_to_first_iteration_s"] or 0 for x in ranks), 1),
        "process_wall_s": round(wall, 1),
    }
    if len(secs) == 2:
        res["window_ratio"] = round(secs[0] / secs[1], 4)
        if frozen and frozen.get("window_ratio"):
            res["window_ratio_vs_frozen"] = round(res["window_ratio"] / frozen["window_ratio"] - 1, 4)
    return res


def strip(run):
    return {k: v for k, v in run.items() if k != "_ranks"} if run else run


def benefit_agent(tree, out, cand_env):
    """Gate 1 (agent check): one candidate run against the frozen reference timing."""
    frozen = json.loads((ROOT / "task" / "timing_ref.json").read_text())
    cand = time_tree(tree, out, "cand", cand_env=cand_env)
    runs = {"cand": strip(cand)}
    if cand["status"] != "ok":
        status = "INVALID" if cand["status"] in ("INVALID", "TIMEOUT") else "ERROR"
        return status, {"speedup": None, "runs": runs, "reason": cand.get("reason")}
    ref_total = float(frozen["total_s"])
    c_total = float(cand["total_s"])
    spread = float(frozen.get("spread", 0.0))
    speedup = round(ref_total / c_total, 4)
    bar = max(CFG["benefit"]["min_speedup"], 1.0 + 2.0 * spread)
    notes = []
    dev = cand.get("window_ratio_vs_frozen")
    if dev is not None and abs(dev) > CFG["timing"]["window_band_candidate"]:
        notes.append(f'window ratio {cand["window_ratio"]} is {dev:+.1%} off frozen ratio {frozen.get("window_ratio")}')
    perf = {
        "reference_mode": "frozen",
        "frozen_reference_s_per_iteration": frozen["s_per_iteration"],
        "candidate_s_per_iteration": cand["s_per_iteration"],
        "speedup": speedup,
        "reference_spread": spread,
        "notes": notes,
        "runs": runs,
        "benefit": {
            "speedup": speedup,
            "speedup_bar": round(bar, 4),
            "peak_memory_allocated_gib": cand["peak_memory_allocated_gib"],
            "reference_peak_memory_allocated_gib": frozen.get("peak_memory_allocated_gib"),
            "passed": speedup > bar,
        },
    }
    return ("ok" if perf["benefit"]["passed"] else "UNAVAILABLE"), perf


def benefit_final(tree, out, cand_env):
    """Gate 1 (final check, harness only): ref_a -> cand -> ref_b with fingerprint and interference enforcement."""
    t = CFG["timing"]
    frozen = json.loads((ROOT / "task" / "timing_ref.json").read_text())
    ref_src = check_reference()
    with scratch_dir("sana-ref-") as tmp:
        ref_tree = materialize(ref_src, Path(tmp) / "reference")
        ref_a = time_tree(ref_tree, out, "ref_a")
        cand = time_tree(tree, out, "cand", cand_env=cand_env) if ref_a["status"] == "ok" else None
        ref_b = time_tree(ref_tree, out, "ref_b") if cand and cand["status"] == "ok" else None
    runs = {"ref_a": strip(ref_a), "cand": strip(cand), "ref_b": strip(ref_b)}
    bad = {k: v for k, v in runs.items() if v is None or v["status"] != "ok"}
    if bad:
        status = "INVALID" if any(v and v["status"] in ("INVALID", "TIMEOUT") for v in bad.values()) else "ERROR"
        return status, {"speedup": None, "runs": runs}
    diffs = {k: fingerprint_diff(ref_a.get("fingerprint"), v.get("fingerprint")) for k, v in runs.items() if k != "ref_a"}
    for k in diffs:
        diffs[k] = [d for d in diffs[k] if d != "candidate_environment"]
    badfp = {k: d for k, d in diffs.items() if d}
    if badfp:
        return "ERROR", {
            "speedup": None,
            "runs": runs,
            "fingerprint_diffs": diffs,
            "reason": f"timing runs not comparable: fingerprint differs from ref_a in {badfp}",
        }
    ra, rb, c = ref_a["total_s"], ref_b["total_s"], cand["total_s"]
    ref = (ra + rb) / 2
    perf = {
        "reference_mode": "paired_ref_a_ref_b",
        "reference_s_per_iteration": round(ref / ref_a["iterations_timed"], 4),
        "candidate_s_per_iteration": cand["s_per_iteration"],
        "speedup": round(ref / c, 4),
        "reference_spread": round(abs(ra - rb) / ref, 4),
        "frozen_reference_s_per_iteration": frozen["s_per_iteration"],
        "runs": runs,
        "fingerprint_diffs": diffs,
    }
    problems = []
    for name, run in (("ref_a", ref_a), ("ref_b", ref_b)):
        off = run["total_s"] / frozen["total_s"] - 1
        if abs(off) > t["frozen_band"]:
            problems.append(
                f'{name} took {run["total_s"]:.1f}s, {off:+.1%} off frozen {frozen["total_s"]:.1f}s (> {t["frozen_band"]:.0%})'
            )
    pair = [abs(a - b) / ((a + b) / 2) for a, b in zip(ref_a["window_s"], ref_b["window_s"])]
    perf["reference_pair_window_spread"] = [round(x, 4) for x in pair]
    if max(pair) > t["pair_band"]:
        problems.append(f'the two reference runs differ by {max(pair):.1%} in a window (> {t["pair_band"]:.0%})')
    for name, run in (("ref_a", ref_a), ("cand", cand), ("ref_b", ref_b)):
        band = t["window_band"] if name != "cand" else t["window_band_candidate"]
        dev = run.get("window_ratio_vs_frozen")
        if dev is not None and abs(dev) > band:
            problems.append(
                f'{name}: window ratio {run["window_ratio"]} is {dev:+.1%} off frozen ratio {frozen["window_ratio"]} (> {band:.0%})'
            )
    perf["problems"] = problems
    if problems:
        return "INVALID", perf
    bar = max(CFG["benefit"]["min_speedup"], 1 + 2 * perf["reference_spread"])
    perf["benefit"] = {
        "speedup": perf["speedup"],
        "speedup_bar": round(bar, 4),
        "peak_memory_allocated_gib": cand["peak_memory_allocated_gib"],
        "reference_peak_memory_allocated_gib": round(
            (ref_a["peak_memory_allocated_gib"] + ref_b["peak_memory_allocated_gib"]) / 2, 2
        ),
        "passed": perf["speedup"] > bar,
    }
    return ("ok" if perf["benefit"]["passed"] else "UNAVAILABLE"), perf


# ---------------------------------------------------------------- verdict


def evaluate(tree, out, skip_timing=False, cand_env=None, quick=False, n1=False, final=False):
    cand_env = cand_env or {}
    failed = coverage(tree)
    if failed:
        return failed[0], failed[1], None
    if n1:
        traj = trajectories(tree, out, cand_env)
        correctness = {"stage": "statistical (n1: gates 1 and 2 skipped)", "trajectory": traj}
        if traj["verdict"] == "equivalent":
            return "PASS", dict(correctness, passed=True, verdict="equivalent", measured_tier="N1"), None
        if traj["verdict"] == "error":
            return ("INVALID" if traj.get("status") == "INVALID" else "ERROR"), dict(correctness, passed=False, verdict="error"), None
        return "INCORRECT", dict(
            correctness,
            passed=False,
            verdict=traj["verdict"],
            note=f'an endpoint deviates beyond {CFG["trajectory"]["sigma_k"]} x reference seed sigma',
        ), None
    if quick:
        num = run_numerics(tree, out, cand_env)
        if num.get("exact_failures"):
            return "INCORRECT", {"passed": False, "stage": "structure", "verdict": "structure", "numerics": num}, None
        if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
            return "ERROR", {"passed": False, "stage": "structure", "numerics": num}, None
        return "PASS", {
            "passed": True,
            "stage": "structure (quick: gates 1 and 3 not run)",
            "bitwise": num["bitwise"],
            "measured_tier": num.get("measured_tier"),
            "numerics": num,
        }, None
    performance = None
    if not skip_timing:
        status, performance = (benefit_final if final else benefit_agent)(tree, out, cand_env)
        if status in ("ERROR", "INVALID"):
            return status, {
                "passed": False,
                "stage": "benefit",
                "note": (performance or {}).get("reason")
                or "; ".join((performance or {}).get("problems") or [])
                or "timing failed",
            }, performance
        if status == "UNAVAILABLE":
            performance["measured_speedup"] = performance.pop("speedup")
            return "UNAVAILABLE", {
                "passed": None,
                "stage": "benefit",
                "verdict": "no_benefit",
                "note": "not faster than the reference: rejected without a precision evaluation",
            }, performance
    num = run_numerics(tree, out, cand_env)
    tier = num.get("measured_tier")
    correctness = {"stage": "bitwise", "measured_tier": tier, "numerics": num}
    if num.get("exact_failures"):
        return "INCORRECT", dict(correctness, passed=False, verdict="structure", note="exact checks failed"), performance
    if num.get("errors") or num.get("status") not in ("PASS", "INCORRECT"):
        return "ERROR", dict(correctness, passed=False, note="numerics could not compare the records"), performance
    if tier not in TIERS or TIERS.index(tier) > TIERS.index(CFG["agent_max_tier"]):
        return "INCORRECT", dict(
            correctness, passed=False, verdict="needs_signoff", note=f'measured tier {tier} exceeds {CFG["agent_max_tier"]}'
        ), performance
    if num["bitwise"]:
        return "PASS", dict(correctness, passed=True, verdict="bitwise"), performance
    traj = trajectories(tree, out, cand_env)
    correctness.update(stage="statistical", trajectory=traj)
    if traj["verdict"] == "equivalent":
        return "PASS", dict(correctness, passed=True, verdict="equivalent"), performance
    if traj["verdict"] == "error":
        return ("INVALID" if traj.get("status") == "INVALID" else "ERROR"), dict(correctness, passed=False, verdict="error"), performance
    return "INCORRECT", dict(
        correctness,
        passed=False,
        verdict=traj["verdict"],
        note=f'an endpoint deviates beyond {CFG["trajectory"]["sigma_k"]} x reference seed sigma',
    ), performance


def guarded(out, fn, tree_src=None):
    out.mkdir(parents=True, exist_ok=True)
    start = verify_inputs("start")
    if not start["ok"]:
        return {
            "status": "ERROR",
            "stage": "inputs",
            "inputs": [start],
            "note": "a frozen input differs from task/formulation.json; nothing was measured",
        }
    if tree_src is None:
        res = fn(None)
    else:
        with scratch_dir("sana-src-") as tmp:
            tree = materialize(tree_src, Path(tmp) / "tree")
            digest = tree_digest(tree)
            try:
                res = fn(tree)
            except SystemExit as exc:
                res = {"status": "ERROR", "note": f"the evaluation stopped: {exc}"}
            res["source_digest"] = digest
            if tree_digest(tree) != digest:
                res["status"] = "ERROR"
                res["note"] = "the tree changed while it was being measured (the code wrote into its own tree)"
    end = verify_inputs("end")
    res["inputs"] = [start, end]
    if not end["ok"]:
        res["status"] = "ERROR"
        res["note"] = "a frozen input changed during the run"
    return res


def finish(out, name, res):
    res["cost"] = cost()
    res["evaluator_sha256"] = evaluator_sha256()
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / name).write_text(json.dumps(res, indent=1, default=str) + "\n")
    log(f'wrote {out / name}: status {res.get("status")}')
    summary = {"status": res.get("status")}
    perf = res.get("performance")
    if isinstance(perf, dict):
        summary["speedup"] = perf.get("speedup", perf.get("measured_speedup"))
    for k in ("speedup_vs_frozen_reference", "s_per_iteration"):
        if k in res:
            summary[k] = res[k]
    print(json.dumps(summary))
    return 0 if res.get("status") in ("PASS", "ok") else 1


def cmd_check(a):
    if a.final and not (ROOT / "task" / "final-evaluation").exists():
        raise SystemExit("check --final is reserved for the outer harness (task/final-evaluation not present)")
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        status, correctness, performance = evaluate(
            tree,
            out,
            skip_timing=a.skip_timing,
            cand_env=cand_env,
            quick=a.quick,
            n1=a.n1,
            final=a.final,
        )
        return {
            "status": status,
            "final": bool(a.final),
            "n1": bool(a.n1),
            "tree": str(Path(a.tree).resolve()),
            "config": a.config,
            "environment": cand_env,
            "correctness": correctness,
            "performance": performance,
        }

    return finish(out, "check.json", guarded(out, run, Path(a.tree).resolve()))


def cmd_time(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)
    if a.freeze:
        if a.config:
            raise SystemExit("time --freeze measures the unmodified reference; omit --config")
        ref_src = check_reference()

        def run_freeze(_):
            n_runs = int(CFG["timing"].get("reference_runs", 2))
            with scratch_dir("sana-ref-") as tmp:
                tree = materialize(ref_src, Path(tmp) / "reference")
                runs = {f"ref_{i}": time_tree(tree, out, f"ref_{i}") for i in range(n_runs)}
            ok = [v for v in runs.values() if v["status"] == "ok"]
            if len(ok) != n_runs:
                return {"status": "ERROR", "runs": {k: strip(v) for k, v in runs.items()}}
            totals = [v["total_s"] for v in ok]
            mean_total = statistics.mean(totals)
            spread = (max(totals) - min(totals)) / mean_total if len(totals) > 1 else 0.0
            ratios = [v["window_ratio"] for v in ok if v.get("window_ratio") is not None]
            r0 = ok[0]["_ranks"]
            rng_same = all([x["rng_fingerprints"] for x in v["_ranks"]] == [x["rng_fingerprints"] for x in r0] for v in ok)
            frozen = {
                "total_s": round(mean_total, 3),
                "s_per_iteration": round(mean_total / ok[0]["iterations_timed"], 4),
                "spread": round(spread, 4),
                "window_ratio": round(statistics.mean(ratios), 4) if ratios else None,
                "peak_memory_allocated_gib": round(statistics.mean(v["peak_memory_allocated_gib"] for v in ok), 2),
                "peak_memory_reserved_gib": round(statistics.mean(v["peak_memory_reserved_gib"] for v in ok), 2),
                "reference_runs": totals,
                "rng_identical_across_runs": rng_same,
                "rng_fingerprints": {str(x["rank"]): x["rng_fingerprints"] for x in r0},
                "timing_plan": [CFG["timing"]["warmup"], CFG["timing"]["window"], CFG["timing"]["repeats"]],
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
            }
            (ROOT / "task" / "timing_ref.json").write_text(json.dumps(frozen, indent=1) + "\n")
            return {"status": "ok", "frozen": frozen, "runs": {k: strip(v) for k, v in runs.items()}}

        return finish(out, "freeze_timing.json", guarded(out, run_freeze))

    def run(tree):
        res = strip(time_tree(tree, out, "tree", cand_env=cand_env))
        ref_file = ROOT / "task" / "timing_ref.json"
        if ref_file.exists():
            frozen = json.loads(ref_file.read_text())
            if res["status"] == "ok":
                res["speedup_vs_frozen_reference"] = round(frozen["total_s"] / res["total_s"], 4)
            res["frozen_reference_s_per_iteration"] = frozen["s_per_iteration"]
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)

    return finish(out, "time.json", guarded(out, run, Path(a.tree).resolve()))


CATEGORIES = [
    ("mccl/comm", r"mccl|nccl|allgather|reducescatter|allreduce|broadcast"),
    ("attention", r"fmha|flash|atten|sdpa|mem_eff"),
    ("gemm", r"gemm|sgemm|hgemm|bf16.*gemm|mublas|wmma|matmul"),
    ("conv", r"conv|dgrad|wgrad|fprop|mudnn"),
    ("norm", r"norm|rms"),
    ("optimizer", r"adam|multi_tensor|foreach"),
    ("memcpy/cast", r"copy|memcpy|cast|convert|_to_copy|memset"),
    ("tilelang/custom", r"tilelang|tvm|gdn|ucpe|musa_custom"),
    ("elementwise/reduce", r"elementwise|reduce|vectorized|unrolled|index|cat|silu|gelu|softmax|mul|add|exp|sigmoid"),
]


def categorize(kernels):
    cats = {}
    for k in kernels:
        n = k["name"].lower()
        cat = next((c for c, rx in CATEGORIES if re.search(rx, n)), "other")
        cats[cat] = cats.get(cat, 0.0) + k["total_ms"]
    return {k: round(v, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}


def cmd_profile(a):
    out = Path(a.out).resolve()
    cand_env = candidate_environment(a.config)

    def run(tree):
        res = time_tree(tree, out, "profile", cand_env=cand_env, profile="torch", profile_iters=a.iters)
        ranks = res.pop("_ranks", None)
        if res["status"] != "ok":
            return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)
        res["note"] = (
            f'{a.iters} iterations after warmup of {CFG["timing"]["warmup"]}; '
            "instrumented: wall time is not comparable with `time`"
        )
        r0 = ranks[0]
        ends = r0["iteration_end_s"]
        first = CFG["timing"]["warmup"]
        res["iteration_s_host"] = [
            round(ends[i] - ends[i - 1], 3) for i in range(first, min(len(ends), first + a.iters))
        ]
        summ = json.loads((out / "timing_profile.torchprof.json").read_text())
        summ["categories_ms"] = categorize(summ["kernels"])
        res["torch_profile"] = summ
        trace_path = out / "timing_profile.trace.json"
        res["trace"] = str(trace_path)

        # Generate kernel-utilization-analysis report next to the trace
        ku_md = out / "kernel_util.md"
        ku_json = out / "kernel_util.json"
        ku_cmd = [
            str(PY),
            str(BENCH / "kernel_util.py"),
            str(trace_path),
            "--out",
            str(ku_md),
            "--json",
            str(ku_json),
        ]
        if CFG.get("peak_bw_gbs"):
            ku_cmd.extend(["--peak-bw-gbs", str(CFG["peak_bw_gbs"])])
        ku_r = subprocess.run(ku_cmd, capture_output=True, text=True, timeout=300)
        if ku_r.returncode == 0 and ku_md.exists():
            res["kernel_util_md"] = str(ku_md)
            res["kernel_util_json"] = str(ku_json)
        else:
            res["kernel_util_error"] = (ku_r.stderr or ku_r.stdout)[-1000:]

        md = [
            "# profile",
            "",
            f"tree {a.tree}, config {a.config}",
            "",
            res["note"],
            "",
            f'window {summ["span_s"]:.2f} s for {summ["iterations"]} iterations; GPU busy '
            f'{summ["gpu_busy_s"]:.2f} s ({(summ["gpu_busy_fraction"] or 0):.0%}); '
            f'kernel time {summ["kernel_time_s"]:.2f} s (rank 0)',
            "",
            "host time per iteration (s): " + ", ".join(map(str, res["iteration_s_host"])),
            "",
            "## kernel time by category (ms, rank 0)",
            "",
            "| category | ms |",
            "|---|---|",
        ]
        md += [f"| {k} | {v} |" for k, v in summ["categories_ms"].items()]
        md += [
            "",
            "## GPU time inside annotated model calls (forward only, rank 0)",
            "",
            "| call | GPU ms | calls | CPU ms |",
            "|---|---|---|---|",
        ]
        md += [f'| {k} | {v["gpu_ms"]} | {v["calls"]} | {v["cpu_ms"]} |' for k, v in summ["annotations"].items()]
        md += ["", "## top kernels", "", "| ms | calls | share | kernel |", "|---|---|---|---|"]
        md += [
            f'| {k["total_ms"]} | {k["calls"]} | {k["share_of_kernel_time"]} | `{k["name"][:140]}` |'
            for k in summ["kernels"][:40]
        ]
        md += ["", "## top CPU ops (self time)", "", "| ms | calls | op |", "|---|---|---|"]
        md += [f'| {o["self_cpu_ms"]} | {o["calls"]} | {o["name"]} |' for o in summ["cpu_ops_self_time"][:25]]
        if ku_md.exists():
            md += ["", f"Kernel utilization report: `{ku_md.name}` (JSON: `{ku_json.name}`)."]
        (out / "profile.md").write_text("\n".join(md) + "\n")
        return dict(res, tree=str(Path(a.tree).resolve()), config=a.config, environment=cand_env)

    return finish(out, "profile.json", guarded(out, run, Path(a.tree).resolve()))


def cmd_baseline(a):
    out = Path(a.out).resolve()

    def run(_):
        ref_src = check_reference()
        with scratch_dir("sana-ref-") as tmp:
            tree = materialize(ref_src, Path(tmp) / "reference")
            runs = {k: time_tree(tree, out, k) for k in ("ref_a", "ref_b")}
        ok = [v for v in runs.values() if v["status"] == "ok"]
        tot = [v["total_s"] for v in ok]
        res = {
            "status": "ok" if len(ok) == len(runs) else "ERROR",
            "runs": {k: strip(v) for k, v in runs.items()},
            "total_s": tot,
            "spread": round((max(tot) - min(tot)) / statistics.mean(tot), 4) if len(tot) > 1 else 0.0,
        }
        if ok:
            r = ok[0]["_ranks"]
            res["rng_fingerprints"] = {str(x["rank"]): x["rng_fingerprints"] for x in r}
            res["rng_identical_across_runs"] = all(
                [x["rng_fingerprints"] for x in v["_ranks"]] == [x["rng_fingerprints"] for x in r] for v in ok
            )
        return res

    return finish(out, "baseline.json", guarded(out, run))


def cmd_record(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(a.cases).read_text()) if a.cases else json.loads(path(CFG["numerics"]["cases"]).read_text())
    tree = Path(a.tree).resolve()
    if a.traj_seeds:
        spec = json.loads(path(CFG["trajectory"]["spec"]).read_text())
        cases = [traj_case(spec, int(s)) for s in a.traj_seeds.split(",")]
    failed = record_cases(
        tree,
        out,
        cases,
        out,
        candidate_environment(a.config),
        "formulate",
        losses_only=bool(a.traj_seeds),
    )
    print(json.dumps({"status": "ok" if not failed else failed, "out": str(out), "cost": cost()}))
    return 1 if failed else 0


def cmd_freeze_inputs(a):
    inputs = freeze_inputs(a.paths)
    f = ROOT / "task" / "formulation.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data["inputs"] = inputs
    f.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({"files": len(inputs), "sha256": sum(v["kind"] == "sha256" for v in inputs.values())}))
    return 0


def cmd_loader(a):
    got, err = loader_digest(Path(a.tree).resolve())
    print(json.dumps(got) if got else err)
    return 0 if got else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("check")
    k.add_argument("--tree", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--config")
    k.add_argument("--quick", action="store_true")
    k.add_argument("--skip-timing", action="store_true")
    k.add_argument("--n1", action="store_true")
    k.add_argument("--final", action="store_true", help=argparse.SUPPRESS)
    t = sub.add_parser("time")
    t.add_argument("--tree", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config")
    t.add_argument("--freeze", action="store_true", help="formulator: freeze reference timing into task/timing_ref.json")
    f = sub.add_parser("profile")
    f.add_argument("--tree", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--config")
    f.add_argument("--iters", type=int, default=2)
    b = sub.add_parser("baseline")
    b.add_argument("--out", required=True)
    r = sub.add_parser("record", help="formulator: record numerics cases or reference trajectories")
    r.add_argument("--tree", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cases")
    r.add_argument("--config")
    r.add_argument("--traj-seeds")
    ld = sub.add_parser("loader", help="formulator: print dataset digest")
    ld.add_argument("--tree", required=True)
    fi = sub.add_parser("freeze-inputs", help="formulator: write frozen input fingerprints")
    fi.add_argument("paths", nargs="+")
    a = p.parse_args()
    return {
        "freeze-inputs": cmd_freeze_inputs,
        "check": cmd_check,
        "time": cmd_time,
        "profile": cmd_profile,
        "baseline": cmd_baseline,
        "record": cmd_record,
        "loader": cmd_loader,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
