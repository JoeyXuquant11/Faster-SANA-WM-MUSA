"""Frozen SANA-WM Stage-1 bidirectional FSDP2 training recipe on 4x MTT S5000."""
from pathlib import Path
import yaml

ENTRY = "train_video_scripts/train_sana_wm_stage1.py"
WORLD_SIZE = 4
DEFAULT_SEED = 3407


def write_config(
    work_dir,
    base_yaml,
    fixture_dir,
    model_root,
    *,
    seed=DEFAULT_SEED,
    num_workers=2,
    max_steps=None,
    lr=None,
):
    """Materialize a per-run YAML config from the frozen base recipe into work_dir."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(base_yaml).read_text())
    fixture_dir = Path(fixture_dir)
    model_root = Path(model_root)

    cfg["data"]["data_dir"] = {"sekai_game": str(fixture_dir / "raw")}
    cfg["data"]["vae_cache_dir"] = str(fixture_dir / "vae_cache")
    cfg["model"]["load_from"] = str(
        model_root / "SANA-WM_bidirectional" / "dit" / "sana_wm_1600m_720p.safetensors"
    )
    cfg["vae"]["vae_pretrained"] = str(model_root / "SANA-WM_bidirectional")

    cfg["work_dir"] = str(work_dir / "out")
    cfg["train"]["work_dir"] = str(work_dir / "out")
    cfg["train"]["seed"] = int(seed)
    cfg["train"]["num_workers"] = int(num_workers)
    cfg["train"]["save_model_steps"] = 0
    cfg["train"]["save_model_epochs"] = 0
    cfg["train"]["max_steps"] = None if max_steps is None else int(max_steps)
    if lr is not None:
        cfg["train"]["optimizer"]["lr"] = float(lr)

    out_yaml = work_dir / "recipe_run.yaml"
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_yaml


def argv(
    work_dir,
    base_yaml,
    fixture_dir,
    model_root,
    *,
    seed=DEFAULT_SEED,
    num_workers=2,
    max_steps=None,
    lr=None,
):
    cfg_path = write_config(
        work_dir,
        base_yaml,
        fixture_dir,
        model_root,
        seed=seed,
        num_workers=num_workers,
        max_steps=max_steps,
        lr=lr,
    )
    return ["--config_path", str(cfg_path)]
