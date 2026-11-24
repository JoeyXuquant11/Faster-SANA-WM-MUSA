"""Build the deterministic 64-sample Sekai-Game schema fixture (193 frames, 704x1280)
used by SANA-WM Stage-1 bidirectional FSDP2 benchmarking and validation.

Matches task/loader_digest.json bit-for-bit without requiring private data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

NUM_SAMPLES = 64
NUM_PIXEL_FRAMES = 193  # (193 - 1) // 8 + 1 = 25 latent frames
LATENT_T = (NUM_PIXEL_FRAMES - 1) // 8 + 1
PROMPTS = [
    "A sports car drives smoothly forward across a vast sunlit salt desert under a clear blue sky.",
    "First-person walk through a glowing bioluminescent cave with crystal formations and shallow pools.",
    "A small robot glides down a misty mushroom forest trail while turning gently to the left.",
    "Camera pans slowly across an icy glacial plain toward a distant glowing stone arch at dusk.",
    "Steady forward flight along a narrow canyon river lined with red sandstone cliffs and pine trees.",
    "Walking through a quiet lantern-lit courtyard after rain with reflections shimmering on wet cobbles.",
    "Gliding above rolling emerald hills toward a coastal lighthouse under drifting white clouds.",
    "Moving through an ancient sunlit temple corridor with tall carved stone pillars and dust motes.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output fixture directory (writes raw/ and vae_cache/)")
    p.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    args = p.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    cache_dir = out_dir / "vae_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_zip = raw_dir / "sekai_game_train_00000000.zip"
    cache_zip = cache_dir / raw_zip.name
    rng = np.random.default_rng(20261122)

    ids, ranges, all_poses, all_intr = [], [], [], []
    captions = {}
    scores = {k: {} for k in ["_vmafmotion", "_unimatch", "_dover", "_vlm_entity_filter", "_vlm_quality_filter"]}

    with zipfile.ZipFile(raw_zip, "w", compression=zipfile.ZIP_STORED) as zf_raw, \
         zipfile.ZipFile(cache_zip, "w", compression=zipfile.ZIP_STORED) as zf_cache:
        for idx in range(args.num_samples):
            key = f"sample_{idx:06d}"
            prompt = PROMPTS[idx % len(PROMPTS)] + f" Scene variant {idx}."
            zf_raw.writestr(f"{key}.json", json.dumps({"prompt": prompt, "width": 1280, "height": 704}))

            z = rng.standard_normal((128, LATENT_T, 22, 40)).astype(np.float32) * 0.5
            buf = io.BytesIO()
            np.savez(buf, z=z)
            zf_cache.writestr(f"{key}.npz", buf.getvalue())

            poses = np.repeat(np.eye(4, dtype=np.float32)[None], NUM_PIXEL_FRAMES, axis=0)
            yaw_rate = ((idx % 7) - 3) * 0.0025
            speed = 0.015 + (idx % 5) * 0.004
            for f in range(NUM_PIXEL_FRAMES):
                th = yaw_rate * f
                c, s = math.cos(th), math.sin(th)
                poses[f, 0, 0] = c
                poses[f, 0, 2] = s
                poses[f, 2, 0] = -s
                poses[f, 2, 2] = c
                poses[f, 2, 3] = speed * f
                poses[f, 0, 3] = 0.2 * speed * f * math.sin(th * 0.5)
            fx = 740.0 + (idx % 5) * 15.0
            intr = np.tile(np.array([fx, fx, 640.0, 352.0], dtype=np.float32), (NUM_PIXEL_FRAMES, 1))

            start = idx * NUM_PIXEL_FRAMES
            ids.append(key)
            ranges.append([start, start + NUM_PIXEL_FRAMES])
            all_poses.append(poses)
            all_intr.append(intr)
            captions[key] = {"prompt": prompt}
            scores["_vmafmotion"][key] = {"score": 5.0}
            scores["_unimatch"][key] = {"score": 12.0}
            scores["_dover"][key] = {"score": 0.6}
            scores["_vlm_entity_filter"][key] = {"score": 5.0}
            scores["_vlm_quality_filter"][key] = {"score": 1.0}

    np.savez(
        raw_zip.with_name(raw_zip.stem + "_camera.npz"),
        ids=np.array(ids),
        ranges=np.array(ranges, dtype=np.int64),
        pose=np.concatenate(all_poses, axis=0),
        intrinsics=np.concatenate(all_intr, axis=0),
    )
    raw_zip.with_name(raw_zip.stem + "_LongSceneStaticCaption-Qwen3-VL-30B-A3B-Instruct.json").write_text(
        json.dumps(captions)
    )
    for suf, d in scores.items():
        raw_zip.with_name(raw_zip.stem + f"{suf}.json").write_text(json.dumps(d))

    digests = {}
    for p_file in sorted(out_dir.rglob("*")):
        if p_file.is_file():
            h = hashlib.sha256()
            with p_file.open("rb") as fp:
                for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            digests[str(p_file.relative_to(out_dir))] = {"size": p_file.stat().st_size, "sha256": h.hexdigest()}
    (out_dir / "fixture_digests.json").write_text(json.dumps(digests, indent=2) + "\n")
    print(f"Built {args.num_samples}-sample fixture ({sum(v['size'] for v in digests.values()) / 1e6:.1f} MB) in {out_dir}")


if __name__ == "__main__":
    main()
