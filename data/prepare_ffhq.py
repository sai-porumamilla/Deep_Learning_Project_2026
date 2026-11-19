"""A1: download FFHQ at 256x256 and split into train / val / test.

Source: huggingface.co/datasets/merkol/ffhq-256 (70k images, 15 parquet shards).
We only need the first N images, so only the shards covering them are downloaded.

Output (deterministic, so A and B get identical splits by running this):
    <out>/images/00000.png ...
    <out>/train.txt, val.txt, test.txt   (one filename per line)

Usage:
    python data/prepare_ffhq.py --out datasets/ffhq256
"""
import argparse
import io
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "merkol/ffhq-256"
SIZE = 256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets/ffhq256")
    p.add_argument("--n-train", type=int, default=18000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    total = a.n_train + a.n_val + a.n_test

    shards = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    i = 0
    for shard in shards:
        if i >= total:
            break
        path = hf_hub_download(REPO, shard, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["image"]):
            for row in batch.column("image").to_pylist():
                if i >= total:
                    break
                dst = img_dir / f"{i:05d}.png"
                if not dst.exists():  # resumable
                    img = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
                    if img.size != (SIZE, SIZE):
                        img = img.resize((SIZE, SIZE), Image.LANCZOS)
                    img.save(dst)
                i += 1
        print(f"{shard}: {i}/{total}")

    if i < total:
        raise SystemExit(f"only {i} images available, need {total}")

    names = [f"{k:05d}.png" for k in range(total)]
    random.Random(a.seed).shuffle(names)
    splits = {
        "train": names[: a.n_train],
        "val": names[a.n_train : a.n_train + a.n_val],
        "test": names[a.n_train + a.n_val :],
    }
    for split, files in splits.items():
        (out / f"{split}.txt").write_text("\n".join(sorted(files)) + "\n")
    print({k: len(v) for k, v in splits.items()}, "->", out)


if __name__ == "__main__":
    main()
