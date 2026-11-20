"""A2: PortraitDataset and lab_to_rgb (see Data Contract in README).

Each item: {"L": Tensor[1, 256, 256], "ab": Tensor[2, 256, 256]}, both in [-1, 1]
    L  = L / 50 - 1     (L  in [0, 100])
    ab = ab / 110       (ab in about [-110, 110])
"""
from pathlib import Path

import numpy as np
import torch
from kornia.color import lab_to_rgb as _lab_to_rgb
from kornia.color import rgb_to_lab as _rgb_to_lab
from PIL import Image
from torch.utils.data import Dataset


def rgb_to_lab(rgb):
    """RGB float [(B,) 3, H, W] in [0, 1] -> normalized (L [(B,) 1, H, W], ab [(B,) 2, H, W])."""
    lab = _rgb_to_lab(rgb)
    return lab[..., :1, :, :] / 50 - 1, lab[..., 1:, :, :] / 110


def lab_to_rgb(L, ab):
    """Normalized L [(B,) 1, H, W] + ab [(B,) 2, H, W] -> RGB [(B,) 3, H, W] in [0, 1].

    Pure torch: runs on whatever device L/ab are on. Feed the result to
    torchvision.utils.save_image for sample grids (call .detach() on model output first).
    """
    lab = torch.cat([(L + 1) * 50, ab * 110], dim=-3)
    return _lab_to_rgb(lab, clip=True)


def load_rgb(path):
    """Image file -> RGB float tensor [3, H, W] in [0, 1]."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(rgb).permute(2, 0, 1)


class PortraitDataset(Dataset):
    """root/ contains images/ and {train,val,test}.txt (output of data/prepare_ffhq.py)."""

    def __init__(self, root, split="train", flip=None):
        self.img_dir = Path(root) / "images"
        self.files = (Path(root) / f"{split}.txt").read_text().split()
        self.flip = split == "train" if flip is None else flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rgb = load_rgb(self.img_dir / self.files[i])
        if self.flip and torch.rand(1).item() < 0.5:
            rgb = rgb.flip(-1)
        L, ab = rgb_to_lab(rgb)
        return {"L": L, "ab": ab}


if __name__ == "__main__":
    # Self-check: python -m data.dataset <root>
    import sys

    from torch.utils.data import DataLoader

    ds = PortraitDataset(sys.argv[1], "val", flip=False)
    batch = next(iter(DataLoader(ds, batch_size=4)))
    L, ab = batch["L"], batch["ab"]
    assert L.shape == (4, 1, 256, 256) and ab.shape == (4, 2, 256, 256), (L.shape, ab.shape)
    assert L.min() >= -1 and L.max() <= 1 and ab.abs().max() <= 1, (L.min(), L.max(), ab.abs().max())

    # RGB -> Lab -> RGB round-trip must match the original image
    orig = torch.stack([load_rgb(ds.img_dir / f) for f in ds.files[:4]])
    err = (lab_to_rgb(L, ab) - orig).abs().max().item() * 255
    assert err < 1.0, f"round-trip error {err:.3f}/255"
    assert lab_to_rgb(L[0], ab[0]).shape == (3, 256, 256)

    # flip actually flips
    ds_f = PortraitDataset(sys.argv[1], "val", flip=True)
    torch.manual_seed(0)
    flipped = [torch.equal(ds_f[0]["L"], ds[0]["L"].flip(-1)) for _ in range(20)]
    assert any(flipped) and not all(flipped), "flip should happen ~50% of the time"

    print(f"ok: shapes, ranges, flip, round-trip max error {err:.4f}/255")
