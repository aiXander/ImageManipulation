#!/usr/bin/env python3
"""
Sort images by visual similarity using LPIPS perceptual distance + TSP solver.
Outputs center-cropped, resized images with clean sequential filenames.

Requirements: pip install tsp_solver2 lpips torchvision pillow tqdm

Example usage:
    python sort_visual_path.py /path/to/images
    python sort_visual_path.py /path/to/images --aspect-ratio 1:1 --max-dim 1080
    python sort_visual_path.py /path/to/images --optim-steps 2000 --output /path/to/output
"""

import os
import argparse
import itertools
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from tsp_solver.greedy import solve_tsp
import lpips

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def centre_crop(img: Image.Image, aspect_ratio: float) -> Image.Image:
    """Center-crop a PIL image to the target aspect ratio (w/h)."""
    w, h = img.size
    if w / h > aspect_ratio:
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / aspect_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def resize_max(img: Image.Image, max_dim: int) -> Image.Image:
    """Resize so max(w, h) == max_dim, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def load_image_paths(directory: str) -> list[str]:
    paths = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in EXTENSIONS:
            paths.append(os.path.join(directory, f))
    return paths


def load_thumbnails(
    paths: list[str], aspect_ratio: float, thumb_pixels: int
) -> tuple[list[str], list[torch.Tensor]]:
    """Load images as cropped+resized tensors for LPIPS. Returns valid paths and tensors."""
    h = int(np.sqrt(thumb_pixels / aspect_ratio))
    w = int(h * aspect_ratio)
    to_tensor = ToTensor()
    valid_paths, tensors = [], []
    for p in tqdm(paths, desc="Loading images"):
        try:
            img = Image.open(p).convert("RGB")
            img = centre_crop(img, aspect_ratio)
            img = img.resize((w, h), Image.LANCZOS)
            tensors.append(to_tensor(img))
            valid_paths.append(p)
        except Exception as e:
            print(f"  Skipping {os.path.basename(p)}: {e}")
    return valid_paths, tensors


@torch.no_grad()
def compute_distance_matrix(
    tensors: list[torch.Tensor],
    device: torch.device,
    lpips_net: lpips.LPIPS,
    batch_size: int = 8,
    lpips_resize: int = 256,
) -> list[list[float]]:
    """Compute full NxN LPIPS distance matrix."""
    n = len(tensors)
    dist = [[0.0] * n for _ in range(n)]
    pairs = list(itertools.combinations(range(n), 2))

    for start in tqdm(range(0, len(pairs), batch_size), desc="Computing LPIPS"):
        batch = pairs[start : start + batch_size]
        imgs1 = torch.stack([tensors[i] for i, _ in batch]).to(device)
        imgs2 = torch.stack([tensors[j] for _, j in batch]).to(device)

        if lpips_resize > 0:
            _, _, h, w = imgs1.shape
            if w >= h:
                target_w, target_h = lpips_resize, int(lpips_resize * h / w)
            else:
                target_h, target_w = lpips_resize, int(lpips_resize * w / h)
            imgs1 = F.interpolate(imgs1, (target_h, target_w), mode="bilinear", align_corners=False)
            imgs2 = F.interpolate(imgs2, (target_h, target_w), mode="bilinear", align_corners=False)

        d = lpips_net(imgs1 * 2 - 1, imgs2 * 2 - 1).squeeze().cpu()
        if d.dim() == 0:
            d = d.unsqueeze(0)

        for (i, j), val in zip(batch, d.tolist()):
            dist[i][j] = val
            dist[j][i] = val

    return dist


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Sort images by visual similarity (LPIPS + TSP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("directory", help="Input directory containing images")
    parser.add_argument(
        "--aspect-ratio", type=str, default="16:9",
        help="Target aspect ratio as W:H (e.g. 16:9, 1:1, 4:3)",
    )
    parser.add_argument(
        "--max-dim", type=int, default=1920,
        help="Max(width, height) for output images",
    )
    parser.add_argument(
        "--optim-steps", type=int, default=1000,
        help="TSP optimization steps (more = better ordering, slower)",
    )
    parser.add_argument(
        "--thumb-pixels", type=int, default=512 * 512,
        help="Total pixels for LPIPS thumbnails",
    )
    parser.add_argument(
        "--lpips-resize", type=int, default=256,
        help="Resize longest edge before LPIPS (0 to disable)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for LPIPS computation",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: <input>_tsp_sorted)",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality (1-100)",
    )
    parser.add_argument(
        "--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network",
    )
    args = parser.parse_args()

    # Parse aspect ratio
    parts = args.aspect_ratio.split(":")
    if len(parts) != 2:
        parser.error("--aspect-ratio must be W:H (e.g. 16:9)")
    aspect_ratio = float(parts[0]) / float(parts[1])

    device = get_device()
    print(f"Device: {device}")

    # Load
    paths = load_image_paths(args.directory)
    if len(paths) < 2:
        print(f"Found {len(paths)} images — need at least 2.")
        return
    print(f"Found {len(paths)} images")

    paths, tensors = load_thumbnails(paths, aspect_ratio, args.thumb_pixels)
    if len(paths) < 2:
        print("Not enough valid images.")
        return

    # LPIPS
    print(f"Loading LPIPS ({args.lpips_net})...")
    lpips_net = lpips.LPIPS(net=args.lpips_net).eval().to(device)

    t0 = time.time()
    dist_matrix = compute_distance_matrix(
        tensors, device, lpips_net,
        batch_size=args.batch_size,
        lpips_resize=args.lpips_resize,
    )
    print(f"Distance matrix: {time.time() - t0:.1f}s")

    # TSP
    print(f"Solving TSP ({args.optim_steps} optim steps)...")
    t0 = time.time()
    order = solve_tsp(dist_matrix, optim_steps=args.optim_steps, endpoints=None)
    print(f"TSP solved: {time.time() - t0:.1f}s")

    # Save
    outdir = args.output or (args.directory.rstrip("/") + "_tsp_sorted")
    os.makedirs(outdir, exist_ok=True)
    n_digits = max(4, len(str(len(order))))

    print(f"Saving {len(order)} images to {outdir}")
    for seq, idx in enumerate(tqdm(order, desc="Saving")):
        img = Image.open(paths[idx]).convert("RGB")
        img = centre_crop(img, aspect_ratio)
        img = resize_max(img, args.max_dim)
        name = f"{seq + 1:0{n_digits}d}.jpg"
        img.save(os.path.join(outdir, name), quality=args.quality)

    print(f"Done — {len(order)} images → {outdir}")


if __name__ == "__main__":
    main()
