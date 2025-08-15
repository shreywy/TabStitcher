#!/usr/bin/env python3

"""
Tab Snippet Extractor for YouTube Guitar Videos
------------------------------------------------
Updated: Downloads video to script directory, auto-generates output filename, optional --keep video.
"""
import argparse
import math
import os
import sys
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

# Third-party
import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash

# Optional import of yt_dlp only if needed
def ensure_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class Snippet:
    start_t: float
    end_t: float
    best_img: np.ndarray  # BGR image of the best/clearest frame
    best_sharpness: float


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_region(frame: np.ndarray, crop: Optional[Tuple[int,int,int,int]], bottom_frac: float) -> np.ndarray:
    h, w = frame.shape[:2]
    if crop is not None:
        x, y, cw, ch = crop
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        cw = max(1, min(cw, w-x))
        ch = max(1, min(ch, h-y))
        return frame[y:y+ch, x:x+cw]
    y0 = int(h * (1.0 - bottom_frac))
    return frame[y0:h, 0:w]


def phash(img: np.ndarray, hash_size: int = 16) -> imagehash.ImageHash:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("L")
    return imagehash.phash(pil, hash_size=hash_size)


def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    return (a - b)


def stitch_vertical(images: List[np.ndarray], pad: int = 12) -> np.ndarray:
    if not images:
        raise ValueError("No images to stitch.")
    widths = [img.shape[1] for img in images]
    target_w = max(widths)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if w != target_w:
            scale = target_w / w
            img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        resized.append(img)
    pads = [np.full((pad, target_w, 3), 255, dtype=np.uint8) for _ in range(len(resized)-1)]
    rows = []
    for i, img in enumerate(resized):
        rows.append(img)
        if i < len(resized) - 1:
            rows.append(pads[i])
    return np.vstack(rows)


def sanitize_filename(name: str) -> str:
    # Keep only letters, numbers, spaces
    return re.sub(r"[^A-Za-z0-9 ]+", "", name).strip()


def download_video(url: str, out_dir: str) -> Tuple[str, str]:
    """Download video to out_dir. Returns (filepath, video_title)."""
    try:
        import yt_dlp
    except Exception:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")

    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        base, ext = os.path.splitext(filepath)
        if ext.lower() != ".mp4":
            mp4_candidate = base + ".mp4"
            if os.path.exists(mp4_candidate):
                filepath = mp4_candidate
        return filepath, info.get("title", "video")


def parse_crop(s: str) -> Tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be 'x,y,w,h'")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description="Extract and stitch guitar tab snippets from a YouTube/local video.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL to download & process")
    src.add_argument("--video", help="Path to a local video file to process")

    ap.add_argument("--out", default=None, help="Output PNG path (default: video title - YYYYMMDD.png)")
    ap.add_argument("--keep", action="store_true", help="Keep downloaded video file")
    ap.add_argument("--sample-fps", type=float, default=3.0)
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--hash-thresh", type=int, default=10)
    ap.add_argument("--min-hold", type=float, default=0.7)
    ap.add_argument("--min-gap", type=float, default=0.4)
    ap.add_argument("--crop", type=parse_crop, default=None)
    ap.add_argument("--crop-bottom-frac", type=float, default=1/3)
    ap.add_argument("--max-segments", type=int, default=0)

    args = ap.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Obtain video path
    if args.url:
        print(f"Downloading video to: {script_dir}")
        video_path, title = download_video(args.url, script_dir)
    else:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}", file=sys.stderr)
            sys.exit(2)
        title = os.path.splitext(os.path.basename(video_path))[0]

    # Default output filename if not specified
    if args.out is None:
        clean_title = sanitize_filename(title)
        date_str = datetime.now().strftime("%Y%m%d")
        args.out = os.path.join(script_dir, f"{clean_title} - {date_str}.png")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.", file=sys.stderr)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames > 0 else None
    print(f"Video FPS: {fps:.3f} | Frames: {total_frames} | Duration: {duration:.2f}s" if duration else f"Video FPS: {fps:.3f}")

    step = max(1, int(round(fps / args.sample_fps)))
    t_per_sample = step / fps

    prev_hash = None
    pending_change_since = None
    block_end_guard = 0.0
    t = 0.0
    frame_idx = 0

    current_imgs: List[Tuple[np.ndarray, float]] = []
    snippets: List[Snippet] = []
    segment_start_t = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        crop = crop_region(frame, args.crop, args.crop_bottom_frac)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(crop_gray)
        h = phash(crop, hash_size=args.hash_size)

        if prev_hash is None:
            prev_hash = h

        dist = hamming(h, prev_hash)

        if t >= block_end_guard and dist >= args.hash_thresh:
            if pending_change_since is None:
                pending_change_since = t
            if (t - pending_change_since) >= args.min_hold:
                if current_imgs:
                    best_img, best_sharp = max(current_imgs, key=lambda x: x[1])
                else:
                    best_img, best_sharp = crop, sharp
                snippets.append(Snippet(segment_start_t, t, best_img, best_sharp))
                if args.max_segments and len(snippets) >= args.max_segments:
                    break
                segment_start_t = t
                current_imgs = [(crop, sharp)]
                prev_hash = h
                pending_change_since = None
                block_end_guard = t + args.min_gap
        else:
            pending_change_since = None
            current_imgs.append((crop, sharp))
            if dist > 0 and dist < args.hash_thresh * 0.5:
                prev_hash = h

        t += t_per_sample
        frame_idx += 1

    if current_imgs:
        best_img, best_sharp = max(current_imgs, key=lambda x: x[1])
        snippets.append(Snippet(segment_start_t, t, best_img, best_sharp))

    if not snippets:
        print("No snippets detected. Try lowering --hash-thresh or increasing --sample-fps.", file=sys.stderr)
        sys.exit(1)

    print(f"Detected {len(snippets)} snippets. Stitching...")
    stitched = stitch_vertical([s.best_img for s in snippets], pad=16)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, stitched)
    print(f"Saved: {args.out}")

    # Delete video if downloaded and --keep not set
    if args.url and not args.keep:
        try:
            os.remove(video_path)
            print(f"Deleted downloaded video: {video_path}")
        except Exception:
            print(f"Could not delete video: {video_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
