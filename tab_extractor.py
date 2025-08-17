#!/usr/bin/env python3

"""
Advanced Tab Snippet Extractor with Segment-based Deduplication
---------------------------------------------------------------
- Downloads video to script directory
- Auto-generates output filename
- Auto-deletes downloaded videos unless --keep is specified
- Optimized for performance
- Improved segment-based deduplication
"""
import argparse
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
    """Compute perceptual hash for an image (optimized)"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageOps.exif_transpose(pil)
    pil = pil.convert("L")
    return imagehash.phash(pil, hash_size=hash_size)


def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    return (a - b)


def stitch_vertical(images: List[np.ndarray], pad: int = 12) -> np.ndarray:
    """Stitch images vertically with padding (optimized)"""
    if not images:
        raise ValueError("No images to stitch.")
    
    # Find maximum width
    target_w = max(img.shape[1] for img in images)
    
    # Resize images to target width
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if w != target_w:
            scale = target_w / w
            new_h = int(h * scale)
            img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
        resized.append(img)
    
    # Create padding once and reuse
    pad_img = np.full((pad, target_w, 3), 255, dtype=np.uint8)
    
    # Stack images with padding
    rows = []
    for i, img in enumerate(resized):
        rows.append(img)
        if i < len(resized) - 1:
            rows.append(pad_img)
            
    return np.vstack(rows)


def sanitize_filename(name: str) -> str:
    """Clean filename to remove special characters"""
    return re.sub(r"[^A-Za-z0-9 \-_]", "", name).strip() or "output"


def download_video(url: str, out_dir: str) -> Tuple[str, str]:
    """Download video to out_dir. Returns (filepath, video_title)."""
    try:
        import yt_dlp
    except Exception:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")

    ydl_opts = {
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        if not os.path.exists(filepath):
            # Try adding .mp4 extension
            mp4_path = filepath + ".mp4"
            if os.path.exists(mp4_path):
                filepath = mp4_path
        return filepath, info.get("title", "video")


def parse_crop(s: str) -> Tuple[int, int, int, int]:
    """Parse crop parameters from string"""
    parts = [int(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be 'x,y,w,h'")
    return tuple(parts)


def create_comparison_region(frame: np.ndarray) -> np.ndarray:
    """Create the comparison region by removing top 15% and right 25%"""
    h, w = frame.shape[:2]
    # Remove top 15%
    top_crop = int(h * 0.15)
    # Remove right 25%
    right_crop = int(w * 0.25)
    return frame[top_crop:, :-right_crop]


def split_into_fifths(frame: np.ndarray) -> List[np.ndarray]:
    """Split frame into 5 horizontal segments (optimized)"""
    h, w = frame.shape[:2]
    segment_width = w // 5
    segments = []
    for i in range(5):
        start_x = i * segment_width
        end_x = (i + 1) * segment_width if i < 4 else w
        segments.append(frame[:, start_x:end_x])
    return segments


def segments_are_similar(seg1: np.ndarray, seg2: np.ndarray, hash_thresh: int) -> bool:
    """Check if two segments are similar using perceptual hash (optimized)"""
    try:
        hash1 = phash(seg1)
        hash2 = phash(seg2)
        return hamming(hash1, hash2) <= hash_thresh
    except:
        # Fallback to pixel difference if hashing fails
        if seg1.shape != seg2.shape:
            return False
        diff = cv2.absdiff(seg1, seg2)
        return np.mean(diff) < 10


def should_stitch(current_frame: np.ndarray, last_stitched_frame: np.ndarray, hash_thresh: int) -> bool:
    """
    Determine if we should stitch current frame by comparing segments.
    Only stitch if at least 3 segments are different (meaning 2 or fewer are the same).
    """
    if last_stitched_frame is None:
        return True  # Always stitch first frame
    
    # Create comparison regions
    current_comp = create_comparison_region(current_frame)
    last_comp = create_comparison_region(last_stitched_frame)
    
    # Split into fifths
    current_segments = split_into_fifths(current_comp)
    last_segments = split_into_fifths(last_comp)
    
    # Count similar segments
    similar_count = 0
    for i in range(5):
        if segments_are_similar(current_segments[i], last_segments[i], hash_thresh):
            similar_count += 1
    
    # Only stitch if 3 or more segments are different (meaning 2 or fewer are the same)
    return similar_count < 3


def main():
    ap = argparse.ArgumentParser(
        description="Extract and stitch guitar tab snippets from YouTube/local videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL to download & process")
    src.add_argument("--video", help="Path to a local video file to process")

    ap.add_argument("--out", default=None, help="Output PNG path (default: video title - YYYYMMDD.png)")
    ap.add_argument("--keep", action="store_true", help="Keep downloaded video file")
    ap.add_argument("--sample-fps", type=float, default=3.0, 
                   help="Frame sampling rate (higher = more frames, slower)")
    ap.add_argument("--hash-size", type=int, default=16, 
                   help="Perceptual hash size (higher = more precise, slower)")
    ap.add_argument("--hash-thresh", type=int, default=10, 
                   help="Hash difference threshold (lower = stricter)")
    ap.add_argument("--min-hold", type=float, default=0.7, 
                   help="Minimum duration to hold a segment (seconds)")
    ap.add_argument("--min-gap", type=float, default=0.4, 
                   help="Minimum gap between segments (seconds)")
    ap.add_argument("--crop", type=parse_crop, default=None, 
                   help="Manual crop region 'x,y,width,height'")
    ap.add_argument("--crop-bottom-frac", type=float, default=1/3, 
                   help="Bottom fraction to automatically crop")
    ap.add_argument("--max-segments", type=int, default=0, 
                   help="Maximum segments to extract (0 = unlimited)")

    args = ap.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Obtain video path
    video_path = ""
    title = ""
    is_downloaded = False
    
    try:
        if args.url:
            print(f"⏳ Downloading video...")
            video_path, title = download_video(args.url, script_dir)
            is_downloaded = True
            print(f"✅ Downloaded: {os.path.basename(video_path)}")
        else:
            video_path = args.video
            if not os.path.exists(video_path):
                print(f"❌ Video not found: {video_path}", file=sys.stderr)
                sys.exit(2)
            title = os.path.splitext(os.path.basename(video_path))[0]
            print(f"📹 Processing local video: {title}")
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(2)

    # Default output filename if not specified
    if args.out is None:
        clean_title = sanitize_filename(title)
        date_str = datetime.now().strftime("%Y%m%d")
        args.out = os.path.join(script_dir, f"{clean_title} - {date_str}.png")
        print(f"📄 Output will be saved to: {args.out}")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.", file=sys.stderr)
        sys.exit(2)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames > 0 else None
    
    if duration:
        print(f"🎬 Video: {fps:.1f} FPS | {total_frames} frames | {duration:.1f} seconds")
    else:
        print(f"🎬 Video: {fps:.1f} FPS")

    # Calculate frame skipping
    step = max(1, int(round(fps / args.sample_fps)))
    t_per_sample = step / fps
    estimated_samples = total_frames // step
    print(f"🔍 Sampling every {step} frames ({args.sample_fps:.1f}/sec) | ~{estimated_samples} samples")

    # Initialize processing variables
    prev_hash = None
    pending_change_since = None
    block_end_guard = 0.0
    t = 0.0
    frame_idx = 0
    processed_count = 0

    current_imgs: List[Tuple[np.ndarray, float]] = []
    snippets: List[Snippet] = []
    segment_start_t = 0.0
    last_stitched_frame = None
    filtered_snippets: List[Snippet] = []

    print("⏳ Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames according to sampling rate
        if frame_idx % step != 0:
            frame_idx += 1
            continue
            
        # Process frame
        crop = crop_region(frame, args.crop, args.crop_bottom_frac)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = variance_of_laplacian(crop_gray)
        h = phash(crop, hash_size=args.hash_size)

        if prev_hash is None:
            prev_hash = h

        dist = hamming(h, prev_hash)

        # Detect segment changes
        if t >= block_end_guard and dist >= args.hash_thresh:
            if pending_change_since is None:
                pending_change_since = t
            if (t - pending_change_since) >= args.min_hold:
                if current_imgs:
                    best_img, best_sharp = max(current_imgs, key=lambda x: x[1])
                else:
                    best_img, best_sharp = crop, sharp
                    
                snippets.append(Snippet(segment_start_t, t, best_img, best_sharp))
                
                # Check segment limit
                if args.max_segments and len(snippets) >= args.max_segments:
                    print(f"ℹ️ Reached max segments ({args.max_segments}), stopping early")
                    break
                    
                # Reset for next segment
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
        processed_count += 1
        
        # Show progress
        if processed_count % 10 == 0:
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"📊 Progress: {progress:.1f}% | Segments: {len(snippets)}", end='\r')

    # Finalize last segment
    if current_imgs:
        best_img, best_sharp = max(current_imgs, key=lambda x: x[1])
        snippets.append(Snippet(segment_start_t, t, best_img, best_sharp))

    if not snippets:
        print("❌ No snippets detected. Try lowering --hash-thresh or increasing --sample-fps.", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Detected {len(snippets)} potential tab segments")

    # Apply segment-based deduplication
    print("🔍 Applying deduplication...")
    for snippet in snippets:
        if should_stitch(snippet.best_img, last_stitched_frame, args.hash_thresh):
            filtered_snippets.append(snippet)
            last_stitched_frame = snippet.best_img

    if not filtered_snippets:
        print("❌ After deduplication, no snippets remain. Try lowering --hash-thresh.", file=sys.stderr)
        sys.exit(1)

    print(f"🎯 Keeping {len(filtered_snippets)} distinct segments after deduplication")
    print("🧵 Stitching segments together...")
    
    # Get only the best images from filtered snippets
    images_to_stitch = [s.best_img for s in filtered_snippets]
    stitched = stitch_vertical(images_to_stitch, pad=16)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, stitched)
    print(f"💾 Saved tab sheet to: {args.out}")

    # Delete video if downloaded and --keep not set
    if is_downloaded and not args.keep:
        try:
            os.remove(video_path)
            print(f"🗑️ Deleted downloaded video: {video_path}")
        except Exception as e:
            print(f"⚠️ Could not delete video: {str(e)}", file=sys.stderr)
    elif is_downloaded and args.keep:
        print(f"💾 Keeping downloaded video: {video_path}")


if __name__ == "__main__":
    main()