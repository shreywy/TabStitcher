# tab_extractor_gui.py
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import re
import threading
import time
import datetime
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash
from typing import List, Tuple, Optional

@dataclass
class Snippet:
    start_t: float
    end_t: float
    best_img: np.ndarray
    best_sharpness: float

class TabExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Guitar Tab Extractor")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Store script directory for output
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_input_tab()
        self.create_settings_tab()
        self.create_crop_tab()
        self.create_output_tab()
        
        # Create progress area
        self.create_progress_area()
        
        # Initialize variables
        self.video_path = ""
        self.crop_coords = {"x1": 0, "y1": 0, "x2": 1, "y2": 1}
        self.crop_rect = None
        self.drawing = False
        self.crop_enabled = False
        self.is_downloaded = False

    def create_input_tab(self):
        # Input tab
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="Input")
        
        # Video source selection
        source_frame = ttk.LabelFrame(input_frame, text="Video Source")
        source_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # YouTube URL
        ttk.Label(source_frame, text="YouTube URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.url_entry = ttk.Entry(source_frame, width=50)
        self.url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        self.url_entry.insert(0, "https://www.youtube.com/watch?v=")
        
        # Local file selection
        ttk.Label(source_frame, text="Local Video File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.file_entry = ttk.Entry(source_frame, width=50)
        self.file_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(source_frame, text="Browse...", command=self.browse_file).grid(row=1, column=2, padx=5, pady=5)
        
        # Keep video checkbox
        self.keep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(source_frame, text="Keep downloaded video", variable=self.keep_var).grid(
            row=2, column=0, columnspan=3, padx=5, pady=5, sticky="w"
        )

    def create_settings_tab(self):
        # Settings tab
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Sampling settings
        sampling_frame = ttk.LabelFrame(settings_frame, text="Sampling Settings")
        sampling_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(sampling_frame, text="Sample Rate (FPS):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sample_fps = tk.DoubleVar(value=3.0)
        ttk.Scale(sampling_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.sample_fps, 
                  command=lambda v: self.sample_fps_label.config(text=f"{float(v):.1f}")).grid(
                      row=0, column=1, padx=5, pady=5, sticky="we")
        self.sample_fps_label = ttk.Label(sampling_frame, text="3.0")
        self.sample_fps_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Hash settings
        hash_frame = ttk.LabelFrame(settings_frame, text="Comparison Settings")
        hash_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(hash_frame, text="Hash Size:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.hash_size = tk.IntVar(value=16)
        ttk.Combobox(hash_frame, textvariable=self.hash_size, values=[8, 16, 32], width=5).grid(
            row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(hash_frame, text="Hash Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.hash_thresh = tk.IntVar(value=10)
        ttk.Scale(hash_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.hash_thresh, 
                  command=lambda v: self.hash_thresh_label.config(text=f"{int(float(v))}")).grid(
                      row=1, column=1, padx=5, pady=5, sticky="we")
        self.hash_thresh_label = ttk.Label(hash_frame, text="10")
        self.hash_thresh_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Timing settings
        timing_frame = ttk.LabelFrame(settings_frame, text="Timing Settings")
        timing_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(timing_frame, text="Min Hold Time (s):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.min_hold = tk.DoubleVar(value=0.7)
        ttk.Scale(timing_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, variable=self.min_hold, 
                  command=lambda v: self.min_hold_label.config(text=f"{float(v):.1f}")).grid(
                      row=0, column=1, padx=5, pady=5, sticky="we")
        self.min_hold_label = ttk.Label(timing_frame, text="0.7")
        self.min_hold_label.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(timing_frame, text="Min Gap Time (s):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.min_gap = tk.DoubleVar(value=0.4)
        ttk.Scale(timing_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.min_gap, 
                  command=lambda v: self.min_gap_label.config(text=f"{float(v):.1f}")).grid(
                      row=1, column=1, padx=5, pady=5, sticky="we")
        self.min_gap_label = ttk.Label(timing_frame, text="0.4")
        self.min_gap_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Segment settings
        segment_frame = ttk.LabelFrame(settings_frame, text="Segment Settings")
        segment_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(segment_frame, text="Max Segments:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.max_segments = tk.IntVar(value=0)
        ttk.Spinbox(segment_frame, from_=0, to=100, textvariable=self.max_segments, width=5).grid(
            row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(segment_frame, text="(0 = unlimited)").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # Bottom crop fraction
        bottom_frame = ttk.LabelFrame(settings_frame, text="Bottom Crop")
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(bottom_frame, text="Bottom Fraction:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.bottom_frac = tk.DoubleVar(value=0.33)
        ttk.Scale(bottom_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, variable=self.bottom_frac, 
                  command=lambda v: self.bottom_frac_label.config(text=f"{float(v):.2f}")).grid(
                      row=0, column=1, padx=5, pady=5, sticky="we")
        self.bottom_frac_label = ttk.Label(bottom_frame, text="0.33")
        self.bottom_frac_label.grid(row=0, column=2, padx=5, pady=5)

    def create_crop_tab(self):
        # Crop tab
        crop_frame = ttk.Frame(self.notebook)
        self.notebook.add(crop_frame, text="Crop")
        
        # Crop selection
        crop_sel_frame = ttk.LabelFrame(crop_frame, text="Crop Selection")
        crop_sel_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Enable crop checkbox
        self.crop_enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(crop_sel_frame, text="Enable Custom Crop", variable=self.crop_enable_var).pack(
            anchor=tk.W, padx=5, pady=5)
        
        # Crop coordinates
        coord_frame = ttk.Frame(crop_sel_frame)
        coord_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(coord_frame, text="X:").grid(row=0, column=0, padx=5, pady=5)
        self.x_var = tk.StringVar(value="0")
        ttk.Entry(coord_frame, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(coord_frame, text="Y:").grid(row=0, column=2, padx=5, pady=5)
        self.y_var = tk.StringVar(value="0")
        ttk.Entry(coord_frame, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(coord_frame, text="Width:").grid(row=0, column=4, padx=5, pady=5)
        self.w_var = tk.StringVar(value="100%")
        ttk.Entry(coord_frame, textvariable=self.w_var, width=8).grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Label(coord_frame, text="Height:").grid(row=0, column=6, padx=5, pady=5)
        self.h_var = tk.StringVar(value="100%")
        ttk.Entry(coord_frame, textvariable=self.h_var, width=8).grid(row=0, column=7, padx=5, pady=5)
        
        # Percent labels
        ttk.Label(coord_frame, text="(pixels or %)", font=("Arial", 9)).grid(row=0, column=8, padx=5, pady=5)
        
        # Help text
        help_frame = ttk.Frame(crop_sel_frame)
        help_frame.pack(fill=tk.X, padx=10, pady=5)
        
        help_text = "Enter values in pixels or percentages. Examples: '100', '50%'"
        ttk.Label(help_frame, text=help_text, font=("Arial", 9)).pack(side=tk.LEFT)

    def create_output_tab(self):
        # Output tab - now just shows info since path is auto-generated
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="Output")
        
        # Output information
        info_frame = ttk.LabelFrame(output_frame, text="Output Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_text = ("Output will be automatically saved in the same directory as this script.\n"
                    "For local files: [video_name]_tabs.png\n"
                    "For YouTube videos: youtube_[video_id]_tabs.png")
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor="w")
        
        # Show the current output path
        self.output_path_var = tk.StringVar(value="No output path generated yet")
        ttk.Label(info_frame, text="Current output path:").pack(padx=10, pady=(20, 5), anchor="w")
        ttk.Label(info_frame, textvariable=self.output_path_var, wraplength=600).pack(padx=10, pady=5, anchor="w")

    def create_progress_area(self):
        # Progress area at bottom
        progress_frame = ttk.Frame(self.root)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Run button
        self.run_btn = ttk.Button(progress_frame, text="Extract Tabs", command=self.run_extraction, width=15)
        self.run_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Console output
        console_frame = ttk.LabelFrame(self.root, text="Processing Log")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=8)
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console.configure(state='disabled')
        
        # Add text with timestamp
        self.log_message("Application started")

    def browse_file(self):
        filetypes = (
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
        filename = filedialog.askopenfilename(title="Select video file", filetypes=filetypes)
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.url_entry.delete(0, tk.END)
            self.video_path = filename
            self.update_output_path()

    def sanitize_filename(self, name: str) -> str:
        """Clean filename to remove special characters"""
        return re.sub(r"[^A-Za-z0-9 \-_]", "", name).strip() or "output"

    def update_output_path(self):
        """Generate output path based on input source"""
        if self.file_entry.get():
            # For local files
            base = os.path.splitext(os.path.basename(self.file_entry.get()))[0]
            clean_name = self.sanitize_filename(base)
            output_path = os.path.join(self.script_dir, f"{clean_name}_tabs.png")
        elif self.url_entry.get() and "youtube.com/watch?v=" in self.url_entry.get():
            # For YouTube URLs
            try:
                video_id = self.url_entry.get().split("v=")[1].split("&")[0]
                output_path = os.path.join(self.script_dir, f"youtube_{video_id}_tabs.png")
            except Exception:
                output_path = os.path.join(self.script_dir, "youtube_tabs.png")
        else:
            # Default name if we can't determine source
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.script_dir, f"tabs_{timestamp}.png")
        
        self.output_path_var.set(output_path)
        return output_path

    def log_message(self, message):
        self.console.configure(state='normal')
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.console.configure(state='disabled')

    def variance_of_laplacian(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def phash(self, img: np.ndarray, hash_size: int = 16) -> imagehash.ImageHash:
        """Compute perceptual hash for an image"""
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("L")
        return imagehash.phash(pil, hash_size=hash_size)

    def crop_region(self, frame: np.ndarray, crop: Optional[Tuple[int, int, int, int]], bottom_frac: float) -> np.ndarray:
        """Crop the specified region from frame"""
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

    def stitch_vertical(self, images: List[np.ndarray], pad: int = 12) -> np.ndarray:
        """Stitch images vertically with padding"""
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

    def download_video(self, url: str) -> Tuple[str, str]:
        """Download video to script directory"""
        try:
            import yt_dlp
        except Exception:
            raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")

        ydl_opts = {
            "outtmpl": os.path.join(self.script_dir, "%(title)s.%(ext)s"),
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

    def parse_crop(self, s: str) -> Tuple[int, int, int, int]:
        """Parse crop parameters from string"""
        parts = [int(p.strip()) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError("Crop must be 'x,y,w,h'")
        return tuple(parts)

    def run_extraction(self):
        # Validate inputs
        url = self.url_entry.get().strip()
        file_path = self.file_entry.get().strip()
        
        if not url and not file_path:
            messagebox.showerror("Error", "Please enter a YouTube URL or select a local video file")
            return
            
        if url and file_path:
            messagebox.showerror("Error", "Please select only one video source (URL or file)")
            return
            
        # Get automatically generated output path
        output_path = self.update_output_path()
        
        # Disable run button during processing
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        self.progress["value"] = 0
        self.log_message("Starting tab extraction...")
        self.log_message(f"Output will be saved to: {output_path}")
        
        # Run in a separate thread
        threading.Thread(
            target=self.process_video, 
            args=(url, file_path, output_path),
            daemon=True
        ).start()

    def process_video(self, url: str, file_path: str, output_path: str):
        """Main processing function that runs in a separate thread"""
        try:
            video_path = ""
            title = ""
            is_downloaded = False
            
            try:
                if url:
                    self.log_message("⏳ Downloading video...")
                    video_path, title = self.download_video(url)
                    is_downloaded = True
                    self.log_message(f"✅ Downloaded: {os.path.basename(video_path)}")
                else:
                    video_path = file_path
                    if not os.path.exists(video_path):
                        self.log_message(f"❌ Video not found: {video_path}")
                        return
                    title = os.path.splitext(os.path.basename(video_path))[0]
                    self.log_message(f"📹 Processing local video: {title}")
            except Exception as e:
                self.log_message(f"❌ Error: {str(e)}")
                return

            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log_message("❌ Failed to open video.")
                return

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = total_frames / fps if total_frames > 0 else None
            
            if duration:
                self.log_message(f"🎬 Video: {fps:.1f} FPS | {total_frames} frames | {duration:.1f} seconds")
            else:
                self.log_message(f"🎬 Video: {fps:.1f} FPS")

            # Calculate frame skipping
            step = max(1, int(round(fps / self.sample_fps.get())))
            t_per_sample = step / fps
            estimated_samples = total_frames // step
            self.log_message(f"🔍 Sampling every {step} frames ({self.sample_fps.get():.1f}/sec) | ~{estimated_samples} samples")

            # Parse crop coordinates if enabled
            crop = None
            if self.crop_enable_var.get():
                try:
                    crop_str = f"{self.x_var.get()},{self.y_var.get()},{self.w_var.get()},{self.h_var.get()}"
                    crop = self.parse_crop(crop_str)
                    self.log_message(f"🔲 Using crop region: {crop}")
                except Exception as e:
                    self.log_message(f"⚠️ Error parsing crop: {str(e)} - Using default bottom fraction")
                    crop = None

            # Initialize processing variables
            prev_hash = None
            pending_change_since = None
            block_end_guard = 0.0
            t = 0.0
            frame_idx = 0
            processed_count = 0

            current_imgs = []  # List[Tuple[np.ndarray, float]]
            snippets = []  # List[Snippet]
            segment_start_t = 0.0
            filtered_snippets = []  # List[Snippet]
            last_stitched_frame = None

            self.log_message("⏳ Processing video...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames according to sampling rate
                if frame_idx % step != 0:
                    frame_idx += 1
                    continue
                    
                # Process frame
                crop_img = self.crop_region(frame, crop, self.bottom_frac.get())
                crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                sharp = self.variance_of_laplacian(crop_gray)
                h = self.phash(crop_img, hash_size=self.hash_size.get())

                if prev_hash is None:
                    prev_hash = h

                # Calculate Hamming distance
                dist = (h - prev_hash)

                # Detect segment changes
                if t >= block_end_guard and dist >= self.hash_thresh.get():
                    if pending_change_since is None:
                        pending_change_since = t
                    if (t - pending_change_since) >= self.min_hold.get():
                        if current_imgs:
                            best_img, best_sharp = max(current_imgs, key=lambda x: x[1])
                        else:
                            best_img, best_sharp = crop_img, sharp
                            
                        snippets.append(Snippet(segment_start_t, t, best_img, best_sharp))
                        
                        # Check segment limit
                        if self.max_segments.get() and len(snippets) >= self.max_segments.get():
                            self.log_message(f"ℹ️ Reached max segments ({self.max_segments.get()}), stopping early")
                            break
                            
                        # Reset for next segment
                        segment_start_t = t
                        current_imgs = [(crop_img, sharp)]
                        prev_hash = h
                        pending_change_since = None
                        block_end_guard = t + self.min_gap.get()
                else:
                    pending_change_since = None
                    current_imgs.append((crop_img, sharp))
                    if dist > 0 and dist < self.hash_thresh.get() * 0.5:
                        prev_hash = h

                t += t_per_sample
                frame_idx += 1
                processed_count += 1
                
                # Update progress
                if processed_count % 10 == 0 and total_frames > 0:
                    progress = (frame_idx / total_frames) * 100
                    self.progress["value"] = progress
                    self.status_var.set(f"Processing: {progress:.1f}% | Segments: {len(snippets)}")
                    self.root.update_idletasks()

            # Finalize last segment
            if current_imgs:
                best_img, best_sharp = max(current_imgs, key=lambda x: x[1])
                snippets.append(Snippet(segment_start_t, t, best_img, best_sharp))

            if not snippets:
                self.log_message("❌ No snippets detected. Try lowering Hash Threshold or increasing Sample Rate.")
                return

            self.log_message(f"✅ Detected {len(snippets)} potential tab segments")

            # Apply deduplication
            self.log_message("🔍 Applying deduplication...")
            for snippet in snippets:
                # Skip deduplication for the first segment
                if last_stitched_frame is None:
                    filtered_snippets.append(snippet)
                    last_stitched_frame = snippet.best_img
                    continue
                    
                # Compare using perceptual hash
                try:
                    hash1 = self.phash(snippet.best_img)
                    hash2 = self.phash(last_stitched_frame)
                    if abs(hash1 - hash2) > self.hash_thresh.get():
                        filtered_snippets.append(snippet)
                        last_stitched_frame = snippet.best_img
                except Exception as e:
                    self.log_message(f"⚠️ Deduplication error: {str(e)} - Keeping segment")
                    filtered_snippets.append(snippet)
                    last_stitched_frame = snippet.best_img

            if not filtered_snippets:
                self.log_message("❌ After deduplication, no snippets remain. Try lowering Hash Threshold.")
                return

            self.log_message(f"🎯 Keeping {len(filtered_snippets)} distinct segments after deduplication")
            self.log_message("🧵 Stitching segments together...")
            
            # Get only the best images from filtered snippets
            images_to_stitch = [s.best_img for s in filtered_snippets]
            stitched = self.stitch_vertical(images_to_stitch, pad=16)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            cv2.imwrite(output_path, stitched)
            self.log_message(f"💾 Saved tab sheet to: {output_path}")

            # Delete video if downloaded and keep not set
            if is_downloaded and not self.keep_var.get():
                try:
                    os.remove(video_path)
                    self.log_message(f"🗑️ Deleted downloaded video: {video_path}")
                except Exception as e:
                    self.log_message(f"⚠️ Could not delete video: {str(e)}")
            elif is_downloaded and self.keep_var.get():
                self.log_message(f"💾 Keeping downloaded video: {video_path}")

            self.status_var.set("Extraction completed successfully")
            self.log_message("✅ Tab extraction completed successfully!")
            self.progress["value"] = 100

        except Exception as e:
            self.log_message(f"❌ Error during processing: {str(e)}")
            self.status_var.set("Error occurred")
        finally:
            # Re-enable run button
            self.run_btn.config(state=tk.NORMAL)
            if 'cap' in locals():
                cap.release()

def install_dependencies():
    """Install required Python packages if missing"""
    required = ["opencv-python", "numpy", "Pillow", "imagehash", "yt-dlp", "tk"]
    
    import importlib.util
    import subprocess
    import sys
    
    missing = []
    for package in required:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
    
    if missing:
        print("Installing missing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def main():
    # First check and install dependencies
    install_dependencies()
    
    # Create GUI
    root = tk.Tk()
    app = TabExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()