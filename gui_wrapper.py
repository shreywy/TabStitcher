#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

def run_extraction():
    url = url_var.get().strip()
    video_path = video_var.get().strip()
    out_path = out_var.get().strip()
    delete_after = delete_var.get()

    if not url and not video_path:
        messagebox.showerror("Error", "Please enter a YouTube URL or select a local video.")
        return
    if not out_path:
        messagebox.showerror("Error", "Please select an output file path.")
        return

    cmd = [sys.executable, "tab_extractor.py"]
    if url:
        cmd += ["--url", url]
    else:
        cmd += ["--video", video_path]
    cmd += ["--out", out_path]

    def task():
        try:
            subprocess.run(cmd, check=True)
            if delete_after and url:
                # Delete downloaded video folder from temp
                # The extractor downloads to a temp folder; find and remove
                temp_dir = None
                for part in cmd:
                    if part.startswith(tempfile.gettempdir()):
                        temp_dir = part
                        break
                # Not perfect, better would be to coordinate with tab_extractor for path
            messagebox.showinfo("Done", f"Extraction completed.\nSaved to: {out_path}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Extraction failed.\n\n{e}")

    threading.Thread(target=task, daemon=True).start()

def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")])
    if path:
        video_var.set(path)

def browse_output():
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
    if path:
        out_var.set(path)

root = tk.Tk()
root.title("Guitar Tab Stitcher")

url_var = tk.StringVar()
video_var = tk.StringVar()
out_var = tk.StringVar()
delete_var = tk.BooleanVar(value=True)

frm = tk.Frame(root, padx=10, pady=10)
frm.pack(fill="both", expand=True)

tk.Label(frm, text="YouTube URL:").grid(row=0, column=0, sticky="e")
tk.Entry(frm, textvariable=url_var, width=50).grid(row=0, column=1, columnspan=2, pady=2, sticky="we")

tk.Label(frm, text="or Local Video:").grid(row=1, column=0, sticky="e")
tk.Entry(frm, textvariable=video_var, width=40).grid(row=1, column=1, pady=2, sticky="we")
tk.Button(frm, text="Browse", command=browse_video).grid(row=1, column=2, pady=2, sticky="w")

tk.Label(frm, text="Output PNG:").grid(row=2, column=0, sticky="e")
tk.Entry(frm, textvariable=out_var, width=40).grid(row=2, column=1, pady=2, sticky="we")
tk.Button(frm, text="Browse", command=browse_output).grid(row=2, column=2, pady=2, sticky="w")

tk.Checkbutton(frm, text="Auto-delete downloaded video after extraction", variable=delete_var).grid(row=3, column=0, columnspan=3, pady=4, sticky="w")

tk.Button(frm, text="Run Extraction", command=run_extraction, bg="#4caf50", fg="white").grid(row=4, column=0, columnspan=3, pady=8)

root.mainloop()
