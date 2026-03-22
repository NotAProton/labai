"""
Batch Image Crop & Annotate Tool
---------------------------------
Requirements: pip install pillow

Usage: python image_annotator.py
Then open a folder of images from the File menu or toolbar.

Tools:
  - Crop:      Drag to select a region, then click Save to save the cropped version.
  - Rectangle: Drag to draw a red rectangle annotation overlay.
  - Save:      Saves the current image (with annotations) to  <original_dir>/annotated/
  - Next/Prev: Navigate between images in the folder.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk
import os
import glob

SUPPORTED = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff", "*.webp")

RED = "#FF3333"
HANDLE_COLOR = "#FFD700"


class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Image Annotator")
        self.root.configure(bg="#1e1e2e")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)

        # State
        self.image_paths: list[str] = []
        self.current_index: int = 0
        self.original_image: Image.Image | None = None   # pristine PIL image
        self.working_image: Image.Image | None = None    # with annotations applied
        self.annotations: list[tuple] = []               # list of ("rect", x1,y1,x2,y2)
        self.crop_rect: tuple | None = None              # pending crop selection
        self.mode: str = "none"                          # "crop" | "rect" | "none"
        self.drag_start: tuple | None = None
        self.current_rubber_band = None

        # Canvas scale (display vs actual pixels)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self._build_ui()
        self._bind_events()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        # ── Top toolbar ──────────────────────────────────────────────────────
        toolbar = tk.Frame(self.root, bg="#2a2a3e", pady=6, padx=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_cfg = dict(
            bg="#3c3c5c", fg="white", activebackground="#5a5a8a",
            activeforeground="white", relief=tk.FLAT, padx=14, pady=5,
            font=("Segoe UI", 10, "bold"), cursor="hand2", bd=0
        )
        sep_cfg = dict(bg="#444466", width=2)

        tk.Button(toolbar, text="Open Folder", command=self.open_folder, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Frame(toolbar, **sep_cfg).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        self.btn_crop = tk.Button(toolbar, text="Crop", command=self.activate_crop, **btn_cfg)
        self.btn_crop.pack(side=tk.LEFT, padx=4)

        self.btn_rect = tk.Button(toolbar, text="Rectangle", command=self.activate_rect, **btn_cfg)
        self.btn_rect.pack(side=tk.LEFT, padx=4)

        tk.Button(toolbar, text="Undo", command=self.undo, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Clear All", command=self.clear_annotations, **btn_cfg).pack(side=tk.LEFT, padx=4)

        tk.Frame(toolbar, **sep_cfg).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        tk.Button(toolbar, text="Save", command=self.save_image,
                  bg="#2e7d32", fg="white", activebackground="#43a047",
                  activeforeground="white", relief=tk.FLAT, padx=14, pady=5,
                  font=("Segoe UI", 10, "bold"), cursor="hand2", bd=0).pack(side=tk.LEFT, padx=4)

        tk.Frame(toolbar, **sep_cfg).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        tk.Button(toolbar, text="Prev", command=self.prev_image, **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Next", command=self.next_image, **btn_cfg).pack(side=tk.LEFT, padx=4)

        # Mode indicator
        self.mode_label = tk.Label(toolbar, text="Mode: None", bg="#2a2a3e",
                                   fg="#aaaacc", font=("Segoe UI", 10))
        self.mode_label.pack(side=tk.LEFT, padx=12)

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Open a folder to begin.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bg="#13131f",
                              fg="#88889a", anchor=tk.W, padx=10,
                              font=("Segoe UI", 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Image counter ────────────────────────────────────────────────────
        self.counter_var = tk.StringVar(value="-")
        counter = tk.Label(self.root, textvariable=self.counter_var, bg="#13131f",
                           fg="#ccccdd", font=("Segoe UI", 9), padx=10)
        counter.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Canvas ───────────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg="#1e1e2e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.canvas = tk.Canvas(canvas_frame, bg="#12121e", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Configure>", self.on_resize)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Escape>", lambda e: self.deactivate_tool())

    # ---------------------------------------------------------- Folder open --
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        paths = []
        for pattern in SUPPORTED:
            paths.extend(glob.glob(os.path.join(folder, pattern)))
            paths.extend(glob.glob(os.path.join(folder, pattern.upper())))
        paths = sorted(set(paths))
        if not paths:
            messagebox.showwarning("No images", "No supported images found in that folder.")
            return
        self.image_paths = paths
        self.current_index = 0
        self.load_current_image()

    # -------------------------------------------------------- Image loading --
    def load_current_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        self.original_image = Image.open(path).convert("RGBA")
        self.working_image = self.original_image.copy()
        self.annotations.clear()
        self.crop_rect = None
        self.current_rubber_band = None
        self.mode = "none"
        self._update_mode_label()
        self.counter_var.set(f"Image {self.current_index + 1} / {len(self.image_paths)}")
        self.status_var.set(f"Loaded: {os.path.basename(path)}  "
                            f"({self.original_image.width}x{self.original_image.height})")
        self.render_canvas()

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_current_image()

    # ------------------------------------------------------- Tool activation --
    def activate_crop(self):
        self.mode = "crop"
        self._update_mode_label()
        self.canvas.config(cursor="cross")

    def activate_rect(self):
        self.mode = "rect"
        self._update_mode_label()
        self.canvas.config(cursor="crosshair")

    def deactivate_tool(self):
        self.mode = "none"
        self._update_mode_label()
        self.canvas.config(cursor="arrow")
        self._clear_rubber_band()

    def _update_mode_label(self):
        labels = {"crop": "Crop", "rect": "Rectangle", "none": "None"}
        colors = {"crop": "#FFD700", "rect": "#FF4444", "none": "#aaaacc"}
        m = self.mode
        self.mode_label.config(text=f"Mode: {labels[m]}", fg=colors[m])
        # Highlight active button
        self.btn_crop.config(bg="#5a5a8a" if m == "crop" else "#3c3c5c")
        self.btn_rect.config(bg="#6a2222" if m == "rect" else "#3c3c5c")

    # ------------------------------------------------------- Mouse handling --
    def on_mouse_press(self, event):
        if self.mode == "none" or self.original_image is None:
            return
        self.drag_start = (event.x, event.y)
        self._clear_rubber_band()

    def on_mouse_drag(self, event):
        if not self.drag_start or self.mode == "none":
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self._clear_rubber_band()
        color = HANDLE_COLOR if self.mode == "crop" else RED
        self.current_rubber_band = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=color, width=2, dash=(4, 2))

    def on_mouse_release(self, event):
        if not self.drag_start or self.mode == "none" or self.original_image is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None
        self._clear_rubber_band()

        # Convert canvas coords → image coords
        ix0 = int((min(x0, x1) - self.offset_x) / self.scale_x)
        iy0 = int((min(y0, y1) - self.offset_y) / self.scale_y)
        ix1 = int((max(x0, x1) - self.offset_x) / self.scale_x)
        iy1 = int((max(y0, y1) - self.offset_y) / self.scale_y)

        # Clamp to image bounds
        W, H = self.original_image.size
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(W, ix1), min(H, iy1)

        if ix1 - ix0 < 4 or iy1 - iy0 < 4:
            return  # too small, ignore

        if self.mode == "crop":
            self.crop_rect = (ix0, iy0, ix1, iy1)
            self.status_var.set(f"Crop selection: ({ix0},{iy0}) -> ({ix1},{iy1}) - click Save to apply")
            self.render_canvas()
        elif self.mode == "rect":
            self.annotations.append(("rect", ix0, iy0, ix1, iy1))
            self.apply_annotations()
            self.render_canvas()

    def _clear_rubber_band(self):
        if self.current_rubber_band:
            self.canvas.delete(self.current_rubber_band)
            self.current_rubber_band = None

    # --------------------------------------------------- Annotation / render --
    def apply_annotations(self):
        """Re-draw all rect annotations onto working_image."""
        base = self.original_image.copy()
        draw = ImageDraw.Draw(base)
        for ann in self.annotations:
            if ann[0] == "rect":
                _, x0, y0, x1, y1 = ann
                for offset in range(3):   # thick border
                    draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                                   outline=(255, 50, 50, 255))
        self.working_image = base

    def render_canvas(self):
        if self.working_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return

        img = self.working_image
        # Overlay pending crop rect if any
        if self.crop_rect:
            overlay = img.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(list(self.crop_rect), outline=(255, 215, 0, 255), width=2)
            img = overlay

        # Fit to canvas
        iw, ih = img.size
        ratio = min(canvas_w / iw, canvas_h / ih)
        dw, dh = int(iw * ratio), int(ih * ratio)
        self.scale_x = ratio
        self.scale_y = ratio
        self.offset_x = (canvas_w - dw) // 2
        self.offset_y = (canvas_h - dh) // 2

        resized = img.resize((dw, dh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y,
                                 anchor=tk.NW, image=self._tk_image)

    def on_resize(self, event):
        self.render_canvas()

    # ----------------------------------------------------------------- Save --
    def save_image(self):
        if self.original_image is None:
            messagebox.showwarning("Nothing to save", "Please open a folder first.")
            return

        src_path = self.image_paths[self.current_index]

        save_img = self.working_image.copy()

        if self.crop_rect:
            save_img = save_img.crop(self.crop_rect)

        # Convert to RGB for JPEG saving; keep RGBA for PNG
        ext = os.path.splitext(src_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            save_img = save_img.convert("RGB")
            save_img.save(src_path, quality=95)
        else:
            save_img.save(src_path)

        # Reload saved image as the new baseline and clear pending state
        self.original_image = Image.open(src_path).convert("RGBA")
        self.working_image = self.original_image.copy()
        self.annotations.clear()
        self.crop_rect = None
        self.render_canvas()

        self.status_var.set(f"Saved: {src_path}")

    # --------------------------------------------------------- Undo / Clear --
    def undo(self):
        if self.crop_rect:
            self.crop_rect = None
            self.status_var.set("Crop selection cleared.")
            self.render_canvas()
        elif self.annotations:
            self.annotations.pop()
            self.apply_annotations()
            self.render_canvas()
            self.status_var.set(f"Undone. {len(self.annotations)} annotation(s) remain.")

    def clear_annotations(self):
        self.annotations.clear()
        self.crop_rect = None
        self.apply_annotations()
        self.render_canvas()
        self.status_var.set("All annotations cleared.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()
