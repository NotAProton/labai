"""
box_adjuster.py — Interactive bounding box adjustment GUI.

Shows the cropped image with AI-detected boxes overlaid, allowing:
  • Move box     : click inside box + drag
  • Resize box   : drag any corner handle
  • Delete box   : right-click on box  (or hover + Delete key)
  • New box      : click + drag on empty canvas area

Bottom panel shows annotation_recommendations for context.
"Confirm" button (top) finalises boxes and unblocks the pipeline.
"Cancel" returns the original AI-detected boxes unchanged.
"""
from __future__ import annotations

import sys
import tkinter as tk
from PIL import Image, ImageTk
from typing import Optional

# ── Visual constants ───────────────────────────────────────────────────────
HANDLE_RADIUS = 8       # px — hit-detection radius for corner handles
HANDLE_HALF   = 5       # px — half-size of the drawn handle square
BOX_COLOR     = "#FF3333"
BOX_HOVER_CLR = "#FF8888"
HANDLE_FILL   = "#FFD700"
HANDLE_OUTLINE= "#A08030"
NEW_BOX_COLOR = "#44FF88"
MIN_BOX_PX    = 4       # minimum box side (px in image space) to keep on release

# cursors – standard X11 / Tk names that work on Linux + Windows
_CUR_DEFAULT   = "crosshair"
_CUR_MOVE      = "fleur"
_CUR_RES_TL    = "top_left_corner"
_CUR_RES_TR    = "top_right_corner"
_CUR_RES_BL    = "bottom_left_corner"
_CUR_RES_BR    = "bottom_right_corner"

_RESIZE_CURSORS = {
    "resize_tl": _CUR_RES_TL,
    "resize_tr": _CUR_RES_TR,
    "resize_bl": _CUR_RES_BL,
    "resize_br": _CUR_RES_BR,
}


# ─────────────────────────────────────────────────────────────────────────────
class BoxAdjuster:
    """
    Tkinter GUI for interactive bounding box adjustment.

    Usage::

        adjuster = BoxAdjuster(cropped_image, initial_boxes, recommendations)
        final_boxes = adjuster.run()   # blocks until Confirm / Cancel

    Returns original boxes if the user cancels or if the display is unavailable.
    """

    def __init__(
        self,
        cropped_image: Image.Image,
        initial_boxes: list[list[int]],
        annotation_recommendations: list[str],
    ) -> None:
        self.original_image = cropped_image.copy()
        self._original_boxes: list[list[int]] = [list(b) for b in initial_boxes]
        self.boxes: list[list[int]] = [list(b) for b in initial_boxes]
        self.annotation_recommendations = annotation_recommendations

        # ── Drag state ─────────────────────────────────────────────────────
        self.drag_mode: Optional[str] = None           # "move" | "resize_tl/tr/bl/br" | "create"
        self.drag_box_index: Optional[int] = None
        self.drag_start_canvas: Optional[tuple[int, int]] = None
        self.drag_start_box: Optional[list[int]] = None  # box snapshot at drag start
        self.new_box_origin_img: Optional[tuple[int, int]] = None  # image coords of press

        # ── Hover state ────────────────────────────────────────────────────
        self.hover_box_index: Optional[int] = None
        self.hover_part: Optional[str] = None           # "move" | "resize_tl" | …

        # ── Canvas display transform ───────────────────────────────────────
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._tk_image = None  # keep reference to prevent GC

        self._build_root()
        self._build_ui()
        self._bind_events()

    # ── Root window ────────────────────────────────────────────────────────

    def _build_root(self) -> None:
        self.root = tk.Tk()
        self.root.title("Manual Bounding Box Adjustment")
        self.root.configure(bg="#1e1e2e")

        img_w, img_h = self.original_image.size
        # Leave room for toolbar (~52px) + rec panel (~28 + 22*n) + status (~22px)
        rec_h = 32 + 22 * max(1, len(self.annotation_recommendations))
        win_w = max(820, min(img_w + 56, 1500))
        win_h = max(580, min(img_h + rec_h + 80, 980))
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(700, 480)

    # ── UI layout ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        btn_base = dict(
            bg="#3c3c5c", fg="white",
            activebackground="#5a5a8a", activeforeground="white",
            relief=tk.FLAT, padx=14, pady=5,
            font=("Segoe UI", 10, "bold"), cursor="hand2", bd=0,
        )

        # ── Toolbar (TOP) ─────────────────────────────────────────────────
        toolbar = tk.Frame(self.root, bg="#2a2a3e", pady=6, padx=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            toolbar, text="✓  Confirm", command=self._confirm,
            bg="#2e7d32", fg="white",
            activebackground="#43a047", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=6,
            font=("Segoe UI", 11, "bold"), cursor="hand2", bd=0,
        ).pack(side=tk.LEFT, padx=(4, 8))

        tk.Frame(toolbar, bg="#444466", width=2).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2
        )

        tk.Button(
            toolbar, text="Cancel", command=self._cancel, **btn_base
        ).pack(side=tk.LEFT, padx=4)

        tk.Frame(toolbar, bg="#444466", width=2).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2
        )

        self.box_count_var = tk.StringVar()
        self._refresh_box_count()
        tk.Label(
            toolbar, textvariable=self.box_count_var,
            bg="#2a2a3e", fg="#aaaacc", font=("Segoe UI", 10),
        ).pack(side=tk.LEFT, padx=12)

        instr = (
            "Drag inside box: move   •   Drag corner handle: resize   "
            "•   Right-click / Delete key: remove   •   Drag empty area: new box"
        )
        tk.Label(
            toolbar, text=instr, bg="#2a2a3e", fg="#555577",
            font=("Segoe UI", 8),
        ).pack(side=tk.RIGHT, padx=10)

        # Pack BOTTOM widgets BEFORE the fill+expand canvas so Tkinter
        # respects their position below it.

        # ── Status bar (BOTTOM) ───────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            self.root, textvariable=self.status_var,
            bg="#0d0d1a", fg="#88889a", anchor=tk.W, padx=10,
            font=("Segoe UI", 9),
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # ── Annotation recommendations (BOTTOM, above status) ─────────────
        rec_frame = tk.Frame(self.root, bg="#161626", pady=5, padx=12)
        rec_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            rec_frame, text="Annotation targets:",
            bg="#161626", fg="#6e6e88",
            font=("Segoe UI", 8, "bold"), anchor=tk.W,
        ).pack(anchor=tk.W)

        if self.annotation_recommendations:
            for i, rec in enumerate(self.annotation_recommendations, 1):
                tk.Label(
                    rec_frame, text=f"  {i}.  {rec}",
                    bg="#161626", fg="#aaaacc",
                    font=("Segoe UI", 9), anchor=tk.W,
                    wraplength=1200, justify=tk.LEFT,
                ).pack(anchor=tk.W)
        else:
            tk.Label(
                rec_frame, text="  (no annotation targets)",
                bg="#161626", fg="#555566",
                font=("Segoe UI", 9, "italic"), anchor=tk.W,
            ).pack(anchor=tk.W)

        # ── Canvas (fills remaining space) ────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg="#1e1e2e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        self.canvas = tk.Canvas(
            canvas_frame, bg="#12121e",
            cursor=_CUR_DEFAULT, highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ── Event bindings ─────────────────────────────────────────────────────

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",        self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",  self._on_release)
        self.canvas.bind("<ButtonPress-3>",    self._on_right_click)
        self.canvas.bind("<Motion>",           self._on_motion)
        self.canvas.bind("<Configure>",        lambda _e: self._render_canvas())

        self.root.bind("<Return>",             lambda _e: self._confirm())
        self.root.bind("<Escape>",             lambda _e: self._cancel())
        self.root.bind("<Delete>",             lambda _e: self._delete_hovered())
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

    # ── Coordinate transforms ──────────────────────────────────────────────

    def _to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.scale_x + self.offset_x, iy * self.scale_y + self.offset_y

    def _to_img(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.offset_x) / self.scale_x, (cy - self.offset_y) / self.scale_y

    def _clamp_img(self, ix: float, iy: float) -> tuple[int, int]:
        W, H = self.original_image.size
        return int(max(0, min(W, ix))), int(max(0, min(H, iy)))

    def _clamp_box(self, box: list[int]) -> list[int]:
        """Normalise and clamp a box to image bounds."""
        W, H = self.original_image.size
        l, t, r, b = box
        l, r = sorted([max(0, min(W, l)), max(0, min(W, r))])
        t, b = sorted([max(0, min(H, t)), max(0, min(H, b))])
        return [l, t, r, b]

    # ── Hit detection ──────────────────────────────────────────────────────

    def _handle_canvas_positions(self, box: list[int]) -> dict[str, tuple[float, float]]:
        l, t, r, b = box
        return {
            "resize_tl": self._to_canvas(l, t),
            "resize_tr": self._to_canvas(r, t),
            "resize_bl": self._to_canvas(l, b),
            "resize_br": self._to_canvas(r, b),
        }

    def _hit_test(self, cx: int, cy: int) -> tuple[Optional[int], Optional[str]]:
        """
        Return (box_index, part) where part is one of:
          "resize_tl", "resize_tr", "resize_bl", "resize_br", "move"
        Returns (None, None) when the cursor is over empty canvas.

        Corner handles are tested first (higher priority), iterating
        front-to-back (last box drawn is on top).
        """
        # Handles first — iterate reverse so visually top box wins
        for ridx in range(len(self.boxes) - 1, -1, -1):
            for part, (hx, hy) in self._handle_canvas_positions(self.boxes[ridx]).items():
                if abs(cx - hx) <= HANDLE_RADIUS and abs(cy - hy) <= HANDLE_RADIUS:
                    return ridx, part

        # Box interiors
        for ridx in range(len(self.boxes) - 1, -1, -1):
            l, t, r, b = self.boxes[ridx]
            cl, ct = self._to_canvas(l, t)
            cr, cb = self._to_canvas(r, b)
            if cl <= cx <= cr and ct <= cy <= cb:
                return ridx, "move"

        return None, None

    # ── Mouse events ───────────────────────────────────────────────────────

    def _on_press(self, event: tk.Event) -> None:
        cx, cy = event.x, event.y
        box_idx, part = self._hit_test(cx, cy)

        if box_idx is not None and part is not None:
            # Interact with an existing box
            self.drag_mode = part
            self.drag_box_index = box_idx
            self.drag_start_canvas = (cx, cy)
            self.drag_start_box = list(self.boxes[box_idx])
        else:
            # Start creating a new box
            ix, iy = self._to_img(cx, cy)
            ox, oy = self._clamp_img(ix, iy)
            self.drag_mode = "create"
            self.new_box_origin_img = (ox, oy)
            self.boxes.append([ox, oy, ox, oy])
            self.drag_box_index = len(self.boxes) - 1
            self.drag_start_canvas = (cx, cy)

    def _on_drag(self, event: tk.Event) -> None:
        if self.drag_mode is None or self.drag_box_index is None:
            return

        cx, cy = event.x, event.y

        if self.drag_mode == "create":
            sx, sy = self.new_box_origin_img
            tx, ty = self._clamp_img(*self._to_img(cx, cy))
            self.boxes[self.drag_box_index] = [
                min(sx, tx), min(sy, ty),
                max(sx, tx), max(sy, ty),
            ]

        elif self.drag_mode == "move":
            # Translate delta from canvas pixels → image pixels
            dx_c = cx - self.drag_start_canvas[0]
            dy_c = cy - self.drag_start_canvas[1]
            dl = dx_c / self.scale_x
            dt = dy_c / self.scale_y
            sl, st, sr, sb = self.drag_start_box
            bw, bh = sr - sl, sb - st
            W, H = self.original_image.size
            nl = max(0, min(W - bw, sl + dl))
            nt = max(0, min(H - bh, st + dt))
            self.boxes[self.drag_box_index] = [
                int(nl), int(nt), int(nl + bw), int(nt + bh)
            ]

        else:
            # Resize: one corner anchored, opposite moves with cursor
            sl, st, sr, sb = self.drag_start_box
            tx, ty = self._clamp_img(*self._to_img(cx, cy))
            mode = self.drag_mode
            if mode == "resize_tl":
                self.boxes[self.drag_box_index] = [tx, ty, sr, sb]
            elif mode == "resize_tr":
                self.boxes[self.drag_box_index] = [sl, ty, tx, sb]
            elif mode == "resize_bl":
                self.boxes[self.drag_box_index] = [tx, st, sr, ty]
            elif mode == "resize_br":
                self.boxes[self.drag_box_index] = [sl, st, tx, ty]

        self._render_canvas()

    def _on_release(self, event: tk.Event) -> None:
        if self.drag_mode is None or self.drag_box_index is None:
            return

        box = self.boxes[self.drag_box_index]
        l, t, r, b = box
        # Normalise so l < r, t < b after resize drags
        l, r = sorted([l, r])
        t, b = sorted([t, b])
        box = [l, t, r, b]

        if (r - l) < MIN_BOX_PX or (b - t) < MIN_BOX_PX:
            # Too small — discard
            self.boxes.pop(self.drag_box_index)
            if self.drag_mode == "create":
                self.status_var.set("Box too small — discarded.")
        else:
            self.boxes[self.drag_box_index] = box

        self.drag_mode = None
        self.drag_box_index = None
        self.drag_start_canvas = None
        self.drag_start_box = None
        self.new_box_origin_img = None

        self._refresh_box_count()
        self._render_canvas()

    def _on_right_click(self, event: tk.Event) -> None:
        box_idx, _ = self._hit_test(event.x, event.y)
        if box_idx is not None:
            self.boxes.pop(box_idx)
            if self.hover_box_index == box_idx:
                self.hover_box_index = None
                self.hover_part = None
            self._refresh_box_count()
            self._render_canvas()
            self.status_var.set(
                f"Box deleted.  {len(self.boxes)} box(es) remaining."
            )

    def _on_motion(self, event: tk.Event) -> None:
        if self.drag_mode is not None:
            return  # mid-drag: cursor already set, skip hover logic

        cx, cy = event.x, event.y
        new_idx, new_part = self._hit_test(cx, cy)

        if new_idx != self.hover_box_index or new_part != self.hover_part:
            self.hover_box_index = new_idx
            self.hover_part = new_part

            if new_part in _RESIZE_CURSORS:
                self.canvas.config(cursor=_RESIZE_CURSORS[new_part])
            elif new_part == "move":
                self.canvas.config(cursor=_CUR_MOVE)
            else:
                self.canvas.config(cursor=_CUR_DEFAULT)

            self._render_canvas()

    def _delete_hovered(self) -> None:
        """Delete the currently hovered box (Delete key shortcut)."""
        if (
            self.hover_box_index is not None
            and self.hover_box_index < len(self.boxes)
        ):
            self.boxes.pop(self.hover_box_index)
            self.hover_box_index = None
            self.hover_part = None
            self._refresh_box_count()
            self._render_canvas()
            self.status_var.set(
                f"Box deleted.  {len(self.boxes)} box(es) remaining."
            )

    # ── Rendering ──────────────────────────────────────────────────────────

    def _render_canvas(self) -> None:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        img = self.original_image
        iw, ih = img.size
        ratio = min(cw / iw, ch / ih)
        dw, dh = int(iw * ratio), int(ih * ratio)

        self.scale_x = ratio
        self.scale_y = ratio
        self.offset_x = (cw - dw) // 2
        self.offset_y = (ch - dh) // 2

        # Resize and display image
        resized = img.convert("RGB").resize((dw, dh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self._tk_image
        )

        # Draw boxes + corner handles
        for i, box in enumerate(self.boxes):
            is_hover = (i == self.hover_box_index)
            color = BOX_HOVER_CLR if is_hover else BOX_COLOR
            width = 3 if is_hover else 2

            cl, ct = self._to_canvas(box[0], box[1])
            cr, cb = self._to_canvas(box[2], box[3])
            self.canvas.create_rectangle(cl, ct, cr, cb, outline=color, width=width)

            # Corner handles
            for _part, (hx, hy) in self._handle_canvas_positions(box).items():
                self.canvas.create_rectangle(
                    hx - HANDLE_HALF, hy - HANDLE_HALF,
                    hx + HANDLE_HALF, hy + HANDLE_HALF,
                    fill=HANDLE_FILL, outline=HANDLE_OUTLINE, width=1,
                )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _refresh_box_count(self) -> None:
        n = len(self.boxes)
        self.box_count_var.set("1 box" if n == 1 else f"{n} boxes")

    # ── Actions ────────────────────────────────────────────────────────────

    def _confirm(self) -> None:
        """Normalise all boxes and exit mainloop."""
        result = []
        for b in self.boxes:
            c = self._clamp_box(b)
            if c[2] - c[0] >= MIN_BOX_PX and c[3] - c[1] >= MIN_BOX_PX:
                result.append(c)
        self.boxes = result
        self.root.quit()

    def _cancel(self) -> None:
        """Revert to original AI boxes and exit mainloop."""
        self.boxes = [list(b) for b in self._original_boxes]
        self.root.quit()

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> list[list[int]]:
        """
        Launch the GUI (blocking).  Returns the confirmed list of boxes.
        On Cancel the original boxes are returned unchanged.
        """
        # Schedule an initial render once the window is mapped
        self.root.after(80, self._render_canvas)
        self.root.lift()
        try:
            self.root.focus_force()
        except tk.TclError:
            pass
        self.root.mainloop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        return self.boxes


# ── Convenience wrapper ────────────────────────────────────────────────────

def show_box_adjuster(
    cropped_image: Image.Image,
    initial_boxes: list[list[int]],
    annotation_recommendations: list[str],
) -> list[list[int]]:
    """
    Launch the manual bounding box adjustment GUI and return confirmed boxes.

    Blocks until the user clicks Confirm or Cancel (or closes the window).
    On Cancel the original *initial_boxes* are returned unchanged.
    Falls back gracefully to *initial_boxes* if a display is unavailable.
    """
    try:
        adjuster = BoxAdjuster(cropped_image, initial_boxes, annotation_recommendations)
        return adjuster.run()
    except Exception as exc:  # noqa: BLE001
        print(
            f"    [bbox/gui] Manual adjustment GUI unavailable: {exc}",
            file=sys.stderr,
        )
        return initial_boxes
