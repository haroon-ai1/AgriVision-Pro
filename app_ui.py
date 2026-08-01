"""Desktop GUI (CustomTkinter).

Fixes carried over from v1:
  * ``state("zoomed")`` is Windows-only and raised TclError elsewhere; the
    window is now maximised per-platform with a graceful fallback.
  * The placeholder advertised drag-and-drop that was never implemented. It is
    implemented now when ``tkinterdnd2`` is installed, and the text honestly
    reflects what is available when it is not.
  * A missing checkpoint used to fail silently at load and only surface when the
    user clicked Run. It is now reported at startup with instructions.
  * Inference ran on the UI thread behind a blocking ``after(500)`` sleep, which
    froze the window. It now runs on a worker thread and posts back safely.
  * Only the top-1 class was shown. Runners-up and an explicit low-confidence
    warning are now displayed.
  * Disabled buttons kept their full accent fill and still looked clickable.
    Enabled and disabled states are now visually distinct.

v1's cursor-tracking spotlight background is preserved. It is built after the
window is mapped rather than during __init__, because the Gaussian blur costs
roughly 250 ms and doing it inline delays first paint by that much.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageTk

_SRC = Path(__file__).resolve().parent / "src"
if not (_SRC / "agrivision" / "__init__.py").exists():
    raise SystemExit(
        f"Cannot find the agrivision package at:\n    {_SRC}\n\n"
        "app_ui.py must sit beside the src/ folder. Expected layout:\n"
        "    AgriVision-Pro/\n"
        "        app_ui.py\n"
        "        src/agrivision/__init__.py\n\n"
        f"Currently running from: {Path(__file__).resolve().parent}"
    )
sys.path.insert(0, str(_SRC))

from agrivision.predict import Predictor  # noqa: E402

# Optional: real drag-and-drop support.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

CHECKPOINT = "artifacts/best_model.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ctk.set_appearance_mode("Dark")

COLOR_BG_DEEP = "#050505"
COLOR_PANEL_BG = "#0f0f12"
COLOR_ACCENT = "#6a5cff"
COLOR_ACCENT_HOVER = "#8275ff"
COLOR_ACCENT_DISABLED = "#2a2740"   # muted accent: clearly not clickable
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_SUB = "#a1a1aa"
COLOR_TEXT_DISABLED = "#5b5b66"
COLOR_BORDER = "#27272a"
COLOR_OK = "#4ade80"
COLOR_WARN = "#fbbf24"
COLOR_ERROR = "#ef4444"


class AgriVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AgriVision Pro")
        self.geometry("1200x800")
        self.minsize(900, 620)
        self.configure(fg_color=COLOR_BG_DEEP)

        self.predictor: Predictor | None = None
        self.current_image_path: str | None = None
        self.tk_image = None
        self._result_queue: queue.Queue = queue.Queue()

        # Spotlight state
        self.spotlight_image = None
        self.spotlight_id = None
        self.last_mouse_move = time.time()
        self.last_frame_time = 0.0
        self.idle_angle = 0.0

        self._build_background()
        self._build_ui()

        self.after(60, self._maximize)
        self.after(90, self._build_spotlight)
        self.after(150, self._load_model)
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # Background spotlight
    # ------------------------------------------------------------------
    def _build_background(self) -> None:
        """A Canvas, because moving an item inside one is far smoother than
        repositioning a Label widget on every mouse event."""
        self.bg_canvas = tk.Canvas(self, bg=COLOR_BG_DEEP, highlightthickness=0, bd=0)
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _build_spotlight(self) -> None:
        """Deferred: the blur costs ~250 ms and would otherwise delay first paint."""
        try:
            size = 1400
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            center = size // 2
            max_radius = size // 2
            rgb = ImageColor.getrgb(COLOR_ACCENT)

            for radius in range(max_radius, 0, -15):
                alpha = int(180 * (radius / max_radius))
                draw.ellipse(
                    (center - radius, center - radius, center + radius, center + radius),
                    fill=(*rgb, alpha),
                )

            image = image.filter(ImageFilter.GaussianBlur(120))
            self.spotlight_image = ImageTk.PhotoImage(image)
            self.spotlight_id = self.bg_canvas.create_image(
                -2000, -2000, image=self.spotlight_image, anchor="center"
            )

            self.bind("<Motion>", self._on_mouse_move)
            self.after(50, self._idle_float)
        except Exception as exc:
            # A missing glow is cosmetic; never let it take the app down.
            print(f"[gui] Spotlight unavailable: {exc}")

    def _on_mouse_move(self, event) -> None:
        """Throttled to ~60 FPS so a flood of motion events cannot choke the UI."""
        now = time.time()
        if now - self.last_frame_time < 0.015:
            return
        self.last_frame_time = now
        self.last_mouse_move = now

        if self.spotlight_id is not None:
            # Motion events bubble up from child widgets, so event.x/y are
            # relative to whichever widget was under the cursor. Convert through
            # screen coordinates to get a position the canvas can use.
            x = event.x_root - self.winfo_rootx()
            y = event.y_root - self.winfo_rooty()
            self.bg_canvas.coords(self.spotlight_id, x, y)

    def _idle_float(self) -> None:
        """Drifts the glow in a slow circle once the cursor has been still."""
        if self.spotlight_id is not None and time.time() - self.last_mouse_move > 1.0:
            self.idle_angle += 0.02
            radius = 50
            cx = self.winfo_width() // 2
            cy = self.winfo_height() // 2
            self.bg_canvas.coords(
                self.spotlight_id,
                cx + math.cos(self.idle_angle) * radius,
                cy + math.sin(self.idle_angle) * radius,
            )
        self.after(30, self._idle_float)

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------
    def _maximize(self) -> None:
        """Maximise across Windows, macOS and Linux without crashing on any."""
        try:
            if sys.platform.startswith("win"):
                self.state("zoomed")
            else:
                try:
                    self.attributes("-zoomed", True)
                except tk.TclError:
                    # Some window managers expose neither; size to the screen.
                    self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
        except tk.TclError:
            pass  # A non-maximised window is fine; a crash is not.

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Generous padding so the spotlight stays visible around the panels.
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=55, pady=45)

        self.frame_top = ctk.CTkFrame(
            container, fg_color=COLOR_PANEL_BG, corner_radius=22,
            border_width=1, border_color=COLOR_BORDER,
        )
        self.frame_top.pack(side="top", fill="both", expand=True, pady=(0, 18))

        placeholder = (
            "DROP AN IMAGE HERE\nOR CLICK OPEN IMAGE"
            if DND_AVAILABLE
            else "CLICK OPEN IMAGE TO BEGIN\n(install tkinterdnd2 for drag-and-drop)"
        )
        self.lbl_image = tk.Label(
            self.frame_top, text=placeholder, bg=COLOR_PANEL_BG, fg="#555",
            font=("Segoe UI", 18, "bold"), justify="center",
        )
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")

        if DND_AVAILABLE:
            self._register_drop_target()

        self.frame_bottom = ctk.CTkFrame(
            container, fg_color=COLOR_PANEL_BG, corner_radius=22,
            border_width=1, border_color=COLOR_BORDER, height=280,
        )
        self.frame_bottom.pack(side="bottom", fill="x")
        self.frame_bottom.pack_propagate(False)

        self.lbl_result = ctk.CTkLabel(
            self.frame_bottom, text="SYSTEM READY",
            font=("Segoe UI", 28, "bold"), text_color=COLOR_TEXT_MAIN,
        )
        self.lbl_result.pack(pady=(22, 4))

        self.lbl_conf = ctk.CTkLabel(
            self.frame_bottom, text="Waiting for image input...",
            font=("Segoe UI", 13), text_color=COLOR_TEXT_SUB,
        )
        self.lbl_conf.pack(pady=(0, 2))

        self.lbl_runners = ctk.CTkLabel(
            self.frame_bottom, text="", font=("Segoe UI", 11),
            text_color="#6b7280", justify="center",
        )
        self.lbl_runners.pack(pady=(0, 14))

        self.progress = ctk.CTkProgressBar(
            self.frame_bottom, mode="indeterminate", height=3,
            progress_color=COLOR_ACCENT, fg_color=COLOR_PANEL_BG,
        )

        btn_group = ctk.CTkFrame(self.frame_bottom, fg_color="transparent")
        btn_group.pack(fill="x", padx=80, pady=(0, 18))

        self.btn_upload = ctk.CTkButton(
            btn_group, text="OPEN IMAGE", font=("Segoe UI", 13, "bold"),
            height=50, corner_radius=10, fg_color="#222225", hover_color="#333336",
            border_width=1, border_color=COLOR_BORDER, command=self.upload_image,
        )
        self.btn_upload.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.btn_run = ctk.CTkButton(
            btn_group, text="RUN DIAGNOSIS", font=("Segoe UI", 13, "bold"),
            height=50, corner_radius=10, command=self.run_diagnosis,
        )
        self.btn_run.pack(side="right", fill="x", expand=True, padx=(12, 0))
        self._set_run_enabled(False)

        self.lbl_status = ctk.CTkLabel(
            self.frame_bottom, text="Loading model...",
            font=("Segoe UI", 10), text_color="#444",
        )
        self.lbl_status.pack(side="bottom", pady=(0, 8))

    def _set_run_enabled(self, enabled: bool) -> None:
        """CustomTkinter keeps the accent fill when a button is disabled, so a
        dead control still reads as clickable. Recolour it explicitly."""
        if enabled:
            self.btn_run.configure(
                state="normal", fg_color=COLOR_ACCENT,
                hover_color=COLOR_ACCENT_HOVER, text_color="white",
            )
        else:
            self.btn_run.configure(
                state="disabled", fg_color=COLOR_ACCENT_DISABLED,
                hover_color=COLOR_ACCENT_DISABLED, text_color=COLOR_TEXT_DISABLED,
            )

    def _register_drop_target(self) -> None:
        try:
            self.lbl_image.drop_target_register(DND_FILES)
            self.lbl_image.dnd_bind("<<Drop>>", self._on_drop)
        except Exception as exc:
            print(f"[gui] Drag-and-drop unavailable: {exc}")

    def _on_drop(self, event) -> None:
        # Paths arrive brace-wrapped when they contain spaces.
        path = event.data.strip().strip("{}").split("} {")[0]
        if Path(path).suffix.lower() not in IMAGE_EXTENSIONS:
            messagebox.showwarning("Unsupported file", f"Not an image file:\n{path}")
            return
        self._set_image(path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        def worker():
            try:
                self._result_queue.put(("model_ok", Predictor(CHECKPOINT, device="cpu")))
            except Exception as exc:
                self._result_queue.put(("model_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------
    def upload_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a leaf image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if path:
            self._set_image(path)

    def _set_image(self, path: str) -> None:
        try:
            pil_img = Image.open(path)
            pil_img.load()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image:\n{exc}")
            return

        self.current_image_path = path
        self.update_idletasks()

        w = max(self.frame_top.winfo_width() - 40, 200)
        h = max(self.frame_top.winfo_height() - 40, 200)
        ratio = min(w / pil_img.width, h / pil_img.height, 1.0)
        size = (max(int(pil_img.width * ratio), 1), max(int(pil_img.height * ratio), 1))

        display = pil_img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        self.tk_image = ctk.CTkImage(light_image=display, dark_image=display, size=size)

        self.lbl_image.destroy()
        self.lbl_image = ctk.CTkLabel(self.frame_top, image=self.tk_image, text="")
        self.lbl_image.place(relx=0.5, rely=0.5, anchor="center")
        if DND_AVAILABLE:
            self._register_drop_target()

        self.lbl_result.configure(text="IMAGE LOADED", text_color=COLOR_TEXT_MAIN)
        self.lbl_conf.configure(text=Path(path).name)
        self.lbl_runners.configure(text="")

        if self.predictor is not None:
            self._set_run_enabled(True)

    # ------------------------------------------------------------------
    # Inference (worker thread)
    # ------------------------------------------------------------------
    def run_diagnosis(self) -> None:
        if self.predictor is None or not self.current_image_path:
            return

        self.lbl_result.configure(text="ANALYZING...", text_color=COLOR_ACCENT)
        self.lbl_conf.configure(text="")
        self.lbl_runners.configure(text="")
        self._set_run_enabled(False)
        self.btn_upload.configure(state="disabled")
        self.progress.pack(fill="x", padx=80, pady=(0, 6))
        self.progress.start()

        path = self.current_image_path

        def worker():
            try:
                self._result_queue.put(("prediction", self.predictor.predict(path, top_k=3)))
            except Exception as exc:
                self._result_queue.put(("prediction_error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_queue(self) -> None:
        """Poll for worker results. Tk widgets may only be touched from this thread."""
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()

                if kind == "model_ok":
                    self.predictor = payload
                    n = len(payload.class_names)
                    self.lbl_status.configure(
                        text=f"{payload.cfg.backbone} · {n} classes · ready",
                        text_color="#444",
                    )
                    if self.current_image_path:
                        self._set_run_enabled(True)

                elif kind == "model_error":
                    self.lbl_status.configure(text="Model not loaded", text_color=COLOR_ERROR)
                    self.lbl_result.configure(text="NO MODEL", text_color=COLOR_ERROR)
                    self.lbl_conf.configure(text="Train a model before running diagnosis")
                    messagebox.showerror(
                        "Model not found",
                        f"{payload}\n\nTrain one with:\n"
                        f"  python -m agrivision.train --data-root dataset/PlantVillage",
                    )

                elif kind == "prediction":
                    self._show_prediction(payload)

                elif kind == "prediction_error":
                    self._finish_run()
                    self.lbl_result.configure(text="ERROR", text_color=COLOR_ERROR)
                    self.lbl_conf.configure(text=str(payload)[:90])

        except queue.Empty:
            pass

        self.after(100, self._drain_queue)

    def _show_prediction(self, result: dict) -> None:
        self._finish_run()
        top = result["top"]

        if result["uncertain"]:
            colour = COLOR_WARN
            note = "  ·  LOW CONFIDENCE — verify manually"
        elif top["healthy"]:
            colour = COLOR_OK
            note = ""
        else:
            colour = COLOR_ACCENT
            note = ""

        self.lbl_result.configure(text=top["label"].upper(), text_color=colour)
        self.lbl_conf.configure(text=f"CONFIDENCE {top['confidence'] * 100:.2f}%{note}")

        runners = result["predictions"][1:]
        if runners:
            text = "   ".join(f"{p['label']} {p['confidence'] * 100:.1f}%" for p in runners)
            self.lbl_runners.configure(text=f"Also considered:  {text}")

    def _finish_run(self) -> None:
        self.progress.stop()
        self.progress.pack_forget()
        self._set_run_enabled(True)
        self.btn_upload.configure(state="normal")


def main() -> None:
    if DND_AVAILABLE:
        # tkinterdnd2 needs its Tk subclass mixed into the root window.
        class DnDApp(AgriVisionApp, TkinterDnD.DnDWrapper):
            def __init__(self):
                super().__init__()
                self.TkdndVersion = TkinterDnD._require(self)

        app = DnDApp()
    else:
        app = AgriVisionApp()

    app.mainloop()


if __name__ == "__main__":
    main()
