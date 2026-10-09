import tkinter as tk
import cv2
import numpy as np
import os
import time
import urllib.request
import threading
from PIL import Image, ImageTk

try:
    import pyvirtualcam
    VIRTUALCAM_AVAILABLE = True
except ImportError:
    VIRTUALCAM_AVAILABLE = False

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
CAT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "cat.png")
MODEL_PATH     = os.path.join(SCRIPT_DIR, "blaze_face_short_range.tflite")
MODEL_URL      = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

SCALE_FACTOR   = 3.0
Y_OFFSET_RATIO = -0.25
LANDMARK_COLOR = (80, 255, 160)
LANDMARK_RADIUS= 3

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading blaze_face_short_range.tflite...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

def load_cat_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        img = cv2.merge([b, g, r, alpha])
    return img

def rotate_image(img, angle):
    if abs(angle) < 0.5: return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_rgba(frame, overlay, x, y, w, h):
    if w <= 0 or h <= 0: return
    ov = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x2 <= x1 or y2 <= y1: return
    ox1, oy1 = x1 - x, y1 - y
    roi     = frame[y1:y2, x1:x2].astype(np.float32)
    ov_crop = ov[oy1:oy1+(y2-y1), ox1:ox1+(x2-x1)]
    alpha   = ov_crop[:, :, 3:4].astype(np.float32) / 255.0
    color   = ov_crop[:, :, :3].astype(np.float32)
    frame[y1:y2, x1:x2] = (alpha * color + (1 - alpha) * roi).astype(np.uint8)

BG      = "#0d0d0d"
PANEL   = "#141414"
ACCENT  = "#e94560"
FG      = "#e0e0e0"
FG_DIM  = "#555"
BTN_BG  = "#1e1e1e"
BTN_HOV = "#2a2a2a"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Face Tracker (Tasks API Detector)")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self._cap_lock = threading.Lock()
        self._start_ms = int(time.time() * 1000)

        self._build_ui()

        ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=0.4
        )
        self.face_detector = mp_vision.FaceDetector.create_from_options(options)

        self.cap = None
        for idx in range(4):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    ok, _ = c.read()
                    if ok:
                        self.cap = c
                        break
                    c.release()
            if self.cap: break

        if self.cap is None:
            self._set_status("Camera not found")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._set_status("Camera connected")

        self.cat_img = load_cat_rgba(CAT_IMAGE_PATH)
        if self.cat_img is None:
            self._set_status("cat.png not found")
            self.var_cat.set(False)

        self._photo = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self.set_fullscreen(False))

        self.available_cams = {}
        threading.Thread(target=self.scan_cameras, daemon=True).start()
        self.update()

    def _build_ui(self):
        self.is_fullscreen = False
        self.virtualcam_active = False
        self.virtualcam = None

        topbar = tk.Frame(self.root, bg=PANEL, pady=0)
        topbar.pack(fill="x")

        tk.Label(topbar, text="CAT TRACKER", font=("Consolas", 11, "bold"), fg=ACCENT, bg=PANEL).pack(side="left", padx=14, pady=8)

        self.btn_vcam = self._btn(topbar, "VCAM OFF", self.toggle_virtualcam, side="right", padx=10)
        self._btn(topbar, "FULLSCREEN", self.toggle_fullscreen, side="right")

        tk.Label(topbar, text="CAM", bg=PANEL, fg=FG_DIM, font=("Consolas", 9)).pack(side="right", padx=(10, 2), pady=8)
        self.cam_var = tk.StringVar()
        self.cam_combo = tk.OptionMenu(topbar, self.cam_var, "...")
        self.cam_combo.config(bg=BTN_BG, fg=FG, relief="flat", activebackground=BTN_HOV, font=("Consolas", 9), highlightthickness=0, bd=0)
        self.cam_combo["menu"].config(bg=BTN_BG, fg=FG, activebackground=ACCENT)
        self.cam_combo.pack(side="right", pady=8)

        ctrl = tk.Frame(self.root, bg=BG, pady=6)
        ctrl.pack(fill="x", padx=12)

        self.var_cat  = tk.BooleanVar(value=True)
        self.var_pts  = tk.BooleanVar(value=False)
        self.var_flip = tk.BooleanVar(value=True)

        cb_style = dict(bg=BG, fg=FG, activebackground=BG, activeforeground=ACCENT, selectcolor=PANEL, font=("Consolas", 10), bd=0, relief="flat", cursor="hand2")
        tk.Checkbutton(ctrl, text="CAT", variable=self.var_cat, **cb_style).pack(side="left", padx=8)
        tk.Checkbutton(ctrl, text="POINTS", variable=self.var_pts, **cb_style).pack(side="left", padx=8)
        tk.Checkbutton(ctrl, text="MIRROR", variable=self.var_flip, **cb_style).pack(side="left", padx=8)

        tk.Label(ctrl, text="SIZE", bg=BG, fg=FG_DIM, font=("Consolas", 9)).pack(side="left", padx=(16, 4))
        self.scale_size = tk.Scale(ctrl, from_=0.5, to=3.5, resolution=0.05, orient="horizontal", length=120, bg=BG, fg=FG, highlightthickness=0, troughcolor="#1e1e1e", activebackground=ACCENT, showvalue=False, bd=0)
        self.scale_size.set(SCALE_FACTOR)
        self.scale_size.pack(side="left")

        tk.Label(ctrl, text="Y-OFFSET", bg=BG, fg=FG_DIM, font=("Consolas", 9)).pack(side="left", padx=(16, 4))
        self.scale_y = tk.Scale(ctrl, from_=-0.8, to=0.4, resolution=0.05, orient="horizontal", length=100, bg=BG, fg=FG, highlightthickness=0, troughcolor="#1e1e1e", activebackground=ACCENT, showvalue=False, bd=0)
        self.scale_y.set(Y_OFFSET_RATIO)
        self.scale_y.pack(side="left")

        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="#000", highlightthickness=0)
        self.canvas.pack(padx=12, pady=(4, 0))

        statusbar = tk.Frame(self.root, bg=PANEL, pady=4)
        statusbar.pack(fill="x")
        self.status_var = tk.StringVar(value="Init...")
        tk.Label(statusbar, textvariable=self.status_var, bg=PANEL, fg=FG_DIM, font=("Consolas", 9), anchor="w").pack(side="left", padx=12)

    def _btn(self, parent, text, cmd, side="left", padx=4):
        b = tk.Button(parent, text=text, font=("Consolas", 9), bg=BTN_BG, fg=FG, relief="flat", bd=0, activebackground=BTN_HOV, activeforeground=ACCENT, cursor="hand2", padx=10, pady=6, command=cmd)
        b.pack(side=side, padx=padx, pady=6)
        return b

    def _set_status(self, msg):
        self.status_var.set(msg)

    def get_camera_names(self):
        try:
            import subprocess
            out = subprocess.check_output('powershell -NoProfile -Command "Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName"', shell=True, stderr=subprocess.DEVNULL, timeout=5).decode("cp866", errors="ignore")
            return [l.strip() for l in out.strip().splitlines() if l.strip()]
        except Exception:
            return []

    def scan_cameras(self):
        self.root.after(0, lambda: self._set_status("Scanning cameras..."))
        cam_names = self.get_camera_names()
        found = []
        for idx in range(6):
            c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if c.isOpened():
                ok, _ = c.read()
                if ok: found.append(idx)
            c.release()
        result = {}
        for i, idx in enumerate(found):
            label = f"[{idx}] {cam_names[i]}" if i < len(cam_names) else f"Camera {idx}"
            result[label] = idx
        self.root.after(0, lambda r=result: self._apply_cameras(r))

    def _apply_cameras(self, result):
        self.available_cams = result
        if not self.available_cams:
            self._set_status("No cameras found")
            return
        menu = self.cam_combo["menu"]
        menu.delete(0, "end")
        for label in self.available_cams:
            menu.add_command(label=label, command=lambda l=label: self.switch_camera(l))
        first = list(self.available_cams.keys())[0]
        self.cam_var.set(first)
        self.switch_camera(first)
        self._set_status(f"Found {len(self.available_cams)} camera(s)")

    def switch_camera(self, label):
        idx = self.available_cams[label]
        with self._cap_lock:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam_var.set(label)
        self._set_status(f"Switched to {label}")

    def toggle_virtualcam(self):
        if not VIRTUALCAM_AVAILABLE:
            self._set_status("pip install pyvirtualcam")
            return
        if self.virtualcam_active:
            self.virtualcam_active = False
            if self.virtualcam:
                try: self.virtualcam.close()
                except Exception: pass
                self.virtualcam = None
            self.btn_vcam.config(text="VCAM OFF", fg=FG)
            self._set_status("Virtual cam off")
        else:
            for b in [None, "obs", "unitycapture"]:
                try:
                    kwargs = dict(width=640, height=480, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
                    if b: kwargs["backend"] = b
                    self.virtualcam = pyvirtualcam.Camera(**kwargs)
                    self.virtualcam_active = True
                    self.btn_vcam.config(text="VCAM ON", fg=ACCENT)
                    self._set_status(f"Virtual cam on: {self.virtualcam.device}")
                    break
                except Exception:
                    continue
            if not self.virtualcam_active:
                self._set_status("No virtual cam driver")

    def toggle_fullscreen(self):
        self.set_fullscreen(not self.is_fullscreen)

    def set_fullscreen(self, state):
        self.is_fullscreen = state
        self.root.attributes("-fullscreen", state)

    def update(self):
        if self.cap and self.cap.isOpened():
            try:
                with self._cap_lock:
                    ret, frame = self.cap.read()
            except Exception:
                self.root.after(16, self.update)
                return

            if ret:
                if self.var_flip.get():
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_c = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_c)
                
                ts = int(time.time() * 1000) - self._start_ms
                result = self.face_detector.detect_for_video(mp_image, ts)

                faces_info = []
                if result.detections:
                    for detection in result.detections:
                        bbox = detection.bounding_box
                        fx = bbox.origin_x
                        fy = bbox.origin_y
                        fw = bbox.width
                        fh = bbox.height

                        kp = detection.keypoints
                        if kp and len(kp) >= 2:
                            rx, ry = kp[0].x * w, kp[0].y * h
                            lx, ly = kp[1].x * w, kp[1].y * h
                            angle = -np.degrees(np.arctan2(ly - ry, lx - rx))
                        else:
                            angle = 0

                        if self.var_pts.get() and kp:
                            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), LANDMARK_COLOR, 2)
                            for point in kp:
                                px, py = int(point.x * w), int(point.y * h)
                                cv2.circle(frame, (px, py), LANDMARK_RADIUS, (0, 0, 255), -1)

                        faces_info.append((fx, fy, fw, fh, angle))

                if self.var_cat.get() and self.cat_img is not None:
                    sc = self.scale_size.get()
                    yo = self.scale_y.get()
                    for (fx, fy, fw, fh, angle) in faces_info:
                        nw = int(fw * sc)
                        nh = int(fh * sc)
                        nx = fx - (nw - fw) // 2
                        ny = fy - (nh - fh) // 2 + int(fh * yo)
                        overlay_rgba(frame, rotate_image(self.cat_img, angle), nx, ny, nw, nh)

                n = len(faces_info)
                self._set_status(f"Faces: {n}" + (" :)" if n else " - none"))

                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._photo = ImageTk.PhotoImage(image=pil_img)
                self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

                if self.virtualcam_active and self.virtualcam:
                    try:
                        self.virtualcam.send(frame)
                    except Exception:
                        pass

        self.root.after(16, self.update)

    def on_close(self):
        if self.cap: self.cap.release()
        if self.virtualcam:
            try: self.virtualcam.close()
            except Exception: pass
        try: self.face_detector.close()
        except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()