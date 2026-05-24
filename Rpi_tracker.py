"""
Tennis Ball Tracker – Raspberry Pi Edition
===========================================
Optimised for ~30 ms end-to-end latency on a Raspberry Pi 3/4/5.

Design choices vs. the generic version
---------------------------------------
* filterpy removed – Kalman math is plain NumPy (faster, no class overhead)
* Processing resolution fixed at 320×240 (configurable) – detection is just
  as reliable and ~4× faster than 640×480
* Spin estimation disabled by default (costs ~8 ms on RPi)
* GPIO output drives a servo or stepper that physically steers the deflector
* Picamera2 is used when available; falls back to V4L2 (USB webcam)

Dependencies
------------
    pip install opencv-python numpy

    # For Raspberry Pi camera module:
    sudo apt install -y python3-picamera2

    # For GPIO / servo:
    pip install RPi.GPIO   (or gpiozero)

Usage
-----
    python rpi_tennis_tracker.py               # RPi camera, no display
    python rpi_tennis_tracker.py --preview     # show OpenCV window (adds ~3ms)
    python rpi_tennis_tracker.py --video t.mp4 # test on a file
    python rpi_tennis_tracker.py --no-gpio     # disable servo output

Controls (preview mode)
    q – quit    r – reset tracker    d – toggle mask window
"""

import argparse
import time
import collections
import sys

import cv2
import numpy as np

# ── Optional GPIO ──────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = False
except ImportError:
    HAS_GPIO = False
    print("[WARN] RPi.GPIO lib not found – GPIO output will be disabled.")

# ── Optional Picamera2 ─────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    HAS_PICAM = True
except ImportError:
    HAS_PICAM = False
    print("[WARN] Picamera2 lib not found – falling back to V4L2 (USB webcam). "
          "Install with: sudo apt install python3-picamera2")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROC_W, PROC_H = 320, 240        # processing resolution
TARGET_FPS     = 60              # request this from camera (actual may differ)
DT             = 1.0 / 60.0     # nominal Kalman timestep (updated at runtime)

# HSV colour range – fluorescent tennis ball
# Press 'd' in preview mode and adjust if the mask looks wrong
HSV_LOWER = np.array([25,  80,  80], dtype=np.uint8)
HSV_UPPER = np.array([75, 255, 255], dtype=np.uint8)

MIN_RADIUS = 6    # px at 320×240
MAX_RADIUS = 80

TRAIL_LEN   = 24  # past positions shown
PREDICT_STEPS = 50  # frames to extrapolate
PANEL_MARGIN  = 10  # px from edge = "panel wall"

# Kalman noise (tune per setup)
Q_STD = 6.0    # process noise std  – higher → snappier, noisier
R_STD = 3.5    # measurement noise  – higher → smoother, more lag

# GPIO / Servo
SERVO_PIN   = 18   # BCM pin for PWM servo signal
SERVO_FREQ  = 50   # Hz
SERVO_MIN   = 2.5  # % duty cycle → leftmost position
SERVO_MAX   = 12.5 # % duty cycle → rightmost position


# ═══════════════════════════════════════════════════════════════════════════
#  LEAN KALMAN FILTER  (pure NumPy, no class overhead per prediction step)
# ═══════════════════════════════════════════════════════════════════════════

def build_kalman_matrices(dt: float):
    """
    State vector: x = [px, py, vx, vy, ax, ay]
    Observation:  z = [px, py]
    Model: constant acceleration (good for rolling / slow-arc ball)
    """
    F = np.array([
        [1, 0, dt, 0,  0.5*dt**2, 0        ],
        [0, 1, 0,  dt, 0,         0.5*dt**2],
        [0, 0, 1,  0,  dt,        0        ],
        [0, 0, 0,  1,  0,         dt       ],
        [0, 0, 0,  0,  1,         0        ],
        [0, 0, 0,  0,  0,         1        ],
    ], dtype=np.float32)

    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float32)

    Q = np.eye(6, dtype=np.float32) * (Q_STD ** 2)
    R = np.eye(2, dtype=np.float32) * (R_STD ** 2)

    return F, H, Q, R


def kalman_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P


def kalman_update(x, P, z, H, R):
    y  = z - H @ x                          # innovation
    S  = H @ P @ H.T + R                    # innovation covariance
    K  = P @ H.T @ np.linalg.inv(S)        # Kalman gain
    x  = x + K @ y
    P  = (np.eye(len(x), dtype=np.float32) - K @ H) @ P
    return x, P


# ═══════════════════════════════════════════════════════════════════════════
#  BALL DETECTION  (HSV masking – runs entirely in C via OpenCV)
# ═══════════════════════════════════════════════════════════════════════════

# Reusable kernel to avoid per-frame allocation
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def detect_ball(frame_bgr):
    """
    Returns (cx, cy, radius_px) in processing-resolution coords, or None.
    Also returns the binary mask for optional debug display.
    """
    blurred = cv2.GaussianBlur(frame_bgr, (7, 7), 0)      # 7×7 is enough at 320×240
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        ((cx, cy), r) = cv2.minEnclosingCircle(c)
        if not (MIN_RADIUS <= r <= MAX_RADIUS):
            continue
        area = cv2.contourArea(c)
        if area / (np.pi * r * r) < 0.40:   # circularity gate
            continue
        return (int(cx), int(cy), int(r)), mask

    return None, mask


# ═══════════════════════════════════════════════════════════════════════════
#  TRAJECTORY EXTRAPOLATION  &  DEFLECTION POINT
# ═══════════════════════════════════════════════════════════════════════════

def extrapolate(x, F, n_steps):
    """Fast forward-simulate the Kalman state n_steps into the future."""
    pts = []
    state = x.copy()
    for _ in range(n_steps):
        state = F @ state
        pts.append((int(state[0]), int(state[1])))
    return pts


def find_wall_intersection(traj, w, h):
    """First predicted point that reaches the panel boundary."""
    for pt in traj:
        px, py = pt
        if px <= PANEL_MARGIN or px >= w - PANEL_MARGIN or \
           py <= PANEL_MARGIN or py >= h - PANEL_MARGIN:
            return pt
    return None


def target_to_servo_duty(target_x, frame_w):
    """Map target X pixel → servo duty cycle (linear)."""
    ratio = np.clip(target_x / frame_w, 0.0, 1.0)
    return SERVO_MIN + ratio * (SERVO_MAX - SERVO_MIN)


# ═══════════════════════════════════════════════════════════════════════════
#  GPIO / SERVO
# ═══════════════════════════════════════════════════════════════════════════

# class ServoController:
#     def __init__(self, pin=SERVO_PIN, freq=SERVO_FREQ):
#         self.enabled = HAS_GPIO
#         if not self.enabled:
#             print("[WARN] RPi.GPIO not found – servo output disabled.")
#             return
#         GPIO.setmode(GPIO.BCM)
#         GPIO.setup(pin, GPIO.OUT)
#         self.pwm = GPIO.PWM(pin, freq)
#         self.pwm.start(7.5)   # neutral
#         self._last_duty = 7.5

#     def set_position(self, duty: float):
#         if not self.enabled:
#             return
#         duty = float(np.clip(duty, SERVO_MIN, SERVO_MAX))
#         if abs(duty - self._last_duty) > 0.05:   # dead-band: avoid jitter
#             self.pwm.ChangeDutyCycle(duty)
#             self._last_duty = duty

#     def cleanup(self):
#         if self.enabled:
#             self.pwm.stop()
#             GPIO.cleanup()


# ═══════════════════════════════════════════════════════════════════════════
#  CAMERA ABSTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class Camera:
    """Wraps Picamera2 or V4L2 with a uniform .read() → frame interface. 
    -Claude (It added this abstraction to allow easy switching between Picamera2 and a USB webcam for testing.) [I h8 OOP]"""

    def __init__(self, source, w=PROC_W, h=PROC_H):
        self._use_picam = False
        if source == "picam" and HAS_PICAM:
            self._init_picam(w, h)
        else:
            self._init_v4l2(source if source != "picam" else 0, w, h)
        self.w, self.h = w, h

    def _init_picam(self, w, h):
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": (w, h), "format": "BGR888"},
            controls={"FrameRate": TARGET_FPS}
        )
        self._cam.configure(cfg)
        self._cam.start()
        self._use_picam = True
        print(f"[INFO] Using Picamera2 at {w}×{h}")

    def _init_v4l2(self, source, w, h):
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
        # V4L2 low-latency: reduce internal buffer to 1 frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[INFO] Using V4L2 source={source} at {w}×{h}")

    def read(self):
        if self._use_picam:
            frame = self._cam.capture_array()
            return True, frame
        return self._cap.read()

    def fps(self):
        if self._use_picam:
            return TARGET_FPS
        f = self._cap.get(cv2.CAP_PROP_FPS)
        return f if f > 0 else 30.0

    def release(self):
        if self._use_picam:
            self._cam.stop()
        else:
            self._cap.release()


# ═══════════════════════════════════════════════════════════════════════════
#  HUD  (only rendered when --preview is active)
# ═══════════════════════════════════════════════════════════════════════════

YELLOW  = (0, 220, 255)
GREEN   = (50, 255, 80)
RED     = (0, 60, 255)
CYAN    = (255, 220, 0)
GREY    = (130, 130, 130)


def draw_hud(frame, x_state, fps, ball_found, duty):
    vx, vy   = x_state[2], x_state[3]
    speed_px = float(np.sqrt(vx**2 + vy**2))
    # rough real-world speed assuming ~0.3 mm/px at typical setup distance
    speed_ms = speed_px * 0.0003 * fps

    lines = [
        f"FPS:   {fps:.1f}",
        f"Speed: {speed_ms:.2f} m/s  ({speed_px:.1f} px/fr)",
        f"Vel:   vx={vx:.1f}  vy={vy:.1f}",
        f"Accel: ax={x_state[4]:.2f}  ay={x_state[5]:.2f}",
        f"Servo: {duty:.1f}% duty" if duty is not None else "Servo: —",
        f"Ball:  {'DETECTED' if ball_found else 'PREDICTING'}",
    ]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (280, 16 + 20 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    col = GREEN if ball_found else RED
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (6, 15 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


def draw_trail(frame, trail):
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = (0, int(255 * alpha), int(80 * alpha))
        cv2.line(frame, pts[i-1], pts[i], c, 2)


def draw_trajectory(frame, traj):
    for i in range(1, len(traj)):
        a = 1.0 - i / len(traj)
        c = (0, int(160 * a), int(255 * a))
        cv2.line(frame, traj[i-1], traj[i], c, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP (may run into issues instantiating IMX500 Camera in the global scope, so it's all wrapped in run())
#  If issue persists, consider moving the Camera class definition and instantiation inside the run() function as well.
#  or contact Marwan for further assistance.
# ═══════════════════════════════════════════════════════════════════════════

def run(args):
    # ── Camera ──
    source = args.video if args.video else ("picam" if args.picam else args.camera)
    cam    = Camera(source)
    actual_fps = cam.fps()
    dt     = 1.0 / actual_fps
    print(f"[INFO] Camera FPS={actual_fps:.1f}  dt={dt*1000:.2f} ms")

    # ── Kalman matrices ──
    F, H, Q, R = build_kalman_matrices(dt)
    x = np.zeros(6, dtype=np.float32)
    P = np.eye(6, dtype=np.float32) * 500.0

    # ── Servo ── (Ignore)
    # servo = ServoController() if (args.gpio and HAS_GPIO) else ServoController.__new__(ServoController)
    # if not args.gpio:
        # servo.enabled = False

    initialized  = False
    frames_lost  = 0
    MAX_LOST     = int(actual_fps * 0.4)   # 400 ms grace period
    trail        = collections.deque(maxlen=TRAIL_LEN)
    show_mask    = args.debug
    last_duty    = None

    # ── FPS counter ──
    fps_val   = actual_fps
    t0        = time.perf_counter()
    fc        = 0

    print("[INFO] Running. Press q to quit.")

    while True:
        # ──────────────────────────────── Capture
        t_frame = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        # Resize only if needed (Picamera2 already delivers at PROC_W×PROC_H)
        if frame.shape[1] != PROC_W or frame.shape[0] != PROC_H:
            frame = cv2.resize(frame, (PROC_W, PROC_H),
                               interpolation=cv2.INTER_LINEAR)

        # ──────────────────────────────── Detection
        detection, mask = detect_ball(frame)

        if detection is not None:
            cx, cy, r = detection
            frames_lost = 0

            if not initialized:
                x = np.array([cx, cy, 0, 0, 0, 0], dtype=np.float32)
                P = np.eye(6, dtype=np.float32) * 500.0
                initialized = True
            else:
                x, P = kalman_predict(x, P, F, Q)
                z    = np.array([cx, cy], dtype=np.float32)
                x, P = kalman_update(x, P, z, H, R)

            trail.append((int(x[0]), int(x[1])))

            if args.preview:
                cv2.circle(frame, (cx, cy), r, YELLOW, 2)
                cv2.circle(frame, (cx, cy), 3, RED, -1)

        elif initialized:
            # Predict-only (ball temporarily occluded)
            x, P = kalman_predict(x, P, F, Q)
            frames_lost += 1
            trail.append((int(x[0]), int(x[1])))

            if frames_lost > MAX_LOST:
                initialized = False
                trail.clear()
                print("[WARN] Ball lost – tracker reset.")

        # ──────────────────────────────── Extrapolation & servo
        last_duty = None
        if initialized and frames_lost < 5:
            traj     = extrapolate(x, F, PREDICT_STEPS)
            wall_pt  = find_wall_intersection(traj, PROC_W, PROC_H)

            if wall_pt is not None:
                duty     = target_to_servo_duty(wall_pt[0], PROC_W)
                last_duty = duty
                servo.set_position(duty)

                if args.preview:
                    draw_trajectory(frame, traj)
                    cv2.circle(frame, wall_pt, 10, CYAN, 2)
                    cv2.putText(frame, f"TARGET ({wall_pt[0]},{wall_pt[1]})",
                                (wall_pt[0] + 12, wall_pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
            elif args.preview:
                draw_trajectory(frame, traj)

        # ──────────────────────────────── Preview rendering
        if args.preview:
            if len(trail) > 1:
                draw_trail(frame, trail)
            # Panel boundary
            cv2.rectangle(frame,
                          (PANEL_MARGIN, PANEL_MARGIN),
                          (PROC_W - PANEL_MARGIN, PROC_H - PANEL_MARGIN),
                          GREY, 1)
            draw_hud(frame, x, fps_val, detection is not None, last_duty)
            cv2.imshow("Tennis Tracker [RPi]", frame)
            if show_mask:
                cv2.imshow("Mask", mask)

        # ──────────────────────────────── Timing & keys
        fc += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= 1.0:
            fps_val = fc / elapsed
            fc, t0 = 0, time.perf_counter()

        t_end = time.perf_counter()
        loop_ms = (t_end - t_frame) * 1000
        # Verbose timing (disable in production)
        if args.verbose:
            print(f"loop={loop_ms:.1f}ms  fps={fps_val:.1f}  "
                  f"ball={'Y' if detection else 'N'}  "
                  f"duty={last_duty:.1f}" if last_duty else
                  f"loop={loop_ms:.1f}ms  fps={fps_val:.1f}  ball={'Y' if detection else 'N'}")

        if args.preview:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                initialized = False
                trail.clear()
                print("[INFO] Tracker reset.")
            elif key == ord('d'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Mask")

    cam.release()
    servo.cleanup()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RPi Tennis Ball Tracker")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="V4L2 camera device index (default 0)")
    src.add_argument("--picam",  action="store_true", help="Use Picamera2 (Raspberry Pi camera module)")
    src.add_argument("--video",  type=str, help="Path to a video file (for testing)")

    ap.add_argument("--preview", action="store_true", help="Show OpenCV window (adds ~3ms, useful for tuning)")
    ap.add_argument("--debug",   action="store_true", help="Also show the HSV mask window (requires --preview)")
    ap.add_argument("--no-gpio", dest="gpio", action="store_false",  help="Disable GPIO/servo output")
    ap.add_argument("--verbose", action="store_true", help="Print per-frame timing to stdout")
    ap.set_defaults(gpio=True)

    args = ap.parse_args()
    if args.debug and not args.preview:
        print("[WARN] --debug requires --preview; enabling preview.")
        args.preview = True

    run(args)