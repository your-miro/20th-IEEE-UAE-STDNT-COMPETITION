Can also be found at https://github.com/your-miro/20th-IEEE-UAE-STDNT-COMPETITION
# Tennis Ball Tracker – IMX500

Real-time tennis ball detection and trajectory estimation using the Raspberry Pi AI Camera (IMX500). Detects the ball via HSV colour masking, tracks it with a Kalman filter, and predicts where it will hit the panel boundary.

---

## Hardware Required

- Raspberry Pi 3B+ / 4 / 5
- Raspberry Pi AI Camera (IMX500)
- Ribbon cable connected to the CSI port

---

## Install

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv
pip install numpy
source /path/to/python/virtual/environment
```
The final command is specific to the rasberry pi in the lab as it may lack some permissions to install python packages


---

## Run

**Headless** – no window, just terminal output:
```bash
python rpi_tennis_tracker.py --no-gpio
```

**With preview window** – recommended for tuning:
```bash
python rpi_tennis_tracker.py --preview --no-gpio
```

**With preview + HSV mask** – use this to tune the colour range:
```bash
python rpi_tennis_tracker.py --preview --debug --no-gpio
```

**Verbose timing** – prints loop time and FPS every frame:
```bash
python rpi_tennis_tracker.py --preview --no-gpio --verbose
```

Stop with `q` in the preview window, or `Ctrl-C` in the terminal.

---

## Preview Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset tracker (if ball is lost or stuck) |
| `d` | Toggle HSV mask window |

---

## What to Expect

When the preview window opens you'll see:

- **Yellow circle** — detected ball position from the camera
- **Green trail** — recent path of the ball
- **Fading blue line** — predicted future trajectory (50 frames ahead)
- **Cyan circle + TARGET label** — where the ball is predicted to hit the panel wall
- **HUD (top-left)** — live FPS, speed estimate, velocity, acceleration, and ball status

If the ball goes out of frame or is occluded, the tracker switches to **PREDICTING** mode (red HUD) and continues estimating position from the Kalman filter for ~400 ms before resetting.

---

## Tuning the Colour Mask

Run with `--debug` to see the HSV mask alongside the preview. The ball should appear as a solid white blob — nothing else. If it's noisy or missing, edit these two lines in `rpi_tennis_tracker.py`:

```python
HSV_LOWER = np.array([25,  80,  80], dtype=np.uint8)
HSV_UPPER = np.array([75, 255, 255], dtype=np.uint8)
```

Adjust under your actual lighting conditions. The three values are **Hue, Saturation, Value**.

---

## Tuning the Kalman Filter

Two constants control the smoothness vs. responsiveness trade-off:

```python
Q_STD = 6.0   # raise if the estimated position lags behind the ball
R_STD = 3.5   # raise if the trajectory line jitters too much
```
