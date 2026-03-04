import cv2
import numpy as np
import time
from math import hypot, sin, cos, pi

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import urllib.request
import os

def download_model():
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")
    return model_path

def get_volume_controller():
    devices = AudioUtilities.GetSpeakers()
    volume = devices._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(volume, POINTER(IAudioEndpointVolume))

def get_hand_detector(model_path):
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)

def _blend_roi(img, x1, y1, x2, y2, color, alpha):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

FINGER_COLORS = {
    "thumb":  (255, 255, 0),
    "index":  (255, 0, 255),
    "middle": (0, 255, 128),
    "ring":   (0, 165, 255),
    "pinky":  (180, 0, 255),
    "palm":   (0, 215, 255),
}
CONNECTIONS_COLORED = [
    ((0, 1), "thumb"), ((1, 2), "thumb"), ((2, 3), "thumb"), ((3, 4), "thumb"),
    ((0, 5), "index"), ((5, 6), "index"), ((6, 7), "index"), ((7, 8), "index"),
    ((0, 9), "middle"), ((9, 10), "middle"), ((10, 11), "middle"), ((11, 12), "middle"),
    ((0, 13), "ring"), ((13, 14), "ring"), ((14, 15), "ring"), ((15, 16), "ring"),
    ((0, 17), "pinky"), ((17, 18), "pinky"), ((18, 19), "pinky"), ((19, 20), "pinky"),
    ((5, 9), "palm"), ((9, 13), "palm"), ((13, 17), "palm"),
]
FINGERTIP_IDS = {4, 8, 12, 16, 20}
_JOINT_COLOR = {}
for (_, end), group in CONNECTIONS_COLORED:
    _JOINT_COLOR[end] = FINGER_COLORS[group]
_JOINT_COLOR[0] = FINGER_COLORS["palm"]

def _draw_glow_line(img, pt1, pt2, color, thickness=3):
    faded = (color[0] // 3, color[1] // 3, color[2] // 3)
    cv2.line(img, pt1, pt2, faded, thickness + 4, cv2.LINE_AA)
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

def _draw_joint(img, cx, cy, lm_id, pulse):
    col = _JOINT_COLOR.get(lm_id, (0, 255, 255))
    if lm_id == 0:
        radius = int(14 + 3 * pulse)
        faded = (col[0] * 2 // 5, col[1] * 2 // 5, col[2] * 2 // 5)
        cv2.circle(img, (cx, cy), radius + 6, faded, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), radius, col, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
    elif lm_id in FINGERTIP_IDS:
        outer = int(9 + 2 * pulse)
        cv2.circle(img, (cx, cy), outer, col, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), 5, col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

def get_landmarks(detection_result, img):
    lm_list = []
    h, w = img.shape[:2]
    if not detection_result.hand_landmarks:
        return lm_list
    t_now = time.time()
    pulse = (sin(t_now * 4 * pi) + 1) * 0.5
    for hand_landmarks in detection_result.hand_landmarks:
        pts = {}
        for idx, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts[idx] = (cx, cy)
            lm_list.append((idx, cx, cy))
        for (s, e), group in CONNECTIONS_COLORED:
            _draw_glow_line(img, pts[s], pts[e], FINGER_COLORS[group])
        for idx, (cx, cy) in pts.items():
            _draw_joint(img, cx, cy, idx, pulse)
        if 0 in pts:
            _draw_holo_reticle(img, pts[0][0], pts[0][1], t_now, pulse)
        if all(k in pts for k in (0, 5, 9, 13, 17)):
            _draw_holo_palm_grid(img, pts, t_now)
    return lm_list

def _draw_holo_reticle(img, cx, cy, t_now, pulse):
    col = (0, 255, 220)
    r_outer = int(28 + 4 * pulse)
    r_inner = int(18 + 2 * pulse)
    angle = t_now * 2
    for i in range(4):
        a = int(np.degrees(angle + i * pi / 2))
        cv2.ellipse(img, (cx, cy), (r_outer, r_outer), 0, a, a + 60, col, 1, cv2.LINE_AA)
    col2 = (200, 200, 0)
    for i in range(4):
        a = int(np.degrees(-angle + i * pi / 2))
        cv2.ellipse(img, (cx, cy), (r_inner, r_inner), 0, a, a + 40, col2, 1, cv2.LINE_AA)
    tick = int(6 + 2 * pulse)
    r2 = r_outer + 2
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (cx + dx * r2, cy + dy * r2),
                 (cx + dx * (r2 + tick), cy + dy * (r2 + tick)), col, 1, cv2.LINE_AA)

def _draw_holo_palm_grid(img, pts, t_now):
    palm_pts = [pts[k] for k in (0, 5, 9, 13, 17)]
    col = (180, 100, 0)
    alpha = 0.12 + 0.05 * (sin(t_now * 3) + 1) * 0.5
    hull = cv2.convexHull(np.array(palm_pts, dtype=np.int32))
    x_min = min(p[0] for p in palm_pts)
    x_max = max(p[0] for p in palm_pts)
    y_min = min(p[1] for p in palm_pts)
    y_max = max(p[1] for p in palm_pts)
    h, w = img.shape[:2]
    x1, y1 = max(0, x_min - 5), max(0, y_min - 5)
    x2, y2 = min(w, x_max + 5), min(h, y_max + 5)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2].copy()
    hull_shifted = hull - np.array([x1, y1])
    cv2.drawContours(roi, [hull_shifted], 0, col, 1, cv2.LINE_AA)
    step = max(8, (y_max - y_min) // 6)
    for y in range(y_min, y_max, step):
        ry = y - y1
        if 0 <= ry < roi.shape[0]:
            cv2.line(roi, (x_min - x1, ry), (x_max - x1, ry), col, 1, cv2.LINE_AA)
    cv2.addWeighted(roi, alpha, img[y1:y2, x1:x2], 1 - alpha, 0, img[y1:y2, x1:x2])

def draw_finger_line(img, x1, y1, x2, y2):
    length = hypot(x2 - x1, y2 - y1)
    t = min(max(length / 220, 0), 1)
    line_col = (int(50 + 205 * (1 - t)), int(255 * (1 - t)), int(255 * t))
    _draw_glow_line(img, (x1, y1), (x2, y2), line_col)
    for px, py in ((x1, y1), (x2, y2)):
        cv2.circle(img, (px, py), 12, line_col, 2, cv2.LINE_AA)
        cv2.circle(img, (px, py), 6, line_col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (mx, my), 8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (mx, my), 4, line_col, -1, cv2.LINE_AA)

_GRAD_LUT = None
_GRAD_LUT_SIZE = 0
def _get_gradient_lut(bar_h):
    global _GRAD_LUT, _GRAD_LUT_SIZE
    if _GRAD_LUT_SIZE == bar_h and _GRAD_LUT is not None:
        return _GRAD_LUT
    lut = np.zeros((bar_h, 3), dtype=np.uint8)
    for row in range(bar_h):
        t = row / bar_h
        if t < 0.25:
            r, g, b = 0, int(220 + t / 0.25 * 35), int(255 - t / 0.25 * 195)
        elif t < 0.5:
            tt = (t - 0.25) / 0.25
            r, g, b = int(240 * tt), 255, 0
        elif t < 0.75:
            tt = (t - 0.5) / 0.25
            r, g, b = int(240 + 15 * tt), int(255 - 115 * tt), 0
        else:
            tt = (t - 0.75) / 0.25
            r, g, b = 255, int(140 - 110 * tt), 0
        lut[row] = (b, g, r)
    _GRAD_LUT = lut
    _GRAD_LUT_SIZE = bar_h
    return lut

def draw_volume_bar(img, vol_percent, hand_detected=False):
    h_img, w_img = img.shape[:2]
    t_now = time.time()
    sc = max(0.5, min(h_img / 480.0, 2.0))
    BAR_W   = int(38 * sc)
    BAR_H   = int(min(260 * sc, h_img * 0.55))
    PAD     = int(18 * sc)
    TICK_W  = int(38 * sc)
    PANEL_W = BAR_W + PAD * 2 + TICK_W
    PANEL_H = BAR_H + PAD * 2 + int(75 * sc)
    PX      = int(12 * sc)
    PY      = max(int(42 * sc), (h_img - PANEL_H) // 2)
    BX      = PX + PAD
    BY      = PY + PAD + int(24 * sc)
    filled  = int(BAR_H * vol_percent / 100)
    _blend_roi(img, PX, PY, PX + PANEL_W, PY + PANEL_H, (8, 8, 14), 0.78)
    cv2.rectangle(img, (PX, PY), (PX + PANEL_W, PY + PANEL_H), (50, 50, 75), 1)
    cl = int(12 * sc)
    cc = (0, 200, 255)
    for (cx, cy, dx, dy) in [
        (PX, PY, 1, 1), (PX + PANEL_W, PY, -1, 1),
        (PX, PY + PANEL_H, 1, -1), (PX + PANEL_W, PY + PANEL_H, -1, -1),
    ]:
        cv2.line(img, (cx, cy), (cx + cl * dx, cy), cc, 2)
        cv2.line(img, (cx, cy), (cx, cy + cl * dy), cc, 2)
    cv2.line(img, (PX + 4, PY + 1), (PX + PANEL_W - 4, PY + 1), (70, 70, 100), 1)
    if vol_percent <= 0:
        icon, icon_col = "MUTE", (80, 80, 220)
    elif vol_percent < 33:
        icon, icon_col = "LOW", (220, 190, 0)
    elif vol_percent < 66:
        icon, icon_col = "MID", (0, 210, 160)
    else:
        icon, icon_col = "HIGH", (0, 160, 255)
    fs_t = 0.35 * sc
    cv2.putText(img, "VOL", (BX, PY + PAD + int(2 * sc)),
                cv2.FONT_HERSHEY_DUPLEX, fs_t, (120, 120, 160), 1, cv2.LINE_AA)
    cv2.putText(img, icon, (BX + int(30 * sc), PY + PAD + int(2 * sc)),
                cv2.FONT_HERSHEY_DUPLEX, fs_t, icon_col, 1, cv2.LINE_AA)
    cv2.rectangle(img, (BX, BY), (BX + BAR_W, BY + BAR_H), (14, 14, 22), -1)
    cv2.rectangle(img, (BX, BY), (BX + BAR_W, BY + BAR_H), (45, 45, 65), 1)
    if filled > 0:
        lut = _get_gradient_lut(BAR_H)
        fill_slice = lut[:filled]
        bar_top = BY + BAR_H - filled
        for i in range(filled):
            fy = bar_top + (filled - 1 - i)
            col = tuple(int(c) for c in fill_slice[i])
            cv2.line(img, (BX + 1, fy), (BX + BAR_W - 1, fy), col, 1)
        top_y = bar_top
        pulse = (sin(t_now * 6 * pi) + 1) * 0.5
        bright = int(180 + 75 * pulse)
        cv2.line(img, (BX, top_y), (BX + BAR_W, top_y), (bright, bright, 255), max(1, int(2 * sc)))
    fs_tk = 0.28 * sc
    for tick in range(0, 11):
        ty      = BY + BAR_H - int(tick / 10 * BAR_H)
        is_key  = tick % 5 == 0
        t_len   = int(10 * sc) if is_key else int(4 * sc)
        t_col   = (120, 120, 160) if is_key else (50, 50, 70)
        tx      = BX + BAR_W + int(3 * sc)
        cv2.line(img, (tx, ty), (tx + t_len, ty), t_col, 1)
        if is_key:
            cv2.putText(img, f"{tick * 10}",
                        (tx + t_len + int(3 * sc), ty + int(4 * sc)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_tk, (110, 110, 150), 1, cv2.LINE_AA)
    fs_pct = 0.55 * sc
    pct_str = f"{int(vol_percent)}%"
    (tw, th), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_DUPLEX, fs_pct, 1)
    badge_cx = BX + BAR_W // 2
    bp = int(8 * sc)
    badge_x = badge_cx - tw // 2 - bp
    badge_y = BY + BAR_H + int(12 * sc)
    badge_w = tw + bp * 2
    badge_h = th + int(12 * sc)
    _blend_roi(img, badge_x, badge_y, badge_x + badge_w, badge_y + badge_h, (16, 16, 24), 0.82)
    cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), icon_col, 1)
    for cx, cy in [(badge_x, badge_y), (badge_x + badge_w, badge_y),
                   (badge_x, badge_y + badge_h), (badge_x + badge_w, badge_y + badge_h)]:
        cv2.circle(img, (cx, cy), max(1, int(2 * sc)), icon_col, -1, cv2.LINE_AA)
    cv2.putText(img, pct_str,
                (badge_cx - tw // 2, badge_y + th + int(5 * sc)),
                cv2.FONT_HERSHEY_DUPLEX, fs_pct, (240, 240, 255), 1, cv2.LINE_AA)

def draw_engagement_banner(img, hand_detected, engage_time):
    h_img, w_img = img.shape[:2]
    t_now = time.time()
    sc = max(0.5, min(h_img / 480.0, 2.0))
    if hand_detected:
        engage_time = t_now
        label = "Hand Gesture Volume Control: Engaged"
        col = (0, 255, 180)
    else:
        label = "Standby  -  Show Hand to Control"
        col = (80, 80, 160)
    dt = t_now - engage_time if engage_time else 999
    alpha = max(0.55, min(0.85, 0.85 - (dt - 0.5) * 0.1)) if hand_detected else 0.55
    fs = 0.40 * sc
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    bpad_x = int(12 * sc)
    bpad_y = int(7 * sc)
    dot_space = int(20 * sc)
    bw = tw + bpad_x * 2 + dot_space
    bh = th + bpad_y * 2
    bx = (w_img - bw) // 2
    by = h_img - bh - int(10 * sc)
    _blend_roi(img, bx, by, bx + bw, by + bh, (10, 10, 16), alpha)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), col, 1)
    dot_x = bx + int(10 * sc)
    dot_y = by + bh // 2
    cv2.circle(img, (dot_x, dot_y), max(2, int(3 * sc)), col, -1, cv2.LINE_AA)
    if hand_detected:
        pr = int(5 * sc + 2 * sc * ((sin(t_now * 8) + 1) * 0.5))
        cv2.circle(img, (dot_x, dot_y), pr, col, 1, cv2.LINE_AA)
    cv2.putText(img, label,
                (dot_x + int(10 * sc), by + th + bpad_y),
                cv2.FONT_HERSHEY_DUPLEX, fs, col, 1, cv2.LINE_AA)
    return engage_time

def draw_fps(img, fps):
    h_img, w_img = img.shape[:2]
    text = f"FPS: {int(fps)}"
    (tw, th), _ = cv2.getTextSize("FPS: 00", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 10
    bx = w_img - tw - pad * 3
    by = 10
    bw = tw + pad * 2
    bh = th + pad * 2
    _blend_roi(img, bx, by, bx + bw, by + bh, (0, 0, 0), 0.6)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
    cv2.putText(img, text, (bx + pad, by + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    try:
        volume = get_volume_controller()
        vol_min, vol_max = volume.GetVolumeRange()[:2]
        print(f"Volume range: {vol_min:.1f} to {vol_max:.1f}")
    except Exception as e:
        print(f"Error initializing volume controller: {e}")
        return
    model_path = download_model()
    detector = get_hand_detector(model_path)
    THUMB_TIP, INDEX_TIP = 4, 8
    DIST_MIN, DIST_MAX = 50, 220
    SNAP_LOW, SNAP_HIGH = 5, 95
    prev_time = 0
    frame_ts = 0
    last_vol_percent = 50.0
    engage_time = 0
    print("Running... Press 'Q' to quit.")
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_ts += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect_for_video(mp_image, frame_ts)
        lm_list = get_landmarks(results, img)
        hand_detected = bool(lm_list)
        if hand_detected:
            x1, y1 = lm_list[THUMB_TIP][1], lm_list[THUMB_TIP][2]
            x2, y2 = lm_list[INDEX_TIP][1], lm_list[INDEX_TIP][2]
            draw_finger_line(img, x1, y1, x2, y2)
            length = hypot(x2 - x1, y2 - y1)
            vol_percent = float(np.interp(length, [DIST_MIN, DIST_MAX], [0, 100]))
            if vol_percent <= SNAP_LOW:
                vol_percent = 0.0
            elif vol_percent >= SNAP_HIGH:
                vol_percent = 100.0
            vol = float(np.interp(vol_percent, [0, 100], [vol_min, vol_max]))
            try:
                volume.SetMasterVolumeLevel(vol, None)
            except Exception:
                pass
            last_vol_percent = vol_percent
        draw_volume_bar(img, last_vol_percent, hand_detected)
        engage_time = draw_engagement_banner(img, hand_detected, engage_time)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        draw_fps(img, fps)
        cv2.imshow("Hand Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Exited cleanly.")

if __name__ == "__main__":
    main()