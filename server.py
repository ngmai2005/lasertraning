import cv2, numpy as np, json, atexit, threading, time
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ================= APP =================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= CONFIG =================
CAM_INDEX = 0
FLIP_MODE = 1
LASER_MIN_AREA = 4
TEMPORAL_FRAMES = 3
WIDTH, HEIGHT = 1280, 720

# ================= CAMERA =================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

latest_frame = None
frame_lock = threading.Lock()
running = True

# ================= DATA =================
laser_buffer = []
laser_pos = None
calibration = {
    "top_left": [0, 0],
    "top_right": [WIDTH, 0],
    "bottom_left": [0, HEIGHT],
    "bottom_right": [WIDTH, HEIGHT]
}

# ================= CAMERA LOOP (CHỈ 1 LẦN) =================
def camera_loop():
    global latest_frame
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, FLIP_MODE)
        with frame_lock:
            latest_frame = frame.copy()

threading.Thread(target=camera_loop, daemon=True).start()

# ================= VIDEO STREAM =================
def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.get("/video")
def video():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ================= LASER DETECT =================
@app.get("/laser")
def get_laser():
    global laser_pos, laser_buffer

    with frame_lock:
        if latest_frame is None:
            return {"x": None, "y": None}
        frame = latest_frame.copy()

    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bright = cv2.inRange(v, 240, 255)
    sat = cv2.inRange(s, 120, 255)
    base = cv2.bitwise_and(bright, sat)

    red1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 120, 200), (180, 255, 255))
    green = cv2.inRange(hsv, (35, 120, 200), (90, 255, 255))
    pink = cv2.inRange(hsv, (140, 80, 200), (165, 255, 255))

    mask = base & (red1 | red2 | green | pink)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pos = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) >= LASER_MIN_AREA:
            (x, y), _ = cv2.minEnclosingCircle(c)
            pos = (int(x), int(y))

    # ===== MAP + TEMPORAL =====
    if pos:
        cx = np.interp(
            pos[0],
            [0, WIDTH],
            [calibration["top_left"][0], calibration["top_right"][0]]
        )
        cy = np.interp(
            pos[1],
            [0, HEIGHT],
            [calibration["top_left"][1], calibration["bottom_left"][1]]
        )
        laser_buffer.append([int(cx), int(cy)])
    else:
        laser_buffer.append(None)

    if len(laser_buffer) > TEMPORAL_FRAMES:
        laser_buffer.pop(0)

    valid = [p for p in laser_buffer if p]
    if valid:
        x = int(sum(p[0] for p in valid) / len(valid))
        y = int(sum(p[1] for p in valid) / len(valid))
        laser_pos = [x, y]
    else:
        laser_pos = None

    return {
        "x": laser_pos[0] if laser_pos else None,
        "y": laser_pos[1] if laser_pos else None
    }

# ================= CALIBRATE =================
@app.get("/calibrate")
def calibrate():
    global calibration
    calibration = {
        "top_left": [0, 0],
        "top_right": [WIDTH, 0],
        "bottom_left": [0, HEIGHT],
        "bottom_right": [WIDTH, HEIGHT]
    }
    with open("calibration.json", "w") as f:
        json.dump(calibration, f)
    return {"status": "ok"}

# ================= INDEX =================
@app.get("/")
def index():
    return FileResponse("index.html")

# ================= CLEANUP =================
def release_camera():
    global running
    running = False
    cap.release()

atexit.register(release_camera)

# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
