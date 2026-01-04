import cv2, numpy as np, json, atexit
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

CAM_INDEX = 0
FLIP_MODE = 1
LASER_MIN_AREA = 4
TEMPORAL_FRAMES = 3

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3,1280)
cap.set(4,720)
laser_buffer=[]
laser_pos=None
calibration={"top_left":[0,0],"top_right":[1280,0],"bottom_left":[0,720],"bottom_right":[1280,720]}

def gen_frames():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, FLIP_MODE)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

@app.get("/laser")
def get_laser():
    global laser_pos, laser_buffer
    ret, frame = cap.read()
    if not ret: return {"x":None,"y":None}
    frame = cv2.flip(frame, FLIP_MODE)
    # detect laser
    blur=cv2.GaussianBlur(frame,(5,5),0)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    bright=cv2.inRange(v,240,255)
    sat=cv2.inRange(s,120,255)
    base=cv2.bitwise_and(bright,sat)
    red1=cv2.inRange(hsv,(0,120,200),(10,255,255))
    red2=cv2.inRange(hsv,(160,120,200),(180,255,255))
    green=cv2.inRange(hsv,(35,120,200),(90,255,255))
    pink=cv2.inRange(hsv,(140,80,200),(165,255,255))
    mask=base & (red1|red2|green|pink)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pos=None
    if cnts:
        c=max(cnts,key=cv2.contourArea)
        if cv2.contourArea(c)>=LASER_MIN_AREA:
            (x,y),_ = cv2.minEnclosingCircle(c)
            pos=(int(x),int(y))
    # map & temporal
    if pos:
        w,h=1280,720
        cx = np.interp(pos[0],[0,w],[calibration["top_left"][0],calibration["top_right"][0]])
        cy = np.interp(pos[1],[0,h],[calibration["top_left"][1],calibration["bottom_left"][1]])
        pos=[int(cx),int(cy)]
        laser_buffer.append(pos)
    else: laser_buffer.append(None)
    if len(laser_buffer)>TEMPORAL_FRAMES: laser_buffer.pop(0)
    valid=[p for p in laser_buffer if p]
    if valid:
        x=int(sum([p[0] for p in valid])/len(valid))
        y=int(sum([p[1] for p in valid])/len(valid))
        laser_pos=[x,y]
    else: laser_pos=None
    return {"x":laser_pos[0] if laser_pos else None, "y":laser_pos[1] if laser_pos else None}

@app.get("/calibrate")
def calibrate():
    global calibration
    calibration={"top_left":[0,0],"top_right":[1280,0],"bottom_left":[0,720],"bottom_right":[1280,720]}
    with open("calibration.json","w") as f: json.dump(calibration,f)
    return {"status":"ok"}

@app.get("/")
def index(): return FileResponse("index.html")

def release_camera(): cap.release()
atexit.register(release_camera)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
