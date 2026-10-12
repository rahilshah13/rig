import sys
import os
import math
import struct
import subprocess
import json
import numpy as np
import trimesh
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

app = FastAPI(title="Consolidated Render & Animation Service")

# ---------------------------------------------------------
# Camera & Render Presets
# ---------------------------------------------------------
PRESETS = {
    "front": {"elevation": 0, "azim": 0, "color": (255, 255, 255)},
    "side": {"elevation": 0, "azim": 90, "color": (200, 200, 200)},
    "isometric": {"elevation": 35.264, "azim": 45, "color": (150, 150, 150)},
    "cinematic": {"elevation": 15, "azim": -60, "color": (220, 220, 220)}
}

RESOLUTION = [360, 360, 3]
W, H = RESOLUTION[0], RESOLUTION[1]
FPS = 30

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, int(v)))

def pixel(frame, x, y, r, g, b):
    if 0 <= x < W and 0 <= y < H:
        i = (y * W + x) * 3
        frame[i + 0], frame[i + 1], frame[i + 2] = clamp(r), clamp(g), clamp(b)

def circle(frame, cx, cy, r, color):
    r2 = r * r
    for y in range(int(max(0, cy - r)), int(min(H, cy + r))):
        for x in range(int(max(0, cx - r)), int(min(W, cx + r))):
            dx, dy = (x - cx), (y - cy)
            if dx * dx + dy * dy <= r2:
                pixel(frame, x, y, *color)

def line(frame, x0, y0, x1, y1, thickness, color):
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if steps == 0:
        circle(frame, x0, y0, thickness, color)
        return
    for i in range(steps):
        t = i / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        circle(frame, x, y, thickness, color)

def project_3d_to_2d(vertices, azim_deg, elev_deg, width, height, scale=140):
    azim, elev = math.radians(azim_deg), math.radians(elev_deg)
    cos_a, sin_a = math.cos(azim), math.sin(azim)
    cos_e, sin_e = math.cos(elev), math.sin(elev)
    projected = []
    for v in vertices:
        x, y, z = v[0], v[1], v[2]
        x1 = x * cos_a + z * sin_a
        z1 = -x * sin_a + z * cos_a
        y2 = y * cos_e - z1 * sin_e
        z2 = y * sin_e + z1 * cos_e
        screen_x = width // 2 + int(x1 * scale)
        screen_y = height // 2 - int(y2 * scale)
        projected.append((screen_x, screen_y, z2))
    return projected

@app.post("/api/render-card")
async def api_render_card(request: Request):
    data = await request.json()
    var_id = data.get("var_id")
    epoch = data.get("epoch", 1)
    preset = data.get("preset", "isometric")
    mass = data.get("mass", 75.0)
    height = data.get("height", 1.75)
    
    output_filename = f"var_{var_id}_epoch_{epoch}_{preset}.mp4"
    output_path = os.path.join("renders", output_filename)
    
    preset_cfg = PRESETS.get(preset, PRESETS["isometric"])
    azim, elev, color = preset_cfg["azim"], preset_cfg["elevation"], preset_cfg["color"]
    total_frames = 60
    
    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", f"temp_{output_filename}.rgb")

    with open(raw_path, "wb") as f_out:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            frame_buffer = [255, 255, 255] * (W * H)
            cx, cy = W // 2, H // 2
            angle = math.sin(t * 4) * 30
            
            line(frame_buffer, cx, cy - 40, cx, cy + 20, 3, color)
            line(frame_buffer, cx, cy, cx - int(35 * math.cos(math.radians(angle))), cy + 30, 2, color)
            line(frame_buffer, cx, cy, cx + int(35 * math.cos(math.radians(angle))), cy + 30, 2, color)
            f_out.write(bytearray(frame_buffer))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)
    return {"status": "success", "video_url": f"/renders/{output_filename}"}

@app.post("/api/render-pointcloud")
async def api_render_pointcloud(request: Request):
    data = await request.json()
    var_id = data.get("var_id")
    epoch = data.get("epoch", 1)
    mass = data.get("mass", 75.0)
    height = data.get("height", 1.75)
    
    output_filename = f"var_{var_id}_epoch_{epoch}_pc.mp4"
    output_path = os.path.join("renders", output_filename)
    total_frames = 60
    
    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", f"temp_pc_{output_filename}.rgb")

    with open(raw_path, "wb") as f_out:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            frame_buffer = [15, 15, 18] * (W * H)
            for i in range(800):
                px = int(W * 0.5 + math.sin(t * 3 + i) * 60)
                py = int(H * 0.5 + math.cos(t * 2 + i) * 60 + (i % 40) - 20)
                if 0 <= px < W and 0 <= py < H:
                    idx = (py * W + px) * 3
                    frame_buffer[idx+0] = 0
                    frame_buffer[idx+1] = 200
                    frame_buffer[idx+2] = 255
            f_out.write(bytearray(frame_buffer))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)
    return {"status": "success", "video_url": f"/renders/{output_filename}"}

@app.post("/api/render-opengl")
async def api_render_opengl(request: Request):
    data = await request.json()
    var_id = data.get("var_id")
    epoch = data.get("epoch", 1)
    mass = data.get("mass", 75.0)
    height = data.get("height", 1.75)
    
    output_filename = f"var_{var_id}_epoch_{epoch}_gl.mp4"
    output_path = os.path.join("renders", output_filename)
    total_frames = 60
    
    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", f"temp_gl_{output_filename}.rgb")

    with open(raw_path, "wb") as f_out:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            frame_buffer = [20, 22, 30] * (W * H)
            for y in range(H):
                for x in range(W):
                    if abs((x - W//2)) < 30 and abs((y - H//2)) < 80:
                        idx = (y * W + x) * 3
                        frame_buffer[idx+0] = 79
                        frame_buffer[idx+1] = 70
                        frame_buffer[idx+2] = 229
            f_out.write(bytearray(frame_buffer))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)
    return {"status": "success", "video_url": f"/renders/{output_filename}"}

os.makedirs("renders", exist_ok=True)
app.mount("/renders", StaticFiles(directory="renders"), name="renders")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Starting Render Microservice on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)