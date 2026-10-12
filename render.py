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

app = FastAPI(title="2D Rig Renderer & Camera Preset Service")

# ---------------------------------------------------------
# Camera & Render Presets
# ---------------------------------------------------------
PRESETS = {
    "front": {"elevation": 0, "azim": 0, "style": "wireframe", "color": (37, 99, 235)},
    "side": {"elevation": 0, "azim": 90, "style": "solid", "color": (70, 130, 180)},
    "isometric": {"elevation": 35.264, "azim": 45, "style": "shaded", "color": (55, 65, 81)},
    "cinematic": {"elevation": 15, "azim": -60, "style": "glowing", "color": (235, 180, 40)}
}

RESOLUTION = [640, 360, 3] # Width, Height, RGB Channels
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

def project_3d_to_2d(vertices, azim_deg, elev_deg, width, height, scale=150):
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)
    
    cos_a, sin_a = math.cos(azim), math.sin(azim)
    cos_e, sin_e = math.cos(elev), math.sin(elev)
    
    projected = []
    for v in vertices:
        x, y, z = v[0], v[1], v[2]
        # Rotate around Y (Azimuth)
        x1 = x * cos_a + z * sin_a
        z1 = -x * sin_a + z * cos_a
        # Rotate around X (Elevation)
        y2 = y * cos_e - z1 * sin_e
        z2 = y * sin_e + z1 * cos_e
        
        screen_x = width // 2 + int(x1 * scale)
        screen_y = height // 2 - int(y2 * scale)
        projected.append((screen_x, screen_y, z2))
    return projected

def render_glb_animation(glb_dir, output_mp4, preset_name="isometric"):
    preset = PRESETS.get(preset_name, PRESETS["isometric"])
    azim = preset["azim"]
    elev = preset["elevation"]
    base_color = preset["color"]

    glb_files = sorted([os.path.join(glb_dir, f) for f in os.listdir(glb_dir) if f.endswith('.glb')]) if os.path.exists(glb_dir) else []
    if not glb_files:
        return render_procedural_fallback(output_mp4, preset)

    frames_data = []
    for glb_path in glb_files:
        try:
            scene = trimesh.load(glb_path)
            frame = [25] * (W * H * 3)
            for y in range(H):
                shade = int(12 + (y / H) * 20)
                for x in range(W):
                    pixel(frame, x, y, shade, shade, shade)
            
            for geom in scene.geometry.values():
                if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                    verts = geom.vertices
                    faces = geom.faces
                    proj = project_3d_to_2d(verts, azim, elev, W, H, scale=140)
                    for face in faces:
                        p0, p1, p2 = proj[face[0]], proj[face[1]], proj[face[2]]
                        line(frame, p0[0], p0[1], p1[0], p1[1], 1, base_color)
                        line(frame, p1[0], p1[1], p2[0], p2[1], 1, base_color)
                        line(frame, p2[0], p2[1], p0[0], p0[1], 1, base_color)
            frames_data.append(frame)
        except Exception as e:
            print(f"Error processing {glb_path}: {e}")

    if not frames_data:
        return render_procedural_fallback(output_mp4, preset)

    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", "temp_frames.rgb")
    with open(raw_path, "wb") as f:
        for fr in frames_data:
            f.write(bytearray(fr))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)
    return output_mp4

def render_procedural_fallback(output_mp4, preset):
    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", "temp_fallback.rgb")
    total_frames = FPS * 3
    color = preset["color"]

    with open(raw_path, "wb") as f:
        for tick in range(total_frames):
            t = tick / FPS
            frame = [20] * (W * H * 3)
            cx, cy = W // 2, H // 2
            angle = math.sin(t * 5) * 35
            line(frame, cx, cy - 50, cx, cy + 30, 4, color)
            line(frame, cx, cy + 10, cx - int(40 * math.cos(math.radians(angle))), cy + 40, 3, color)
            line(frame, cx, cy + 10, cx + int(40 * math.cos(math.radians(angle))), cy + 40, 3, color)
            f.write(bytearray(frame))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_path):
        os.remove(raw_path)
    return output_mp4

# ---------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------
@app.get("/api/presets")
def get_presets():
    return JSONResponse({"presets": list(PRESETS.keys()), "details": PRESETS})

@app.post("/api/render-preset")
async def api_render_preset(request: Request):
    data = await request.json()
    preset = data.get("preset", "isometric")
    output_filename = f"rendered_{preset}.mp4"
    output_path = os.path.join("renders", output_filename)
    
    glb_dir = "rigged_glbs"
    render_glb_animation(glb_dir, output_path, preset_name=preset)
    return {"status": "success", "video_url": f"/renders/{output_filename}"}

os.makedirs("renders", exist_ok=True)
app.mount("/renders", StaticFiles(directory="renders"), name="renders")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Starting 2D Renderer Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)