import sys
import os
import math
import subprocess
import json
import asyncio
import numpy as np
import trimesh
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="Procedural GLB & Realistic Render Service")
executor = ThreadPoolExecutor(max_workers=4)

PRESETS = {
    "front": {"elevation": 0, "azim": 0},
    "side": {"elevation": 0, "azim": 90},
    "isometric": {"elevation": 35.264, "azim": 45},
    "cinematic": {"elevation": 15, "azim": -60}
}

RESOLUTION = [360, 360, 3]
W, H = RESOLUTION[0], RESOLUTION[1]
FPS = 30

def generate_procedural_glb(var_id, epoch, bone_transforms=None):
    scene = trimesh.Scene()
    
    hips = trimesh.creation.icosphere(subdivisions=3, radius=0.22)
    hips.visual.vertex_colors = [210, 215, 220, 255]
    scene.add_geometry(hips, node_name="Hips")
    
    spine = trimesh.creation.cylinder(radius=0.09, height=1.0, sections=16)
    spine.apply_translation([0, 0.5, 0])
    spine.visual.vertex_colors = [100, 110, 125, 255]
    scene.add_geometry(spine, node_name="Spine")
    
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.26)
    head.apply_translation([0, 1.1, 0])
    head.visual.vertex_colors = [240, 220, 205, 255]
    scene.add_geometry(head, node_name="Head")
    
    for side, x_sign in [("Left", -1), ("Right", 1)]:
        arm = trimesh.creation.cylinder(radius=0.06, height=0.8, sections=12)
        arm.apply_translation([x_sign * 0.5, 0.6, 0])
        arm.visual.vertex_colors = [90, 100, 115, 255]
        scene.add_geometry(arm, node_name=f"{side}Arm")
        
        leg = trimesh.creation.cylinder(radius=0.07, height=0.9, sections=12)
        leg.apply_translation([x_sign * 0.25, -0.5, 0])
        leg.visual.vertex_colors = [70, 80, 95, 255]
        scene.add_geometry(leg, node_name=f"{side}Leg")
        
    os.makedirs("renders", exist_ok=True)
    glb_path = os.path.join("renders", f"var_{var_id}_epoch_{epoch}.glb")
    scene.export(glb_path)
    return glb_path

def project_vertex(v, azim_deg, elev_deg, fov_scale=130.0):
    rad_az = math.radians(azim_deg)
    rad_el = math.radians(elev_deg)
    
    cos_a, sin_a = math.cos(rad_az), math.sin(rad_az)
    x1 = v[0] * cos_a - v[2] * sin_a
    z1 = v[0] * sin_a + v[2] * cos_a
    y1 = v[1]
    
    cos_e, sin_e = math.cos(rad_el), math.sin(rad_el)
    y2 = y1 * cos_e - z1 * sin_e
    z2 = y1 * sin_e + z1 * cos_e
    x2 = x1
    
    dist = 4.5
    factor = fov_scale / (z2 + dist)
    px = int(W / 2 + x2 * factor)
    py = int(H / 2 - y2 * factor)
    return px, py, z2

def rasterize_triangle(frame_buffer, depth_buffer, p0, p1, p2, color, normal):
    light_dir = np.array([0.5, 0.8, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    diffuse = max(0.2, np.dot(normal, light_dir))
    shaded_color = (np.array(color[:3], dtype=np.float32) * diffuse).astype(np.uint8)

    min_x = max(0, min(p0[0], p1[0], p2[0]))
    max_x = min(W - 1, max(p0[0], p1[0], p2[0]))
    min_y = max(0, min(p0[1], p1[1], p2[1]))
    max_y = min(H - 1, max(p0[1], p1[1], p2[1]))

    def edge_fn(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    area = edge_fn(p0, p1, p2)
    if area == 0:
        return

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = (x, y)
            w0 = edge_fn(p1, p2, p)
            w1 = edge_fn(p2, p0, p)
            w2 = edge_fn(p0, p1, p)
            
            if area > 0:
                inside = (w0 >= 0 and w1 >= 0 and w2 >= 0)
            else:
                inside = (w0 <= 0 and w1 <= 0 and w2 <= 0)
            
            if inside:
                b0 = w0 / area
                b1 = w1 / area
                b2 = (area - w0 - w1) / area
                z_depth = b0 * p0[2] + b1 * p1[2] + b2 * p2[2]
                
                if z_depth < depth_buffer[y, x]:
                    depth_buffer[y, x] = z_depth
                    frame_buffer[y, x] = shaded_color

@app.post("/api/render-card")
async def api_render_card(request: Request):
    data = await request.json()
    def task():
        var_id = data.get("var_id", 1)
        epoch = data.get("epoch", 1)
        preset = data.get("preset", "isometric")
        duration_sec = max(4, min(30, int(data.get("duration_sec", 6))))
        
        glb_path = generate_procedural_glb(var_id, epoch, data.get("bone_transforms"))
        output_filename = f"var_{var_id}_epoch_{epoch}_{preset}.mp4"
        output_path = os.path.join("renders", output_filename)
        
        scene = trimesh.load(glb_path)
        preset_cfg = PRESETS.get(preset, PRESETS["isometric"])
        base_azim, base_elev = preset_cfg["azim"], preset_cfg["elevation"]
        
        total_frames = duration_sec * FPS
        raw_path = os.path.join("renders", f"temp_{output_filename}.rgb")

        with open(raw_path, "wb") as f_out:
            for frame_idx in range(total_frames):
                t = frame_idx / FPS
                frame_buffer = np.ones((H, W, 3), dtype=np.uint8) * 240
                depth_buffer = np.ones((H, W), dtype=np.float32) * np.inf
                azim = base_azim + math.sin(t * 2.0) * 20.0
                
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        verts = geom.vertices
                        faces = geom.faces
                        vertex_colors = getattr(geom.visual, 'vertex_colors', np.array([[200, 200, 200, 255]] * len(verts)))
                        
                        for face in faces:
                            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
                            edge1 = v1 - v0
                            edge2 = v2 - v0
                            normal = np.cross(edge1, edge2)
                            norm_len = np.linalg.norm(normal)
                            normal = normal / norm_len if norm_len > 0 else np.array([0, 0, 1.0])

                            p0 = project_vertex(v0, azim, base_elev)
                            p1 = project_vertex(v1, azim, base_elev)
                            p2 = project_vertex(v2, azim, base_elev)
                            
                            color = vertex_colors[face[0]] if len(vertex_colors) > face[0] else [200, 200, 200, 255]
                            rasterize_triangle(frame_buffer, depth_buffer, p0, p1, p2, color, normal)
                
                f_out.write(frame_buffer.tobytes())

        subprocess.run([
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(raw_path):
            os.remove(raw_path)
        return {"status": "success", "video_url": f"/renders/{output_filename}", "duration_sec": duration_sec}

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, task)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/render-pointcloud")
async def api_render_pointcloud(request: Request):
    data = await request.json()
    def task():
        var_id = data.get("var_id", 1)
        epoch = data.get("epoch", 1)
        duration_sec = int(data.get("duration_sec", 6))
        glb_path = generate_procedural_glb(var_id, epoch)
        scene = trimesh.load(glb_path)
        
        output_filename = f"var_{var_id}_epoch_{epoch}_pc.mp4"
        output_path = os.path.join("renders", output_filename)
        total_frames = duration_sec * FPS
        raw_path = os.path.join("renders", f"temp_pc_{output_filename}.rgb")

        with open(raw_path, "wb") as f_out:
            for frame_idx in range(total_frames):
                t = frame_idx / FPS
                frame_buffer = np.ones((H, W, 3), dtype=np.uint8) * 15
                azim = t * 40.0
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices'):
                        for v in geom.vertices:
                            px, py, _ = project_vertex(v, azim, 25.0, fov_scale=140.0)
                            if 0 <= px < W and 0 <= py < H:
                                frame_buffer[max(0, py-1):min(H, py+2), max(0, px-1):min(W, px+2)] = [0, 200, 255]
                f_out.write(frame_buffer.tobytes())

        subprocess.run([
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return {"status": "success", "video_url": f"/renders/{output_filename}"}

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, task)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/render-opengl")
async def api_render_opengl(request: Request):
    data = await request.json()
    def task():
        var_id = data.get("var_id", 1)
        epoch = data.get("epoch", 1)
        duration_sec = int(data.get("duration_sec", 6))
        glb_path = generate_procedural_glb(var_id, epoch)
        scene = trimesh.load(glb_path)
        
        output_filename = f"var_{var_id}_epoch_{epoch}_gl.mp4"
        output_path = os.path.join("renders", output_filename)
        total_frames = duration_sec * FPS
        raw_path = os.path.join("renders", f"temp_gl_{output_filename}.rgb")

        with open(raw_path, "wb") as f_out:
            for frame_idx in range(total_frames):
                t = frame_idx / FPS
                frame_buffer = np.ones((H, W, 3), dtype=np.uint8) * 25
                azim = -t * 50.0
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        verts = geom.vertices
                        faces = geom.faces
                        for face in faces:
                            p0 = project_vertex(verts[face[0]], azim, 30.0, fov_scale=130.0)
                            p1 = project_vertex(verts[face[1]], azim, 30.0, fov_scale=130.0)
                            p2 = project_vertex(verts[face[2]], azim, 30.0, fov_scale=130.0)
                            for pa, pb in [(p0, p1), (p1, p2), (p2, p0)]:
                                if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H:
                                    frame_buffer[pa[1], pa[0]] = [99, 102, 241]
                                    frame_buffer[pb[1], pb[0]] = [99, 102, 241]
                f_out.write(frame_buffer.tobytes())

        subprocess.run([
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return {"status": "success", "video_url": f"/renders/{output_filename}"}

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, task)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

os.makedirs("renders", exist_ok=True)
app.mount("/renders", StaticFiles(directory="renders"), name="renders")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Starting Realistic Render Microservice on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)