import sys
import os
import math
import subprocess
import json
import numpy as np
import trimesh
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Procedural GLB & Render Service")

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

def generate_procedural_glb(var_id, epoch):
    """Procedurally generates a 3D skeletal trimesh scene and exports it as a GLB file."""
    scene = trimesh.Scene()
    
    # Base hips geometry
    hips = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    hips.visual.vertex_colors = [200, 200, 200, 255]
    scene.add_geometry(hips, node_name="Hips")
    
    # Spine and head geometry
    spine = trimesh.creation.cylinder(radius=0.08, height=1.0)
    spine.apply_translation([0, 0.5, 0])
    scene.add_geometry(spine, node_name="Spine")
    
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
    head.apply_translation([0, 1.1, 0])
    head.visual.vertex_colors = [230, 230, 230, 255]
    scene.add_geometry(head, node_name="Head")
    
    # Limbs (arms and legs)
    for side, x_sign in [("Left", -1), ("Right", 1)]:
        arm = trimesh.creation.cylinder(radius=0.05, height=0.8)
        arm.apply_translation([x_sign * 0.5, 0.6, 0])
        scene.add_geometry(arm, node_name=f"{side}Arm")
        
        leg = trimesh.creation.cylinder(radius=0.06, height=0.9)
        leg.apply_translation([x_sign * 0.25, -0.5, 0])
        scene.add_geometry(leg, node_name=f"{side}Leg")
        
    os.makedirs("renders", exist_ok=True)
    glb_path = os.path.join("renders", f"var_{var_id}_epoch_{epoch}.glb")
    scene.export(glb_path)
    return glb_path

def project_vertex(v, azim_deg, elev_deg, fov_scale=120.0):
    """Projects 3D vertices using azimuth/elevation rotation and perspective scaling."""
    rad_az = math.radians(azim_deg)
    rad_el = math.radians(elev_deg)
    
    # Azimuth rotation (Y-axis)
    cos_a, sin_a = math.cos(rad_az), math.sin(rad_az)
    x1 = v[0] * cos_a - v[2] * sin_a
    z1 = v[0] * sin_a + v[2] * cos_a
    y1 = v[1]
    
    # Elevation rotation (X-axis)
    cos_e, sin_e = math.cos(rad_el), math.sin(rad_el)
    y2 = y1 * cos_e - z1 * sin_e
    z2 = y1 * sin_e + z1 * cos_e
    x2 = x1
    
    # Perspective projection
    dist = 4.0
    factor = fov_scale / (z2 + dist)
    px = int(W / 2 + x2 * factor)
    py = int(H / 2 - y2 * factor)
    return px, py, z2

@app.post("/api/render-card")
async def api_render_card(request: Request):
    try:
        data = await request.json()
        var_id = data.get("var_id", 1)
        epoch = data.get("epoch", 1)
        preset = data.get("preset", "isometric")
        
        glb_path = generate_procedural_glb(var_id, epoch)
        output_filename = f"var_{var_id}_epoch_{epoch}_{preset}.mp4"
        output_path = os.path.join("renders", output_filename)
        
        scene = trimesh.load(glb_path)
        preset_cfg = PRESETS.get(preset, PRESETS["isometric"])
        base_azim, base_elev, color = preset_cfg["azim"], preset_cfg["elevation"], preset_cfg["color"]
        
        total_frames = 60
        raw_path = os.path.join("renders", f"temp_{output_filename}.rgb")

        with open(raw_path, "wb") as f_out:
            for frame_idx in range(total_frames):
                t = frame_idx / FPS
                frame_buffer = np.ones((H, W, 3), dtype=np.uint8) * 255
                azim = base_azim + math.sin(t * 3.0) * 25.0
                
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        verts = geom.vertices
                        faces = geom.faces
                        for face in faces:
                            p0 = project_vertex(verts[face[0]], azim, base_elev)
                            p1 = project_vertex(verts[face[1]], azim, base_elev)
                            p2 = project_vertex(verts[face[2]], azim, base_elev)
                            
                            # Simple edge drawing for 2D wireframe representation
                            for pa, pb in [(p0, p1), (p1, p2), (p2, p0)]:
                                if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H:
                                    frame_buffer[pa[1], pa[0]] = color
                                    frame_buffer[pb[1], pb[0]] = color
                
                f_out.write(frame_buffer.tobytes())

        subprocess.run([
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(raw_path):
            os.remove(raw_path)
            
        return {"status": "success", "video_url": f"/renders/{output_filename}"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/render-pointcloud")
async def api_render_pointcloud(request: Request):
    try:
        data = await request.json()
        var_id = data.get("var_id", 1)
        epoch = data.get("epoch", 1)
        
        glb_path = generate_procedural_glb(var_id, epoch)
        scene = trimesh.load(glb_path)
        
        output_filename = f"var_{var_id}_epoch_{epoch}_pc.mp4"
        output_path = os.path.join("renders", output_filename)
        total_frames = 60
        raw_path = os.path.join("renders", f"temp_pc_{output_filename}.rgb")

        with open(raw_path, "wb") as f_out:
            for frame_idx in range(total_frames):
                t = frame_idx / FPS
                frame_buffer = np.ones((H, W, 3), dtype=np.uint8) * 15
                azim = t * 45.0  # continuous rotation
                
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
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/render-opengl")
async def api_render_opengl(request: Request):
    try:
        data = await request.json()
        var_id = data.get("var_id", 1)
        epoch = data.get("epoch", 1)
        
        glb_path = generate_procedural_glb(var_id, epoch)
        scene = trimesh.load(glb_path)
        
        output_filename = f"var_{var_id}_epoch_{epoch}_gl.mp4"
        output_path = os.path.join("renders", output_filename)
        total_frames = 60
        raw_path = os.path.join("renders", f"temp_gl_{output_filename}.rgb")

        with open(raw_path, "wb") as f_out:
            for frame_idx in range(total_frames):
                t = frame_idx / FPS
                frame_buffer = np.ones((H, W, 3), dtype=np.uint8) * 25
                azim = -t * 60.0
                
                for geom in scene.geometry.values():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        verts = geom.vertices
                        faces = geom.faces
                        for face in faces:
                            p0 = project_vertex(verts[face[0]], azim, 30.0, fov_scale=130.0)
                            p1 = project_vertex(verts[face[1]], azim, 30.0, fov_scale=130.0)
                            p2 = project_vertex(verts[face[2]], azim, 30.0, fov_scale=130.0)
                            
                            for pa, pb in [(p0, p1), (p1, p2)]:
                                if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H:
                                    frame_buffer[pa[1], pa[0]] = [79, 70, 229]
                                    frame_buffer[pb[1], pb[0]] = [79, 70, 229]
                                    
                f_out.write(frame_buffer.tobytes())

        subprocess.run([
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(raw_path):
            os.remove(raw_path)
            
        return {"status": "success", "video_url": f"/renders/{output_filename}"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

os.makedirs("renders", exist_ok=True)
app.mount("/renders", StaticFiles(directory="renders"), name="renders")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Starting Render Microservice on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)