import os
import math
import subprocess
import numpy as np
import trimesh

def render_point_cloud_animation(glb_dir, output_mp4, mass=75.0, height=1.75):
    W, H, FPS = 360, 360, 30
    total_frames = 60
    
    glb_files = sorted([os.path.join(glb_dir, f) for f in os.listdir(glb_dir) if f.endswith('.glb')]) if os.path.exists(glb_dir) else []
    
    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", f"temp_pc_{os.path.basename(output_mp4)}.rgb")

    with open(raw_path, "wb") as f_out:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            frame_buffer = [15, 15, 18] * (W * H) # dark studio backdrop
            
            # Load corresponding glb or fallback to procedural sample points
            glb_path = glb_files[frame_idx % len(glb_files)] if glb_files else None
            points = []
            
            if glb_path and os.path.exists(glb_path):
                try:
                    scene = trimesh.load(glb_path)
                    for geom in scene.geometry.values():
                        if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                            # Sample point cloud distribution from mesh vertices
                            pts = geom.vertices
                            if len(pts) > 300:
                                indices = np.random.choice(len(pts), 300, replace=False)
                                pts = pts[indices]
                            for p in pts:
                                points.append((p[0], p[1], p[2], 1.0))
                except Exception:
                    pass

            # Fallback procedural point cloud density if empty
            if not points:
                for i in range(1500):
                    px = math.sin(t * 3 + i) * 0.8
                    py = (i / 1500.0) * height - (height / 2)
                    pz = math.cos(t * 2 + i) * 0.8
                    points.append((px, py, pz, 0.8))

            # Camera parameters (dynamic orbit)
            dist = 3.5 + 0.5 * math.cos(t * 0.5)
            yaw = t * 0.8
            pitch = -0.2 + 0.1 * math.sin(t * 0.4)
            
            ca, sa = math.cos(yaw), math.sin(yaw)
            cp, sp = math.cos(pitch), math.sin(pitch)

            for pt in points:
                x, y, z, d = pt[0], pt[1], pt[2], pt[3]
                # Rotate Y
                x1 = x * ca + z * sa
                z1 = -x * sa + z * ca
                # Rotate X
                y2 = y * cp - z1 * sp
                z2 = y * sp + z1 * cp
                
                final_z = z2 + dist
                if final_z <= 0.1:
                    continue
                
                focal = 300.0
                sx = int(W * 0.5 + (x1 * focal) / final_z)
                sy = int(H * 0.5 - (y2 * focal) / final_z)
                
                if 0 <= sx < W and 0 <= sy < H:
                    idx = (sy * W + sx) * 3
                    # Density-based coloring (cyan/blue/white point cloud glow)
                    c = int(120 + 135 * d)
                    frame_buffer[idx+0] = c // 3
                    frame_buffer[idx+1] = c
                    frame_buffer[idx+2] = 255

            f_out.write(bytearray(frame_buffer))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)
    return output_mp4