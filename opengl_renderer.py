import os
import subprocess
import numpy as np
import trimesh

def render_opengl_mesh_animation(glb_dir, output_mp4, mass=75.0, height=1.75):
    W, H, FPS = 360, 360, 30
    total_frames = 60
    
    glb_files = sorted([os.path.join(glb_dir, f) for f in os.listdir(glb_dir) if f.endswith('.glb')]) if os.path.exists(glb_dir) else []
    
    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", f"temp_gl_{os.path.basename(output_mp4)}.rgb")

    with open(raw_path, "wb") as f_out:
        for frame_idx in range(total_frames):
            t = frame_idx / FPS
            frame_buffer = [20, 22, 30] * (W * H) # sleek dark PBR studio backdrop
            
            glb_path = glb_files[frame_idx % len(glb_files)] if glb_files else None
            
            if glb_path and os.path.exists(glb_path):
                try:
                    scene = trimesh.load(glb_path)
                    for geom in scene.geometry.values():
                        if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                            verts, faces = geom.vertices, geom.faces
                            # Simple orthographic/perspective projection for OpenGL style shaded triangles
                            scale = 140.0
                            azim = t * 0.5
                            ca, sa = math.cos(azim), math.sin(azim)
                            
                            proj = []
                            for v in verts:
                                x1 = v[0] * ca + v[2] * sa
                                z1 = -v[0] * sa + v[2] * ca
                                sx = W // 2 + int(x1 * scale)
                                sy = H // 2 - int(v[1] * scale)
                                proj.append((sx, sy, z1))
                                
                            for face in faces:
                                p0, p1, p2 = proj[face[0]], proj[face[1]], proj[face[2]]
                                # Rasterize shaded triangle edges (OpenGL wireframe / solid approximation)
                                for pa, pb in [(p0, p1), (p1, p2), (p2, p0)]:
                                    steps = int(max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]))) + 1
                                    for i in range(steps):
                                        st = i / max(steps, 1)
                                        x = int(pa[0] + (pb[0] - pa[0]) * st)
                                        y = int(pa[1] + (pb[1] - pa[1]) * st)
                                        if 0 <= x < W and 0 <= y < H:
                                            idx = (y * W + x) * 3
                                            frame_buffer[idx+0] = 79
                                            frame_buffer[idx+1] = 70
                                            frame_buffer[idx+2] = 229 # Indigo shade
                except Exception:
                    pass

            f_out.write(bytearray(frame_buffer))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)
    return output_mp4