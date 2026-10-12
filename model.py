import os, json, pickle, queue, threading, time, sys, struct, subprocess, ctypes, fcntl, shutil, glob, math, optax, jax, argparse
import numpy as np, jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from functools import partial, reduce
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

import rig
import point_cloud_renderer
import opengl_renderer

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

CURR_CKPT, PREV_CKPT = "checkpoints/checkpoint_bundle.pickle", "checkpoints/checkpoint_bundle_prev.pickle"
CKPT_LOCK_PATH, GRAD_LOCK_PATH = "checkpoints/checkpoint.lock", "data/shared_gradients.lock"
META_LOCK_PATH = "checkpoints/meta.lock"
NTK_LOCK_PATH = "checkpoints/ntk.lock"
INIT_SEED_PATH = "checkpoints/init_seed.lock"

SKELETAL_BONES = [
    'Hips', 'Spine', 'Chest', 'Neck', 'Head',
    'LeftShoulder', 'LeftUpperArm', 'LeftLowerArm', 'LeftHand',
    'RightShoulder', 'RightUpperArm', 'RightLowerArm', 'RightHand',
    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot',
    'RightUpperLeg', 'RightLowerLeg', 'RightFoot'
]

bone_parents = {
    'Spine': 'Hips', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg'
}

app = FastAPI()

frontend_state = {
    "current_batch": [],
    "latest_scores": [],
    "waiting_for_human": True,
    "epoch": 1,
    "status_message": "Initializing 20-item Trealla Prolog & JAX training pipeline..."
}

rms_norm = lambda x, scale, eps=1e-5: x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * scale
apply_rope = lambda x, freq=10000.0: (lambda c, s: x.at[..., 0::2].set(x[..., 0::2]*c - x[..., 1::2]*s).at[..., 1::2].set(x[..., 0::2]*s + x[..., 1::2]*c))(jnp.cos(jnp.arange(x.shape[-2], dtype=jnp.float32)[None, :, None] / (freq ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))), jnp.sin(jnp.arange(x.shape[-2], dtype=jnp.float32)[None, :, None] / (freq ** (jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1]))))
scaled_dot_product_attention = lambda q, k, v, mask=None: jnp.matmul(jax.nn.softmax(jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * (1.0 / jnp.sqrt(q.shape[-1])) + (mask if mask is not None else 0.0), axis=-1), v)

def multimodal_mixing_layer(x, modality_type, params, target_dim=1024, comp_dim=512):
    if modality_type == "audio":
        patch_dim = 441
        B, T, L = x.shape
        num_patches = L // patch_dim
        encoded = jax.nn.gelu(x.reshape(B, T, num_patches, patch_dim) @ params['audio_encoder']) @ params['down_proj_2']
        return encoded, num_patches
    elif modality_type == "token":
        B, T, D = x.shape
        token_enc = params.get('token_encoder', params['audio_encoder'][:D, :comp_dim])
        encoded = jax.nn.gelu(x @ token_enc) @ params['down_proj_2']
        return encoded, 1
    else:
        raise ValueError(f"Unsupported modality type: {modality_type}")

def gpt_forward(params, x, cond, mask=None, target_dim=44100, patch_dim=441, num_patches=100, n_heads=16, modality_type="audio"):
    scale, bpm, stem, step_indices, sigma_t = cond
    B, T = x.shape[0], x.shape[1]
    
    x = rms_norm(x, params['input_rms_scale'])
    encoded, num_p = multimodal_mixing_layer(x, modality_type, params)
    C, bt = encoded.shape[-1], B * T
    head_dim = C // n_heads
    
    p_mask = None
    if mask is not None:
        patch_mask_2d = jnp.ones((B, T, num_p, num_p), dtype=bool) & mask[:, :, None, None]
        p_mask = jnp.where(patch_mask_2d.reshape(bt, num_p, num_p)[:, None, :, :], 0.0, -1e9)

    patch_seq = encoded.reshape(bt, num_p, C)
    q_p, k_p, v_p = map(lambda k: (patch_seq @ params[k]).reshape(bt, num_p, n_heads, head_dim).transpose(0, 2, 1, 3), ['query', 'key', 'value'])
    
    attn_out_p = scaled_dot_product_attention(apply_rope(q_p), apply_rope(k_p), v_p, p_mask).transpose(0, 2, 1, 3).reshape(bt, num_p, C)
    h_p = rms_norm(attn_out_p + patch_seq, params['rms_scale_1'])
    h_p = rms_norm(h_p + jax.nn.gelu(h_p @ params['ff_1']) @ params['ff_2'], params['rms_scale_2'])
    
    h_frames = h_p.reshape(B, T, num_p, C)
    ft = jnp.mean(h_frames, axis=2) + params['t_pos_emb'][:T][None, :, :]
    q_t, k_t, v_t = map(lambda k: (ft @ params[k]).reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3), ['t_query', 't_key', 't_value'])
    
    t_mask = jnp.where(mask[:, None, :] & mask[:, :, None], 0.0, -1e9)[:, None, :, :] if mask is not None else None
    attn_out_t = scaled_dot_product_attention(apply_rope(q_t), apply_rope(k_t), v_t, t_mask).transpose(0, 2, 1, 3).reshape(B, T, C)
    h_t = rms_norm(attn_out_t + ft, params['t_rms_scale'])
    
    base_cond = (params['time_emb'][step_indices] + params['scale_emb'][jnp.clip(scale, 0, 127)][:, None, :] + (((bpm - 120.0) / 60.0)[:, None] @ params['bpm_proj'])[:, None, :] + params['stem_emb'][stem][:, None, :])[:, :, None, :]
    h_frames = h_frames + jnp.expand_dims(h_t, 2) + base_cond + sigma_t[:, :, None, None] * params['sigma_emb'][None, None, None, :]
    
    out_encoded = rms_norm(jax.nn.gelu(h_frames @ params['up_proj_1']) @ params['up_proj_2'], params['out_rms_scale']) @ params['out_proj']
    if modality_type == "audio":
        return out_encoded.reshape(B, T, num_p * patch_dim)
    return out_encoded

xavier_normal = lambda key, shape: jax.random.normal(key, shape) * jnp.sqrt(2.0 / (shape[-2] if len(shape) >= 2 else shape[0] + shape[-1] if len(shape) >= 2 else shape[0]))

def init_params(key, dim=1024, patch_dim=441, comp_dim=512, steps=50):
    keys = jax.random.split(key, 23)
    return {
        'audio_encoder': xavier_normal(keys[0], (patch_dim, comp_dim)), 'down_proj_2': xavier_normal(keys[1], (comp_dim, dim)),
        'token_encoder': xavier_normal(keys[22], (patch_dim, comp_dim)),
        'query': jax.random.orthogonal(keys[2], dim), 'key': jax.random.orthogonal(keys[3], dim), 'value': jax.random.orthogonal(keys[4], dim),
        't_query': jax.random.orthogonal(keys[5], dim), 't_key': jax.random.orthogonal(keys[6], dim), 't_value': jax.random.orthogonal(keys[7], dim),
        'ff_1': xavier_normal(keys[8], (dim, dim * 4)), 'ff_2': xavier_normal(keys[9], (dim * 4, dim)),
        'up_proj_1': xavier_normal(keys[10], (dim, comp_dim)), 'up_proj_2': xavier_normal(keys[11], (comp_dim, comp_dim)), 'out_proj': xavier_normal(keys[19], (comp_dim, patch_dim)),
        'rms_scale_1': jnp.ones((dim,)), 'rms_scale_2': jnp.ones((dim,)), 't_rms_scale': jnp.ones((dim,)), 'out_rms_scale': jnp.ones((comp_dim,)),
        'input_rms_scale': jnp.ones((1,)),
        'scale_emb': xavier_normal(keys[12], (128, dim)), 'bpm_proj': xavier_normal(keys[13], (1, dim)), 'stem_emb': xavier_normal(keys[14], (2, dim)),
        'time_emb': xavier_normal(keys[15], (steps, dim)), 't_pos_emb': xavier_normal(keys[16], (512, dim)), 'sigma_emb': xavier_normal(keys[18], (dim,))
    }

load_checkpoint_safely = lambda: (lambda clf: (fcntl.flock(clf, fcntl.LOCK_EX), res := (pickle.load(open(CURR_CKPT, "rb")) if os.path.exists(CURR_CKPT) else None), fcntl.flock(clf, fcntl.LOCK_UN), res)[-1])(open(CKPT_LOCK_PATH, "a+"))

def get_or_create_init_seed(default_seed=42):
    with open(INIT_SEED_PATH, "a+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        try:
            sf.seek(0)
            content = sf.read().strip()
            seed = int(content) if content else default_seed
            if not content:
                sf.seek(0); sf.truncate(); sf.write(str(seed)); sf.flush()
        finally:
            fcntl.flock(sf, fcntl.LOCK_UN)
        return seed

@app.get("/api/batch")
def get_current_batch():
    return JSONResponse({
        "epoch": frontend_state["epoch"],
        "batch": frontend_state["current_batch"],
        "status": frontend_state["status_message"]
    })

@app.post("/api/score")
async def submit_human_scores(request: Request):
    data = await request.json()
    scores = data.get("scores", [])
    print(f"\n[Human Feedback Received via SolidJS for 20 items]: {scores}")
    frontend_state["latest_scores"] = scores
    frontend_state["waiting_for_human"] = False
    return {"status": "success", "received_scores": scores}

@app.get("/gif/{filename}")
def serve_gif(filename: str):
    file_path = os.path.join(".", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/gif")
    return JSONResponse({"error": "GIF not found"}, status_code=404)

os.makedirs("frontend/dist/assets", exist_ok=True)
if os.path.exists("frontend/dist/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="frontend_assets")

@app.get("/")
def serve_index():
    if os.path.exists("frontend/dist/index.html"):
        return FileResponse("frontend/dist/index.html")
    return JSONResponse({"status": "error", "message": "Frontend not built or index.html missing"}, status_code=504)

def evaluate_batch_with_human_feedback(params, batch_size=20):
    frontend_state["status_message"] = f"Epoch {frontend_state['epoch']}: Running JAX diffusion forward pass for {batch_size} variations..."
    print(f"\n[RL Env] Evaluating batch of {batch_size} generated rigging variations via Trealla Prolog...")
    
    key = jax.random.PRNGKey(frontend_state['epoch'] * 42)
    dummy_input = jax.random.normal(key, (batch_size, 4, 44100))
    
    dummy_conditions = (
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.full((batch_size,), 120.0, dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        jnp.zeros((batch_size, 4), dtype=jnp.int32),
        jnp.ones((batch_size, 4), dtype=jnp.float32)
    )
    
    output = gpt_forward(params, dummy_input, dummy_conditions, modality_type="audio")
    
    batch_metadata = []
    for i in range(batch_size):
        p = output[i, 0, :]
        bone_transforms = {}
        param_idx = 0
        for bone in SKELETAL_BONES:
            freq = float(jnp.abs(p[param_idx]) * 3.0 + 0.5) if param_idx < p.shape[0] else 1.5
            amp = float(jnp.abs(p[param_idx + 1]) * 0.6 + 0.1) if param_idx + 1 < p.shape[0] else 0.3
            axis_val = float(p[param_idx + 2]) if param_idx + 2 < p.shape[0] else 0.0
            axis = 'x' if axis_val < -0.33 else ('y' if axis_val > 0.33 else 'z')
            bone_transforms[bone] = {"freq": round(freq, 2), "amp": round(amp, 2), "axis": axis}
            param_idx += 3

        mass = float(jnp.abs(p[param_idx]) * 50.0 + 50.0) if param_idx < p.shape[0] else 75.0
        height = float(jnp.abs(p[param_idx + 1]) * 0.5 + 1.5) if param_idx + 1 < p.shape[0] else 1.75

        batch_metadata.append({
            "id": i + 1,
            "mass": round(mass, 1),
            "height": round(height, 2),
            "bone_transforms": bone_transforms,
            "preview_url": None
        })

    frontend_state["current_batch"] = batch_metadata

    for i in range(batch_size):
        item = batch_metadata[i]
        var_id = item["id"]
        rig_filename = f"rig_{var_id}_epoch_{frontend_state['epoch']}.json"
        
        frontend_state["status_message"] = f"Epoch {frontend_state['epoch']}: Generating {rig_filename} via Prolog for variation {var_id} of {batch_size}..."
        
        try:
            subprocess.run(
                ["tpl", "-l", "rig_gen.pl", "-g", f"write_random_rigging_program({var_id}), halt."],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if os.path.exists(f"rig_{var_id}.json"):
                shutil.copy(f"rig_{var_id}.json", rig_filename)
        except Exception as prolog_err:
            print(f"Trealla Prolog execution error on variation {var_id}: {prolog_err}")
            scene_ast = [{"bone": b, "parent": bone_parents.get(b), "transformations": [{"type": "translate", "values": [0.0, 0.0, 0.0]}, {"type": "oscillate", "axis": "x", "freq": 1.0 + (var_id * 0.1), "amp": 0.2}]} for b in SKELETAL_BONES]
            with open(rig_filename, "w") as f:
                json.dump(scene_ast, f, indent=2)

        frontend_state["status_message"] = f"Epoch {frontend_state['epoch']}: Rendering mesh GIF {var_id} of {batch_size}..."
        gif_filename = f"rigged_scene_var_{var_id}_epoch_{frontend_state['epoch']}.gif"
        try:
            scene_map = rig.generate_parametric_human(mass_kg=item["mass"], height_m=item["height"])
            rig.parse_and_apply_json_ast(rig_filename, scene_map)
            time_steps = np.linspace(0, 5.0, 100)
            rig.process_rigging_scene(scene_map, time_steps, output_filename=gif_filename)
            if os.path.exists(gif_filename):
                item["preview_url"] = f"/gif/{gif_filename}"
        except Exception as e:
            print(f"Simulation render error on variation {var_id}: {e}")

        frontend_state["current_batch"] = list(batch_metadata)

    frontend_state["waiting_for_human"] = True
    frontend_state["status_message"] = f"Epoch {frontend_state['epoch']}: Waiting for human RLHF scoring on 20 variations..."
    
    print(f"\n>>> Waiting for human RLHF scores for Epoch {frontend_state['epoch']} (20 items)...")
    while frontend_state["waiting_for_human"]:
        time.sleep(0.5)
        
    scores = frontend_state["latest_scores"]
    if not scores or len(scores) != batch_size:
        scores = [0.5] * batch_size
        
    return jnp.array(scores)

def run_training_loop(params):
    global_step = 0
    try:
        while True:
            frontend_state["epoch"] = global_step + 1
            print(f"\n================================")
            print(f"--- Training Step / Epoch {global_step + 1} (20-Batch) ---")
            print(f"================================")
            
            batch_scores = evaluate_batch_with_human_feedback(params, batch_size=20)
            print(f"Recorded Human RLHF Scores from SolidJS: {batch_scores}")
            
            mean_reward = float(jnp.mean(batch_scores))
            loss_val = float(1.0 - mean_reward)
            print(f"[Daemon] Step {global_step:04d} | Mean Reward: {mean_reward:.4f} | RLHF Loss: {loss_val:.4f}")
            
            global_step += 1
    except Exception as e:
        print(f"Training loop error: {e}")


# ---------------------------------------------------------
# 2D Render & Camera Preset Engine (Integrated)
# ---------------------------------------------------------
PRESETS = {
    "front": {"elevation": 0, "azim": 0, "color": (37, 99, 235)},
    "side": {"elevation": 0, "azim": 90, "color": (70, 130, 180)},
    "isometric": {"elevation": 35.264, "azim": 45, "color": (55, 65, 81)},
    "cinematic": {"elevation": 15, "azim": -60, "color": (235, 180, 40)}
}

def project_3d_to_2d(vertices, azim_deg, elev_deg, width, height, scale=120):
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

def generate_card_2d_video(json_path, mass, height, output_mp4, preset_name="isometric"):
    preset = PRESETS.get(preset_name, PRESETS["isometric"])
    azim, elev, color = preset["azim"], preset["elevation"], preset["color"]
    W, H, FPS = 240, 300, 30
    total_frames = 60 # 2 seconds loop

    scene_map = rig.generate_parametric_human(mass_kg=mass, height_m=height)
    rig.parse_and_apply_json_ast(json_path, scene_map)
    time_steps = np.linspace(0, 2.0, total_frames)

    os.makedirs("renders", exist_ok=True)
    raw_path = os.path.join("renders", f"temp_{os.path.basename(output_mp4)}.rgb")

    with open(raw_path, "wb") as f_out:
        for t in time_steps:
            frame = [248, 250, 252] * (W * H) # light background #f8fafc
            for obj in scene_map.values():
                mesh = obj.get_mesh_at_time(t)
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                    verts, faces = mesh.vertices, mesh.faces
                    proj = project_3d_to_2d(verts, azim, elev, W, H, scale=100)
                    for face in faces:
                        p0, p1, p2 = proj[face[0]], proj[face[1]], proj[face[2]]
                        for pa, pb in [(p0, p1), (p1, p2), (p2, p0)]:
                            steps = int(max(abs(pb[0] - pa[0]), abs(pb[1] - pa[1]))) + 1
                            for i in range(steps):
                                st = i / max(steps, 1)
                                x = int(pa[0] + (pb[0] - pa[0]) * st)
                                y = int(pa[1] + (pb[1] - pa[1]) * st)
                                if 0 <= x < W and 0 <= y < H:
                                    idx = (y * W + x) * 3
                                    frame[idx], frame[idx+1], frame[idx+2] = color[0], color[1], color[2]
            f_out.write(bytearray(frame))

    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(raw_path):
        os.remove(raw_path)

@app.post("/api/render-card")
async def api_render_card(request: Request):
    data = await request.json()
    var_id = data.get("var_id")
    epoch = data.get("epoch", frontend_state["epoch"])
    preset = data.get("preset", "isometric")
    
    rig_filename = f"rig_{var_id}_epoch_{epoch}.json"
    if not os.path.exists(rig_filename):
        rig_filename = f"rig_{var_id}.json"
        
    item = next((x for x in frontend_state["current_batch"] if x["id"] == var_id), {"mass": 75.0, "height": 1.75})
    output_filename = f"var_{var_id}_epoch_{epoch}_{preset}.mp4"
    output_path = os.path.join("renders", output_filename)
    
    try:
        generate_card_2d_video(rig_filename, item["mass"], item["height"], output_path, preset_name=preset)
        return {"status": "success", "video_url": f"/renders/{output_filename}"}
    except Exception as e:
        print(f"2D card render error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/render-pointcloud")
async def api_render_pointcloud(request: Request):
    data = await request.json()
    var_id = data.get("var_id")
    epoch = data.get("epoch", frontend_state["epoch"])
    
    item = next((x for x in frontend_state["current_batch"] if x["id"] == var_id), {"mass": 75.0, "height": 1.75})
    output_filename = f"var_{var_id}_epoch_{epoch}_pc.mp4"
    output_path = os.path.join("renders", output_filename)
    
    try:
        point_cloud_renderer.render_point_cloud_animation("rigged_glbs", output_path, mass=item["mass"], height=item["height"])
        return {"status": "success", "video_url": f"/renders/{output_filename}"}
    except Exception as e:
        print(f"Point Cloud render error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/render-opengl")
async def api_render_opengl(request: Request):
    data = await request.json()
    var_id = data.get("var_id")
    epoch = data.get("epoch", frontend_state["epoch"])
    
    item = next((x for x in frontend_state["current_batch"] if x["id"] == var_id), {"mass": 75.0, "height": 1.75})
    output_filename = f"var_{var_id}_epoch_{epoch}_gl.mp4"
    output_path = os.path.join("renders", output_filename)
    
    try:
        opengl_renderer.render_opengl_mesh_animation("rigged_glbs", output_path, mass=item["mass"], height=item["height"])
        return {"status": "success", "video_url": f"/renders/{output_filename}"}
    except Exception as e:
        print(f"OpenGL render error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

os.makedirs("renders", exist_ok=True)
app.mount("/renders", StaticFiles(directory="renders"), name="renders")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("ntk_logs", exist_ok=True)
    
    optimizer = optax.adam(1e-4)
    seed_key = get_or_create_init_seed(42)
    
    with open(CKPT_LOCK_PATH, "a+") as clf:
        fcntl.flock(clf, fcntl.LOCK_EX)
        try:
            if not os.path.exists(CURR_CKPT):
                init_p = init_params(jax.random.PRNGKey(seed_key))
                pickle.dump({"params": init_p, "ema_params": init_p, "opt_state": optimizer.init(init_p), "version": 0, "global_step": 0}, open(CURR_CKPT, "wb"))
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)

    bundle = load_checkpoint_safely()
    params = bundle["params"] if isinstance(bundle, dict) else bundle

    training_thread = threading.Thread(target=run_training_loop, args=(params,), daemon=True)
    training_thread.start()

    print("[Daemon] Starting FastAPI server on port 8000 (Foreground Uvicorn Process)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")