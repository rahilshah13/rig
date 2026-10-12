import os, json, pickle, queue, threading, time, sys, struct, subprocess, ctypes, fcntl, shutil, glob, math, optax, jax, argparse
import numpy as np, jax.numpy as jnp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from functools import partial, reduce
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

CONCURRENT_MODELS = 3
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
    "status_message": "Initializing 16-item Trealla Prolog & JAX training pipeline...",
    "real_loss": 0.0,
    "normalized_loss": 0.0,
    "current_operation": "Idle"
}

# -----------------------------------------------------------------------------
# 1. Transformer Architecture & Diffusion Core
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2. VAE Engine & RLHF Latent Alignment
# -----------------------------------------------------------------------------
def init_vae_params(key, latent_dim=128, input_dim=60, hidden_dim=512):
    keys = jax.random.split(key, 5)
    return {
        'enc_dense': xavier_normal(keys[0], (input_dim, hidden_dim)),
        'enc_mean': xavier_normal(keys[1], (hidden_dim, latent_dim)),
        'enc_logvar': xavier_normal(keys[2], (hidden_dim, latent_dim)),
        'dec_dense': xavier_normal(keys[3], (latent_dim, hidden_dim)),
        'dec_out': xavier_normal(keys[4], (hidden_dim, input_dim))
    }

def vae_encode(vae_params, x):
    h = jax.nn.gelu(x @ vae_params['enc_dense'])
    mean = h @ vae_params['enc_mean']
    logvar = h @ vae_params['enc_logvar']
    return mean, logvar

def vae_reparameterize(key, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, mean.shape)
    return mean + eps * std

def vae_decode(vae_params, z):
    h = jax.nn.gelu(z @ vae_params['dec_dense'])
    return h @ vae_params['dec_out']

def vae_loss_fn(vae_params, vae_state_key, x, human_scores, beta=0.01):
    mean, logvar = vae_encode(vae_params, x)
    z = vae_reparameterize(vae_state_key, mean, logvar)
    recon_x = vae_decode(vae_params, z)
    
    recon_loss = jnp.mean(jnp.square(recon_x - x))
    kl_loss = -0.5 * jnp.mean(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
    
    reward_weight = (1.0 - human_scores)[:, None]
    rlhf_penalty = jnp.mean(reward_weight * jnp.square(z))
    
    total_loss = recon_loss + beta * kl_loss + 0.1 * rlhf_penalty
    return total_loss, (recon_loss, kl_loss, rlhf_penalty)

@jax.jit
def vae_update_step(vae_params, opt_state, opt_def, key, x, scores):
    grad_fn = jax.value_and_grad(vae_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(vae_params, key, x, scores)
    updates, new_opt_state = opt_def.update(grads, opt_state, vae_params)
    new_params = optax.apply_updates(vae_params, updates)
    return new_params, new_opt_state, loss, aux

# -----------------------------------------------------------------------------
# 3. ARM64 NEON SIMD Compilation Engine
# -----------------------------------------------------------------------------
def compile_closed_jaxpr_to_arm64(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    sym_name = "_jax_arm64_simd_kernel" if sys.platform == "darwin" else "jax_arm64_simd_kernel"
    
    asm = [
        ".text",
        ".align 4",
        f".global {sym_name}",
        f"{sym_name}:",
        "    sub sp, sp, #65536",
        "    stp x29, x30, [sp]",
        "    mov x29, sp"
    ]

    literals = {}
    for eqn in jaxpr.eqns:
        for invar in eqn.invars:
            if type(invar).__name__ == "Literal" or hasattr(invar, "val"):
                val = float(invar.val)
                if np.isfinite(val) and val not in literals:
                    literals[val] = None

    free_regs = [f"v{i}.4s" for i in range(8, 32)]
    total_vector_regs = len(free_regs)
    
    for val in literals:
        if not free_regs: break
        literals[val] = free_regs.pop(0)

    last_use = {}
    for i, eqn in enumerate(jaxpr.eqns):
        for invar in eqn.invars:
            if not (type(invar).__name__ == "Literal" or hasattr(invar, "val")):
                last_use[invar] = i
    for outvar in jaxpr.outvars:
        last_use[outvar] = len(jaxpr.eqns)

    reg_map = {}
    spilled_vars = {}
    stack_ptr = 16
    live_trace = []

    def alloc_reg(var, step_idx):
        nonlocal stack_ptr
        if not free_regs:
            var_to_spill = list(reg_map.keys())[0]
            reg = reg_map.pop(var_to_spill)
            stack_ptr += 16
            asm.append(f"    str {reg.replace('.4s', '').replace('v', 'q')}, [sp, #{stack_ptr}]")
            spilled_vars[var_to_spill] = stack_ptr
            free_regs.append(reg)
        reg = free_regs.pop(0)
        reg_map[var] = reg
        return reg

    def get_reg(var, step_idx):
        if var in reg_map: return reg_map[var]
        elif var in spilled_vars:
            offset = spilled_vars[var]
            reg = alloc_reg(var, step_idx)
            asm.append(f"    ldr {reg.replace('.4s', '').replace('v', 'q')}, [sp, #{offset}]")
            del spilled_vars[var]
            return reg
        return alloc_reg(var, step_idx)

    def free_dead(step):
        for var, reg in list(reg_map.items()):
            if last_use.get(var, -1) == step:
                free_regs.append(reg)
                del reg_map[var]

    asm.append("    mov x8, #0")
    asm.append("    lsl x9, x1, #2")
    asm.append(".loop_start:")
    asm.append("    cmp x8, x9")
    asm.append("    b.ge .loop_end")

    for i, invar in enumerate(jaxpr.invars):
        reg = alloc_reg(invar, 0)
        asm.append(f"    ldr {reg.replace('.4s', '').replace('v', 'q')}, [x{i + 2}, x8]")

    prim_map = {"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv"}

    for step, eqn in enumerate(jaxpr.eqns):
        prim = eqn.primitive.name
        if prim not in prim_map:
            free_dead(step)
            live_trace.append(len(reg_map))
            continue
        
        in_regs = [literals[float(inv.val)] if (type(inv).__name__ == "Literal" or hasattr(inv, "val")) else get_reg(inv, step) for inv in eqn.invars]
        out_reg = alloc_reg(eqn.outvars[0], step)
        asm.append(f"    {prim_map[prim]} {out_reg}, {in_regs[0]}, {in_regs[1]}")
        free_dead(step)
        live_trace.append(len(reg_map))

    final_reg = get_reg(jaxpr.outvars[0], len(jaxpr.eqns))
    asm.append(f"    str {final_reg.replace('.4s', '').replace('v', 'q')}, [x0, x8]")
    asm.append("    add x8, x8, #16")
    asm.append("    b .loop_start")
    asm.append(".loop_end:")
    asm.append("    ldp x29, x30, [sp]")
    asm.append("    add sp, sp, #65536")
    asm.append("    ret")

    return "\n".join(asm), {"peak_registers": max(live_trace) if live_trace else 0, "total_regs": total_vector_regs}

# -----------------------------------------------------------------------------
# 4. API Endpoints & Training Loop
# -----------------------------------------------------------------------------
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
        "status": frontend_state["status_message"],
        "real_loss": round(frontend_state["real_loss"], 5),
        "normalized_loss": round(frontend_state["normalized_loss"], 5),
        "current_operation": frontend_state["current_operation"]
    })

@app.post("/api/score")
async def submit_human_scores(request: Request):
    data = await request.json()
    scores = data.get("scores", [])
    print(f"\n[Human Feedback Received via SolidJS for 16 items]: {scores}")
    frontend_state["latest_scores"] = scores
    frontend_state["waiting_for_human"] = False
    return {"status": "success", "received_scores": scores}

@app.post("/api/benchmark-simd")
def benchmark_simd():
    try:
        key = jax.random.PRNGKey(42)
        batch_x = jnp.ones((2, 4, 44100), dtype=jnp.float32)
        sigmas = jnp.ones((2, 4), dtype=jnp.float32)
        scales = jnp.zeros((2,), dtype=jnp.int32)
        bpms = jnp.full((2,), 120.0, dtype=jnp.float32)
        stems = jnp.zeros((2,), dtype=jnp.int32)
        steps = jnp.zeros((2, 4), dtype=jnp.int32)
        masks = jnp.ones((2, 4), dtype=bool)
        
        bundle = load_checkpoint_safely()
        current_params = bundle["params"] if isinstance(bundle, dict) else bundle

        def trace_fn(x):
            cond = (scales, bpms, stems, steps, sigmas)
            return gpt_forward(current_params, x, cond, masks, modality_type="audio")

        closed_jaxpr = jax.make_jaxpr(trace_fn)(batch_x)
        arm_asm, metrics = compile_closed_jaxpr_to_arm64(closed_jaxpr)

        with open("kernel.s", "w") as f:
            f.write(arm_asm)

        link_flags = ["-shared", "-o", "./libarm.so", "kernel.s"]
        if sys.platform == "darwin":
            link_flags.extend(["-Wl,-exported_symbol,_jax_arm64_simd_kernel"])
        subprocess.run(["gcc"] + link_flags, check=True)

        lib_arm = ctypes.CDLL("./libarm.so")
        data_in = np.ones(44100, dtype=np.float32)
        data_out = np.zeros(44100, dtype=np.float32)

        start = time.time()
        for _ in range(20):
            lib_arm.jax_arm64_simd_kernel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            lib_arm.jax_arm64_simd_kernel(data_out.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(44100), data_in.ctypes.data_as(ctypes.c_void_p))
        arm_time_ms = (time.time() - start) / 20 * 1000

        start = time.time()
        for _ in range(20):
            _ = trace_fn(batch_x)
        jax_time_ms = (time.time() - start) / 20 * 1000

        return {
            "status": "success",
            "jax_latency_ms": round(jax_time_ms, 3),
            "arm64_neon_latency_ms": round(arm_time_ms, 3),
            "speedup": round(jax_time_ms / max(arm_time_ms, 0.001), 2),
            "peak_registers": metrics["peak_registers"],
            "total_regs": metrics["total_regs"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

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

def evaluate_batch_with_human_feedback(params, vae_params, batch_size=16):
    frontend_state["current_operation"] = "JAX Diffusion Forward Pass"
    frontend_state["status_message"] = f"Epoch {frontend_state['epoch']}: Running JAX diffusion forward pass for {batch_size} variations..."
    
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
            "preview_url": f"/gif/rigged_scene_var_{i+1}_epoch_{frontend_state['epoch']}.gif"
        })

    frontend_state["current_batch"] = batch_metadata
    frontend_state["current_operation"] = "Waiting for Human RLHF Scoring"
    frontend_state["waiting_for_human"] = True
    frontend_state["status_message"] = f"Epoch {frontend_state['epoch']}: Waiting for human RLHF scoring on {batch_size} variations..."
    
    while frontend_state["waiting_for_human"]:
        time.sleep(0.5)
        
    scores = frontend_state["latest_scores"]
    if not scores or len(scores) != batch_size:
        scores = [0.5] * batch_size
        
    return jnp.array(scores)

def run_training_loop(params, vae_params):
    global_step = 0
    vae_opt_def = optax.adam(3e-4)
    vae_opt_state = vae_opt_def.init(vae_params)
    
    try:
        while True:
            frontend_state["epoch"] = global_step + 1
            frontend_state["current_operation"] = "Batch Evaluation & JAX Forward Pass"
            
            batch_scores = evaluate_batch_with_human_feedback(params, vae_params, batch_size=16)
            
            frontend_state["current_operation"] = "Feature Extraction & VAE Update"
            batch_features = []
            for item in frontend_state["current_batch"]:
                feat = []
                for bone in SKELETAL_BONES:
                    b_cfg = item["bone_transforms"].get(bone, {"freq": 1.0, "amp": 0.2, "axis": 'x'})
                    axis_val = -1.0 if b_cfg["axis"] == 'x' else (1.0 if b_cfg["axis"] == 'y' else 0.0)
                    feat.extend([b_cfg["freq"], b_cfg["amp"], axis_val])
                feat.extend([item["mass"] / 100.0, item["height"] / 2.0])
                batch_features.append(feat)
            
            x_batch = jnp.array(batch_features, dtype=jnp.float32)
            key = jax.random.PRNGKey(global_step)
            
            vae_params, vae_opt_state, vae_loss, aux = vae_update_step(
                vae_params, vae_opt_state, vae_opt_def, key, x_batch, batch_scores
            )
            
            frontend_state["real_loss"] = float(aux[0])
            frontend_state["normalized_loss"] = float(vae_loss)
            
            print(f"[VAE RLHF Training] Step {global_step:04d} | Real Loss: {aux[0]:.4f} | Normalized Loss: {vae_loss:.4f} | KL: {aux[1]:.4f} | Penalty: {aux[2]:.4f}")
            global_step += 1
    except Exception as e:
        print(f"Training loop error: {e}")

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
                init_vae_p = init_vae_params(jax.random.PRNGKey(seed_key + 1))
                pickle.dump({
                    "params": init_p, 
                    "ema_params": init_p,
                    "vae_params": init_vae_p,
                    "opt_state": optimizer.init(init_p), 
                    "version": 0, 
                    "global_step": 0
                }, open(CURR_CKPT, "wb"))
        finally:
            fcntl.flock(clf, fcntl.LOCK_UN)

    bundle = load_checkpoint_safely()
    params = bundle["params"] if isinstance(bundle, dict) and "params" in bundle else bundle
    vae_params = bundle["vae_params"] if isinstance(bundle, dict) and "vae_params" in bundle else init_vae_params(jax.random.PRNGKey(43))

    training_thread = threading.Thread(target=run_training_loop, args=(params, vae_params), daemon=True)
    training_thread.start()

    print("[Daemon] Starting Decoupled FastAPI server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")