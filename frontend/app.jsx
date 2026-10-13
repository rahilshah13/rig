import { render } from "solid-js/web";
import { createStore } from "solid-js/store";
import { onMount, onCleanup, For } from "solid-js";

function RigCanvas(props) {
  let canvasRef;
  let animationFrameId;

  const boneParents = {
    'Spine': 'Hips', 'Chest': 'Spine', 'Neck': 'Chest', 'Head': 'Neck',
    'LeftShoulder': 'Chest', 'LeftUpperArm': 'LeftShoulder', 'LeftLowerArm': 'LeftUpperArm', 'LeftHand': 'LeftLowerArm',
    'RightShoulder': 'Chest', 'RightUpperArm': 'RightShoulder', 'RightLowerArm': 'RightUpperArm', 'RightHand': 'RightLowerArm',
    'LeftUpperLeg': 'Hips', 'LeftLowerLeg': 'LeftUpperLeg', 'LeftFoot': 'LeftLowerLeg',
    'RightUpperLeg': 'Hips', 'RightLowerLeg': 'RightUpperLeg', 'RightFoot': 'RightLowerLeg'
  };

  const baseOffsets = {
    'Hips': [0, 0, 0], 'Spine': [0, 20, 0], 'Chest': [0, 40, 0], 'Neck': [0, 60, 0], 'Head': [0, 75, 0],
    'LeftShoulder': [-20, 45, 0], 'LeftUpperArm': [-35, 30, 0], 'LeftLowerArm': [-50, 10, 0], 'LeftHand': [-65, -10, 0],
    'RightShoulder': [20, 45, 0], 'RightUpperArm': [35, 30, 0], 'RightLowerArm': [50, 10, 0], 'RightHand': [65, -10, 0],
    'LeftUpperLeg': [-15, -20, 0], 'LeftLowerLeg': [-15, -55, 0], 'LeftFoot': [-15, -90, 15],
    'RightUpperLeg': [15, -20, 0], 'RightLowerLeg': [15, -55, 0], 'RightFoot': [15, -90, 15]
  };

  onMount(() => {
    const ctx = canvasRef.getContext("2d");
    let startTime = Date.now();

    const renderLoop = () => {
      const t = (Date.now() - startTime) / 1000;
      ctx.clearRect(0, 0, canvasRef.width, canvasRef.height);

      ctx.save();
      ctx.translate(canvasRef.width / 2, canvasRef.height / 2 + 10);
      ctx.scale(1.0, -1.0);

      const transforms = props.transforms || {};
      const computedPositions = {};

      Object.keys(baseOffsets).forEach(bone => {
        const base = baseOffsets[bone];
        const cfg = transforms[bone] || { freq: 1.0, amp: 0.2, axis: 'x' };
        let dx = 0, dy = 0, dz = 0;
        const oscillation = Math.sin(2 * Math.PI * cfg.freq * t) * cfg.amp * 20;
        if (cfg.axis === 'x') dx = oscillation;
        else if (cfg.axis === 'y') dy = oscillation;
        else dz = oscillation;

        computedPositions[bone] = [base[0] + dx, base[1] + dy, base[2] + dz];
      });

      ctx.lineWidth = 2.0;
      ctx.strokeStyle = "#0f172a";
      ctx.fillStyle = "#0f172a";

      Object.keys(boneParents).forEach(bone => {
        const parent = boneParents[bone];
        const p1 = computedPositions[bone];
        const p2 = computedPositions[parent];

        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1[0], p1[1]);
          ctx.lineTo(p2[0], p2[1]);
          ctx.stroke();
        }
      });

      Object.keys(computedPositions).forEach(bone => {
        const p = computedPositions[bone];
        ctx.beginPath();
        ctx.arc(p[0], p[1], 2.8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });

      ctx.restore();
      animationFrameId = requestAnimationFrame(renderLoop);
    };

    renderLoop();
  });

  onCleanup(() => cancelAnimationFrame(animationFrameId));

  return (
    <div style="width: 100%; height: 100%; min-height: 220px; max-height: 260px; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #ffffff; border-radius: 6px; box-sizing: border-box;">
      <canvas ref={canvasRef} width="280" height="240" style="width: 100%; height: 100%; object-fit: contain; display: block;" />
    </div>
  );
}

function ExplainerModal(props) {
  return (
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0, 0, 0, 0.5); backdrop-filter: blur(2px); display: flex; align-items: center; justify-content: center; z-index: 1000; padding: 20px;">
      <div style="background: #ffffff; width: 100%; max-width: 700px; border-radius: 8px; padding: 24px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12); display: flex; flex-direction: column; gap: 14px; max-height: 85vh; overflow-y: auto; text-align: left; color: #0f172a; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px;">
          <h2 style="font-size: 16px; font-weight: 700; margin: 0; text-transform: uppercase;">Machine Learning Objectives</h2>
          <button onClick={props.onClose} style="background: #f8fafc; border: none; font-size: 14px; font-weight: bold; width: 28px; height: 28px; border-radius: 50%; cursor: pointer; color: #0f172a; display: flex; align-items: center; justify-content: center;">✕</button>
        </div>
        <div style="font-size: 13px; line-height: 1.5; display: flex; flex-direction: column; gap: 8px;">
          <p><strong>Store-backed State:</strong> Application view states and telemetry logs are synchronized via SolidJS store bindings.</p>
          <p><strong>Selective RLHF:</strong> Unscored samples are masked out during optimization, ensuring human feedback is exclusively incorporated in active alignment passes.</p>
          <p><strong>AOT ARM64 Compilation:</strong> JAXPR execution graphs are compiled to ARM64 NEON assembly kernels for high-throughput latent sampling.</p>
        </div>
        <button onClick={props.onClose} style="padding: 8px 16px; background: #0f172a; color: #ffffff; border: none; border-radius: 6px; font-weight: 700; cursor: pointer; align-self: flex-end; text-transform: uppercase; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">Close</button>
      </div>
    </div>
  );
}

function App() {
  const [state, setState] = createStore({
    batchData: { 
      epoch: 1, 
      global_step: 0,
      repeat_step: 0,
      batch: [], 
      currentIndex: 0,
      status: "Initializing...", 
      actual_loss: 0.0, 
      normalized_loss: 0.0,
      current_operation: "Idle" 
    },
    currentScore: 0.5,
    showModal: false,
    viewMode: "rig",
    preset: "isometric",
    url: "",
    isRendering: false,
    isPaused: false,
    selectedCheckpoint: "checkpoint_bundle.pickle",
    vaeSamplingMode: "jax",
    inferenceLatentScale: 1.0,
    inferenceStepCount: 10,
    inferenceResult: null,
    isInferencing: false
  });

  onMount(() => {
    document.title = "RigStudio";
  });

  const fetchData = async () => {
    if (!state.isPaused) {
      try {
        const res = await fetch("/api/batch");
        if (res.ok) {
          const data = await res.json();
          if (data) setState("batchData", data);
        }
      } catch (err) {
        console.error("Batch fetch error:", err);
      }
    }
  };

  onMount(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    onCleanup(() => clearInterval(interval));
  });

  const togglePause = async () => {
    const nextState = !state.isPaused;
    setState("isPaused", nextState);
    try {
      await fetch("/api/training/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: nextState ? "pause" : "resume" })
      });
    } catch (err) {
      console.error("Training control error:", err);
    }
  };

  const handleCheckpointChange = async (e) => {
    const ckpt = e.target.value;
    setState("selectedCheckpoint", ckpt);
    try {
      await fetch("/api/training/checkpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint: ckpt })
      });
    } catch (err) {
      console.error("Checkpoint change error:", err);
    }
  };

  const runFunctionalInference = async () => {
    setState("isInferencing", true);
    try {
      const res = await fetch("/api/vae/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: state.vaeSamplingMode, scale: state.inferenceLatentScale, steps: state.inferenceStepCount })
      });
      const data = await res.json();
      setState("inferenceResult", data);
    } catch (err) {
      console.error("Inference error:", err);
      setState("inferenceResult", { status: "error", message: err.toString() });
    } finally {
      setState("isInferencing", false);
    }
  };

  const switchView = async (mode, selectedPreset = "isometric") => {
    setState("viewMode", mode);
    if (mode === "rig") return;
    setState("preset", selectedPreset);
    const batch = state.batchData.batch;
    const item = batch[state.batchData.currentIndex];
    if (!item) return;

    if (mode === "gif") {
      setState("isRendering", true);
      try {
        const res = await fetch("/api/render-card", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ var_id: item.id, epoch: state.batchData.epoch, preset: selectedPreset, duration_sec: item.duration_sec || 4, bone_transforms: item.bone_transforms })
        });
        const data = await res.json();
        if (data.status === "success") setState("url", data.gif_url);
      } catch (err) {
        console.error("GIF Render error:", err);
      } finally {
        setState("isRendering", false);
      }
      return;
    }

    setState("isRendering", true);
    let endpoint = "/api/render-card";
    let payload = { var_id: item.id, epoch: state.batchData.epoch, preset: selectedPreset, duration_sec: item.duration_sec || 6, bone_transforms: item.bone_transforms };
    if (mode === "pc") endpoint = "/api/render-pointcloud";
    else if (mode === "opengl") endpoint = "/api/render-opengl";

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === "success") setState("url", data.video_url);
    } catch (err) {
      console.error("Render error:", err);
    } finally {
      setState("isRendering", false);
    }
  };

  const submitScoreAndNext = async () => {
    const batch = state.batchData.batch;
    const item = batch[state.batchData.currentIndex];
    if (!item) return;

    try {
      await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_id: item.id, score: state.currentScore })
      });
    } catch (err) {
      console.error("Score submit error:", err);
    }

    setState({ currentScore: 0.5, url: "", viewMode: "rig" });
    fetchData();
  };

  return (
    <div style="font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #0f172a; height: 100vh; max-height: 100vh; overflow: hidden; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 8px 12px; box-sizing: border-box;">
      <div style="width: 100%; max-width: 1100px; display: flex; flex-direction: column; align-items: center; text-align: center; height: 100%; gap: 6px; box-sizing: border-box;">
        
        {/* Header */}
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; background: #ffffff; padding: 8px 14px; border-radius: 8px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04); box-sizing: border-box;">
          <h1 style="font-size: 13px; font-weight: 800; color: #0f172a; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">RigStudio</h1>
          <button onClick={() => setState("showModal", true)} style="padding: 4px 10px; background: #f1f5f9; color: #0f172a; border: none; border-radius: 6px; font-size: 10px; font-weight: 700; cursor: pointer;">
            ℹ️ Objectives
          </button>
        </div>

        {/* Main Content Area */}
        {state.batchData.batch.length === 0 ? (
          <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04); width: 100%; flex-grow: 1; box-sizing: border-box;">
            <div style="width: 24px; height: 24px; border: 3px solid #0f172a; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 6px;"></div>
            <p style="font-size: 12px; color: #0f172a; font-weight: 700;">Streaming procedural variation from JAX VAE...</p>
            <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          </div>
        ) : (
          (() => {
            const batch = state.batchData.batch;
            const idx = state.batchData.currentIndex % batch.length;
            const item = batch[idx];
            if (!item) return null;

            return (
              <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 12px; width: 100%; flex-grow: 1; min-height: 0; align-items: stretch; box-sizing: border-box; text-align: left;">
                
                {/* Left Column: RLHF Grading Component */}
                <div style="background: #ffffff; padding: 12px 14px; border-radius: 8px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04); display: flex; flex-direction: column; justify-content: space-between; color: #0f172a; box-sizing: border-box; height: 100%;">
                  
                  <div style="display: flex; flex-direction: column; gap: 4px; width: 100%;">
                    <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                      <h3 style="font-size: 12px; font-weight: 700; margin: 0;">Variation #{item.id} (Item {idx + 1} of {batch.length})</h3>
                      <div style="display: flex; gap: 3px;">
                        <button onClick={() => switchView("rig")} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; background: ${state.viewMode === 'rig' ? '#0f172a' : '#f1f5f9'}; color: ${state.viewMode === 'rig' ? '#ffffff' : '#0f172a'}; cursor: pointer; font-weight: bold;`}>RIG</button>
                        <button onClick={() => switchView("gif", "isometric")} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; background: ${state.viewMode === 'gif' ? '#0f172a' : '#f1f5f9'}; color: ${state.viewMode === 'gif' ? '#ffffff' : '#0f172a'}; cursor: pointer; font-weight: bold;`}>GIF</button>
                        <button onClick={() => switchView("2d", state.preset)} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; background: ${state.viewMode === '2d' ? '#0f172a' : '#f1f5f9'}; color: ${state.viewMode === '2d' ? '#ffffff' : '#0f172a'}; cursor: pointer; font-weight: bold;`}>2D</button>
                        <button onClick={() => switchView("pc")} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; background: ${state.viewMode === 'pc' ? '#0f172a' : '#f1f5f9'}; color: ${state.viewMode === 'pc' ? '#ffffff' : '#0f172a'}; cursor: pointer; font-weight: bold;`}>PC</button>
                        <button onClick={() => switchView("opengl")} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; background: ${state.viewMode === 'opengl' ? '#0f172a' : '#f1f5f9'}; color: ${state.viewMode === 'opengl' ? '#ffffff' : '#0f172a'}; cursor: pointer; font-weight: bold;`}>GL</button>
                      </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                      <p style="font-size: 10px; color: #64748b; margin: 0;">Mass: <strong>{item.mass}kg</strong> | Height: <strong>{item.height}m</strong> | Length: <strong>{item.duration_sec || 6}s</strong></p>
                      {state.viewMode === "2d" && (
                        <select value={state.preset} onChange={(e) => switchView("2d", e.target.value)} style="font-size: 9px; padding: 2px 4px; border-radius: 4px; border: 1px solid #e2e8f0; background: #f8fafc; color: #0f172a;">
                          <option value="isometric">Isometric</option>
                          <option value="cinematic">Cinematic</option>
                          <option value="front">Front</option>
                          <option value="side">Side</option>
                        </select>
                      )}
                    </div>
                  </div>

                  {/* Viewport Container */}
                  <div style="display: flex; width: 100%; flex-grow: 1; align-items: center; justify-content: center; margin: 4px 0; overflow: hidden; box-sizing: border-box; background: #000000; border-radius: 6px; min-height: 220px;">
                    {state.viewMode === 'rig' ? (
                      <RigCanvas transforms={item.bone_transforms} />
                    ) : state.viewMode === 'gif' ? (
                      state.isRendering ? (
                        <div style="width: 100%; height: 100%; background: #f8fafc; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 10px;">Rendering GIF...</div>
                      ) : (
                        state.url ? (
                          <img src={state.url} alt="Rigged 3D Human GIF" style="width: 100%; height: 100%; max-height: 240px; object-fit: contain; display: block;" />
                        ) : (
                          <div style="width: 100%; height: 100%; background: #f8fafc; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 10px;">Click GIF button to render animation</div>
                        )
                      )
                    ) : (
                      state.isRendering ? (
                        <div style="width: 100%; height: 100%; background: #f8fafc; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 10px;">Rendering...</div>
                      ) : (
                        state.url ? (
                          <video controls autoplay loop muted playsinline preload="auto" controlslist="nodownload nofullscreen noremoteplayback" disablePictureInPicture src={state.url} style="width: 100%; height: 100%; max-height: 240px; object-fit: contain; display: block;" />
                        ) : (
                          <div style="width: 100%; height: 100%; background: #f8fafc; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 10px;">Ready</div>
                        )
                      )
                    )}
                  </div>

                  <div style="width: 100%; display: flex; flex-direction: column; gap: 4px;">
                    <label style="font-size: 10px; font-weight: 700; display: flex; justify-content: space-between;">
                      <span>Score Reward:</span>
                      <span>{state.currentScore}</span>
                    </label>
                    <input type="range" min="0" max="1" step="0.1" value={state.currentScore} 
                      onInput={(e) => setState("currentScore", parseFloat(e.target.value))} 
                      style="width: 100%; cursor: pointer; accent-color: #0f172a; height: 14px;" />
                    
                    <button onClick={submitScoreAndNext} style="margin-top: 2px; padding: 8px 12px; background: #0f172a; color: #ffffff; border: none; border-radius: 6px; cursor: pointer; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; width: 100%; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.2);">
                      Submit Score & Continue
                    </button>
                  </div>

                </div>

                {/* Right Column: Training Dashboard stacked above Inference Component */}
                <div style="display: flex; flex-direction: column; gap: 12px; height: 100%; box-sizing: border-box;">
                  
                  {/* Training Status & Dashboard Component with Active Training Run Label, Pause, and Checkpoint Selection */}
                  <div style={`background: ${state.isPaused ? '#fef2f2' : '#ffffff'}; border: ${state.isPaused ? '1px solid #ef4444' : '1px solid transparent'}; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04); color: #0f172a; padding: 10px 14px; border-radius: 8px; font-size: 10px; font-weight: 600; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box; transition: all 0.2s ease;`}>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 4px;">
                      <span style="font-weight: 800; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; color: #0f172a;">Active Training Run</span>
                      {state.isPaused && (
                        <span style="background: #ef4444; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-weight: 800; font-size: 8px; text-transform: uppercase;">PAUSED</span>
                      )}
                    </div>

                    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                      <span><strong>Op:</strong> {state.batchData.current_operation || "Initializing..."}</span>
                      <span><strong>Epoch:</strong> {state.batchData.epoch}</span>
                    </div>

                    <div style="display: flex; justify-content: space-around; width: 100%; border-top: 1px solid #f1f5f9; padding-top: 4px;">
                      <span>Actual Loss: <strong style="color: #0f172a;">{state.batchData.actual_loss ?? 0.0000}</strong></span>
                      <span>Normalized Loss: <strong style="color: #0f172a;">{state.batchData.normalized_loss ?? 0.0000}</strong></span>
                    </div>

                    {/* Checkpoint & Pause/Resume Controls inside Training Component */}
                    <div style="display: flex; gap: 6px; align-items: center; justify-content: space-between; background: #f8fafc; padding: 4px 6px; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: 2px; box-sizing: border-box;">
                      <select value={state.selectedCheckpoint} onChange={handleCheckpointChange} style="font-size: 9px; padding: 3px 4px; border-radius: 4px; border: 1px solid #e2e8f0; background: #ffffff; color: #0f172a; cursor: pointer; flex-grow: 1;">
                        <option value="checkpoint_bundle.pickle">Latest Bundle</option>
                        <option value="checkpoint_bundle_prev.pickle">Previous Bundle</option>
                      </select>
                      <button onClick={togglePause} style={`padding: 4px 8px; border-radius: 4px; border: none; font-size: 9px; font-weight: 800; cursor: pointer; background: ${state.isPaused ? '#ef4444' : '#0f172a'}; color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);`}>
                        {state.isPaused ? "▶ RESUME" : "⏸ PAUSE"}
                      </button>
                    </div>
                  </div>

                  {/* Inference Component */}
                  <div style="background: #ffffff; padding: 12px 14px; border-radius: 8px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04); display: flex; flex-direction: column; justify-content: space-between; color: #0f172a; box-sizing: border-box; flex-grow: 1;">
                    
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                      <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="font-size: 12px; font-weight: 700; margin: 0; text-transform: uppercase;">Inference</h3>
                        <div style="display: flex; gap: 3px;">
                          <button onClick={() => setState("vaeSamplingMode", "jax")} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; cursor: pointer; font-weight: bold; background: ${state.vaeSamplingMode === 'jax' ? '#0f172a' : '#f1f5f9'}; color: ${state.vaeSamplingMode === 'jax' ? '#ffffff' : '#0f172a'};`}>JAX</button>
                          <button onClick={() => setState("vaeSamplingMode", "arm64")} style={`font-size: 9px; padding: 2px 6px; border-radius: 4px; border: none; cursor: pointer; font-weight: bold; background: ${state.vaeSamplingMode === 'arm64' ? '#0f172a' : '#f1f5f9'}; color: ${state.vaeSamplingMode === 'arm64' ? '#ffffff' : '#0f172a'};`}>ARM64</button>
                        </div>
                      </div>

                      <div style="display: flex; flex-direction: column; gap: 2px;">
                        <label style="font-size: 10px; font-weight: 700; display: flex; justify-content: space-between;">
                          <span>Latent Scale:</span>
                          <span>{state.inferenceLatentScale}x</span>
                        </label>
                        <input type="range" min="0.1" max="3.0" step="0.1" value={state.inferenceLatentScale}
                          onInput={(e) => setState("inferenceLatentScale", parseFloat(e.target.value))}
                          style="width: 100%; cursor: pointer; accent-color: #0f172a; height: 14px;" />
                      </div>

                      <div style="display: flex; flex-direction: column; gap: 2px;">
                        <label style="font-size: 10px; font-weight: 700; display: flex; justify-content: space-between;">
                          <span>Sampling Steps:</span>
                          <span>{state.inferenceStepCount}</span>
                        </label>
                        <input type="range" min="1" max="50" step="1" value={state.inferenceStepCount}
                          onInput={(e) => setState("inferenceStepCount", parseInt(e.target.value))}
                          style="width: 100%; cursor: pointer; accent-color: #0f172a; height: 14px;" />
                      </div>
                    </div>

                    {/* Inference Preview & Latent Distance Statistics Box */}
                    <div style="background: #f8fafc; border-radius: 6px; padding: 6px; display: flex; flex-direction: column; gap: 4px; align-items: center; justify-content: center; box-sizing: border-box; margin: 4px 0; min-height: 90px;">
                      {state.isInferencing ? (
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 4px; font-size: 10px; font-weight: 600; color: #64748b;">
                          <div style="width: 14px; height: 14px; border: 2px solid #0f172a; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                          <span>Sampling asset & computing stats...</span>
                        </div>
                      ) : (state.inferenceResult && state.inferenceResult.preview_url) ? (
                        <div style="display: flex; flex-direction: column; align-items: center; gap: 4px; width: 100%; height: 100%;">
                          <video controls autoplay loop muted playsinline preload="auto" controlslist="nodownload nofullscreen noremoteplayback" disablePictureInPicture src={state.inferenceResult.preview_url} style="width: 100%; height: 65px; object-fit: contain; background: #000000; border-radius: 6px;" />
                          <div style="font-size: 9px; color: #0f172a; width: 100%; display: flex; flex-direction: column; gap: 1px; background: #ffffff; padding: 4px; border-radius: 4px; border: 1px solid #e2e8f0; text-align: left;">
                            <div><strong>Avg L2 Dist:</strong> {state.inferenceResult.stats?.avg_distance}</div>
                            <div><strong>Min/Max Dist:</strong> {state.inferenceResult.stats?.min_distance} / {state.inferenceResult.stats?.max_distance}</div>
                            <div style="color: #64748b;">{state.inferenceResult.latency_ms}ms ({state.inferenceResult.engine})</div>
                          </div>
                        </div>
                      ) : (
                        <div style="font-size: 10px; color: #64748b; text-align: center; padding: 4px;">Execute inference to sample asset and view latent statistics.</div>
                      )}
                    </div>

                    <button onClick={runFunctionalInference} style="padding: 8px 12px; background: #0f172a; color: #ffffff; border: none; border-radius: 6px; cursor: pointer; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; width: 100%; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.2);">
                      Inference
                    </button>

                  </div>

                </div>

              </div>
            );
          })()
        )}

      </div>
      {state.showModal && <ExplainerModal onClose={() => setState("showModal", false)} />}
    </div>
  );
}

render(() => <App />, document.getElementById("app"));