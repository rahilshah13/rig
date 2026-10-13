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
      ctx.translate(canvasRef.width / 2, canvasRef.height / 2 + 25);
      ctx.scale(1.3, -1.3);

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
      ctx.strokeStyle = "#000000";
      ctx.fillStyle = "#000000";

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
        ctx.arc(p[0], p[1], 3.0, 0, 2 * Math.PI);
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
    <canvas ref={canvasRef} width="220" height="240" style="background: #ffffff; border-radius: 4px; border: 1px solid #000000;" />
  );
}

function ExplainerModal(props) {
  return (
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(2px); display: flex; align-items: center; justify-content: center; z-index: 1000; padding: 20px;">
      <div style="background: #ffffff; width: 100%; max-width: 700px; border-radius: 6px; padding: 24px; border: 1px solid #000000; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); display: flex; flex-direction: column; gap: 14px; max-height: 85vh; overflow-y: auto; text-align: left; color: #000000; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #000000; padding-bottom: 10px;">
          <h2 style="font-size: 16px; font-weight: 700; margin: 0; text-transform: uppercase;">Machine Learning Objectives</h2>
          <button onClick={props.onClose} style="background: #ffffff; border: 1px solid #000000; font-size: 14px; font-weight: bold; width: 28px; height: 28px; border-radius: 4px; cursor: pointer; color: #000000;">✕</button>
        </div>
        <div style="font-size: 13px; line-height: 1.5; display: flex; flex-direction: column; gap: 8px;">
          <p><strong>Store-backed State:</strong> Application view states and telemetry logs are synchronized via SolidJS store bindings.</p>
          <p><strong>Disambiguated Losses:</strong> Actual Loss tracks unconstrained optimization error in <code style="background: #f1f1f1; padding: 2px 4px; border-radius: 3px;">[0, inf)</code>. Normalized Loss tracks scaled progress strictly bounded within <code style="background: #f1f1f1; padding: 2px 4px; border-radius: 3px;">[0, 1]</code>.</p>
        </div>
        <button onClick={props.onClose} style="padding: 8px 16px; background: #000000; color: #ffffff; border: 1px solid #000000; border-radius: 4px; font-weight: 700; cursor: pointer; align-self: flex-end; text-transform: uppercase; font-size: 12px;">Close</button>
      </div>
    </div>
  );
}

function App() {
  const [state, setState] = createStore({
    batchData: { 
      epoch: 1, 
      batch: [], 
      currentIndex: 0,
      status: "Initializing...", 
      actual_loss: 0.0, 
      actual_loss_range: "[0, inf)",
      normalized_loss: 0.0, 
      normalized_loss_range: "[0, 1]",
      current_operation: "Idle" 
    },
    currentScore: 0.5,
    showModal: false,
    viewMode: "gif",
    preset: "isometric",
    url: "",
    isRendering: false,
    isPaused: false,
    selectedCheckpoint: "checkpoint_bundle.pickle"
  });

  const fetchData = async () => {
    if (!state.isPaused) {
      try {
        const res = await fetch("/api/batch");
        const data = await res.json();
        if (data) setState("batchData", data);
      } catch (err) {
        console.error("Batch fetch error:", err);
      }
    }
  };

  onMount(() => {
    fetchData();
    const interval = setInterval(fetchData, 1000);
    onCleanup(() => clearInterval(interval));
  });

  const togglePause = async () => {
    const nextState = !state.isPaused;
    setState("isPaused", nextState);
    await fetch("/api/training/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: nextState ? "pause" : "resume" })
    });
  };

  const handleCheckpointChange = async (e) => {
    const ckpt = e.target.value;
    setState("selectedCheckpoint", ckpt);
    await fetch("/api/training/checkpoint", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint: ckpt })
    });
  };

  const switchView = async (mode, selectedPreset = "isometric") => {
    setState("viewMode", mode);
    if (mode === "gif") return;
    setState("preset", selectedPreset);
    const batch = state.batchData.batch;
    const item = batch[state.batchData.currentIndex];
    if (!item) return;

    setState("isRendering", true);
    let endpoint = "/api/render-card";
    let payload = { var_id: item.id, epoch: state.batchData.epoch, preset: selectedPreset };
    if (mode === "pc") { endpoint = "/api/render-pointcloud"; payload = { var_id: item.id, epoch: state.batchData.epoch }; }
    else if (mode === "opengl") { endpoint = "/api/render-opengl"; payload = { var_id: item.id, epoch: state.batchData.epoch }; }

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

    await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item_id: item.id, score: state.currentScore })
    });

    setState({
      currentScore: 0.5,
      url: "",
      viewMode: "gif"
    });
    fetchData();
  };

  return (
    <div style="font-family: system-ui, -apple-system, sans-serif; background: #ffffff; color: #000000; height: 100vh; max-height: 100vh; overflow: hidden; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 12px 16px; box-sizing: border-box;">
      <div style="width: 100%; max-width: 980px; display: flex; flex-direction: column; align-items: center; text-align: center; height: 100%; gap: 8px;">
        
        {/* Header & Control Toolbar */}
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
          <h1 style="font-size: 16px; font-weight: 800; color: #000000; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">Rig Stream - Store State</h1>
          <div style="display: flex; gap: 8px; align-items: center;">
            <select value={state.selectedCheckpoint} onChange={handleCheckpointChange} style="font-size: 11px; padding: 4px 8px; border-radius: 4px; border: 1px solid #000000; background: #ffffff; color: #000000; cursor: pointer;">
              <option value="checkpoint_bundle.pickle">Latest Bundle</option>
              <option value="checkpoint_bundle_prev.pickle">Previous Bundle</option>
            </select>
            <button onClick={togglePause} style={`padding: 4px 10px; border-radius: 4px; border: 1px solid #000000; font-size: 11px; font-weight: 700; cursor: pointer; background: ${state.isPaused ? '#000000' : '#ffffff'}; color: ${state.isPaused ? '#ffffff' : '#000000'};`}>
              {state.isPaused ? "Resume Training" : "Pause Training"}
            </button>
            <button onClick={() => setState("showModal", true)} style="padding: 4px 10px; background: #ffffff; color: #000000; border: 1px solid #000000; border-radius: 4px; font-size: 11px; font-weight: 700; cursor: pointer;">
              ℹ️ Objectives
            </button>
          </div>
        </div>

        {/* Status & Disambiguated Loss Dashboard */}
        <div style="background: #ffffff; border: 1px solid #000000; color: #000000; padding: 8px 12px; border-radius: 4px; font-size: 12px; font-weight: 600; display: flex; flex-direction: column; gap: 4px; width: 100%; box-sizing: border-box;">
          <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
            <span><strong>Operation:</strong> {state.batchData.current_operation || "Initializing..."}</span>
            <span><strong>Epoch:</strong> {state.batchData.epoch}</span>
          </div>
          <div style="display: flex; justify-content: space-around; width: 100%; border-top: 1px solid #e0e0e0; padding-top: 4px; font-size: 11px;">
            <span>Actual Loss <span style="color:#666;">[0, inf)</span>: <strong style="color: #000000;">{state.batchData.actual_loss ?? 0.0000}</strong></span>
            <span>Normalized Loss <span style="color:#666;">[0, 1]</span>: <strong style="color: #000000;">{state.batchData.normalized_loss ?? 0.0000}</strong></span>
          </div>
        </div>

        {/* Architecture Pipeline Position Tracker (replaces terminal output log) */}
        <div style="width: 100%; background: #f8f9fa; border: 1px solid #000000; border-radius: 4px; padding: 8px 12px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box; text-align: left;">
          <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; display: flex; justify-content: space-between; align-items: center;">
            <span>Model Pipeline Phase</span>
            <span style="font-weight: normal; color: #555;">model.py execution stack</span>
          </div>
          <div style="display: flex; justify-content: space-between; gap: 4px; font-size: 11px; font-weight: 600;">
            <div style={`flex: 1; text-align: center; padding: 4px; border-radius: 3px; border: 1px solid #000000; background: ${state.batchData.current_operation?.includes('Init') ? '#000000' : '#ffffff'}; color: ${state.batchData.current_operation?.includes('Init') ? '#ffffff' : '#000000'};`}>
              1. Checkpoint & Init
            </div>
            <div style={`flex: 1; text-align: center; padding: 4px; border-radius: 3px; border: 1px solid #000000; background: ${state.batchData.current_operation?.includes('Scan') || state.batchData.current_operation?.includes('Training') ? '#000000' : '#ffffff'}; color: ${state.batchData.current_operation?.includes('Scan') || state.batchData.current_operation?.includes('Training') ? '#ffffff' : '#000000'};`}>
              2. Lax.Scan Transformer
            </div>
            <div style={`flex: 1; text-align: center; padding: 4px; border-radius: 3px; border: 1px solid #000000; background: ${state.batchData.current_operation?.includes('Batch') ? '#000000' : '#ffffff'}; color: ${state.batchData.current_operation?.includes('Batch') ? '#ffffff' : '#000000'};`}>
              3. VAE & Latent Sampling
            </div>
            <div style={`flex: 1; text-align: center; padding: 4px; border-radius: 3px; border: 1px solid #000000; background: ${state.batchData.current_operation?.includes('Paused') ? '#000000' : '#ffffff'}; color: ${state.batchData.current_operation?.includes('Paused') ? '#ffffff' : '#000000'};`}>
              4. Server State Sync
            </div>
          </div>
        </div>

        {/* Maximized Single Item Card */}
        {state.batchData.batch.length === 0 ? (
          <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 20px; background: #ffffff; border-radius: 4px; border: 1px solid #000000; width: 100%; flex-grow: 1; box-sizing: border-box;">
            <div style="width: 32px; height: 32px; border: 3px solid #000000; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 10px;"></div>
            <p style="font-size: 14px; color: #000000; font-weight: 700;">Streaming procedural variation from JAX VAE...</p>
            <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          </div>
        ) : (
          (() => {
            const batch = state.batchData.batch;
            const idx = state.batchData.currentIndex % batch.length;
            const item = batch[idx];
            if (!item) return null;

            return (
              <div style="background: #ffffff; padding: 14px 18px; border-radius: 6px; display: flex; flex-direction: column; align-items: center; text-align: left; border: 1px solid #000000; color: #000000; width: 100%; box-sizing: border-box; flex-grow: 1; justify-content: space-between;">
                
                <div style="display: flex; justify-content: space-between; width: 100%; align-items: center; margin-bottom: 4px;">
                  <h3 style="font-size: 14px; font-weight: 700; margin: 0;">Variation #{item.id} (Item {idx + 1} of {batch.length})</h3>
                  <div style="display: flex; gap: 4px;">
                    <button onClick={() => switchView("gif")} style={`font-size: 11px; padding: 3px 8px; border-radius: 3px; border: 1px solid #000000; background: ${state.viewMode === 'gif' ? '#000000' : '#ffffff'}; color: ${state.viewMode === 'gif' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>GIF</button>
                    <button onClick={() => switchView("2d", state.preset)} style={`font-size: 11px; padding: 3px 8px; border-radius: 3px; border: 1px solid #000000; background: ${state.viewMode === '2d' ? '#000000' : '#ffffff'}; color: ${state.viewMode === '2d' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>2D</button>
                    <button onClick={() => switchView("pc")} style={`font-size: 11px; padding: 3px 8px; border-radius: 3px; border: 1px solid #000000; background: ${state.viewMode === 'pc' ? '#000000' : '#ffffff'}; color: ${state.viewMode === 'pc' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>PC</button>
                    <button onClick={() => switchView("opengl")} style={`font-size: 11px; padding: 3px 8px; border-radius: 3px; border: 1px solid #000000; background: ${state.viewMode === 'opengl' ? '#000000' : '#ffffff'}; color: ${state.viewMode === 'opengl' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>GL</button>
                  </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; width: 100%; align-items: center; margin-bottom: 6px;">
                  <p style="font-size: 12px; color: #333333; margin: 0;">Mass: <strong>{item.mass}kg</strong> | Height: <strong>{item.height}m</strong></p>
                  {state.viewMode === "2d" && (
                    <select value={state.preset} onChange={(e) => switchView("2d", e.target.value)} style="font-size: 11px; padding: 2px 6px; border-radius: 3px; border: 1px solid #000000; background: #ffffff; color: #000000;">
                      <option value="isometric">Isometric</option>
                      <option value="cinematic">Cinematic</option>
                      <option value="front">Front</option>
                      <option value="side">Side</option>
                    </select>
                  )}
                </div>

                <div style="display: flex; gap: 16px; width: 100%; justify-content: center; align-items: center; margin-bottom: 8px;">
                  <div style="display: flex; flex-direction: column; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; margin-bottom: 4px; text-transform: uppercase;">Bone Rig (Canvas)</span>
                    <RigCanvas transforms={item.bone_transforms} />
                  </div>
                  <div style="display: flex; flex-direction: column; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; margin-bottom: 4px; text-transform: uppercase;">{state.viewMode.toUpperCase()} Viewport</span>
                    {state.viewMode === 'gif' ? (
                      item.preview_url ? (
                        <img src={item.preview_url} alt="Mesh Render" style="width: 220px; height: 240px; object-fit: contain; background: #ffffff; border-radius: 4px; border: 1px solid #000000;" />
                      ) : (
                        <div style="width: 220px; height: 240px; background: #ffffff; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000000; font-size: 12px; border: 1px solid #000000; text-align: center;">Rendering...</div>
                      )
                    ) : (
                      state.isRendering ? (
                        <div style="width: 220px; height: 240px; background: #ffffff; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000000; font-size: 12px; border: 1px solid #000000; text-align: center;">Rendering...</div>
                      ) : (
                        state.url ? (
                          <video controls autoplay loop src={state.url} style="width: 220px; height: 240px; object-fit: contain; background: #000000; border-radius: 4px; border: 1px solid #000000;" />
                        ) : (
                          <div style="width: 220px; height: 240px; background: #ffffff; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000000; font-size: 12px; border: 1px solid #000000; text-align: center;">Ready</div>
                        )
                      )
                    )}
                  </div>
                </div>

                <div style="width: 100%; display: flex; flex-direction: column; gap: 4px;">
                  <label style="font-size: 11px; font-weight: 700; display: flex; justify-content: space-between;">
                    <span>Score Reward:</span>
                    <span>{state.currentScore}</span>
                  </label>
                  <input type="range" min="0" max="1" step="0.1" value={state.currentScore} 
                    onInput={(e) => setState("currentScore", parseFloat(e.target.value))} 
                    style="width: 100%; cursor: pointer; accent-color: #000000; height: 18px;" />
                  
                  <button onClick={submitScoreAndNext} style="margin-top: 4px; padding: 10px 16px; background: #000000; color: #ffffff; border: 1px solid #000000; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; width: 100%;">
                    Submit Score & Continue to Next
                  </button>
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