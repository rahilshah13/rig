import { render } from "solid-js/web";
import { createSignal, onMount, onCleanup, For } from "solid-js";

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
      ctx.translate(canvasRef.width / 2, canvasRef.height / 2 + 15);
      ctx.scale(0.8, -0.8);

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

      ctx.lineWidth = 2;
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
        ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI);
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
    <canvas ref={canvasRef} width="120" height="150" style="background: #ffffff; border-radius: 4px; border: 1px solid #000000;" />
  );
}

function ExplainerModal(props) {
  return (
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(2px); display: flex; align-items: center; justify-content: center; z-index: 1000;">
      <div style="background: #ffffff; width: 90%; max-width: 750px; border-radius: 8px; padding: 28px; border: 2px solid #000000; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); display: flex; flex-direction: column; gap: 16px; max-height: 85vh; overflow-y: auto; text-align: left; color: #000000;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #000000; padding-bottom: 12px;">
          <h2 style="font-size: 18px; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">Machine Learning Objectives & Grading Guide</h2>
          <button onClick={props.onClose} style="background: #ffffff; border: 2px solid #000000; font-size: 16px; font-weight: bold; width: 32px; height: 32px; border-radius: 4px; cursor: pointer; color: #000000;">✕</button>
        </div>

        <div style="display: flex; flex-direction: column; gap: 14px; font-size: 13px; line-height: 1.5;">
          <div>
            <h3 style="font-size: 14px; font-weight: 700; margin-bottom: 4px; text-transform: uppercase;">1. Core Machine Learning Objective (JAX Diffusion & VAE)</h3>
            <p>The backend trains a <strong>JAX Diffusion Transformer</strong> paired with a <strong>Variational Autoencoder (VAE)</strong> to generate structured parameters (bone oscillation frequencies, amplitudes, body mass, and height) for parametric human rigs.</p>
          </div>

          <div>
            <h3 style="font-size: 14px; font-weight: 700; margin-bottom: 4px; text-transform: uppercase;">2. What Your Grading Goes Towards (RLHF Latent Alignment)</h3>
            <p>When you adjust the 0 to 1 sliders and click <strong>Submit Batch Scores</strong>, your ratings act as direct human feedback (RLHF). High scores reinforce the VAE latent vectors for those specific kinematic properties, while low scores penalize them.</p>
          </div>

          <div>
            <h3 style="font-size: 14px; font-weight: 700; margin-bottom: 4px; text-transform: uppercase;">3. Learned Data vs. Verification Renders</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12px;">
              <thead>
                <tr style="background: #000000; color: #ffffff; text-align: left;">
                  <th style="padding: 6px; border: 1px solid #000000;">Component / View</th>
                  <th style="padding: 6px; border: 1px solid #000000;">Role in Training</th>
                  <th style="padding: 6px; border: 1px solid #000000;">Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style="padding: 6px; border: 1px solid #000000;"><strong>Bone Rig & Features</strong></td>
                  <td style="padding: 6px; border: 1px solid #000000; font-weight: 700;">Actively Learned on</td>
                  <td style="padding: 6px; border: 1px solid #000000;">Extracted directly into feature vectors fed into the VAE and RLHF loss function.</td>
                </tr>
                <tr>
                  <td style="padding: 6px; border: 1px solid #000000;"><strong>GIF Previews</strong></td>
                  <td style="padding: 6px; border: 1px solid #000000; font-weight: 700;">Verification Render</td>
                  <td style="padding: 6px; border: 1px solid #000000;">Generated via Trealla Prolog & procedural scripts for visual inspection.</td>
                </tr>
                <tr>
                  <td style="padding: 6px; border: 1px solid #000000;"><strong>2D, PC, & OpenGL Views</strong></td>
                  <td style="padding: 6px; border: 1px solid #000000; font-weight: 700;">Verification Render</td>
                  <td style="padding: 6px; border: 1px solid #000000;">On-demand camera presets and point-cloud / OpenGL video renders for structural validation.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <button onClick={props.onClose} style="padding: 10px 20px; background: #000000; color: #ffffff; border: 2px solid #000000; border-radius: 4px; font-weight: 700; cursor: pointer; align-self: flex-end; text-transform: uppercase;">Close</button>
      </div>
    </div>
  );
}

function VariationCard(props) {
  const item = props.item;
  const epoch = props.epoch;
  const [viewMode, setViewMode] = createSignal("gif");
  const [preset, setPreset] = createSignal("isometric");
  const [urls, setUrls] = createSignal({ "2d": "", "pc": "", "opengl": "" });
  const [isRendering, setIsRendering] = createSignal(false);

  const switchView = async (mode, selectedPreset = "isometric") => {
    setViewMode(mode);
    if (mode === "gif") return;
    
    setPreset(selectedPreset);
    const currentUrls = urls();
    if (!currentUrls[mode]) {
      setIsRendering(true);
      let endpoint = "/api/render-card";
      let payload = { var_id: item.id, epoch: epoch, preset: selectedPreset };
      
      if (mode === "pc") {
        endpoint = "/api/render-pointcloud";
        payload = { var_id: item.id, epoch: epoch };
      } else if (mode === "opengl") {
        endpoint = "/api/render-opengl";
        payload = { var_id: item.id, epoch: epoch };
      }

      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.status === "success") {
          setUrls({ ...urls(), [mode]: data.video_url });
        }
      } catch (err) {
        console.error("Render error:", err);
      } finally {
        setIsRendering(false);
      }
    }
  };

  return (
    <div style="background: #ffffff; padding: 12px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; flex-direction: column; align-items: center; text-align: left; border: 2px solid #000000; color: #000000;">
      <div style="display: flex; justify-content: space-between; width: 100%; align-items: center; margin-bottom: 2px;">
        <h3 style="font-size: 14px; font-weight: 700; margin: 0;">Var #{item.id}</h3>
        <div style="display: flex; gap: 2px;">
          <button onClick={() => switchView("gif")} style={`font-size: 8px; padding: 2px 5px; border-radius: 3px; border: 1px solid #000000; background: ${viewMode() === 'gif' ? '#000000' : '#ffffff'}; color: ${viewMode() === 'gif' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>GIF</button>
          <button onClick={() => switchView("2d", preset())} style={`font-size: 8px; padding: 2px 5px; border-radius: 3px; border: 1px solid #000000; background: ${viewMode() === '2d' ? '#000000' : '#ffffff'}; color: ${viewMode() === '2d' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>2D</button>
          <button onClick={() => switchView("pc")} style={`font-size: 8px; padding: 2px 5px; border-radius: 3px; border: 1px solid #000000; background: ${viewMode() === 'pc' ? '#000000' : '#ffffff'}; color: ${viewMode() === 'pc' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>PC</button>
          <button onClick={() => switchView("opengl")} style={`font-size: 8px; padding: 2px 5px; border-radius: 3px; border: 1px solid #000000; background: ${viewMode() === 'opengl' ? '#000000' : '#ffffff'}; color: ${viewMode() === 'opengl' ? '#ffffff' : '#000000'}; cursor: pointer; font-weight: bold;`}>GL</button>
        </div>
      </div>
      <p style="font-size: 11px; color: #333333; margin-bottom: 6px; width: 100%;">Mass: {item.mass}kg | H: {item.height}m</p>
      
      {viewMode() === "2d" && (
        <div style="display: flex; gap: 4px; margin-bottom: 6px; width: 100%; justify-content: center;">
          <select value={preset()} onChange={(e) => switchView("2d", e.target.value)} style="font-size: 9px; padding: 2px 4px; border-radius: 3px; border: 1px solid #000000; background: #ffffff; color: #000000;">
            <option value="isometric">Isometric</option>
            <option value="cinematic">Cinematic</option>
            <option value="front">Front</option>
            <option value="side">Side</option>
          </select>
        </div>
      )}

      <div style="display: flex; gap: 6px; width: 100%; justify-content: center; align-items: center; margin-bottom: 10px;">
        <div style="display: flex; flex-direction: column; align-items: center;">
          <span style="font-size: 9px; font-weight: 700; margin-bottom: 2px;">Bone Rig</span>
          <RigCanvas transforms={item.bone_transforms} />
        </div>
        <div style="display: flex; flex-direction: column; align-items: center;">
          <span style="font-size: 9px; font-weight: 700; margin-bottom: 2px;">{viewMode().toUpperCase()} View</span>
          {viewMode() === 'gif' ? (
            item.preview_url ? (
              <img src={item.preview_url} alt="Mesh Render" style="width: 120px; height: 150px; object-fit: contain; background: #ffffff; border-radius: 4px; border: 1px solid #000000;" />
            ) : (
              <div style="width: 120px; height: 150px; background: #ffffff; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000000; font-size: 10px; border: 1px solid #000000; text-align: center; padding: 4px;">Rendering...</div>
            )
          ) : (
            isRendering() ? (
              <div style="width: 120px; height: 150px; background: #ffffff; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000000; font-size: 10px; border: 1px solid #000000; text-align: center; padding: 4px;">Rendering...</div>
            ) : (
              urls()[viewMode()] ? (
                <video controls autoplay loop src={urls()[viewMode()]} style="width: 120px; height: 150px; object-fit: contain; background: #000000; border-radius: 4px; border: 1px solid #000000;" />
              ) : (
                <div style="width: 120px; height: 150px; background: #ffffff; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: #000000; font-size: 10px; border: 1px solid #000000; text-align: center; padding: 4px;">Ready</div>
              )
            )
          )}
        </div>
      </div>

      <label style="font-size: 11px; font-weight: 700; margin-bottom: 3px; width: 100%; display: flex; justify-content: space-between;">
        <span>Score:</span>
        <span>{props.score ?? 0.5}</span>
      </label>
      <input type="range" min="0" max="1" step="0.1" value={props.score ?? 0.5} 
        onInput={(e) => props.onScoreChange(item.id, e.target.value)} 
        style="width: 100%; cursor: pointer; accent-color: #000000;" />
    </div>
  );
}

function SimdBenchmarkPanel() {
  const [metrics, setMetrics] = createSignal(null);
  const [loading, setLoading] = createSignal(false);

  const runBenchmark = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/benchmark-simd", { method: "POST" });
      const data = await res.json();
      if (data.status === "success") {
        setMetrics(data);
      }
    } catch (err) {
      console.error("SIMD benchmark error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style="background: #ffffff; border: 2px solid #000000; border-radius: 6px; padding: 16px; margin-top: 30px; width: 100%; max-width: 1300px; box-sizing: border-box; display: flex; flex-direction: column; align-items: center;">
      <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 12px;">
        <h3 style="font-size: 15px; font-weight: 700; color: #000000; margin: 0; text-transform: uppercase;">Apple Silicon ARM64 NEON SIMD Kernel Benchmark</h3>
        <button onClick={runBenchmark} disabled={loading()} style="padding: 8px 16px; background: #000000; color: #ffffff; border: 2px solid #000000; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 700;">
          {loading() ? "Compiling & Running..." : "Run ARM64 NEON Benchmark"}
        </button>
      </div>
      {metrics() && (
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; width: 100%; background: #ffffff; padding: 12px; border-radius: 4px; border: 1px solid #000000; text-align: center;">
          <div>
            <span style="font-size: 11px; color: #333333; display: block; font-weight: 600;">Python JAX Latency</span>
            <strong style="font-size: 14px; color: #000000;">{metrics().jax_latency_ms} ms</strong>
          </div>
          <div>
            <span style="font-size: 11px; color: #333333; display: block; font-weight: 600;">ARM64 NEON Latency</span>
            <strong style="font-size: 14px; color: #000000;">{metrics().arm64_neon_latency_ms} ms</strong>
          </div>
          <div>
            <span style="font-size: 11px; color: #333333; display: block; font-weight: 600;">SIMD Speedup</span>
            <strong style="font-size: 14px; color: #000000;">{metrics().speedup}x</strong>
          </div>
          <div>
            <span style="font-size: 11px; color: #333333; display: block; font-weight: 600;">Vector Regs Used</span>
            <strong style="font-size: 14px; color: #000000;">{metrics().peak_registers} / {metrics().total_regs}</strong>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [batchData, setBatchData] = createSignal({ 
    epoch: 1, 
    batch: [], 
    status: "Initializing...", 
    real_loss: 0.0, 
    normalized_loss: 0.0, 
    current_operation: "Idle" 
  });
  const [scores, setScores] = createSignal({});
  const [showModal, setShowModal] = createSignal(false);

  const fetchBatch = async () => {
    try {
      const res = await fetch("/api/batch");
      const data = await res.json();
      if (data) {
        setBatchData(data);
        if (data.batch && data.batch.length > 0) {
          const initialScores = {};
          data.batch.forEach(item => {
            if (scores()[item.id] === undefined) {
              initialScores[item.id] = 0.5;
            } else {
              initialScores[item.id] = scores()[item.id];
            }
          });
          setScores(initialScores);
        }
      }
    } catch (err) {
      console.error("Waiting for backend status...", err);
    }
  };

  onMount(() => {
    fetchBatch();
    const interval = setInterval(fetchBatch, 1000);
    onCleanup(() => clearInterval(interval));
  });

  const handleScoreChange = (id, val) => {
    setScores({ ...scores(), [id]: parseFloat(val) });
  };

  const submitScores = async () => {
    const currentBatch = batchData().batch;
    const scoreList = currentBatch.map(item => scores()[item.id] ?? 0.5);
    
    await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scores: scoreList })
    });
    
    setBatchData({ ...batchData(), status: "Submitting 16 scores to JAX VAE trainer..." });
  };

  return (
    <div style="font-family: system-ui, -apple-system, sans-serif; background: #ffffff; color: #000000; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 30px 20px; box-sizing: border-box;">
      <div style="width: 100%; max-width: 1300px; display: flex; flex-direction: column; align-items: center; text-align: center;">
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 6px;">
          <div></div>
          <h1 style="font-size: 24px; font-weight: 800; color: #000000; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">Rig Environment - Human-in-the-Loop VAE Scoring Dashboard</h1>
          <button onClick={() => setShowModal(true)} style="padding: 6px 14px; background: #ffffff; color: #000000; border: 2px solid #000000; border-radius: 4px; font-size: 13px; font-weight: 700; cursor: pointer;">
            ℹ️ ML Objectives
          </button>
        </div>
        <p style="font-size: 14px; color: #333333; margin-bottom: 12px; font-weight: 600;">Training Epoch: <span>{batchData().epoch}</span></p>
        
        <div style="background: #ffffff; border: 2px solid #000000; color: #000000; padding: 12px 18px; border-radius: 6px; font-size: 13px; font-weight: 600; margin-bottom: 20px; display: flex; flex-direction: column; gap: 6px; width: 100%; max-width: 900px; box-sizing: border-box;">
          <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <div style="width: 8px; height: 8px; background: #000000; border-radius: 50%; animation: pulse 1.5s infinite;"></div>
              <span><strong>Operation:</strong> {batchData().current_operation || "Initializing..."}</span>
            </div>
            <span><strong>Epoch:</strong> {batchData().epoch}</span>
          </div>
          <div style="display: flex; justify-content: space-around; width: 100%; border-top: 1px solid #000000; padding-top: 6px; font-size: 12px;">
            <span>Real Loss: <strong style="color: #000000;">{batchData().real_loss ?? 0.0000}</strong></span>
            <span>Normalized Loss: <strong style="color: #000000;">{batchData().normalized_loss ?? 0.0000}</strong></span>
            <span>Status: {batchData().status}</span>
          </div>
        </div>
        <style>{`@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.2; } 100% { opacity: 1; } }`}</style>

        {batchData().batch.length === 0 ? (
          <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 20px; background: #ffffff; border-radius: 6px; border: 2px solid #000000; width: 100%;">
            <div style="width: 40px; height: 40px; border: 4px solid #000000; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 16px;"></div>
            <p style="font-size: 15px; color: #000000; font-weight: 700;">Generating 16 procedural variations via Trealla Prolog & JAX...</p>
            <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          </div>
        ) : (
          <div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
            <div style="display: grid; grid-template-columns: repeat(4, minmax(260px, 1fr)); gap: 16px; width: 100%; margin-bottom: 20px;">
              <For each={batchData().batch}>
                {(item) => (
                  <VariationCard item={item} epoch={batchData().epoch} score={scores()[item.id]} onScoreChange={handleScoreChange} />
                )}
              </For>
            </div>
            <button onClick={submitScores} style="padding: 12px 28px; background: #000000; color: #ffffff; border: 2px solid #000000; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">
              Submit RLHF Results & Continue Training
            </button>
          </div>
        )}

        <SimdBenchmarkPanel />
      </div>

      {showModal() && <ExplainerModal onClose={() => setShowModal(false)} />}
    </div>
  );
}

render(() => <App />, document.getElementById("app"));