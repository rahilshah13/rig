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
      ctx.strokeStyle = "#2563eb";
      ctx.fillStyle = "#1e293b";

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
    <canvas ref={canvasRef} width="120" height="150" style="background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;" />
  );
}

function App() {
  const [batchData, setBatchData] = createSignal({ epoch: 1, batch: [], status: "Initializing..." });
  const [scores, setScores] = createSignal({});

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
    const interval = setInterval(fetchBatch, 1500);
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
    
    setBatchData({ ...batchData(), status: "Submitting 20 scores to JAX trainer..." });
  };

  return (
    <div style="font-family: system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 40px 20px; box-sizing: border-box;">
      <div style="width: 100%; max-width: 1400px; display: flex; flex-direction: column; align-items: center; text-align: center;">
        <h1 style="font-size: 28px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Rig Environment - 20-Variation Human-in-the-Loop Scoring Dashboard</h1>
        <p style="font-size: 16px; color: #64748b; margin-bottom: 8px;">Training Epoch: <strong style="color: #2563eb;">{batchData().epoch}</strong></p>
        
        {/* Live Status Progress Banner */}
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; padding: 10px 20px; border-radius: 8px; font-size: 14px; font-weight: 500; margin-bottom: 24px; display: flex; align-items: center; gap: 8px;">
          <div style="width: 8px; height: 8px; background: #2563eb; border-radius: 50%; animation: pulse 1.5s infinite;"></div>
          <span>Status: {batchData().status}</span>
        </div>
        <style>{`@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }`}</style>
        
        {batchData().batch.length === 0 ? (
          <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 20px; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); width: 100%; margin-top: 20px;">
            <div style="width: 44px; height: 44px; border: 4px solid #e2e8f0; border-top-color: #2563eb; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 20px;"></div>
            <p style="font-size: 16px; color: #475569; font-weight: 500;">Generating 20 procedural variations via Trealla Prolog & JAX...</p>
            <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
          </div>
        ) : (
          <div style="width: 100%; display: flex; flex-direction: column; align-items: center;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; width: 100%; margin-bottom: 32px;">
              <For each={batchData().batch}>
                {(item) => (
                  <div style="background: #ffffff; padding: 14px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); display: flex; flex-direction: column; align-items: center; text-align: left; border: 1px solid #e2e8f0;">
                    <h3 style="font-size: 16px; font-weight: 600; color: #1e293b; margin-bottom: 2px; width: 100%;">Var #{item.id}</h3>
                    <p style="font-size: 12px; color: #64748b; margin-bottom: 10px; width: 100%;">Mass: {item.mass}kg | H: {item.height}m</p>
                    
                    <div style="display: flex; gap: 8px; width: 100%; justify-content: center; align-items: center; margin-bottom: 12px;">
                      <div style="display: flex; flex-direction: column; align-items: center;">
                        <span style="font-size: 10px; font-weight: 600; color: #64748b; margin-bottom: 2px;">Bone Rig</span>
                        <RigCanvas transforms={item.bone_transforms} />
                      </div>
                      <div style="display: flex; flex-direction: column; align-items: center;">
                        <span style="font-size: 10px; font-weight: 600; color: #64748b; margin-bottom: 2px;">Mesh GIF</span>
                        {item.preview_url ? (
                          <img src={item.preview_url} alt="Mesh Render" style="width: 120px; height: 150px; object-fit: contain; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0;" />
                        ) : (
                          <div style="width: 120px; height: 150px; background: #f8fafc; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: #94a3b8; font-size: 11px; border: 1px solid #e2e8f0; text-align: center; padding: 4px;">Rendering...</div>
                        )}
                      </div>
                    </div>

                    <label style="font-size: 12px; font-weight: 500; color: #334155; margin-bottom: 4px; width: 100%; display: flex; justify-content: space-between;">
                      <span>Score:</span>
                      <strong style="color: #2563eb;">{scores()[item.id] ?? 0.5}</strong>
                    </label>
                    <input type="range" min="0" max="1" step="0.1" value={scores()[item.id] ?? 0.5} 
                      onInput={(e) => handleScoreChange(item.id, e.target.value)} 
                      style="width: 100%; cursor: pointer; accent-color: #2563eb;" />
                  </div>
                )}
              </For>
            </div>
            <button onClick={submitScores} style="padding: 14px 32px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; box-shadow: 0 4px 6px -1px rgba(37,99,235,0.3); transition: background 0.2s;">
              Submit Batch 20 Scores & Continue Training
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

render(() => <App />, document.getElementById("app"));