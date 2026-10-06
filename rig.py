import sys
import subprocess
import os
import shutil
import random
import json
import trimesh
import numpy as np

class RiggedObject:
    def __init__(self, name, mesh, pivot=None):
        self.name = name
        self.base_mesh = mesh
        self.pivot = np.array(pivot) if pivot is not None else np.array([0.0, 0.0, 0.0])
        self.parent = None
        self.children = []
        self.transform_funcs = []

    def add_child(self, child_obj):
        child_obj.parent = self
        self.children.append(child_obj)

    def add_transformation(self, func):
        self.transform_funcs.append(func)

    def get_local_matrix_at_time(self, t):
        matrix = np.eye(4)
        for func in self.transform_funcs:
            matrix = matrix @ func(t)
        return matrix

    def get_global_matrix_at_time(self, t):
        local_matrix = self.get_local_matrix_at_time(t)
        if self.parent is not None:
            return self.parent.get_global_matrix_at_time(t) @ local_matrix
        return local_matrix

    def get_mesh_at_time(self, t):
        mesh = self.base_mesh.copy()
        global_matrix = self.get_global_matrix_at_time(t)
        mesh.apply_transform(global_matrix)
        return mesh


def generate_parametric_human(mass_kg, height_m):
    density = 1000.0
    total_volume = mass_kg / density
    scale_factor = (total_volume / (height_m * 0.1**2)) ** (1.0 / 3.0)

    chest_mesh = trimesh.creation.capsule(radius=0.15 * scale_factor, height=0.4 * height_m)
    head_mesh = trimesh.creation.icosphere(radius=0.08 * scale_factor, subdivisions=2)
    head_mesh.apply_translation([0.0, 0.0, (0.4 * height_m / 2.0) + (0.08 * scale_factor)])

    arm_mesh = trimesh.creation.capsule(radius=0.04 * scale_factor, height=0.3 * height_m)
    leg_mesh = trimesh.creation.capsule(radius=0.05 * scale_factor, height=0.4 * height_m)
    joint_mesh = trimesh.creation.icosphere(radius=0.03 * scale_factor, subdivisions=1)

    offsets = {
        'Hips': [0.0, 0.0, 0.5 * height_m],
        'Spine': [0.0, 0.0, 0.65 * height_m],
        'Chest': [0.0, 0.0, 0.8 * height_m],
        'Neck': [0.0, 0.0, 1.05 * height_m],
        'Head': [0.0, 0.0, 1.15 * height_m],
        'LeftShoulder': [-0.2 * scale_factor, 0.0, 0.95 * height_m],
        'LeftUpperArm': [-0.25 * scale_factor, 0.0, 0.8 * height_m],
        'LeftLowerArm': [-0.25 * scale_factor, 0.0, 0.5 * height_m],
        'LeftHand': [-0.25 * scale_factor, 0.0, 0.2 * height_m],
        'RightShoulder': [0.2 * scale_factor, 0.0, 0.95 * height_m],
        'RightUpperArm': [0.25 * scale_factor, 0.0, 0.8 * height_m],
        'RightLowerArm': [0.25 * scale_factor, 0.0, 0.5 * height_m],
        'RightHand': [0.25 * scale_factor, 0.0, 0.2 * height_m],
        'LeftUpperLeg': [-0.1 * scale_factor, 0.0, 0.3 * height_m],
        'LeftLowerLeg': [-0.1 * scale_factor, 0.0, 0.0 * height_m],
        'LeftFoot': [-0.1 * scale_factor, 0.0, -0.2 * height_m],
        'RightUpperLeg': [0.1 * scale_factor, 0.0, 0.3 * height_m],
        'RightLowerLeg': [0.1 * scale_factor, 0.0, 0.0 * height_m],
        'RightFoot': [0.1 * scale_factor, 0.0, -0.2 * height_m],
    }

    mesh_map = {
        'Hips': joint_mesh, 'Spine': joint_mesh, 'Chest': chest_mesh,
        'Neck': joint_mesh, 'Head': head_mesh,
        'LeftShoulder': joint_mesh, 'LeftUpperArm': arm_mesh, 'LeftLowerArm': arm_mesh, 'LeftHand': joint_mesh,
        'RightShoulder': joint_mesh, 'RightUpperArm': arm_mesh, 'RightLowerArm': arm_mesh, 'RightHand': joint_mesh,
        'LeftUpperLeg': leg_mesh, 'LeftLowerLeg': leg_mesh, 'LeftFoot': joint_mesh,
        'RightUpperLeg': leg_mesh, 'RightLowerLeg': leg_mesh, 'RightFoot': joint_mesh
    }

    parts = {}
    for name, geom in mesh_map.items():
        pivot = offsets.get(name, [0.0, 0.0, 0.0])
        m = geom.copy()
        m.apply_translation(pivot)
        parts[name] = RiggedObject(name, m, pivot=pivot)

    return parts


def generate_rig_script_with_trealla(run_index):
    prolog_file = "rig_gen.pl"
    if not os.path.exists(prolog_file):
        raise FileNotFoundError(f"Could not find Prolog file: {prolog_file}")

    query = f"write_rigging_program({run_index}), halt."
    try:
        subprocess.run(
            ["tpl", "-l", prolog_file, "-g", query],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Prolog script generation failed:\n{e.stderr}")
        return False


def render_scene_to_image(scene, filepath):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        all_verts = []
        for geom in scene.geometry.values():
            if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                verts = geom.vertices
                faces = geom.faces
                if len(verts) > 0 and len(faces) > 0:
                    poly = Poly3DCollection(verts[faces], alpha=0.9)
                    poly.set_facecolor('#4682B4')
                    poly.set_edgecolor('#2c5270')
                    poly.set_linewidth(0.15)
                    ax.add_collection3d(poly)
                    all_verts.append(verts)
                    
        if all_verts:
            all_verts = np.vstack(all_verts)
            min_val = all_verts.min(axis=0)
            max_val = all_verts.max(axis=0)
            center = (min_val + max_val) / 2.0
            max_range = np.max(max_val - min_val) * 0.70
            if max_range < 1e-3:
                max_range = 1.0
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()
        
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Software frame rendering error: {e}")
        return False


def process_rigging_scene(rigged_objects, time_steps, output_filename="rigged_scene.gif"):
    output_dir = "rigged_glbs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("_frames", exist_ok=True)
    frame_files = []

    print(f"Processing {len(time_steps)} frames (GLBs + Image snapshots)...")
    
    for i, t in enumerate(time_steps):
        frame_scene = trimesh.Scene()
        
        for obj in rigged_objects.values():
            mesh_at_t = obj.get_mesh_at_time(t)
            frame_scene.add_geometry(mesh_at_t, node_name=obj.name, geom_name=obj.name)

        glb_path = os.path.join(output_dir, f"frame_{i:04d}_t{t:.2f}.glb")
        frame_scene.export(glb_path)

        frame_path = f"_frames/frame_{i:04d}.png"
        if render_scene_to_image(frame_scene, frame_path):
            frame_files.append(frame_path)

    if frame_files:
        print(f"Stitched {len(frame_files)} frames into {output_filename} using ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "30", "-i", "_frames/frame_%04d.png",
            "-vf", "fps=30,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            output_filename
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for f in frame_files:
            os.remove(f)
        os.rmdir("_frames")
        print(f"Successfully exported: {output_filename}")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    else:
        print("Warning: Could not render frame snapshots for GIF creation.")
        if os.path.exists("_frames"):
            os.rmdir("_frames")


def parse_and_apply_json_ast(json_filepath, rigged_objects_map):
    with open(json_filepath, "r") as f:
        scene_ast = json.load(f)

    for bone_node in scene_ast:
        obj_name = bone_node["bone"]
        parent_name = bone_node.get("parent")

        if obj_name not in rigged_objects_map:
            raise ValueError(f"Unknown object target in JSON AST: {obj_name}")

        obj = rigged_objects_map[obj_name]

        # Link parent-child FK relationship if specified
        if parent_name and parent_name in rigged_objects_map:
            rigged_objects_map[parent_name].add_child(obj)

        pivot = obj.pivot

        for tx in bone_node.get("transformations", []):
            txtype = tx["type"]

            if txtype == "translate":
                vals = tx["values"]
                x, y, z = vals[0], vals[1], vals[2]
                def make_trans(tx, ty, tz):
                    return lambda t: trimesh.transformations.translation_matrix([tx, ty, tz])
                obj.add_transformation(make_trans(x, y, z))

            elif txtype == "uniform_scale":
                s = float(tx["factor"])
                def make_scale(factor):
                    return lambda t: trimesh.transformations.scale_matrix(factor)
                obj.add_transformation(make_scale(s))

            elif txtype == "oscillate":
                axis = tx["axis"]
                freq = float(tx["freq"])
                amp = float(tx["amp"])
                def make_osc(ax, f, a, pvt):
                    ax_lower = ax.lower()
                    angle_func = lambda t: a * np.sin(2 * np.pi * f * t)
                    dir_vec = [1, 0, 0] if ax_lower == 'x' else ([0, 1, 0] if ax_lower == 'y' else [0, 0, 1])
                    return lambda t: trimesh.transformations.rotation_matrix(angle_func(t), dir_vec, point=pvt)
                obj.add_transformation(make_osc(axis, freq, amp, pivot))


if __name__ == "__main__":
    num_variations = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    for run_idx in range(1, num_variations + 1):
        duration_sec = round(random.uniform(3.0, 20.0), 1)
        total_frames = int(duration_sec * 30)
        
        print(f"\n==========================================")
        print(f"Generating JSON variation {run_idx}/{num_variations} via rig_gen.pl...")
        print(f"Duration: {duration_sec}s ({total_frames} frames)")
        print(f"==========================================")

        if not generate_rig_script_with_trealla(run_idx):
            print(f"Prolog JSON generation failed for variation {run_idx}. Skipping.")
            continue

        json_path = "script.json"
        if not os.path.exists(json_path):
            print(f"Error: Rigging AST file '{json_path}' not found.")
            continue

        scene_map = generate_parametric_human(mass_kg=75.0, height_m=1.75)

        print(f"Parsing and applying JSON AST from '{json_path}'...")
        parse_and_apply_json_ast(json_path, scene_map)
        print("JSON AST successfully applied with Forward Kinematics.")
        
        time_steps = np.linspace(0, duration_sec, total_frames)
        output_filename = f"rigged_scene_{run_idx}_{duration_sec}s.gif" if num_variations > 1 else "rigged_scene.gif"
        process_rigging_scene(scene_map, time_steps, output_filename=output_filename)

    print("\nAll requested JSON-driven variations completed successfully!")