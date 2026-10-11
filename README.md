- `git clone https://github.com/trealla-prolog/trealla.git && cd trealla && make && mkdir -p ~/.local/bin && ln -sf "$(pwd)/tpl" ~/.local/bin/tpl`
- `python3 -m venv .venv; source ./.venv/bin/activate; pip install trimesh numpy ffmpeg-python "pyglet<2" matplotlib`
- `python3 rig.py N`
---

- `docker build -t rig .`
- `docker run -d -p 8000:8000 rig`

---

<img width="800" height="800" alt="rigged_scene_1" src="https://github.com/user-attachments/assets/d4f14e7b-2c6a-4c2d-96a5-032ebf75a85b" />


---

<img width="800" height="800" alt="rigged_scene_2" src="https://github.com/user-attachments/assets/8cec5ad2-6308-4e42-b9f2-3499b829f6cb" />

---

<img width="800" height="800" alt="rigged_scene_3" src="https://github.com/user-attachments/assets/501dc722-6e63-4c40-9c0a-36a6bd4392b9" />

---

<img width="800" height="800" alt="rigged_scene_4" src="https://github.com/user-attachments/assets/5c282f42-c368-475a-989d-e45bfa4673ba" />
