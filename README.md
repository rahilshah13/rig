- `docker build -t rig .`
- `docker run -d -p 8000:8000 -p 8001:8001 -v "$(pwd)/checkpoints:/app/checkpoints" rig`

---