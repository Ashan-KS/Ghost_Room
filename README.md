# Ghost Room: Multi-Modal Room Occupancy Detection

An autonomous edge-AI system for detecting "Ghost Bookings" (rooms booked but unoccupied) and monitoring room utilization. It uses a fusion of Vision, Audio, and Anomaly Detection to accurately determine occupancy without requiring expensive proprietary hardware. 

Designed to run on edge devices like the Raspberry Pi 4, as well as laptops for development.

---

## 🏗️ Architecture & Modules

The system uses three primary models feeding into a central fusion logic via thread-safe queues.

| Subsystem | Approach | Owner | Location |
|---|---|---|---|
| **Vision** | Object detection (YOLOv8 / MobileNet SSD) to count persons. | Sachith | `vision/` |
| **Audio** | WebRTC VAD (Voice Activity Detection) + Feature Extraction. | Rahul | `audio/` |
| **Anomaly** | Isolation Forest on environmental audio features to detect non-vocal activity. | Ginura | `anomaly/` |
| **Fusion & Cloud** | Decision logic combining the above + MQTT AWS publishing. | Ashan | `fusion/`, `cloud/`, `main.py` |

---

## 🚀 Setup Instructions (Using `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and environment manager.

### 1. Install `uv`
If you haven't installed `uv` yet:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Environment Setup

Clone the repository and set up your virtual environment:

```bash
git clone <repo-url>
cd Ghost_Room

# Create a Python 3.10 virtual environment
uv venv --python 3.10

# Install all dependencies from pyproject.toml
uv sync
```

> **Note on Network Timeouts**: 
> If `uv` times out downloading large ML packages like TensorFlow or PyTorch, extend the timeout limit:
> * **Windows**: `set UV_HTTP_TIMEOUT=120` inside CMD before running `uv sync`.

### 3. Model Weights Delivery
Download the corresponding vision weights and place them inside the `models/` directory:
- `yolov8n.pt` (Ultralytics)
- `ssd_mobilenet_v1_coco_quant.tflite` (TensorFlow Lite)

---

## 🧪 Running Tests

Ensure your module passes tests before committing anything to the `main` branch. 
Using the `.venv` Python explicitly ensures it isolates from global Anaconda/System Python versions.

```bash
# Run all tests
.venv\Scripts\python -m pytest tests/ -v

# Or run a specific module's tests
.venv\Scripts\python -m pytest tests/test_vision.py -v
```

---

## 💻 Running the System

### Local Development (Laptops)
To run the full system locally without deploying to AWS:
1. Open `config.py`.
2. Ensure `USE_PI_HARDWARE = False`.
3. Set `MQTT_BROKER_HOST = "localhost"` (this simulates a cloud failure and safely proceeds with local logging).
4. Run the main agent process:

```bash
.venv\Scripts\python main.py
```
*Note: On the first ever run, if `models/baseline.pkl` does not exist, the system will initialize a 60-second audio calibration period to baseline ambient room noise. Keep the room quiet during this!*

### Raspberry Pi Deployment
1. Set `USE_PI_HARDWARE = True` in `config.py`.
2. Configure `MQTT_BROKER_HOST` to the proper AWS EC2 instance IP.
3. Start the agent:
```bash
.venv\Scripts\python main.py
```

To run as a systemd service (auto-starts on boot):
```bash
sudo cp scripts/workspace-agent.service /etc/systemd/system/
sudo systemctl enable workspace-agent
sudo systemctl start workspace-agent
sudo journalctl -u workspace-agent -f   # view logs
```

---

## 🔀 Decision Logic
The `fusion/fusion.py` module evaluates inputs roughly 30 times a second using:
```text
audio_signal = vad_fired AND anomaly_score >= 0.50
vision_signal = confidence >= 0.50
in_use = vision_signal OR audio_signal

state = EMPTY declared after 10 continuous minutes of no signal
```
