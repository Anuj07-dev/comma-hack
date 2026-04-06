# Openpilot on NAVSIM

**Comma Hackathon 2026** — Evaluating comma.ai's openpilot driving models inside the NAVSIM pseudo-simulation benchmark.

---

## Project Idea

[NAVSIM](https://github.com/autonomousvision/navsim) is a data-driven autonomous driving benchmark that combines the speed of open-loop evaluation with the realism of closed-loop simulation. It evaluates a driving agent's planned trajectory against real-world driving data using the Extended PDM Score (EPDMS).

The goal of this project was to plug **openpilot's actual production models** — the same vision and policy ONNX models that run on comma hardware — directly into NAVSIM and see how they score against other AV baselines.

---

## What Was Done

### 1. Data Preprocessing (`navsim/preprocessing/openpilot_model_inputs.py`)

NAVSIM scenes use nuPlan/OpenScene sensor format. Openpilot expects a specific tensor layout derived from its camera pipeline (YUV image streams, desire, traffic convention, lateral control params, curvature history).

We built a preprocessing pipeline that:
- Reads NAVSIM scenes (navhard / navhard_two_stage layout)
- Converts camera frames to openpilot's YUV image stream format
- Constructs all auxiliary tensors (desire, traffic convention, lateral control params, previous desired curvature)
- Logs before/after images for debugging

### 2. Openpilot Agent (`navsim/agents/open_pilot_agent.py`)

A NAVSIM-compatible agent (`OpenpilotAgent`) that:
- Loads openpilot's **vision ONNX** and **policy ONNX** models
- Runs the two-stage inference: vision model → feature extraction → policy model → trajectory
- Parses comma's flat policy output blob using `output_slices` metadata (same MDN decoding as openpilot's `modeld`)
- Converts the predicted plan (x, y, heading) into a NAVSIM `Trajectory`
- Supports recurrent state (GRU hidden states) across frames
- Supports fisheye and rolling-shutter camera variants

### 3. Policy Output Parsing (`navsim/agents/openpilot_policy_parse.py`)

Implements comma's MDN (Mixture Density Network) plan decoding:
- Slices the flat policy output tensor using openpilot's `output_slices` metadata
- Selects the best plan from the mixture
- Converts to (x, y, heading) poses aligned with NAVSIM's coordinate frame

### 4. Evaluation Scripts

- `run_preprocess_openpilot_inputs.py` — preprocess NAVSIM scenes into openpilot tensors offline
- `run_openpilot_agent_compute.py` — run the full agent inference pipeline
- `run_check_openpilot_onnx_shapes.py` — validate ONNX input/output shapes against preprocessed tensors

---

## Architecture

```
NAVSIM Scene
    │
    ▼
Preprocessing (YUV conversion, tensor construction)
    │
    ▼
Vision ONNX (comma supercombo vision stage)
    │  features
    ▼
Policy ONNX (comma driving policy)
    │  flat output blob
    ▼
MDN Plan Decode → (x, y, heading) trajectory
    │
    ▼
NAVSIM EPDMS Evaluation
```

---

## Setup

Follow the base NAVSIM installation:

```bash
pip install -r requirements.txt
pip install -e .
```

You will need:
- A NAVSIM dataset (navhard_two_stage recommended)
- Openpilot vision and policy ONNX model files
- `NUPLAN_MAPS_ROOT` environment variable set

---

## Running the Agent

```bash
# Preprocess scenes into openpilot tensors
python -m navsim.planning.script.run_preprocess_openpilot_inputs \
    --output-dir /path/to/out \
    --max-scenes 100

# Run evaluation
python -m navsim.planning.script.run_openpilot_agent_compute \
    --vision-model /path/to/vision.onnx \
    --policy-model /path/to/policy.onnx
```

Configure the agent via `scripts/config/common/agent/openpilot_agent.yaml`.

---

## Based On

- [NAVSIM](https://github.com/autonomousvision/navsim) — Pseudo-Simulation for Autonomous Driving (CoRL 2025)
- [openpilot](https://github.com/commaai/openpilot) — comma.ai's open source driver assistance system
