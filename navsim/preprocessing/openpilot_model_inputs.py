"""
Convert NAVSIM `Scene` / `AgentInput` samples into tensor layouts matching comma
openpilot's driving model interface (vision YUV420 6-plane stacks + policy vectors).

NAVSIM caveats (handled explicitly here):
- Logs are 2 Hz (0.5 s), not 20 Hz. Policy buffers of length 100 are filled by
  holding each history frame's value for 10 consecutive 20 Hz steps, left-padded
  with zeros when fewer than 5 s of history exist.
- `feature_buffer` (100 x 512) is produced by the on-device vision encoder in
  openpilot; raw NAVSIM has no equivalent — this field is zeros unless you inject
  features from another model.
- `previous desired curvatures` are approximated from ego history geometry
  (not model predictions).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from navsim.common.dataclasses import AgentInput, Camera, Scene

# openpilot temporal context: 5 s at 20 Hz
OPENPILOT_TEMPORAL_STEPS: int = 25
# NAVSIM history interval (seconds)
NAVSIM_DT: float = 0.5
# How many 20 Hz steps correspond to one NAVSIM frame
STEPS_PER_NAVSIM_FRAME: int = int(round(NAVSIM_DT * 20))  # 10

# Target resolution for RGB before YUV packing (H, W) — matches 256 x 512 RGB spec
TARGET_RGB_HW: Tuple[int, int] = (256, 512)


def preprocessed_openpilot_tensor_shapes(
    *,
    flatten_images: bool = True,
    concatenate_image_streams: bool = False,
) -> Dict[str, Tuple[int, ...]]:
    """
    Shapes of tensors in ``OpenpilotInputs.as_dict()`` — the same layout as
    ``run_preprocess_openpilot_inputs`` / ``merge_original_and_openpilot``.

    Used to validate ONNX model input element counts against preprocessing.
    """
    h, w = TARGET_RGB_HW
    two_six_plane: Tuple[int, ...] = (2, 6, h // 2, w // 2)
    flat = int(np.prod(two_six_plane))
    shapes: Dict[str, Tuple[int, ...]] = {
        "desire": (OPENPILOT_TEMPORAL_STEPS, 8),
        "traffic_convention": (2,),
        "lateral_control_params": (2,),
        "prev_desired_curvature": (OPENPILOT_TEMPORAL_STEPS,),
        "feature_buffer": (OPENPILOT_TEMPORAL_STEPS, 512),
    }
    if concatenate_image_streams:
        shapes["image_concat"] = (flat * 2,)
    else:
        if flatten_images:
            shapes["image_stream"] = (flat,)
            shapes["wide_image_stream"] = (flat,)
        else:
            shapes["image_stream"] = two_six_plane
            shapes["wide_image_stream"] = two_six_plane
    return shapes


@dataclass
class OpenpilotInputs:
    """Named view of openpilot-shaped arrays (see field shapes in docstrings)."""

    image_stream: npt.NDArray[np.float32]  # (393216,) or (2, 6, 128, 256)
    wide_image_stream: npt.NDArray[np.float32]
    desire: npt.NDArray[np.float32]  # (100, 8)
    traffic_convention: npt.NDArray[np.float32]  # (2,)
    lateral_control_params: npt.NDArray[np.float32]  # (2,) speed, steering_delay
    prev_desired_curvature: npt.NDArray[np.float32]  # (100,)
    feature_buffer: npt.NDArray[np.float32]  # (100, 512)

    def as_dict(self, flatten_images: bool = True) -> Dict[str, npt.NDArray[np.float32]]:
        out: Dict[str, npt.NDArray[np.float32]] = {
            "desire": self.desire,
            "traffic_convention": self.traffic_convention,
            "lateral_control_params": self.lateral_control_params,
            "prev_desired_curvature": self.prev_desired_curvature,
            "feature_buffer": self.feature_buffer,
        }
        if flatten_images:
            out["image_stream"] = np.ascontiguousarray(self.image_stream.reshape(-1))
            out["wide_image_stream"] = np.ascontiguousarray(self.wide_image_stream.reshape(-1))
        else:
            out["image_stream"] = self.image_stream
            out["wide_image_stream"] = self.wide_image_stream
        return out


def map_name_to_traffic_convention(map_name: str) -> npt.NDArray[np.float32]:
    """
    One-hot [right_hand, left_hand]. NAVSIM / nuPlan release is mostly US (RHT).
    Extend the heuristic if you add LHT maps.
    """
    mn = (map_name or "").lower()
    left_hand_keys = ("gb-", "uk-", "london", "japan", "tokyo", "left_hand", "lht")
    is_left = any(k in mn for k in left_hand_keys)
    if is_left:
        return np.array([0.0, 1.0], dtype=np.float32)
    return np.array([1.0, 0.0], dtype=np.float32)


def driving_command_to_desire8(driving_command: npt.NDArray[np.int_]) -> npt.NDArray[np.float32]:
    """
    Map NAVSIM `driving_command` to 8-D desire slice (openpilot uses 8 one-hot slots).

    NAVSIM typically uses a 4-way one-hot: left / straight / right / unknown.
    Those are copied into the first four dimensions; remaining slots stay zero.
    If a single class index is provided instead, it is one-hot encoded into dim 0..7.
    """
    cmd = np.asarray(driving_command).astype(np.int64).ravel()
    out = np.zeros(8, dtype=np.float32)
    if cmd.size == 0:
        return out
    if cmd.size >= 4:
        out[: min(8, cmd.size)] = cmd[: min(8, cmd.size)].astype(np.float32)
        return out
    idx = int(cmd[0])
    if 0 <= idx < 8:
        out[idx] = 1.0
    return out


def rgb_uint8_to_yuv6planes(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """
    RGB uint8 (H, W, 3) resized to TARGET_RGB_HW, then packed as 6 x (H/2) x (W/2) float32 in [0, 1].

    Planes 0–3 are the four Y subsamples; plane 4 is half-res U (Cb); plane 5 is half-res V (Cr),
    using OpenCV RGB→YCrCb and area downsample for chroma.
    """
    if rgb is None or rgb.size == 0:
        h, w = TARGET_RGB_HW
        return np.zeros((6, h // 2, w // 2), dtype=np.float32)

    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {rgb_u8.shape}")

    target_w, target_h = TARGET_RGB_HW[1], TARGET_RGB_HW[0]
    resized = cv2.resize(rgb_u8, (target_w, target_h), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0
    cr = ycrcb[:, :, 1].astype(np.float32) / 255.0
    cb = ycrcb[:, :, 2].astype(np.float32) / 255.0

    h, w = y.shape
    u = cv2.resize(cb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(cr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    planes = np.stack(
        [
            y[::2, ::2],
            y[::2, 1::2],
            y[1::2, ::2],
            y[1::2, 1::2],
            u,
            v,
        ],
        axis=0,
    )
    return planes.astype(np.float32)


def _stack_two_frame_yuv(
    cam_t0: Optional[Camera],
    cam_t1: Optional[Camera],
) -> npt.NDArray[np.float32]:
    """Two consecutive frames -> (2, 6, 128, 256) float32."""
    imgs: List[npt.NDArray[np.float32]] = []
    for cam in (cam_t0, cam_t1):
        if cam is None or cam.image is None:
            imgs.append(rgb_uint8_to_yuv6planes(np.zeros((*TARGET_RGB_HW, 3), dtype=np.uint8)))
        else:
            imgs.append(rgb_uint8_to_yuv6planes(cam.image.astype(np.uint8)))
    return np.stack(imgs, axis=0).astype(np.float32)


def _get_camera_image(agent_input: AgentInput, time_idx: int, name: str) -> Optional[Camera]:
    if time_idx < 0 or time_idx >= len(agent_input.cameras):
        return None
    cams = agent_input.cameras[time_idx]
    return getattr(cams, name, None)


def _upsample_history_to_100(
    per_navsim_values: npt.NDArray[np.float32],
    navsim_count: int,
    steps_per_frame: int = STEPS_PER_NAVSIM_FRAME,
    total_steps: int = OPENPILOT_TEMPORAL_STEPS,
) -> npt.NDArray[np.float32]:
    """
    `per_navsim_values` shape (navsim_count, D). Right-align repeats; left-pad zeros.
    """
    if per_navsim_values.ndim == 1:
        per_navsim_values = per_navsim_values[:, np.newaxis]
    d = per_navsim_values.shape[1]
    out = np.zeros((total_steps, d), dtype=np.float32)
    filled = navsim_count * steps_per_frame
    used = min(filled, total_steps)
    navsim_used = (used + steps_per_frame - 1) // steps_per_frame
    start_out = total_steps - used
    for i in range(navsim_used):
        sl = slice(
            start_out + i * steps_per_frame,
            start_out + (i + 1) * steps_per_frame,
        )
        out[sl] = per_navsim_values[i]
    return out


def _history_curvatures_from_poses(poses: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Scalar curvature per segment from rear-axle (x, y, heading) rows in the current-time local frame."""
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2:
        return np.zeros(0, dtype=np.float32)
    kappas: List[float] = []
    for i in range(len(poses) - 1):
        x0, y0, th0 = poses[i]
        x1, y1, th1 = poses[i + 1]
        dx, dy = x1 - x0, y1 - y0
        ds = float(np.hypot(dx, dy))
        dth = float(np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0)))
        if ds > 1e-4:
            kappas.append(dth / ds)
        else:
            kappas.append(0.0)
    return np.asarray(kappas, dtype=np.float32)


def _kappa_segments_to_per_frame(kappa_segments: npt.NDArray[np.float32], n_frames: int) -> npt.NDArray[np.float32]:
    """Assign each history frame a scalar curvature (outgoing segment; last frame repeats previous)."""
    out = np.zeros(n_frames, dtype=np.float32)
    if n_frames == 0:
        return out
    ks = np.asarray(kappa_segments, dtype=np.float32).ravel()
    for i in range(n_frames - 1):
        out[i] = ks[i] if i < len(ks) else 0.0
    if n_frames >= 2:
        out[-1] = ks[n_frames - 2] if n_frames - 2 < len(ks) else out[-2]
    elif len(ks):
        out[-1] = ks[0]
    return out


def build_openpilot_inputs_from_scene(
    scene: Scene,
    agent_input: AgentInput,
    *,
    road_camera: str = "cam_f0",
    wide_camera: str = "cam_f0",
    steering_delay_s: float = 0.0,
) -> OpenpilotInputs:
    """
    Build openpilot-shaped inputs from one NAVSIM scene sample.

    :param road_camera: Sensor name on `Cameras` for the main (road) stream, default forward `cam_f0`.
    :param wide_camera: Second stream; NAVSIM has no comma-style wide FCAM — default duplicates `cam_f0`.
        Set e.g. `cam_l0` if you want a different field of view.
    :param steering_delay_s: Placeholder for openpilot's steering delay channel (not in NAVSIM logs).
    """
    n_hist = len(agent_input.cameras)
    idx_prev = max(0, n_hist - 2)
    idx_curr = n_hist - 1

    road_prev = _get_camera_image(agent_input, idx_prev, road_camera)
    road_curr = _get_camera_image(agent_input, idx_curr, road_camera)
    wide_prev = _get_camera_image(agent_input, idx_prev, wide_camera)
    wide_curr = _get_camera_image(agent_input, idx_curr, wide_camera)

    image_stack = _stack_two_frame_yuv(road_prev, road_curr)
    wide_stack = _stack_two_frame_yuv(wide_prev, wide_curr)

    desire_rows = np.stack(
        [driving_command_to_desire8(es.driving_command) for es in agent_input.ego_statuses],
        axis=0,
    )
    desire = _upsample_history_to_100(desire_rows, len(agent_input.ego_statuses))

    n_frames = len(agent_input.ego_statuses)
    poses_hist = np.stack([es.ego_pose for es in agent_input.ego_statuses], axis=0)
    kappa_segments = _history_curvatures_from_poses(poses_hist)
    per_frame_k = _kappa_segments_to_per_frame(kappa_segments, n_frames)

    prev_curv = _upsample_history_to_100(per_frame_k.astype(np.float32), n_frames).reshape(-1)

    vel = np.asarray(agent_input.ego_statuses[-1].ego_velocity, dtype=np.float32).ravel()
    speed = float(np.linalg.norm(vel[:2])) if vel.size >= 2 else float(np.linalg.norm(vel))
    lateral = np.array([speed, steering_delay_s], dtype=np.float32)

    traffic = map_name_to_traffic_convention(scene.scene_metadata.map_name or "")

    features = np.zeros((OPENPILOT_TEMPORAL_STEPS, 512), dtype=np.float32)

    return OpenpilotInputs(
        image_stream=image_stack,
        wide_image_stream=wide_stack,
        desire=desire,
        traffic_convention=traffic,
        lateral_control_params=lateral,
        prev_desired_curvature=prev_curv,
        feature_buffer=features,
    )


def scene_to_compact_original(scene: Scene) -> Dict[str, Any]:
    """JSON-friendly summary of the NAVSIM sample (no images / point clouds)."""
    meta = asdict(scene.scene_metadata)
    frames_out: List[Dict[str, Any]] = []
    for fr in scene.frames:
        es = fr.ego_status
        frames_out.append(
            {
                "token": fr.token,
                "timestamp": fr.timestamp,
                "driving_command": np.asarray(es.driving_command).tolist(),
                "ego_pose": np.asarray(es.ego_pose).tolist(),
                "ego_velocity": np.asarray(es.ego_velocity).tolist(),
                "ego_acceleration": np.asarray(es.ego_acceleration).tolist(),
            }
        )
    return {"scene_metadata": meta, "frames": frames_out}


def merge_original_and_openpilot(
    scene: Scene,
    agent_input: AgentInput,
    *,
    road_camera: str = "cam_f0",
    wide_camera: str = "cam_f0",
    steering_delay_s: float = 0.0,
    flatten_images: bool = True,
    original_mode: Literal["compact", "none"] = "compact",
) -> Dict[str, Any]:
    """
    One dict per dataset row: original (compact) NAVSIM fields + openpilot-shaped arrays.

    Arrays are numpy; use `pickle` or `np.savez` for persistence.
    """
    op = build_openpilot_inputs_from_scene(
        scene,
        agent_input,
        road_camera=road_camera,
        wide_camera=wide_camera,
        steering_delay_s=steering_delay_s,
    )
    out: Dict[str, Any] = {"openpilot": op.as_dict(flatten_images=flatten_images)}
    if original_mode == "compact":
        out["original"] = scene_to_compact_original(scene)
    elif original_mode == "none":
        out["original"] = {}
    else:
        raise ValueError(original_mode)
    return out
