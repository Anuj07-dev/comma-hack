"""
NAVSIM agent that runs comma-style driving as two ONNX models: vision then policy.

Uses the same preprocessing as `run_preprocess_openpilot_inputs.py` /
`build_openpilot_inputs_from_scene`. Vision inputs are image streams; policy
inputs add desire, traffic, lateral params, curvature history, and (typically)
features produced by the vision model.

Configure `policy_inputs_from_vision` to map policy ONNX input names to vision
ONNX output names. If omitted and the vision model has a single output, it is
auto-wired to any policy input whose name heuristically matches `feature_buffer`.

For that feature-buffer path, the vision tensor is flattened and sliced to
``feature_buffer_vision_flat_slice`` (default ``[1064:1576]``, length 512) before
reshaping to the policy ONNX input.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.openpilot_policy_parse import (
    CommaPolicyOutputParser,
    plan_to_xy_heading,
    slice_flat_policy_output,
)
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.preprocessing.openpilot_model_inputs import build_openpilot_inputs_from_scene

logger = logging.getLogger(__name__)

CAMERA_NAMES = (
    "cam_f0",
    "cam_l0",
    "cam_l1",
    "cam_l2",
    "cam_r0",
    "cam_r1",
    "cam_r2",
    "cam_b0",
)


def _sensor_config_for_cameras(enabled: frozenset) -> SensorConfig:
    kwargs = {name: (name in enabled) for name in CAMERA_NAMES}
    kwargs["lidar_pc"] = False
    return SensorConfig(**kwargs)


def _static_dims(shape: Sequence[object]) -> List[int]:
    """Replace symbolic dims with 1 for concrete numpy shapes."""
    static: List[int] = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            static.append(d)
        else:
            static.append(1)
    return static


def _logical_tensors_from_openpilot(
    op_dict: Dict[str, np.ndarray],
    concatenate_image_streams: bool,
) -> Dict[str, np.ndarray]:
    """Logical names -> arrays for binding (batchless)."""
    d = dict(op_dict)
    if concatenate_image_streams:
        d["image_concat"] = np.concatenate(
            [d["image_stream"].reshape(-1), d["wide_image_stream"].reshape(-1)],
            axis=0,
        ).astype(np.float32, copy=False)
    return d


def _guess_logical_key(onnx_name: str) -> Optional[str]:
    n = onnx_name.lower()
    if "feature" in n and "buffer" in n:
        return "feature_buffer"
    if "traffic" in n:
        return "traffic_convention"
    if "prev" in n and "curv" in n:
        return "prev_desired_curvature"
    if "lateral" in n or ("steer" in n and "delay" in n) or n.endswith("_delay"):
        return "lateral_control_params"
    if "desire" in n:
        return "desire"
    if "big" in n or ("wide" in n and "img" in n):
        return "wide_image_stream"
    if "img" in n or "yuv" in n or "input_imgs" in n:
        return "image_stream"
    if "state" in n or "hidden" in n or "gru" in n or "memory" in n or "initial" in n:
        return None  # recurrent / manual
    return None


def _reshape_for_onnx_input(
    arr: np.ndarray,
    target_static_shape: List[int],
    onnx_tensor_type: str,
) -> np.ndarray:
    """
    Reshape `arr` to ONNX static shape with dtype matching the session input.

    Preprocessing uses float32 YUV planes in [0, 1]; comma's vision ONNX often expects
    ``tensor(uint8)``, in which case values are scaled to 0–255.
    """
    expected = int(np.prod(target_static_shape))
    tl = onnx_tensor_type.replace(" ", "").lower()
    if "uint8" in tl:
        flat = np.asarray(arr).reshape(-1)
        if flat.size != expected:
            raise ValueError(
                f"Cannot bind array of size {flat.size} to ONNX input shape {target_static_shape} "
                f"(expected {expected} elements)"
            )
        if flat.dtype == np.uint8 or np.issubdtype(flat.dtype, np.unsignedinteger):
            out = flat.astype(np.uint8, copy=False)
        else:
            x64 = flat.astype(np.float64, copy=False)
            mx = float(np.max(x64)) if x64.size else 0.0
            if mx <= 1.5:
                out = np.clip(np.round(x64 * 255.0), 0, 255).astype(np.uint8)
            else:
                out = np.clip(np.round(x64), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out.reshape(target_static_shape))

    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.size != expected:
        raise ValueError(
            f"Cannot bind array of size {x.size} to ONNX input shape {target_static_shape} (expected {expected} elements)"
        )
    out = np.ascontiguousarray(x.reshape(target_static_shape))
    if "float16" in tl:
        return out.astype(np.float16, copy=False)
    return out


def _interpolate_plan_to_poses(plan: np.ndarray, num_poses: int) -> np.ndarray:
    """Resample plan rows to `num_poses` (x, y, heading)."""
    p = np.asarray(plan, dtype=np.float64)
    if p.ndim == 1:
        if p.size % 3 != 0:
            raise ValueError(f"1D plan length {p.size} not divisible by 3")
        p = p.reshape(-1, 3)
    if p.shape[1] < 2:
        raise ValueError(f"Plan must have at least 2 columns, got {p.shape}")
    if p.shape[1] == 2:
        xy = p[:, :2]
        d = np.diff(xy, axis=0)
        heading = np.arctan2(d[:, 1], d[:, 0])
        heading = np.concatenate([heading[:1], heading])
        p = np.column_stack([xy, heading])
    n = p.shape[0]
    if n == num_poses:
        out = p[:, :3]
    elif n > num_poses:
        idx = np.linspace(0, n - 1, num=num_poses).astype(np.int64)
        out = p[idx, :3]
    else:
        t_old = np.linspace(0.0, 1.0, n)
        t_new = np.linspace(0.0, 1.0, num_poses)
        out = np.zeros((num_poses, 3), dtype=np.float64)
        for k in range(3):
            out[:, k] = np.interp(t_new, t_old, p[:, k])
    return out.astype(np.float32)


def _print_nan_in_onnx_feed(model: str, feed: Dict[str, np.ndarray]) -> None:
    """Print a line per ONNX input tensor that contains NaNs (floating dtypes only)."""
    for name in sorted(feed.keys()):
        arr = np.asarray(feed[name])
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.floating):
            continue
        mask = np.isnan(arr)
        if mask.any():
            print(
                f"[OpenpilotAgent] {model} ONNX input {name!r}: "
                f"{int(mask.sum())} NaN / {arr.size} (shape {tuple(arr.shape)}, dtype {arr.dtype})"
            )


def _infer_policy_inputs_from_vision(
    policy_session: ort.InferenceSession,
    vision_output_names: List[str],
) -> Dict[str, str]:
    """
    If the vision model exposes exactly one output, map it to the first policy
    input that looks like a feature buffer (by name heuristic).
    """
    if len(vision_output_names) != 1:
        return {}
    vis_out = vision_output_names[0]
    for inp in policy_session.get_inputs():
        if _guess_logical_key(inp.name) == "feature_buffer":
            return {inp.name: vis_out}
    return {}


class OpenpilotAgent(AbstractAgent):
    """Runs vision ONNX then policy ONNX with NAVSIM-aligned preprocessing."""

    requires_scene = True

    def __init__(
        self,
        vision_model_path: str,
        policy_model_path: str,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        road_camera: str = "cam_f0",
        wide_camera: str = "cam_f0",
        steering_delay_s: float = 0.0,
        providers: Optional[List[str]] = None,
        vision_input_name_map: Optional[Dict[str, str]] = None,
        policy_input_name_map: Optional[Dict[str, str]] = None,
        policy_inputs_from_vision: Optional[Dict[str, str]] = None,
        concatenate_image_streams: bool = False,
        image_concat_input_name: Optional[str] = None,
        recurrent_input_names: Optional[List[str]] = None,
        recurrent_output_names: Optional[List[str]] = None,
        plan_output_index: Optional[int] = None,
        plan_slice: Optional[slice] = None,
        feature_buffer_vision_flat_slice: Optional[Tuple[int, int]] = (1064, 1576),
        policy_metadata_path: Optional[str] = None,
    ):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._vision_model_path = vision_model_path
        self._policy_model_path = policy_model_path
        self._policy_metadata_path = policy_metadata_path
        self._road_camera = road_camera
        self._wide_camera = wide_camera
        self._steering_delay_s = steering_delay_s
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._vision_input_name_map = dict(vision_input_name_map) if vision_input_name_map else {}
        self._policy_input_name_map = dict(policy_input_name_map) if policy_input_name_map else {}
        self._policy_inputs_from_vision = (
            dict(policy_inputs_from_vision) if policy_inputs_from_vision else None
        )
        self._concatenate_image_streams = concatenate_image_streams
        self._image_concat_input_name = image_concat_input_name
        self._recurrent_in = list(recurrent_input_names) if recurrent_input_names else []
        self._recurrent_out = list(recurrent_output_names) if recurrent_output_names else []
        self._plan_output_index = plan_output_index
        self._plan_slice = plan_slice
        if feature_buffer_vision_flat_slice is None:
            self._feature_buffer_vision_flat_slice: Optional[Tuple[int, int]] = None
        else:
            fbs = tuple(feature_buffer_vision_flat_slice)
            if len(fbs) != 2 or fbs[0] < 0 or fbs[1] < fbs[0]:
                raise ValueError(
                    f"feature_buffer_vision_flat_slice must be (start, end) with end>start; got {fbs}"
                )
            self._feature_buffer_vision_flat_slice = (int(fbs[0]), int(fbs[1]))

        self._vision_session: Optional[ort.InferenceSession] = None
        self._policy_session: Optional[ort.InferenceSession] = None
        self._vision_input_names: List[str] = []
        self._vision_output_names: List[str] = []
        self._policy_input_names: List[str] = []
        self._policy_output_names: List[str] = []
        self._recurrent_state: Dict[str, np.ndarray] = {}
        self._resolved_vision_to_policy: Dict[str, str] = {}
        self._policy_output_slices: Optional[Dict[str, slice]] = None
        self._comma_policy_parser: Optional[CommaPolicyOutputParser] = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._vision_session = ort.InferenceSession(
            self._vision_model_path, sess_options=so, providers=self._providers
        )
        self._policy_session = ort.InferenceSession(
            self._policy_model_path, sess_options=so, providers=self._providers
        )
        self._vision_input_names = [i.name for i in self._vision_session.get_inputs()]
        self._vision_output_names = [o.name for o in self._vision_session.get_outputs()]
        self._policy_input_names = [i.name for i in self._policy_session.get_inputs()]
        self._policy_output_names = [o.name for o in self._policy_session.get_outputs()]

        self._resolved_vision_to_policy = dict(self._policy_inputs_from_vision or {})
        inferred = _infer_policy_inputs_from_vision(self._policy_session, self._vision_output_names)
        for k, v in inferred.items():
            self._resolved_vision_to_policy.setdefault(k, v)

        self._policy_output_slices = None
        self._comma_policy_parser = None
        meta_path = self._resolve_policy_metadata_path()
        if meta_path is not None:
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                slices = meta.get("output_slices")
                if isinstance(slices, dict) and "plan" in slices:
                    self._policy_output_slices = slices
                    self._comma_policy_parser = CommaPolicyOutputParser()
                    logger.info("Loaded comma policy metadata from %s", meta_path)
                else:
                    logger.warning(
                        "Policy metadata at %s has no usable output_slices['plan']; using legacy plan extraction",
                        meta_path,
                    )
            except Exception as e:
                logger.warning(
                    "Could not load policy metadata from %s (%s); using legacy plan extraction",
                    meta_path,
                    e,
                )

        self._recurrent_state.clear()
        for inp in self._policy_session.get_inputs():
            if inp.name in self._recurrent_in:
                static = _static_dims(inp.shape)
                self._recurrent_state[inp.name] = np.zeros(static, dtype=np.float32)

        logger.info(
            "OpenpilotAgent vision: %s inputs=%s outputs=%s",
            self._vision_model_path,
            self._vision_input_names,
            self._vision_output_names,
        )
        logger.info(
            "OpenpilotAgent policy: %s inputs=%s outputs=%s vision_to_policy=%s",
            self._policy_model_path,
            self._policy_input_names,
            self._policy_output_names,
            self._resolved_vision_to_policy,
        )

    def get_sensor_config(self) -> SensorConfig:
        for cam in (self._road_camera, self._wide_camera):
            if cam not in CAMERA_NAMES:
                raise ValueError(f"Unknown camera {cam}; expected one of {CAMERA_NAMES}")
        return _sensor_config_for_cameras(frozenset({self._road_camera, self._wide_camera}))

    def _resolve_policy_metadata_path(self) -> Optional[Path]:
        if self._policy_metadata_path:
            p = Path(self._policy_metadata_path)
            if not p.is_file():
                logger.warning("policy_metadata_path is not a file: %s", p)
                return None
            return p
        cand = Path(self._policy_model_path).parent / "driving_policy_metadata.pkl"
        return cand if cand.is_file() else None

    def _policy_input_logical_key(self, policy_in_name: str) -> Optional[str]:
        return self._policy_input_name_map.get(policy_in_name) or _guess_logical_key(policy_in_name)

    def _bind_vision_inputs(self, logical: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        assert self._vision_session is not None
        feed: Dict[str, np.ndarray] = {}
        for inp in self._vision_session.get_inputs():
            name = inp.name
            static_shape = _static_dims(inp.shape)
            key: Optional[str] = self._vision_input_name_map.get(name)
            if key is None:
                key = _guess_logical_key(name)
            if key is None:
                raise RuntimeError(
                    f"Cannot bind vision ONNX input '{name}' (shape {inp.shape}). "
                    f"Set `vision_input_name_map`."
                )
            if self._image_concat_input_name and name == self._image_concat_input_name:
                arr = logical["image_concat"]
            else:
                if key not in logical:
                    raise KeyError(f"Missing logical tensor '{key}' for vision input '{name}'")
                arr = logical[key]
            feed[name] = _reshape_for_onnx_input(arr, static_shape, inp.type)
        return feed

    def _bind_policy_inputs(
        self,
        logical: Dict[str, np.ndarray],
        vision_outputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        assert self._policy_session is not None
        feed: Dict[str, np.ndarray] = {}
        for inp in self._policy_session.get_inputs():
            name = inp.name
            static_shape = _static_dims(inp.shape)
            if name in self._recurrent_in:
                feed[name] = _reshape_for_onnx_input(self._recurrent_state[name], static_shape, inp.type)
                continue

            arr: Optional[np.ndarray] = None
            if name in self._resolved_vision_to_policy:
                vis_key = self._resolved_vision_to_policy[name]
                if vis_key not in vision_outputs:
                    raise KeyError(
                        f"Policy input '{name}' expects vision output '{vis_key}', "
                        f"but vision produced: {list(vision_outputs.keys())}"
                    )
                arr = np.asarray(vision_outputs[vis_key], dtype=np.float32)
                expected_elems = int(np.prod(static_shape))
                flat = arr.reshape(-1)

                if flat.size == expected_elems:
                    pass
                elif (
                    self._feature_buffer_vision_flat_slice is not None
                    and self._policy_input_logical_key(name) == "feature_buffer"
                ):
                    lo, hi = self._feature_buffer_vision_flat_slice
                    if flat.size < hi:
                        raise ValueError(
                            f"Vision output '{vis_key}' for policy input '{name}' has length {flat.size}, "
                            f"need at least {hi} for feature_buffer slice [{lo}:{hi}]"
                        )
                    arr = flat[lo:hi]
                else:
                    arr = flat

                arr_flat = np.asarray(arr, dtype=np.float32).reshape(-1)
                if arr_flat.size != expected_elems:
                    if expected_elems % arr_flat.size != 0:
                        raise ValueError(
                            f"Vision output '{vis_key}' for policy input '{name}': "
                            f"{arr_flat.size} elements after wiring, ONNX expects {expected_elems}"
                        )
                    arr_flat = np.tile(arr_flat, expected_elems // arr_flat.size)
                arr = arr_flat

            if arr is None:
                key: Optional[str] = self._policy_input_name_map.get(name)
                if key is None:
                    key = _guess_logical_key(name)
                if key is None:
                    raise RuntimeError(
                        f"Cannot bind policy ONNX input '{name}' (shape {inp.shape}). "
                        f"Set `policy_input_name_map`, `policy_inputs_from_vision`, or `recurrent_input_names`."
                    )
                if key not in logical:
                    raise KeyError(f"Missing logical tensor '{key}' for policy input '{name}'")
                arr = logical[key]

            feed[name] = _reshape_for_onnx_input(arr, static_shape, inp.type)
        return feed

    def _update_recurrent(self, outputs: List[np.ndarray]) -> None:
        if not self._recurrent_in or not self._recurrent_out:
            return
        for rin, rout in zip(self._recurrent_in, self._recurrent_out):
            if rout not in self._policy_output_names:
                continue
            idx = self._policy_output_names.index(rout)
            if idx < len(outputs):
                self._recurrent_state[rin] = np.asarray(outputs[idx], dtype=np.float32, copy=True)

    def _extract_plan(self, outputs: List[np.ndarray]) -> np.ndarray:
        if self._plan_output_index is not None:
            plan = np.asarray(outputs[self._plan_output_index])
            if self._plan_slice is not None:
                plan = plan[self._plan_slice]
            return plan
        best: Optional[Tuple[int, float]] = None
        for i, o in enumerate(outputs):
            a = np.asarray(o)
            if a.dtype.kind not in "fc" or a.ndim < 2:
                continue
            score = float(np.prod(a.shape))
            if best is None or score > best[1]:
                best = (i, score)
        if best is None:
            for i, o in enumerate(outputs):
                a = np.asarray(o)
                if a.dtype.kind in "fc" and a.ndim == 1 and a.size >= 6:
                    best = (i, float(a.size))
                    break
        if best is None:
            raise RuntimeError(
                "Could not guess plan output. Set `plan_output_index` (and optional `plan_slice`) in config."
            )
        plan = np.asarray(outputs[best[0]])
        if self._plan_slice is not None:
            plan = plan[self._plan_slice]
        logger.debug("Using policy output index %s shape %s for plan", best[0], plan.shape)
        return plan

    def _trajectory_from_comma_policy_parse(self, policy_outputs: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Decode comma's flat policy blob via ``output_slices`` + MDN (same as openpilot ``modeld``).
        Returns ``(num_poses, 3)`` float32 or ``None`` to fall back to legacy extraction.
        """
        if self._policy_output_slices is None or self._comma_policy_parser is None:
            return None
        if len(policy_outputs) != 1:
            logger.warning(
                "Comma plan parsing expects one policy output tensor; got %d. Using legacy plan extraction.",
                len(policy_outputs),
            )
            return None
        flat = np.asarray(policy_outputs[0], dtype=np.float32).reshape(-1)
        try:
            sliced = slice_flat_policy_output(flat, self._policy_output_slices)
            parsed = self._comma_policy_parser.parse_policy_outputs(sliced)
            plan = parsed["plan"]
            xyh = plan_to_xy_heading(plan)
            if not np.all(np.isfinite(xyh)):
                logger.warning(
                    "Parsed comma plan has non-finite values (vision/policy inputs may be NaN or out of range)."
                )
            return _interpolate_plan_to_poses(xyh, self._trajectory_sampling.num_poses)
        except Exception as e:
            logger.warning(
                "Comma policy plan parse failed (%s). Using legacy plan extraction.",
                e,
            )
            return None

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        if self._vision_session is None or self._policy_session is None:
            self.initialize()

        op = build_openpilot_inputs_from_scene(
            scene,
            agent_input,
            road_camera=self._road_camera,
            wide_camera=self._wide_camera,
            steering_delay_s=self._steering_delay_s,
        )
        logical = _logical_tensors_from_openpilot(
            op.as_dict(flatten_images=True),
            concatenate_image_streams=self._concatenate_image_streams
            or bool(self._image_concat_input_name),
        )
        if self._image_concat_input_name and "image_concat" not in logical:
            logical["image_concat"] = np.concatenate(
                [logical["image_stream"].reshape(-1), logical["wide_image_stream"].reshape(-1)],
                axis=0,
            ).astype(np.float32, copy=False)

        vision_feed = self._bind_vision_inputs(logical)
        assert self._vision_session is not None
        _print_nan_in_onnx_feed("vision", vision_feed)
        vision_out_list = self._vision_session.run(self._vision_output_names, vision_feed)
        vision_outputs = dict(zip(self._vision_output_names, vision_out_list))

        policy_feed = self._bind_policy_inputs(logical, vision_outputs)
        assert self._policy_session is not None
        _print_nan_in_onnx_feed("policy", policy_feed)
        policy_outputs = self._policy_session.run(self._policy_output_names, policy_feed)
        self._update_recurrent(policy_outputs)

        poses = self._trajectory_from_comma_policy_parse(policy_outputs)
        if poses is None:
            plan = self._extract_plan(policy_outputs)
            poses = _interpolate_plan_to_poses(plan, self._trajectory_sampling.num_poses)
        if poses.shape != (self._trajectory_sampling.num_poses, 3):
            poses = poses.reshape(self._trajectory_sampling.num_poses, 3)
        return Trajectory(poses, self._trajectory_sampling)
