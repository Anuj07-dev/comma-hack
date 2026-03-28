"""
Run ``OpenpilotAgent.compute_trajectory(agent_input, scene)`` without PDM scoring or metric cache.

Loads a ``Scene`` and ``AgentInput`` from either:
  * a synthetic NAVSIM scene pickle (``--scene-pkl``), or
  * a log shard (``--log-shard-pkl``: one ``.pkl`` containing a list of frames).

Requires ``NUPLAN_MAPS_ROOT``. The agent is built from the same YAML as evaluation
(``navsim/planning/script/config/common/agent/openpilot_agent.yaml`` by default).

Example (synthetic scene + sensors under navhard_two_stage):
  export NUPLAN_MAPS_ROOT=/path/to/maps
  python -m navsim.planning.script.run_openpilot_agent_compute \\
    --scene-pkl /path/to/synthetic_scene_pickles/foo.pkl \\
    --sensor-blobs /path/to/sensor_blobs

Example (openscene_meta_datas shard, 4+1 frames):
  python -m navsim.planning.script.run_openpilot_agent_compute \\
    --log-shard-pkl /path/to/openscene_meta_datas/bar.pkl \\
    --sensor-blobs /path/to/sensor_blobs \\
    --num-history-frames 4 --num-future-frames 1

To dump loaded camera frames as PNG (before inference):
  ... --log-input-images-dir /tmp/openpilot_input_pngs
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from navsim.common.dataclasses import AgentInput, Camera, Scene, SensorConfig

# Must match attribute names on ``Cameras`` (navsim.common.dataclasses).
_CAMERAS_FIELD_NAMES = (
    "cam_f0",
    "cam_l0",
    "cam_l1",
    "cam_l2",
    "cam_r0",
    "cam_r1",
    "cam_r2",
    "cam_b0",
)

DEFAULT_AGENT_YAML = (
    Path(__file__).resolve().parent / "config" / "common" / "agent" / "openpilot_agent.yaml"
)

logger = logging.getLogger(__name__)


def _image_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert camera array to HxWx3 uint8 for PNG."""
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("empty image array")
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if np.issubdtype(a.dtype, np.floating):
        mx = float(np.nanmax(a)) if a.size else 0.0
        if mx <= 1.0 + 1e-3:
            u8 = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
        else:
            u8 = np.clip(np.round(a), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(u8)
    return np.ascontiguousarray(np.clip(a, 0, 255).astype(np.uint8))


def _log_agent_input_images_png(agent_input: AgentInput, dest: Path, stem: str) -> None:
    """Write each loaded camera image in ``AgentInput`` history as a PNG."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for t, cams in enumerate(agent_input.cameras):
        for cam_name in _CAMERAS_FIELD_NAMES:
            cam: Camera = getattr(cams, cam_name)
            if cam.image is None or np.asarray(cam.image).size == 0:
                continue
            rgb = _image_to_uint8_rgb(cam.image)
            out = dest / f"{stem}_t{t:02d}_{cam_name}.png"
            Image.fromarray(rgb, mode="RGB").save(out)
            n += 1
            logger.info("Wrote %s", out)
    if n == 0:
        logger.warning("No camera images to save under %s", dest)


def _load_agent(agent_yaml: Path):
    return instantiate(OmegaConf.load(agent_yaml))


def _load_scene_and_input_from_pickle(
    scene_pkl: Path,
    sensor_blobs: Path,
    sensor_config: SensorConfig,
) -> Tuple[Scene, AgentInput]:
    scene = Scene.load_from_disk(scene_pkl, sensor_blobs, sensor_config)
    return scene, scene.get_agent_input()


def _load_scene_and_input_from_log_shard(
    log_shard_pkl: Path,
    sensor_blobs: Path,
    num_history_frames: int,
    num_future_frames: int,
    sensor_config: SensorConfig,
) -> Tuple[Scene, AgentInput]:
    with open(log_shard_pkl, "rb") as f:
        frames = pickle.load(f)
    if not isinstance(frames, list) or len(frames) == 0:
        raise SystemExit(f"{log_shard_pkl} must contain a non-empty list of frames")
    need = num_history_frames + num_future_frames
    if len(frames) < need:
        raise SystemExit(
            f"{log_shard_pkl} has {len(frames)} frames; need at least "
            f"{need} (num_history_frames + num_future_frames)"
        )
    scene = Scene.from_scene_dict_list(
        frames,
        sensor_blobs,
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
        sensor_config=sensor_config,
    )
    agent_input = AgentInput.from_scene_dict_list(
        frames,
        sensor_blobs,
        num_history_frames=num_history_frames,
        sensor_config=sensor_config,
    )
    return scene, agent_input


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--scene-pkl",
        type=Path,
        help="Synthetic scene pickle (Scene.save_to_disk format)",
    )
    src.add_argument(
        "--log-shard-pkl",
        type=Path,
        help="Log shard: pickle of list[frame_dict] (openscene_meta_datas style)",
    )
    p.add_argument(
        "--sensor-blobs",
        type=Path,
        required=True,
        help="Root directory of sensor blobs (images, etc.)",
    )
    p.add_argument(
        "--agent-yaml",
        type=Path,
        default=DEFAULT_AGENT_YAML,
        help="Hydra-style agent config (default: openpilot_agent.yaml)",
    )
    p.add_argument("--num-history-frames", type=int, default=4)
    p.add_argument("--num-future-frames", type=int, default=10)
    p.add_argument(
        "--log-input-images-dir",
        type=Path,
        default=None,
        help="If set, save AgentInput camera images as PNG files in this directory",
    )
    args = p.parse_args(argv)

    agent = _load_agent(args.agent_yaml)
    sensor_config = agent.get_sensor_config()

    if args.scene_pkl is not None:
        scene, agent_input = _load_scene_and_input_from_pickle(
            args.scene_pkl, args.sensor_blobs, sensor_config
        )
    else:
        scene, agent_input = _load_scene_and_input_from_log_shard(
            args.log_shard_pkl,
            args.sensor_blobs,
            args.num_history_frames,
            args.num_future_frames,
            sensor_config,
        )

    logger.info("Scene token=%s log=%s", scene.scene_metadata.initial_token, scene.scene_metadata.log_name)
    if args.log_input_images_dir is not None:
        stem = scene.scene_metadata.initial_token
        _log_agent_input_images_png(agent_input, args.log_input_images_dir, stem)
    trajectory = agent.compute_trajectory(agent_input, scene)
    logger.info(
        "Trajectory: %d poses, interval=%.3f s",
        trajectory.poses.shape[0],
        trajectory.trajectory_sampling.interval_length,
    )
    logger.info("First pose (x, y, heading): %s", trajectory.poses[0])


if __name__ == "__main__":
    main()
