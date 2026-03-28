"""
Preprocess NAVSIM log scenes into comma openpilot driving-model-shaped tensors.

By default reads the **navhard_two_stage** layout under ``--dataset-root``:
  ``openscene_meta_datas/*.pkl`` (one list of frames per file), ``sensor_blobs/``,
  and optionally ``synthetic_scene_pickles/``.

Requires NUPLAN_MAPS_ROOT (Scene loading builds the map API like the rest of NAVSIM).

To compare these tensors to ONNX inputs, run
``python -m navsim.planning.script.run_check_openpilot_onnx_shapes`` (after setting model paths).

Example (navhard_two_stage checkout):
  export NUPLAN_MAPS_ROOT=/path/to/maps
  python -m navsim.planning.script.run_preprocess_openpilot_inputs \\
    --output-dir /path/to/out \\
    --max-scenes 10

Example (classic log shards + OPENSCENE_DATA_ROOT sensors):
  export OPENSCENE_DATA_ROOT=/path/to/openscene
  python -m navsim.planning.script.run_preprocess_openpilot_inputs \\
    --legacy-log-shards \\
    --navsim-log-path /path/to/navsim_logs/test \\
    --sensor-path /path/to/sensor_blobs/test \\
    --num-future-frames 10 \\
    --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from tqdm import tqdm

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.preprocessing.openpilot_model_inputs import merge_original_and_openpilot

logger = logging.getLogger(__name__)

DEFAULT_NAVHARD_DATASET_ROOT = Path("/home/akshat/navsim/dataset/navhard_two_stage")

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


def _print_openpilot_shapes(openpilot: Dict[str, Any]) -> None:
    """Print dtype and shape for each tensor in the openpilot branch (once per run)."""
    lines = ["openpilot preprocessed tensors (name, dtype, shape):"]
    for name in sorted(openpilot.keys()):
        arr = openpilot[name]
        if hasattr(arr, "dtype") and hasattr(arr, "shape"):
            lines.append(f"  {name:28}  {str(arr.dtype):10}  {tuple(arr.shape)}")
        else:
            lines.append(f"  {name:28}  {type(arr).__name__}")
    tqdm.write("\n".join(lines))


def _sensor_config_for_cameras(enabled: Set[str]) -> SensorConfig:
    kwargs = {name: (name in enabled) for name in CAMERA_NAMES}
    kwargs["lidar_pc"] = False
    return SensorConfig(**kwargs)


def _infer_num_future_frames(meta_dir: Path, num_history_frames: int) -> int:
    """
    Infer future frame count from one shard so num_history + num_future matches file length.
    navhard_two_stage ``openscene_meta_datas`` shards are length-5 lists (4 history + 1 future).
    """
    pkls = sorted(meta_dir.glob("*.pkl"))
    if not pkls:
        return 10
    with open(pkls[0], "rb") as f:
        frames = pickle.load(f)
    if not isinstance(frames, list):
        return 10
    n = len(frames)
    if n < num_history_frames:
        raise SystemExit(
            f"First pickle in {meta_dir} has {n} frames, fewer than --num-history-frames={num_history_frames}"
        )
    return n - num_history_frames


def _resolve_navhard_paths(
    dataset_root: Optional[Path],
    navsim_log_path: Optional[Path],
    sensor_path: Optional[Path],
    synthetic_scenes_path: Optional[Path],
    include_synthetic: bool,
) -> Tuple[Path, Path, Optional[Path]]:
    """
    :return: (meta_pickle_dir, sensor_blobs_root, synthetic_scenes_dir or None)
    """
    if dataset_root is not None:
        root = dataset_root.resolve()
        meta = (navsim_log_path.resolve() if navsim_log_path is not None else root / "openscene_meta_datas")
        if not meta.is_dir():
            raise SystemExit(f"Scene meta directory does not exist: {meta}")
        if sensor_path is not None:
            sensor = sensor_path.resolve()
        else:
            candidates = [
                root / "sensor_blobs",
                Path(os.environ.get("OPENSCENE_DATA_ROOT", "") or "") / "navhard_two_stage" / "sensor_blobs",
                Path(os.environ.get("OPENSCENE_DATA_ROOT", "") or ""),
            ]
            sensor = next((c.resolve() for c in candidates if c.is_dir()), None)
            if sensor is None:
                raise SystemExit(
                    f"No sensor directory found under {root / 'sensor_blobs'} or OPENSCENE_DATA_ROOT fallbacks. "
                    "Set --sensor-path or OPENSCENE_DATA_ROOT."
                )
        if not sensor.is_dir():
            raise SystemExit(f"Sensor directory does not exist: {sensor}")
        synthetic: Optional[Path] = None
        if include_synthetic:
            if synthetic_scenes_path is not None:
                synthetic = synthetic_scenes_path.resolve()
            else:
                cand = root / "synthetic_scene_pickles"
                synthetic = cand if cand.is_dir() else None
            if synthetic is None or not synthetic.is_dir():
                raise SystemExit(
                    "Synthetic scenes requested but no directory found. "
                    f"Expected {root / 'synthetic_scene_pickles'} or pass --synthetic-scenes-path"
                )
        return meta, sensor, synthetic

    if navsim_log_path is None:
        raise SystemExit("Pass --dataset-root (navhard layout) or --navsim-log-path (log shard directory)")
    meta = navsim_log_path.resolve()
    if not meta.is_dir():
        raise SystemExit(f"navsim log path is not a directory: {meta}")
    if sensor_path is None:
        root = os.environ.get("OPENSCENE_DATA_ROOT")
        if not root:
            raise SystemExit("Set --sensor-path or OPENSCENE_DATA_ROOT when using --navsim-log-path without --dataset-root")
        sensor = Path(root).resolve()
    else:
        sensor = sensor_path.resolve()
    if not sensor.is_dir():
        raise SystemExit(f"Sensor directory does not exist: {sensor}")
    synthetic = synthetic_scenes_path.resolve() if synthetic_scenes_path is not None else None
    if include_synthetic and (synthetic is None or not synthetic.is_dir()):
        raise SystemExit("--include-synthetic requires a valid --synthetic-scenes-path")
    return meta, sensor, synthetic


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_NAVHARD_DATASET_ROOT,
        help=f"navhard_two_stage bundle root (openscene_meta_datas, sensor_blobs, ...). Default: {DEFAULT_NAVHARD_DATASET_ROOT}",
    )
    p.add_argument(
        "--legacy-log-shards",
        action="store_true",
        help="Load classic NAVSIM log pickles: requires --navsim-log-path (and --sensor-path or OPENSCENE_DATA_ROOT)",
    )
    p.add_argument(
        "--navsim-log-path",
        type=Path,
        default=None,
        help="Override scene pickle directory (default under dataset-root: openscene_meta_datas)",
    )
    p.add_argument(
        "--sensor-path",
        type=Path,
        default=None,
        help="Sensor blob root (default: dataset-root/sensor_blobs if present, else OPENSCENE_DATA_ROOT)",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write one pickle per scene token")
    p.add_argument("--num-history-frames", type=int, default=4)
    p.add_argument(
        "--num-future-frames",
        type=int,
        default=-1,
        help="Future frame count; default -1 infers from first meta pickle (navhard meta: 5 frames -> 1 future)",
    )
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--road-camera", type=str, default="cam_f0")
    p.add_argument("--wide-camera", type=str, default="cam_f0")
    p.add_argument("--steering-delay-s", type=float, default=0.0)
    p.add_argument(
        "--no-flatten-images",
        action="store_true",
        help="Keep image_stream as (2, 6, 128, 256) instead of length-393216 vectors",
    )
    p.add_argument(
        "--synthetic-sensor-path",
        type=Path,
        default=None,
        help="Sensor root for synthetic scenes (if used)",
    )
    p.add_argument("--synthetic-scenes-path", type=Path, default=None, help="Directory of synthetic scene pickles")
    p.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Load synthetic scenes from dataset-root/synthetic_scene_pickles (navhard_two_stage)",
    )
    args = p.parse_args()

    dataset_root: Optional[Path] = None if args.legacy_log_shards else args.dataset_root.resolve()

    meta_path, sensor_path, synthetic_dir = _resolve_navhard_paths(
        dataset_root=dataset_root,
        navsim_log_path=args.navsim_log_path,
        sensor_path=args.sensor_path,
        synthetic_scenes_path=args.synthetic_scenes_path,
        include_synthetic=args.include_synthetic,
    )

    num_future = args.num_future_frames
    if num_future < 0:
        num_future = _infer_num_future_frames(meta_path, args.num_history_frames)
        logger.info(
            "Inferred --num-future-frames=%s from shards in %s (with num_history_frames=%s)",
            num_future,
            meta_path,
            args.num_history_frames,
        )

    enabled_cams: Set[str] = {args.road_camera, args.wide_camera}
    unknown = enabled_cams - set(CAMERA_NAMES)
    if unknown:
        raise SystemExit(f"Unknown camera name(s): {unknown}; expected one of {CAMERA_NAMES}")

    scene_filter = SceneFilter(
        num_history_frames=args.num_history_frames,
        num_future_frames=num_future,
        max_scenes=args.max_scenes,
        include_synthetic_scenes=args.include_synthetic,
    )

    syn_sensor = args.synthetic_sensor_path.resolve() if args.synthetic_sensor_path is not None else sensor_path

    scene_loader = SceneLoader(
        data_path=meta_path,
        original_sensor_path=sensor_path,
        synthetic_sensor_path=syn_sensor,
        synthetic_scenes_path=synthetic_dir,
        scene_filter=scene_filter,
        sensor_config=_sensor_config_for_cameras(enabled_cams),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flatten_images = not args.no_flatten_images

    printed_shapes = False
    for token in tqdm(scene_loader.tokens, desc="openpilot preprocess"):
        scene = scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        row = merge_original_and_openpilot(
            scene,
            agent_input,
            road_camera=args.road_camera,
            wide_camera=args.wide_camera,
            steering_delay_s=args.steering_delay_s,
            flatten_images=flatten_images,
            original_mode="compact",
        )
        if not printed_shapes:
            _print_openpilot_shapes(row["openpilot"])
            printed_shapes = True
        out_path = args.output_dir / f"{token}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(row, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "Wrote %d samples under %s (meta=%s sensors=%s)",
        len(scene_loader.tokens),
        args.output_dir,
        meta_path,
        sensor_path,
    )


if __name__ == "__main__":
    main()
