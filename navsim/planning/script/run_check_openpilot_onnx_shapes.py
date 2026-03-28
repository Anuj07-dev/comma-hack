"""
Load vision + policy ONNX models and verify that preprocessing tensor shapes match
ONNX inputs the same way ``OpenpilotAgent`` binds them.

ONNX path resolution (first wins): ``--vision-model-path`` / ``--policy-model-path``,
then ``OPENPILOT_VISION_ONNX`` / ``OPENPILOT_POLICY_ONNX``, then keys in ``--agent-yaml``.

Example:
  python -m navsim.planning.script.run_check_openpilot_onnx_shapes \\
    --vision-model-path /path/to/driving_vision.onnx \\
    --policy-model-path /path/to/driving_policy.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import yaml

from navsim.agents.open_pilot_agent import (
    _guess_logical_key,
    _infer_policy_inputs_from_vision,
)
from navsim.preprocessing.openpilot_model_inputs import preprocessed_openpilot_tensor_shapes

DEFAULT_AGENT_YAML = (
    Path(__file__).resolve().parent / "config" / "common" / "agent" / "openpilot_agent.yaml"
)


def _static_onnx_shape(shape: Sequence[Any]) -> List[int]:
    out: List[int] = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            out.append(d)
        else:
            out.append(1)
    return out


def _elem_count(shape: Sequence[int]) -> int:
    return int(np.prod(shape))


def _compatible(
    prep_shape: Tuple[int, ...],
    onnx_static: List[int],
) -> Tuple[bool, str]:
    pe = _elem_count(prep_shape)
    oe = _elem_count(onnx_static)
    if pe == oe:
        return True, "element count matches"
    if len(onnx_static) >= 1 and onnx_static[0] == 1 and _elem_count(onnx_static[1:]) == pe:
        return True, "preprocessing matches ONNX after leading batch dim 1"
    return False, f"preprocessing {prep_shape} ({pe} elems) vs ONNX {tuple(onnx_static)} ({oe} elems)"


def _static_elem_match(a: List[int], b: List[int]) -> Tuple[bool, str]:
    """Compare two ONNX-style static shapes (e.g. vision output vs policy input)."""
    ae = _elem_count(a)
    be = _elem_count(b)
    if ae == be:
        return True, f"{tuple(a)} vs {tuple(b)}"
    if len(b) >= 1 and b[0] == 1 and _elem_count(b[1:]) == ae:
        return True, f"{tuple(a)} vs {tuple(b)} (leading batch 1 on policy input)"
    return False, f"{tuple(a)} ({ae} elems) vs {tuple(b)} ({be} elems)"


def _optional_str_dict(cfg: Dict[str, Any], key: str) -> Dict[str, str]:
    raw = cfg.get(key)
    if not raw or not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _resolve_onnx_path(cli: Optional[Path], env_key: str, yaml_value: Any) -> Optional[Path]:
    if cli is not None:
        return cli.resolve()
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val).resolve()
    if yaml_value:
        return Path(str(yaml_value)).resolve()
    return None


def _vision_logical_key(
    onnx_name: str,
    vision_input_name_map: Dict[str, str],
    image_concat_input_name: Optional[str],
    concatenate: bool,
) -> Optional[str]:
    if onnx_name in vision_input_name_map:
        return vision_input_name_map[onnx_name]
    if concatenate and image_concat_input_name and onnx_name == image_concat_input_name:
        return "image_concat"
    return _guess_logical_key(onnx_name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--agent-yaml",
        type=Path,
        default=DEFAULT_AGENT_YAML,
        help="Agent YAML (optional name maps; ONNX paths if not set on CLI / env)",
    )
    p.add_argument(
        "--vision-model-path",
        type=Path,
        default=None,
        help="Vision ONNX file (highest priority for vision model path)",
    )
    p.add_argument(
        "--policy-model-path",
        type=Path,
        default=None,
        help="Policy ONNX file (highest priority for policy model path)",
    )
    p.add_argument("--no-flatten-images", action="store_true", help="Match unflattened (2,6,128,256) streams")
    p.add_argument(
        "--concatenate-image-streams",
        action="store_true",
        help="Assume a single image_concat buffer (road+wide)",
    )
    args = p.parse_args()

    if not args.agent_yaml.is_file():
        print(f"ERROR: agent YAML not found: {args.agent_yaml}", file=sys.stderr)
        return 1
    with open(args.agent_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        print("ERROR: agent YAML must parse to a mapping", file=sys.stderr)
        return 1
    cfg = raw

    vision_path = _resolve_onnx_path(
        args.vision_model_path, "OPENPILOT_VISION_ONNX", cfg.get("vision_model_path")
    )
    policy_path = _resolve_onnx_path(
        args.policy_model_path, "OPENPILOT_POLICY_ONNX", cfg.get("policy_model_path")
    )
    if vision_path is None:
        print(
            "ERROR: vision ONNX path not set. Use --vision-model-path, OPENPILOT_VISION_ONNX, "
            "or vision_model_path in the YAML.",
            file=sys.stderr,
        )
        return 1
    if policy_path is None:
        print(
            "ERROR: policy ONNX path not set. Use --policy-model-path, OPENPILOT_POLICY_ONNX, "
            "or policy_model_path in the YAML.",
            file=sys.stderr,
        )
        return 1

    if not vision_path.is_file():
        print(f"ERROR: vision ONNX not found: {vision_path}", file=sys.stderr)
        return 1
    if not policy_path.is_file():
        print(f"ERROR: policy ONNX not found: {policy_path}", file=sys.stderr)
        return 1

    flatten = not args.no_flatten_images

    vision_map = _optional_str_dict(cfg, "vision_input_name_map")
    policy_map = _optional_str_dict(cfg, "policy_input_name_map")
    policy_from_vis_raw = cfg.get("policy_inputs_from_vision")
    policy_from_vis: Optional[Dict[str, str]] = None
    if isinstance(policy_from_vis_raw, dict) and policy_from_vis_raw:
        policy_from_vis = {str(k): str(v) for k, v in policy_from_vis_raw.items()}
    image_concat_name = cfg.get("image_concat_input_name")
    image_concat_name = str(image_concat_name) if image_concat_name else None
    concatenate_cfg = bool(cfg.get("concatenate_image_streams", False))
    concatenate = bool(args.concatenate_image_streams or concatenate_cfg or bool(image_concat_name))
    rec = cfg.get("recurrent_input_names")
    recurrent_in: List[str] = list(rec) if isinstance(rec, list) else []

    prep_shapes = preprocessed_openpilot_tensor_shapes(
        flatten_images=flatten,
        concatenate_image_streams=concatenate,
    )

    so = ort.SessionOptions()
    so.log_severity_level = 3
    vision_sess = ort.InferenceSession(str(vision_path), sess_options=so, providers=["CPUExecutionProvider"])
    policy_sess = ort.InferenceSession(str(policy_path), sess_options=so, providers=["CPUExecutionProvider"])

    vision_out_names = [o.name for o in vision_sess.get_outputs()]
    vision_out_by_name = {o.name: _static_onnx_shape(o.shape) for o in vision_sess.get_outputs()}

    print(f"Vision model:  {vision_path}")
    print(f"Policy model: {policy_path}")
    print(f"Preprocess: flatten_images={flatten} concatenate_image_streams={concatenate}")
    print("Preprocessed tensor shapes (reference):")
    for k in sorted(prep_shapes.keys()):
        print(f"  {k:28}  {prep_shapes[k]}")
    print()

    ok_all = True
    print("Vision ONNX inputs vs preprocessing:")
    for inp in vision_sess.get_inputs():
        static = _static_onnx_shape(inp.shape)
        lkey = _vision_logical_key(
            inp.name,
            vision_map,
            str(image_concat_name) if image_concat_name else None,
            concatenate,
        )
        if lkey is None or lkey not in prep_shapes:
            print(f"  [SKIP] {inp.name}: no logical key / not in preprocess dict (heuristic={lkey})")
            ok_all = False
            continue
        good, msg = _compatible(prep_shapes[lkey], static)
        status = "OK  " if good else "FAIL"
        print(f"  [{status}] {inp.name} ONNX{tuple(static)} {inp.type} <- logical[{lkey}] {prep_shapes[lkey]} :: {msg}")
        ok_all = ok_all and good

    resolved_v2p = dict(policy_from_vis or {})
    inferred = _infer_policy_inputs_from_vision(policy_sess, vision_out_names)
    for k, v in inferred.items():
        resolved_v2p.setdefault(k, v)

    print()
    print("Policy ONNX inputs vs preprocessing (or vision output):")
    for inp in policy_sess.get_inputs():
        static = _static_onnx_shape(inp.shape)
        if inp.name in recurrent_in:
            print(
                f"  [SKIP] {inp.name} ONNX{tuple(static)} recurrent state "
                f"(runtime zeros; not from preprocess)"
            )
            continue
        if inp.name in resolved_v2p:
            vout = resolved_v2p[inp.name]
            if vout not in vision_out_by_name:
                print(f"  [FAIL] {inp.name}: vision output '{vout}' not found; have {vision_out_names}")
                ok_all = False
                continue
            vshape = vision_out_by_name[vout]
            good, msg = _static_elem_match(vshape, static)
            status = "OK  " if good else "FAIL"
            print(f"  [{status}] {inp.name} <- {vout} :: {msg}")
            ok_all = ok_all and good
            continue

        lkey = policy_map.get(inp.name) or _guess_logical_key(inp.name)
        if lkey is None or lkey not in prep_shapes:
            print(f"  [SKIP] {inp.name}: no policy map / heuristic for preprocess (key={lkey})")
            ok_all = False
            continue
        good, msg = _compatible(prep_shapes[lkey], static)
        status = "OK  " if good else "FAIL"
        print(f"  [{status}] {inp.name} ONNX{tuple(static)} {inp.type} <- logical[{lkey}] {prep_shapes[lkey]} :: {msg}")
        ok_all = ok_all and good

    print()
    if ok_all:
        print("All checked bindings match (element counts).")
        return 0
    print("Some checks failed or were skipped; fix maps in YAML or preprocessing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
