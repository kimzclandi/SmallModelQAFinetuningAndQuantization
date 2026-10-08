"""One prespecified same-input logit replay; never loads a model or updates it.

Private logits/targets/teacher cache remain under work/. Output contains numeric
summaries only. This separates local arithmetic stages, not historical quality
causality, upstream parameter-VJP kernels, or a replacement acceptance gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform


VECTORS = ("ce", "full_kl", "gated_kl", "combined_full", "combined_gated")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_executing_sources(run, root):
    for name in ("scripts/replay_private_gradient_capture.py", "scripts/verify_gradient_capture.py",
                 "scripts/verify_gradient_diagnostics.py"):
        if digest(root / name) != run["source_sha256"].get(name):
            raise ValueError("Executing replay source differs from precommitted capture source")


def objectives(z, targets, teacher):
    import torch.nn.functional as functional
    logp = functional.log_softmax(z / 2.0, dim=-1)
    per_token = functional.kl_div(logp, teacher, reduction="none", log_target=True).sum(-1)
    gate = teacher.argmax(-1).eq(targets)
    ce = .5 * functional.cross_entropy(z, targets)
    full = 2.0 * per_token.sum() / len(targets)
    gated = 2.0 * (per_token * gate).sum() / len(targets)
    return {"ce": ce, "full_kl": full, "gated_kl": gated,
            "combined_full": ce + full, "combined_gated": ce + gated}, logp


def formula(z, targets, teacher):
    import torch
    z, q = z.detach().double(), teacher.double().exp()
    gold = torch.zeros_like(z).scatter_(1, targets[:, None], 1.)
    ce = .5 / len(targets) * (z.softmax(-1) - gold)
    full = (q.sum(-1, keepdim=True) * (z / 2.).softmax(-1) - q) / len(targets)
    gated = full * teacher.argmax(-1).eq(targets)[:, None]
    return {"ce": ce, "full_kl": full, "gated_kl": gated,
            "combined_full": ce + full, "combined_gated": ce + gated}


def comparison(actual, reference):
    import torch
    a, b = actual.detach().double(), reference.detach().double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Invalid replay comparison")
    delta = (a - b).abs()
    failed = delta > 1e-5 + 1e-4 * b.abs()
    indices = torch.nonzero(failed.reshape(-1)).flatten()
    return {"allclose": not bool(failed.any()), "failed_elements": int(failed.sum()),
            "first_failure_flat": int(indices[0]) if len(indices) else None,
            "max_abs_error": float(delta.max()), "row_max_abs_error": delta.max(-1).values.tolist()}


def analyze(selected_logits, targets, teacher, public_arrays):
    """Inputs [N,V] f32, [N] i64, [N,V] f16; output scalar summaries."""
    import torch
    import torch.nn.functional as functional
    if (selected_logits.dtype != torch.float32 or targets.dtype != torch.int64
            or teacher.dtype != torch.float16 or selected_logits.ndim != 2
            or selected_logits.shape != teacher.shape or targets.shape != (len(selected_logits),)
            or len(targets) == 0 or not torch.isfinite(selected_logits).all()
            or not torch.isfinite(teacher).all()):
        raise ValueError("Private array contract differs")
    z = selected_logits.detach().clone().requires_grad_()
    reference = formula(z, targets, teacher)
    values32, logp = objectives(z, targets, teacher.float())
    grad32 = {key: torch.autograd.grad(value, z, retain_graph=True)[0]
              for key, value in values32.items()}
    # Capture the actual upstream derivative to avoid assuming an FP32
    # reduction/scalar multiplication order in this diagnostic decomposition.
    upstream = torch.autograd.grad(values32["full_kl"], logp, retain_graph=True)[0]
    z64 = selected_logits.double().requires_grad_()
    values64, _ = objectives(z64, targets, teacher.double())
    grad64 = {key: torch.autograd.grad(value, z64, retain_graph=True)[0]
              for key, value in values64.items()}
    matches, reference_matches = {}, {}
    for key in VECTORS:
        actual = torch.from_numpy(public_arrays[key]["logit_actual"])
        original_reference = torch.from_numpy(public_arrays[key]["logit_reference"])
        matches[key] = torch.equal(grad32[key].detach(), actual)
        reference_matches[key] = torch.equal(reference[key], original_reference)
    if not all(matches.values()) or not all(reference_matches.values()):
        raise ValueError("Same-input replay differs from captured logit gradients/reference")
    p32 = teacher.float().exp().double()
    q64 = (selected_logits.double() / 2.).softmax(-1)
    qforward = functional.log_softmax(selected_logits / 2., dim=-1).double().exp()
    teacher_only = (p32.sum(-1, keepdim=True) * q64 - p32) / len(targets)
    teacher_forward = (p32.sum(-1, keepdim=True) * qforward - p32) / len(targets)
    # Hybrid 64-bit reconstruction of the observed upstream and saved f32
    # forward values; not an alternative implementation or causal intervention.
    backward64 = (upstream.double() - qforward * upstream.double().sum(-1, keepdim=True)) / 2.
    return {"schema": 1, "scope": "One fixed captured TRAIN logit input; no model execution, optimizer, generation or heldout",
            "tolerance": {"atol": 1e-5, "rtol": 1e-4},
            "shape": list(selected_logits.shape), "threads": torch.get_num_threads(),
            "fp32_replay_bitwise_matches_capture": matches,
            "fp64_formula_bitwise_matches_capture": reference_matches,
            "fp32_autograd_vs_formula": {key: comparison(grad32[key], reference[key]) for key in VECTORS},
            "fp64_autograd_vs_formula": {key: comparison(grad64[key], reference[key]) for key in VECTORS},
            "teacher_exp32_only_vs_formula": comparison(teacher_only, reference["full_kl"]),
            "teacher_and_forward32_vs_formula": comparison(teacher_forward, reference["full_kl"]),
            "actual_fp32_vs_teacher_and_forward32": comparison(grad32["full_kl"], teacher_forward),
            "actual_fp32_vs_observed_upstream_hybrid64": comparison(grad32["full_kl"], backward64),
            "log_softmax32_exp_mass": qforward.sum(-1).tolist(),
            "teacher_half_mass": teacher.double().exp().sum(-1).tolist(),
            "teacher_exp32_mass": p32.sum(-1).tolist(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
            "limits": ["FP64 is a local diagnostic on identical captured logits, not a new model/gradient study or changed v1 tolerance.",
                       "Hybrid formulas describe arithmetic stages, not a proof of a specific upstream kernel bug.",
                       "No parameter VJP is recomputed; no claim about training quality or acceleration."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--private-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    root = Path(__file__).resolve().parents[1]
    if not args.private_input.resolve().is_relative_to(root / "work"):
        parser.error("Private tensors must stay below repository work/.")
    if not args.output.resolve().is_relative_to(root / "work"):
        parser.error("New replay output must be below repository work/.")
    if args.output.exists():
        parser.error("Refusing to overwrite a replay result.")
    run = json.loads((args.capture / "run.json").read_text())
    verify_executing_sources(run, root)
    from verify_gradient_capture import verify
    verify(args.capture, repository=root)
    manifest = json.loads((args.capture / "capture.json").read_text())
    private_receipt = manifest["private_inputs"]
    if digest(args.private_input) != private_receipt["sha256"]:
        raise ValueError("Private input hash differs")
    with np.load(args.private_input, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if set(arrays) != set(private_receipt["arrays"]):
        raise ValueError("Private keys differ")
    for key, value in arrays.items():
        meta = private_receipt["arrays"][key]
        if (str(value.dtype) != meta["dtype"] or list(value.shape) != meta["shape"]
                or hashlib.sha256(value.tobytes()).hexdigest() != meta["sha256"]):
            raise ValueError("Private array metadata differs")
    public = {}
    for key in VECTORS:
        with np.load(args.capture / manifest["objectives"][key]["path"], allow_pickle=False) as archive:
            public[key] = {name: archive[name] for name in ("logit_actual", "logit_reference")}
    torch.set_num_threads(8)
    result = analyze(torch.from_numpy(arrays["selected_logits"]), torch.from_numpy(arrays["targets"]),
                     torch.from_numpy(arrays["teacher_log_probs"]), public)
    result.update(capture_run_sha256=digest(args.capture / "run.json"),
                  private_input_sha256=digest(args.private_input), script_sha256=digest(__file__),
                  protocol_commit=run["protocol_commit"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
