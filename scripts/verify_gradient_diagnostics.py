"""Independent offline arithmetic/partial-vector replay; no models or autograd.

Only the explicitly archived parameter blocks permit independent vector replay.
Hashes and unarchived Gram receipts do not reproduce remaining model gradients.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


SEEDS = (20261009, 20261010, 20261011)
STATES = ("initial", "gold", "full_kd", "gated_kd")
SHA = re.compile(r"[0-9a-f]{64}\Z")
ARITHMETIC_ATOL = 1e-12
ARITHMETIC_RTOL = 1e-10
COMPONENTS = ("ce", "full_kl", "gated_kl")
VECTORS = (*COMPONENTS, "combined_full", "combined_gated")
PARAMETER_PATTERN = re.compile(r"\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.default\.weight$")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairs(items):
    value = {}
    for key, item in items:
        need(key not in value, "Duplicate JSON key")
        value[key] = item
    return value


def _number(value, name="number"):
    need(type(value) in (int, float) and math.isfinite(value), "Invalid finite " + name)
    return value


def _float(text):
    return _number(float(text), "JSON number")


def _invalid(text):
    raise ValueError("Non-finite JSON constant: " + text)


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_float=_float, parse_constant=_invalid)


def rows(path):
    return [json.loads(line, object_pairs_hook=_pairs, parse_float=_float,
                       parse_constant=_invalid)
            for line in Path(path).read_text().splitlines() if line]


def _relative(name):
    need(isinstance(name, str) and name and "\\" not in name, "Invalid archive path")
    path = PurePosixPath(name)
    need(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
         "Noncanonical or escaping archive path")
    return name


def _sha(value):
    need(isinstance(value, str) and SHA.fullmatch(value), "Invalid SHA256")
    return value


def close(stored, computed, label):
    if computed is None:
        need(stored is None, label + " must be null for an undefined quantity")
    else:
        _number(stored, label)
        need(math.isclose(stored, computed, rel_tol=ARITHMETIC_RTOL,
                          abs_tol=ARITHMETIC_ATOL), label + " arithmetic differs")


def check_archive(root):
    need(root.is_dir() and not root.is_symlink(), "Archive must be a real directory")
    manifest = read(root / "archive-manifest.json")
    need(manifest.get("schema") == 1 and isinstance(manifest.get("files"), dict),
         "Invalid archive manifest")
    expected = manifest["files"]
    need(expected and "archive-manifest.json" not in expected, "Invalid archive file map")
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "Archive symlinks are forbidden")
        if path.is_file() and path != root / "archive-manifest.json":
            actual.add(path.relative_to(root).as_posix())
    need(actual == set(expected), "Archive file set differs")
    for name, sha in expected.items():
        _relative(name)
        need(Path(name).suffix not in {".safetensors", ".pt", ".bin", ".pth"},
             "Model weights or full tensor caches are not allowed")
        need(digest(root / name) == _sha(sha), "Archive hash differs: " + name)
    return expected


def checked_gram(value):
    """Validate a 3x3 real Gram matrix, retaining exact signs for conclusions."""
    import numpy as np
    need(isinstance(value, list) and len(value) == 3
         and all(isinstance(row, list) and len(row) == 3 for row in value),
         "Gram must be 3 x 3")
    for row in value:
        for item in row:
            _number(item, "Gram element")
    gram = np.asarray(value, dtype=np.float64)
    need(np.all(np.diag(gram) >= 0), "Negative squared gradient norm")
    need(np.allclose(gram, gram.T, rtol=ARITHMETIC_RTOL, atol=ARITHMETIC_ATOL),
         "Asymmetric Gram matrix")
    scale = max(1.0, float(np.max(np.abs(gram))))
    need(float(np.linalg.eigvalsh((gram + gram.T) / 2).min()) >= -1e-10 * scale,
         "Gram matrix is not positive semidefinite")
    # A true zero norm has an exactly zero vector and therefore zero dot products.
    for index in range(3):
        if gram[index, index] == 0:
            need(bool(np.all(gram[index] == 0) and np.all(gram[:, index] == 0)),
                 "Zero gradient has nonzero dot products")
    return gram


def geometry(gram):
    """Calculate weighted-component geometry; undefined ratios/cosines stay null."""
    norms = [math.sqrt(float(gram[i, i])) for i in range(3)]
    result = {"norms": norms}
    for index, name in ((1, "full"), (2, "gated")):
        left, right = norms[0], norms[index]
        result[name] = {
            "dot": float(gram[0, index]),
            "cosine": float(gram[0, index]) / (left * right) if left and right else None,
            "norm_ratio": right / left if left else None,
            "zero_ce": left == 0,
            "zero_kl": right == 0,
        }
    return result


def compare_tree(stored, expected, label="value"):
    """Tight arithmetic tolerance only; integers, booleans, signs and null exact."""
    if isinstance(expected, dict):
        need(isinstance(stored, dict) and set(stored) == set(expected), label + " keys differ")
        for key in expected:
            compare_tree(stored[key], expected[key], label + "." + str(key))
    elif isinstance(expected, list):
        need(isinstance(stored, list) and len(stored) == len(expected), label + " shape differs")
        for index, (left, right) in enumerate(zip(stored, expected)):
            compare_tree(left, right, label + "." + str(index))
    elif type(expected) is float:
        close(stored, expected, label)
    else:
        need(type(stored) is type(expected) and stored == expected, label + " differs")


def producer_geometry(gram):
    norms = {key: math.sqrt(float(gram[index][index])) for index, key in enumerate(COMPONENTS)}
    pairs = {}
    for index, key in enumerate(COMPONENTS[1:], 1):
        zeros = [name for name in ("ce", key) if norms[name] == 0]
        pairs[key] = {
            "dot": float(gram[0][index]),
            "cosine": float(gram[0][index]) / (norms["ce"] * norms[key]) if not zeros else None,
            "cosine_undefined_reason": "zero_gradient:" + ",".join(zeros) if zeros else None,
            "norm_ratio_to_ce": norms[key] / norms["ce"] if norms["ce"] else None,
            "norm_ratio_undefined_reason": "zero_ce_gradient" if not norms["ce"] else None,
            "norm_larger_than_ce": bool(gram[index][index] > gram[0][0]),
            "negative_dot": bool(gram[0][index] < 0),
        }
    return {"gram": [[float(x) for x in row] for row in gram], "norms": norms, "pairs": pairs}


def grouped_geometry(parameters):
    groups = {"all": parameters}
    for layer in sorted({p["layer"] for p in parameters}):
        groups[f"layer:{layer}"] = [p for p in parameters if p["layer"] == layer]
    for module in ("q_proj", "v_proj"):
        groups["module:" + module] = [p for p in parameters if p["module"] == module]
    for matrix in ("A", "B"):
        groups["matrix:" + matrix] = [p for p in parameters if p["matrix"] == matrix]
    return {key: producer_geometry([[math.fsum(p["gram"][i][j] for p in rows)
                                    for j in range(3)] for i in range(3)])
            for key, rows in groups.items()}


def _integer(value, label, minimum=0):
    need(type(value) is int and value >= minimum, "Invalid " + label)
    return value


def check_comparison(value, numel, tolerance):
    need(type(value.get("allclose")) is bool, "Invalid allclose flag")
    failed = _integer(value.get("failed_elements"), "failed elements")
    need(failed <= numel and value["allclose"] == (failed == 0), "Inconsistent failure count")
    first = value.get("first_failure_flat")
    need(first is None if failed == 0 else type(first) is int and 0 <= first < numel,
         "Invalid first failure coordinate")
    need(value.get("atol") == tolerance["atol"] and value.get("rtol") == tolerance["rtol"],
         "Autograd comparison tolerance changed")
    need(value.get("comparison") == "abs(actual-reference)<=atol+rtol*abs(reference), evaluated in float64",
         "Comparison formula changed")
    for key in ("max_abs_error", "l2_error", "reference_norm", "reconstructed_norm"):
        need(_number(value.get(key), key) >= 0, "Negative comparison magnitude")
    maximum, error, reference = (value[key] for key in ("max_abs_error", "l2_error", "reference_norm"))
    close(value.get("relative_l2_error"), error / reference if reference else None, "Relative L2 error")
    slack = ARITHMETIC_ATOL + ARITHMETIC_RTOL * max(maximum, error)
    need(maximum <= error + slack and error <= math.sqrt(numel) * maximum + slack,
         "Impossible residual L2/max relation")
    need(abs(value["reconstructed_norm"] - reference) <= error + slack,
         "Residual violates triangle inequality")
    if value["allclose"]:
        need(maximum <= tolerance["atol"] + tolerance["rtol"] * reference + slack,
             "Passing max error exceeds any possible element tolerance")
    return value["allclose"]


def check_probe(probe, tolerance, expected_layers=24, vocabulary=151936):
    parameters = probe["parameters"]
    need(isinstance(parameters, list) and len(parameters) == expected_layers * 4,
         "Parameter coverage differs")
    names, coordinates = [], set()
    checks = []
    for parameter in parameters:
        name = parameter["name"]
        need(isinstance(name, str), "Invalid parameter name")
        match = PARAMETER_PATTERN.search(name)
        need(match is not None, "Unexpected parameter name")
        layer, module, matrix = int(match[1]), match[2], match[3]
        need((parameter["layer"], parameter["module"], parameter["matrix"]) == (layer, module, matrix),
             "Parameter metadata differs from name")
        need(0 <= layer < expected_layers, "Unexpected parameter layer")
        names.append(name); coordinates.add((layer, module, matrix))
        shape = parameter["shape"]
        need(isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape),
             "Invalid parameter shape")
        need(type(parameter["numel"]) is int and math.prod(shape) == parameter["numel"],
             "Parameter shape/size mismatch")
        gram = checked_gram(parameter["gram"])
        need(set(parameter["gradient_sha256"]) == set(VECTORS), "Missing gradient digest")
        need(set(parameter["zero_gradient"]) == set(VECTORS), "Missing zero-gradient flag")
        for key in VECTORS:
            _sha(parameter["gradient_sha256"][key])
            need(type(parameter["zero_gradient"][key]) is bool, "Invalid zero-gradient flag")
        for index, key in enumerate(COMPONENTS):
            need(parameter["zero_gradient"][key] == (gram[index, index] == 0),
                 "Zero-gradient flag conflicts with Gram")
        need(set(parameter["reconstruction"]) == {"full", "gated"}, "Missing reconstruction check")
        checks += [check_comparison(item, parameter["numel"], tolerance)
                   for item in parameter["reconstruction"].values()]
    need(names == sorted(names) and len(set(names)) == len(names), "Parameter ordering or uniqueness differs")
    need(len(coordinates) == expected_layers * 4, "Duplicate parameter coordinates")
    compare_tree(probe["groups"], grouped_geometry(parameters), "Grouped geometry")
    supervised = _integer(probe["supervised_tokens"], "supervised tokens", 1)
    need(_integer(probe["input_tokens"], "input tokens", 2) > supervised, "Invalid supervised token count")
    need(_integer(probe["teacher_top1_correct"], "teacher agreement") <= supervised, "Too many retained tokens")
    need(set(probe["losses"]) == set(VECTORS), "Missing objective")
    for value in probe["losses"].values():
        _number(value, "loss")
    need(probe["losses"]["ce"] >= 0, "Negative cross entropy")
    for suffix, key in (("full", "full_kl"), ("gated", "gated_kl")):
        # The producer adds the two scalars in float32, after both reductions.
        import numpy as np
        expected = float(np.float32(np.float32(probe["losses"]["ce"]) + np.float32(probe["losses"][key])))
        close(probe["losses"]["combined_" + suffix], expected, "Combined objective")
    need(set(probe["logit_gradient_checks"]) == set(VECTORS), "Missing analytic logit check")
    need(set(probe["legacy_loss_checks"]) == {"full", "gated"}, "Missing historical loss check")
    checks += [check_comparison(item, supervised * vocabulary, tolerance)
               for item in probe["logit_gradient_checks"].values()]
    checks += [check_comparison(item, 1, tolerance) for item in probe["legacy_loss_checks"].values()]
    need(type(probe["correctness_passed"]) is bool and probe["correctness_passed"] == all(checks),
         "Probe correctness flag differs from component checks")
    return probe["correctness_passed"]


def replay_public_block(path, parameter, tolerance):
    """Recompute all five original float32 digests, Gram and linearity checks."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        need(len(archive.files) == len(VECTORS) and set(archive.files) == set(VECTORS), "Raw gradient keys differ")
        arrays = {key: archive[key] for key in VECTORS}
    for key, value in arrays.items():
        need(value.dtype == np.dtype("float32") and list(value.shape) == parameter["shape"]
             and bool(np.isfinite(value).all()), "Invalid public gradient array")
        need(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == parameter["gradient_sha256"][key],
             "Public gradient digest differs")
        need(parameter["zero_gradient"][key] == (np.count_nonzero(value) == 0), "Public zero-gradient flag differs")
    vectors = [arrays[key].astype(np.float64).reshape(-1) for key in COMPONENTS]
    # np.sum of elementwise products is independent of Torch's dot implementation.
    gram = [[float(np.sum(left * right, dtype=np.float64)) for right in vectors] for left in vectors]
    compare_tree(parameter["gram"], gram, "Public block Gram")
    for suffix, key in (("full", "full_kl"), ("gated", "gated_kl")):
        actual = (arrays["ce"] + arrays[key]).astype(np.float64).reshape(-1)
        reference = arrays["combined_" + suffix].astype(np.float64).reshape(-1)
        delta = actual - reference
        failed = np.flatnonzero(np.abs(delta) > tolerance["atol"] + tolerance["rtol"] * np.abs(reference))
        norm = float(np.sqrt(np.sum(reference * reference, dtype=np.float64)))
        residual = float(np.sqrt(np.sum(delta * delta, dtype=np.float64)))
        expected = {"allclose": len(failed) == 0, "failed_elements": int(len(failed)),
                    "first_failure_flat": int(failed[0]) if len(failed) else None,
                    "max_abs_error": float(np.abs(delta).max()), "l2_error": residual,
                    "relative_l2_error": residual / norm if norm else None,
                    "reference_norm": norm,
                    "reconstructed_norm": float(np.sqrt(np.sum(actual * actual, dtype=np.float64))),
                    "atol": tolerance["atol"], "rtol": tolerance["rtol"],
                    "comparison": "abs(actual-reference)<=atol+rtol*abs(reference), evaluated in float64"}
        compare_tree(parameter["reconstruction"][suffix], expected, "Public reconstruction")
    return parameter["numel"] * len(VECTORS)


def summarize(probes):
    """Independent strict-sign counts, not a statistical or quality acceptance."""
    def describe(items):
        pairs = [p["groups"]["all"]["pairs"] for p in items]
        return {"probes": len(items), "distinct_train_ids": len({p["train_id"] for p in items}),
                "full_norm_larger_count": sum(p["full_kl"]["norm_larger_than_ce"] for p in pairs),
                "full_negative_dot_count": sum(p["full_kl"]["negative_dot"] for p in pairs),
                "gated_negative_dot_count": sum(p["gated_kl"]["negative_dot"] for p in pairs),
                "full_cosine_undefined_count": sum(p["full_kl"]["cosine"] is None for p in pairs),
                "gated_cosine_undefined_count": sum(p["gated_kl"]["cosine"] is None for p in pairs)}
    initial = [p for p in probes if p["state"] == "initial"]
    primary = describe(initial)
    complete = len(initial) == 48 and len(probes) == 192 and all(p["correctness_passed"] for p in probes)
    primary["hypotheses"] = {
        "initial_full_kl_larger_than_ce_majority": primary["full_norm_larger_count"] > 24 if complete else None,
        "initial_gate_eliminates_negative_dot": (primary["gated_negative_dot_count"] == 0
                                                  and primary["gated_cosine_undefined_count"] == 0) if complete else None}
    return {"diagnostic_complete": complete, "primary_initial": primary,
            "by_state": {state: describe([p for p in probes if p["state"] == state]) for state in STATES},
            "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
            "quality_or_performance_conclusion": False}


def _time(value):
    need(isinstance(value, str), "Invalid timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(result.tzinfo is not None, "Timestamp must have timezone")
    return result


def verify(root):
    root = Path(root)
    need(not root.is_symlink(), "Archive root symlinks are forbidden")
    root = root.resolve()
    files = check_archive(root)
    run = read(root / "run.json")
    need(run.get("schema") == 1 and run.get("study") == "ce-kl-gradient-v1", "Unexpected study")
    need(run.get("status") == "complete" and run.get("diagnostic_complete") is True,
         "Failed/partial diagnostic is preserved evidence, not a complete diagnostic")
    need(run.get("phase") == "complete" and run.get("probes_completed") == 192
         and run.get("states_completed") == 12, "Incomplete diagnostic coverage")
    need(run.get("artifacts") == {name: sha for name, sha in files.items() if name != "run.json"},
         "Run artifact manifest differs from archive")
    for key in ("optimizer_steps", "free_generation", "heldout_predictions"):
        need(type(run.get(key)) is int and run[key] == 0, "Diagnostic scope changed")
    start, finish = _time(run["started_utc"]), _time(run["finished_utc"])
    need(start <= finish, "Run timestamps reversed")
    need(isinstance(run.get("protocol_commit"), str)
         and re.fullmatch(r"[0-9a-f]{40}", run["protocol_commit"]), "Invalid frozen commit")
    source_hashes = run.get("source_sha256")
    required = {"qa_lab/gradient_diagnostics.py", "qa_lab/confirmation_training.py", "qa_lab/logits_distillation.py",
                "qa_lab/train_artifact.py", "qa_lab/model_identity.py", "qa_lab/common.py",
                "configs/ce-kl-gradient-v1.json"}
    need(isinstance(source_hashes, dict) and required <= set(source_hashes), "Missing required frozen source")
    source_files = {name.removeprefix("source/") for name in files if name.startswith("source/")}
    need(source_files == set(source_hashes), "Frozen source map coverage differs")
    for name, sha in source_hashes.items():
        need(files["source/" + _relative(name)] == _sha(sha), "Frozen source digest differs")
    protocol_file = root / "source/configs/ce-kl-gradient-v1.json"
    need(digest(protocol_file) == run["protocol_sha256"], "Protocol digest differs")
    protocol = read(protocol_file)
    need(protocol == run["protocol"], "Embedded protocol differs from frozen source")
    need(protocol["study"] == "ce-kl-gradient-v1" and protocol["status"] == "frozen_before_execution",
         "Unexpected protocol identity")
    need(protocol["seeds"] == list(SEEDS) and protocol["states"] == list(STATES), "Study design changed")
    selected = protocol["selected_ids"]
    need(isinstance(selected, list) and len(selected) == 16 and len(set(selected)) == 16, "Selected TRAIN IDs differ")
    need(protocol["runtime"] == {"device": "cpu", "dtype": "float32", "attention_implementation": "eager", "cpu_threads": 8},
         "Runtime contract changed")
    need(protocol["weights"] == {"ce": .5, "kl": 2., "temperature": 2.,
                                   "gated_denominator": "all supervised positions; do not renormalize by retained count"},
         "Objective weights or denominator changed")
    tolerance = {"atol": protocol["correctness"]["gradient_atol"], "rtol": protocol["correctness"]["gradient_rtol"]}
    need(tolerance == {"atol": 1e-5, "rtol": 1e-4}, "Frozen gradient tolerance changed")
    for key in ("optimizer_steps", "free_generation", "heldout_predictions"):
        need(type(protocol["scope"].get(key)) is int and protocol["scope"][key] == 0, "Protocol scope changed")
    need(protocol["scope"]["quality_or_performance_claims"] is False
         and protocol["budget"]["paid_resources"] is False, "Forbidden diagnostic claim or resource")
    need(run["input_sha256"] == protocol["hashes"], "Input binding differs")
    inputs = {"student_config": "student_config.json", "artifact_manifest": "artifact_manifest.json",
              "artifact_train": "artifact_train.jsonl", "source_train": "source_train.jsonl"}
    for key, name in inputs.items():
        need(digest(root / "inputs" / name) == protocol["hashes"][key], "Frozen TRAIN input differs: " + key)
    source = rows(root / "inputs/source_train.jsonl")
    artifact = rows(root / "inputs/artifact_train.jsonl")
    need(len(source) == len(artifact) == 242, "TRAIN cohort size changed")
    source_by_id = {item["id"]: item for item in source}
    need(len(source_by_id) == 242 and len({item["id"] for item in artifact}) == 242
         and {item["id"] for item in artifact} == set(source_by_id), "TRAIN ID coverage differs")
    expected_selected = []
    for flag in (False, True):
        candidates = [item["id"] for item in source if item["is_impossible"] is flag]
        expected_selected += sorted(candidates, key=lambda key: hashlib.sha256(("ce-kl-gradient-v1|" + key).encode()).hexdigest())[:8]
    need(selected == expected_selected, "Predeclared stratified hash selection differs")
    cache_receipt = read(root / "inputs/cache.receipt.json")
    need(cache_receipt["original_manifest_sha256"] == protocol["hashes"]["cache_manifest"]
         and cache_receipt["redactions"] == ["artifact_path"], "Cache receipt binding differs")
    cache = cache_receipt["redacted_manifest"]
    need("artifact_path" not in cache and cache["status"] == "complete" and cache["temperature"] == 2.,
         "Cache identity or temperature differs")
    need(cache["model_identities"] == run["model_identities"], "Model identities differ")
    need(cache["student_model_config"] == read(root / "inputs/student_config.json"), "Student configuration differs")
    records = {item["id"]: item for item in cache["records"]}
    need(len(records) == len(cache["records"]) == 242 and set(records) == set(source_by_id), "Cache coverage differs")
    need(sum(item["supervised_tokens"] for item in records.values()) == 1289,
         "Cached supervised token count differs")
    need(all(item["vocabulary"] == 151936 for item in records.values()), "Cached vocabulary differs")
    state_order = [(seed, state) for seed in SEEDS for state in STATES]
    expected_states = [f"states/{state}-{seed}.json" for seed, state in state_order]
    expected_probes = [f"probes/{index:03d}.json" for index in range(192)]
    need(run["state_files"] == expected_states and run["probe_files"] == expected_probes, "Run order or coverage differs")
    need({name for name in files if name.startswith("states/")} == set(expected_states)
         and {name for name in files if name.startswith("probes/")} == set(expected_probes), "Extra or missing state/probe record")
    need({name for name in files if name.startswith("arrays/")} == {f"arrays/{state}-{seed}.npz" for seed, state in state_order},
         "Public gradient array selection differs")
    expected_inputs = {"inputs/" + name for name in inputs.values()} | {"inputs/cache.receipt.json"}
    expected_inputs |= {f"inputs/endpoint-{state}-{seed}.json" for seed in SEEDS for state in STATES[1:]}
    allowed = {"run.json", "summary.json"} | {"source/" + name for name in source_hashes}
    allowed |= expected_inputs | set(expected_states) | set(expected_probes)
    allowed |= {f"arrays/{state}-{seed}.npz" for seed, state in state_order}
    need(set(files) == allowed, "Unexpected file in diagnostic archive")
    endpoints = {}
    for seed in SEEDS:
        for state in STATES[1:]:
            key = f"{state}:{seed}"; path = root / f"inputs/endpoint-{state}-{seed}.json"
            need(digest(path) == protocol["hashes"]["endpoint_training"][key], "Endpoint receipt digest differs")
            receipt = read(path); endpoints[key] = receipt
            need(receipt["status"] == "complete" and receipt["arm"] == state and receipt["seed"] == seed
                 and receipt["steps_completed"] == 242, "Endpoint state differs")
            need(receipt["cache_manifest_sha256"] == protocol["hashes"]["cache_manifest"]
                 and receipt["model_identities"] == run["model_identities"], "Endpoint model/cache differs")
        need(len({endpoints[f"{state}:{seed}"]["initial_trainable_state_sha256"] for state in STATES[1:]}) == 1,
             "Same-seed initial model identities differ")
    probes, raw_values, base_hashes = [], 0, set()
    for state_index, (seed, state) in enumerate(state_order):
        receipt = read(root / expected_states[state_index])
        need(receipt["status"] == "complete" and receipt["state"] == state and receipt["seed"] == seed
             and receipt["probes_completed"] == 16 and receipt["parameter_blocks"] == 96
             and receipt["trainable_parameters"] == 540672, "State receipt coverage differs")
        need(start <= _time(receipt["started_utc"]) <= _time(receipt["finished_utc"]) <= finish, "State timestamps outside run")
        need(receipt["parameters_unchanged"] is True and receipt["parameters_before"] == receipt["parameters_after"],
             "Parameter mutation or incomplete parameter audit")
        before = receipt["parameters_before"]
        need(set(before) == {"adapter", "base"} and before["adapter"]["numel"] == 540672, "Parameter state coverage differs")
        for part in before.values():
            _sha(part["sha256"]); _integer(part["numel"], "Parameter count", 1)
        base_hashes.add(before["base"]["sha256"])
        if state == "initial":
            need(receipt["initial_trainable_state_sha256"] == before["adapter"]["sha256"]
                 == endpoints[f"gold:{seed}"]["initial_trainable_state_sha256"], "Initial parameter identity differs")
        else:
            need(receipt["adapter_files"] == endpoints[f"{state}:{seed}"]["adapter_files"], "Endpoint adapter identity differs")
        for selected_index, key in enumerate(selected):
            index = state_index * 16 + selected_index
            probe = read(root / expected_probes[index]); probes.append(probe)
            need(probe["schema"] == 1 and probe["status"] == "complete" and probe["index"] == index
                 and probe["seed"] == seed and probe["state"] == state and probe["train_id"] == key,
                 "Probe identity/order differs")
            need(probe["is_impossible"] is source_by_id[key]["is_impossible"]
                 and probe["role"] == ("primary_initial" if state == "initial" else "secondary_endpoint"),
                 "Probe strata or primary/secondary scope differs")
            need(probe["cache_record_sha256"] == records[key]["sha256"]
                 and probe["input_tokens"] == records[key]["sequence_tokens"]
                 and probe["supervised_tokens"] == records[key]["supervised_tokens"], "Probe cache identity differs")
            need(check_probe(probe, tolerance), "Probe correctness failure")
            need(sum(p["numel"] for p in probe["parameters"]) == 540672, "Trainable parameter size differs")
            for parameter in probe["parameters"]:
                expected_shape = [8, 896] if parameter["matrix"] == "A" else ([896, 8] if parameter["module"] == "q_proj" else [128, 8])
                need(parameter["shape"] == expected_shape, "Qwen LoRA parameter shape differs")
            classes = probe["token_classes"]
            for field in ("eos_count", "other_special_count", "other_count"):
                _integer(classes[field], field)
            need(classes["supervised_tokens"] == probe["supervised_tokens"]
                 == sum(classes[field] for field in ("eos_count", "other_special_count", "other_count")), "Token class totals differ")
            if selected_index == 0:
                raw = probe["raw_artifact"]
                name = f"arrays/{state}-{seed}.npz"
                need(raw["path"] == name and raw["sha256"] == files[name] and raw["keys"] == list(VECTORS), "Raw gradient identity differs")
                parameter = next(p for p in probe["parameters"] if (p["layer"], p["module"], p["matrix"]) == (0, "q_proj", "B"))
                need(raw["parameter"] == probe["public_parameter"] == parameter["name"], "Public parameter selection differs")
                raw_values += replay_public_block(root / name, parameter, tolerance)
            else:
                need(probe["raw_artifact"] is None and probe["public_parameter"] is None, "Unexpected public vector selection")
    need(len(base_hashes) == 1, "Base model parameter identity changed across states")
    summary = summarize(probes)
    compare_tree(read(root / "summary.json"), summary, "Summary")
    need(summary["diagnostic_complete"], "Incomplete summary")
    for field, limit in (("wall_seconds", "wall_seconds"), ("peak_process_rss_bytes", "max_rss_bytes"),
                         ("output_bytes_last_check", "max_output_bytes")):
        need(0 <= _number(run[field], field) <= protocol["budget"][limit], "Resource budget exceeded")
    need(sum((root / name).stat().st_size for name in files) <= protocol["budget"]["max_output_bytes"],
         "Actual archived bytes exceed disk budget")
    return {"status": "verified", "study": "ce-kl-gradient-v1", "protocol_commit": run["protocol_commit"],
            "protocol_sha256": run["protocol_sha256"], "probes": 192, "parameter_blocks_per_probe": 96,
            "distinct_train_questions": 16, "distinct_train_contexts": len({source_by_id[key]["context_id"] for key in selected}),
            "public_blocks_replayed": 12, "public_gradient_values_replayed": raw_values,
            "summary": summary, "limits": [
                "Independent vector replay covers only first selected TRAIN ID, layer 0 q_proj LoRA B, in each state.",
                "Other parameter/model autograd, teacher cache contents, model/adapter weight bytes and parameter immutability are recorded identities, not independently reproduced.",
                "Hash consistency is not an authenticity signature or independent proof of execution/label-access order.",
                "Gradient geometry is before clipping/AdamW; 192 repeated probes contain only 16 distinct TRAIN questions. No quality or performance conclusion."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/ce-kl-gradient-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        need(not args.output.exists(), "Output must be new")
        need(not args.output.resolve().is_relative_to(args.root.resolve()), "Do not write into frozen archive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
