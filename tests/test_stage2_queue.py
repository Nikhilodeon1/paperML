"""The Stage 2 queue: one protocol everywhere, and the three pods together run every manifest sweep exactly once."""
import json

from evaluation import profile_lik, stage2_queue
from evaluation.manifest import stage2_sweeps


def test_every_sweep_uses_the_converged_protocol():
    for name, config in stage2_sweeps().items():
        assert config["inner_solver"] == "lbfgsb", name
        assert config["grid_points"] == 29 and config["polish"] is True, name
        assert config["extension_points"] == 0 and config["lbfgsb_restarts"] == 2, name


def test_pods_partition_the_sweeps():
    queued = [name for names in stage2_queue.PODS.values() for name in names]
    assert sorted(queued) == sorted(stage2_sweeps())


def test_runner_arguments_rebuild_the_exact_configuration():
    for name, config in stage2_sweeps().items():
        rebuilt = profile_lik.default_config()
        args = stage2_queue.runner_args(config, workers=8)
        for flag, value in zip(args, args[1:]):
            if flag == "--set":
                key, raw = value.split("=", 1)
                rebuilt[key] = json.loads(raw)
        assert rebuilt == config, name
        assert "--restart" in args and args[args.index("--workers") + 1] == "8"


def test_required_replicas_and_perturbations_are_listed():
    sweeps = stage2_sweeps()
    replicas = [c["replica"] for c in sweeps.values() if c["objective"] == "iauc" and c["replica"]]
    assert {r["seed"] for r in replicas if "truth" not in r and "perturb" not in r} == {0, 1, 2, 3, 4}
    perturbed = {(k, v) for r in replicas for k, v in r.get("perturb", {}).items()}
    names = ("glucose_effectiveness", "x_decay", "insulin_clearance", "insulin_secretion", "x_gain")
    assert perturbed == {(n, f) for n in names for f in (0.7, 1.3)}
    boxes = {c["bounds_scale"] for c in sweeps.values()
             if c["objective"] == "iauc" and c["replica"] is None and c["cohort"] == "cgmacros"
             and c["parameterization"] == "rates"}
    assert boxes == {1.0, 2.0, 4.0}


def test_sweeps_are_distinct_to_the_result_matcher():
    """Two sweeps that the matcher cannot tell apart would overwrite or mix each other's rows."""
    keys = set()
    for config in stage2_sweeps().values():
        key = json.dumps({k: config[k] for k in ("objective", "parameterization", "cohort", "bounds_scale", "replica")},
                         sort_keys=True)
        assert key not in keys, key
        keys.add(key)
