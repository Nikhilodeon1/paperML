"""The pod queue: one protocol everywhere, and the three pods together run every registered sweep exactly once."""
import importlib
import json

from evaluation import profile_lik, stage2_queue
from evaluation.manifest import other_sweeps, phase21_sweeps, profile_sweeps, stage2_sweeps, stage3_sweeps


def test_every_sweep_uses_the_one_converged_protocol():
    for name, (_, config) in stage2_queue.all_sweeps().items():
        assert config["inner_solver"] == "lbfgsb" and config["lbfgsb_restarts"] == 2, name
        assert config["grid_points"] == 29, name
    for name, config in profile_sweeps().items():
        assert config["polish"] is True, name
    for name, config in stage2_sweeps().items():
        assert config["extension_points"] == 0, name


def test_pods_partition_the_sweeps():
    queued = [name for names in stage2_queue.PODS.values() for name in names]
    assert sorted(queued) == sorted(stage2_queue.all_sweeps())
    parts = (phase21_sweeps(), stage2_sweeps(), stage3_sweeps(), other_sweeps())
    assert len(stage2_queue.all_sweeps()) == sum(map(len, parts)), "a name is used twice"


def test_runner_arguments_rebuild_the_exact_configuration():
    for name, (module, config) in stage2_queue.all_sweeps().items():
        rebuilt = importlib.import_module(module).default_config()
        args = stage2_queue.runner_args(module, config, workers=8)
        assert args[0] == module
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


def test_stage3_covers_each_experiment_of_the_decision_rules():
    sweeps = stage3_sweeps()
    assert sum((c["replica"] or {}).get("calibrate") == "regression" for c in sweeps.values()) == 4          # R13
    assert sorted({c["min_gap_h"] for c in sweeps.values() if c["min_gap_h"]}) == [4.0, 6.0]                # R14
    assert sum(c["basal"] == "subject" for c in sweeps.values()) == 4                                       # R15
    assert sum(bool(c["sensor_lag_min"]) for c in sweeps.values()) == 2                                     # R16
    dm = [c["replica"] for c in sweeps.values() if (c["replica"] or {}).get("generator") == "dalla_man"]
    assert sorted(r["cgm"] for r in dm) == [False, False, True, True]                                       # R11


def test_sweeps_are_distinct_to_the_result_matcher():
    """A payload written under one sweep's configuration must match that sweep and no other."""
    sweeps = profile_sweeps()
    for name, config in sweeps.items():
        payload = {**{k: config.get(k) for k in ("objective", "bounds_scale", "cohort", "parameterization", "replica",
                                                 "carb_scale", *profile_lik.VARIANT_KEYS)},
                   "inner_solver": "lbfgsb", "lbfgsb_restarts": 2, "grid_points": 29,
                   "polish": {"start_values": [0] * 6}, "inner_steps": config["steps"]}
        matched = [other for other, c in sweeps.items() if profile_lik._matches(payload, c)]
        assert matched == [name], (name, matched)
