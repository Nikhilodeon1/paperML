"""A box width with only some subjects computed once reached Table 3 under a caption saying all 45."""
from evaluation import phase21_summary


def _summary(n):
    return {"n_subjects": n, "interior_for_parameter": {"gastric_emptying": 3},
            "parameters": {"gastric_emptying": {"fraction": 0.0, "bounded": 0}}}


def test_a_partial_box_sweep_is_left_out(monkeypatch):
    counts = {1.0: 45, 2.0: 45, 4.0: 29}
    monkeypatch.setattr(phase21_summary.profile_lik, "summarize", lambda config: _summary(counts[config["bounds_scale"]]))
    monkeypatch.setattr(phase21_summary, "_expected", lambda cohort, min_meals: 45)
    assert set(phase21_summary.box_four()) == {"One", "Two"}
