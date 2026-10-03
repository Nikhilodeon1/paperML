"""Dalla Man 2007 meal glucose-insulin model — differentiable JAX implementation.

Reference: Dalla Man C, Rizza RA, Cobelli C. "Meal simulation model of the glucose-insulin
system." IEEE Trans Biomed Eng. 2007;54(10):1740-9. Population constants from Table I.

Why this model (vs the Bergman engine in `jax_engine.py`): the error decomposition attributes ~79%
of held-out iAUC error to model expressiveness. Dalla Man is more expressive exactly where Bergman
is thin — a real 3-compartment gut (stomach solid -> stomach liquid -> intestine) with a
glucose-dependent gastric-emptying rate instead of a lumped blunting factor, 2-compartment glucose
and insulin kinetics, EGP regulated by delayed insulin, and a SUBCUTANEOUS glucose compartment,
which matters because CGM measures interstitial (not plasma) glucose with a ~10-15 min lag.

Differentiability: Dalla Man's gastric-emptying law is already a smooth tanh blend, so it needs no
smoothing. The only nonsmooth pieces (EGP floor, secretion rectifier) use softplus.

SCOPE (documented reductions, not omissions by accident):
  * Portal-insulin (Ipo) and the dynamic beta-cell secretion model (Y, K*dG/dt) are reduced to a
    static secretion S = Sb + beta*softplus(G - Gb); the kp4*Ipo term in EGP is folded into kp1.
  * Renal excretion E omitted (zero below the ~180 mg/dl threshold; irrelevant for these meals).
  * Glucagon omitted: it is a counter-regulatory (hypoglycaemia) signal and does not shape the
    postprandial iAUC these experiments score.

STEADY STATE: rather than fitting an absolute basal-EGP parameter, we take the subject's basal
glucose Gb (observable — it is the CGM pre-meal baseline) and DERIVE Gtb and Vm0 from the basal
balance so every parameter draw starts exactly at steady state. Without this the model drifts at
t=0 and corrupts iAUC.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

# --- population constants (Dalla Man 2007, Table I) --------------------------------------------
C = {
    "Vg": 1.88,        # dl/kg   glucose distribution volume
    "k1": 0.065,       # 1/min   plasma -> tissue
    "k2": 0.079,       # 1/min   tissue -> plasma
    "Vi": 0.05,        # l/kg    insulin distribution volume
    "m1": 0.190,       # 1/min
    "m2": 0.484,       # 1/min
    "m4": 0.194,       # 1/min
    "HEb": 0.6,        # basal hepatic extraction
    "Fcns": 1.0,       # mg/kg/min  insulin-independent (CNS) uptake
    "Km0": 225.59,     # mg/kg
    "p2u": 0.0331,     # 1/min   remote insulin action
    "ki": 0.0079,      # 1/min   delayed insulin signal for EGP
    "kp2": 0.0021,     # mg/kg/min per mg/kg  (glucose effect on EGP)
    "kp3": 0.009,      # mg/kg/min per pmol/l (insulin effect on EGP)
    "kgri": 0.0558,    # 1/min   grinding (solid -> liquid) rate
    "b": 0.82,         # kempt shape
    "c": 0.010,        # kempt shape
    "EGPb": 2.4,       # mg/kg/min  basal endogenous glucose production
    "Ib": 25.0,        # pmol/l     basal plasma insulin
    "beta_sec": 0.30,  # pmol/kg/min per mg/dl  static secretion slope (default; per-user param)
    "kcr": 0.20,       # mg/kg/min per mg/dl  counter-regulatory (glucagon-lite) EGP defence
}
C["m3"] = C["HEb"] * C["m1"] / (1.0 - C["HEb"])     # derived (Dalla Man): 0.285

STATE_VARS = ["Qsto1", "Qsto2", "Qgut", "Gp", "Gt", "Ip", "Il", "X", "I1", "Id", "Gsc"]
IDX = {n: i for i, n in enumerate(STATE_VARS)}
N_STATE = len(STATE_VARS)
GSC = IDX["Gsc"]


def _softplus(x, beta=10.0):
    return jax.nn.softplus(beta * x) / beta


class DallaManParams(eqx.Module):
    """Per-user differentiable subset; everything else stays at population values in `C`."""
    Vmx: jax.Array        # mg/kg/min per pmol/l — insulin sensitivity (analogue of Bergman Si)
    kabs: jax.Array       # 1/min  intestinal absorption
    kmax: jax.Array       # 1/min  max gastric emptying
    kmin: jax.Array       # 1/min  min gastric emptying
    f: jax.Array          # fraction of ingested glucose appearing in plasma
    Td: jax.Array         # min    subcutaneous (CGM) lag
    Gb: jax.Array         # mg/dl  basal glucose (observable: pre-meal CGM baseline)
    beta_sec: jax.Array   # pmol/kg/min per mg/dl — static insulin-secretion slope
    weight_kg: jax.Array

    @staticmethod
    def defaults(weight_kg: float = 78.0, Gb: float = 100.0) -> "DallaManParams":
        f32 = lambda v: jnp.asarray(v, jnp.float32)
        # kmin and beta_sec (and constant kcr) are CALIBRATED so the 75 g OGTT reproduces the
        # published response (peak 160-180 mg/dl at 40-50 min, back to ~basal by 180 min); with the
        # nominal kmin=0.008 / beta_sec=0.30 this reconstruction peaked at only ~123 mg/dl and
        # undershot basal. See evaluation/dalla_man_validation.py. Everything else is Table I.
        return DallaManParams(Vmx=f32(0.047), kabs=f32(0.057), kmax=f32(0.0558),
                              kmin=f32(0.038), f=f32(0.90), Td=f32(10.0),
                              Gb=f32(Gb), beta_sec=f32(0.14), weight_kg=f32(weight_kg))

    @staticmethod
    def from_demographics(weight_kg, age, sex, Gb=100.0) -> "DallaManParams":
        """Population init; insulin sensitivity declines ~1%/yr after 30 (Dalla Man cohort trend)."""
        age_factor = jnp.where(age > 30, 1.0 - 0.01 * (age - 30.0), 1.0)
        p = DallaManParams.defaults(weight_kg, Gb)
        return eqx.tree_at(lambda m: m.Vmx, p, jnp.asarray(0.047, jnp.float32) * age_factor)


def basal(params: DallaManParams):
    """Derive the steady-state basal quantities so dX/dt == 0 at t=0 for ANY parameter draw."""
    Gpb = params.Gb * C["Vg"]                                  # mg/kg
    # Gp balance at basal (Ra=0): EGPb - Fcns - k1*Gpb + k2*Gtb = 0
    Gtb = (C["Fcns"] + C["k1"] * Gpb - C["EGPb"]) / C["k2"]    # mg/kg
    Uidb = C["EGPb"] - C["Fcns"]                               # = k1*Gpb - k2*Gtb
    Vm0 = Uidb * (C["Km0"] + Gtb) / Gtb                        # so Uid(basal) == Uidb
    Ipb = C["Ib"] * C["Vi"]                                    # pmol/kg
    Ilb = C["m2"] * Ipb / (C["m1"] + C["m3"])
    Sb = (C["m2"] + C["m4"]) * Ipb - C["m1"] * Ilb             # basal secretion
    kp1 = C["EGPb"] + C["kp2"] * Gpb + C["kp3"] * C["Ib"]      # so EGP(basal) == EGPb
    return Gpb, Gtb, Vm0, Ipb, Ilb, Sb, kp1


def initial_state(params: DallaManParams) -> jnp.ndarray:
    Gpb, Gtb, _Vm0, Ipb, Ilb, _Sb, _kp1 = basal(params)
    y = jnp.zeros(N_STATE, dtype=jnp.float32)
    y = y.at[IDX["Gp"]].set(Gpb)
    y = y.at[IDX["Gt"]].set(Gtb)
    y = y.at[IDX["Ip"]].set(Ipb)
    y = y.at[IDX["Il"]].set(Ilb)
    y = y.at[IDX["X"]].set(0.0)
    y = y.at[IDX["I1"]].set(C["Ib"])
    y = y.at[IDX["Id"]].set(C["Ib"])
    y = y.at[IDX["Gsc"]].set(params.Gb)
    return y


def k_empt(Qsto, D, params):
    """Dalla Man gastric emptying (eq. 8): smooth tanh blend, kmax -> kmin -> kmax. Already
    differentiable in the original formulation — no smoothing needed."""
    D_ = jnp.maximum(D, 1.0)
    alpha = 5.0 / (2.0 * D_ * (1.0 - C["b"]))
    beta = 5.0 / (2.0 * D_ * C["c"])
    return params.kmin + 0.5 * (params.kmax - params.kmin) * (
        jnp.tanh(alpha * (Qsto - C["b"] * D_)) - jnp.tanh(beta * (Qsto - C["c"] * D_)) + 2.0)


def rhs(t, y, args):
    params, meal_time, D_mg, sigma, solid_frac = args
    Gpb, Gtb, Vm0, Ipb, Ilb, Sb, kp1 = basal(params)

    Qsto1, Qsto2, Qgut = y[0], y[1], y[2]
    Gp, Gt, Ip, Il, X, I1, Id, Gsc = y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]

    G = Gp / C["Vg"]                       # mg/dl
    I = Ip / C["Vi"]                       # pmol/l

    # meal input: Gaussian pulse of total mass D_mg into the solid stomach compartment
    amp = D_mg / (sigma * jnp.sqrt(2.0 * jnp.pi))
    meal = amp * jnp.exp(-0.5 * ((t - meal_time) / sigma) ** 2)

    # Solid food enters the grinding compartment; a LIQUID load (e.g. a 75 g OGTT drink) bypasses
    # it and lands directly in the liquid stomach — routing a drink through grinding would add a
    # spurious ~18 min serial delay and flatten the peak.
    Qsto = Qsto1 + Qsto2
    kempt = k_empt(Qsto, D_mg, params)
    dQsto1 = solid_frac * meal - C["kgri"] * Qsto1
    dQsto2 = (1.0 - solid_frac) * meal + C["kgri"] * Qsto1 - kempt * Qsto2
    dQgut = kempt * Qsto2 - params.kabs * Qgut
    Ra = params.f * params.kabs * Qgut / params.weight_kg       # mg/kg/min

    # glucose subsystem
    Uid = (Vm0 + params.Vmx * X) * Gt / (C["Km0"] + Gt)
    # EGP: suppressed by glucose and delayed insulin, plus a counter-regulatory (glucagon-lite)
    # term that defends against falling below basal. Omitting counter-regulation was a mistake:
    # with ki=0.0079 the delayed insulin signal suppresses EGP for ~2h after the insulin peak, so
    # without it every parameterisation undershoots basal by 15-25 mg/dl (non-physiological).
    EGP = _softplus(kp1 - C["kp2"] * Gp - C["kp3"] * Id
                    + C["kcr"] * _softplus(params.Gb - G))
    dGp = EGP + Ra - C["Fcns"] - C["k1"] * Gp + C["k2"] * Gt
    dGt = -Uid + C["k1"] * Gp - C["k2"] * Gt

    # insulin subsystem (static secretion; see module docstring for the reduction)
    S = Sb + params.beta_sec * _softplus(G - params.Gb)
    dIp = -(C["m2"] + C["m4"]) * Ip + C["m1"] * Il + S
    dIl = -(C["m1"] + C["m3"]) * Il + C["m2"] * Ip

    # insulin action + delayed insulin signal driving EGP
    dX = -C["p2u"] * X + C["p2u"] * (I - C["Ib"])
    dI1 = -C["ki"] * (I1 - I)
    dId = -C["ki"] * (Id - I1)

    # subcutaneous (CGM) compartment — first-order lag on plasma glucose
    dGsc = (G - Gsc) / params.Td

    return jnp.array([dQsto1, dQsto2, dQgut, dGp, dGt, dIp, dIl, dX, dI1, dId, dGsc])


def run_meal(params: DallaManParams, carbs_g, meal_time: float = 30.0,
             duration_min: float = 210.0, step_min: float = 5.0, sigma: float = 2.0,
             plasma: bool = False, solid_frac: float = 1.0,
             rtol: float = 1e-4, atol: float = 1e-4, max_steps: int = 20000):
    """Simulate one meal. Returns (times, Gsc) — subcutaneous glucose, i.e. what CGM measures.
    `plasma=True` returns plasma glucose (for validation against the paper's figures).
    `solid_frac=0.0` for a liquid load such as the 75 g OGTT drink; 1.0 for solid/mixed meals."""
    D_mg = jnp.asarray(carbs_g, jnp.float32) * 1000.0          # g -> mg of glucose
    ts = jnp.arange(0.0, duration_min + 1e-6, step_min)
    sol = diffeqsolve(
        ODETerm(rhs), Tsit5(), t0=0.0, t1=float(duration_min), dt0=1.0,
        y0=initial_state(params), args=(params, meal_time, D_mg, sigma, solid_frac),
        saveat=SaveAt(ts=ts),
        stepsize_controller=PIDController(rtol=rtol, atol=atol), max_steps=max_steps,
    )
    if plasma:
        return ts, sol.ys[:, IDX["Gp"]] / C["Vg"]
    return ts, sol.ys[:, GSC]
