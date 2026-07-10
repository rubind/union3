"""Check a candidate log-density implementation against the BridgeStan reference.

The JAX/NumPyro port imports `check`:

    from scripts.numpyro_port.check_parity import check
    check(fn)   # fn(params: dict) -> float

where `params` is a constrained-parameter dict (contents of params_k.json, lists
already converted to numpy arrays) and the return value must be the fully
normalized joint log density WITHOUT the constraining-transform Jacobian —
i.e. it must match BridgeStan's log_density(propto=False, jacobian=False).
Pass criterion (handoff §4 phase 2): max relative error ~1e-8 over all points.

Self-test (validates the frozen reference itself, no JAX needed):
    uv run python scripts/numpyro_port/check_parity.py --self-test
re-evaluates every point with a fresh BridgeStan build (must reproduce exactly)
and finite-differences lp along random directions against the stored gradient.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ART = HERE / "artifacts"
STAN_FILE = HERE.parent.parent / "src" / "unity" / "models" / "unity_1.8.stan"


def load_reference():
    ref = dict(np.load(ART / "reference.npz", allow_pickle=True))
    n = len(ref["lp_nojac"])
    params = []
    for k in range(n):
        raw = json.loads((ART / f"params_{k:03d}.json").read_text())
        params.append({k_: np.asarray(v) if isinstance(v, list) else v for k_, v in raw.items()})
    ref["params"] = params
    return ref


def check(fn, rtol=1e-8, label="candidate"):
    """Compare fn(params_dict) -> lp against the stored lp_nojac at every point."""
    ref = load_reference()
    lp_ref = ref["lp_nojac"]
    lp = np.array([fn(p) for p in ref["params"]])
    abs_err = np.abs(lp - lp_ref)
    rel_err = abs_err / np.abs(lp_ref)
    worst = int(np.argmax(rel_err))
    print(f"[{label}] {len(lp)} points | max abs err {abs_err.max():.3e} | "
          f"max rel err {rel_err.max():.3e} (point {worst}: "
          f"ref {lp_ref[worst]:.10f}, got {lp[worst]:.10f})")
    ok = rel_err.max() < rtol
    print(f"[{label}] {'PASS' if ok else 'FAIL'} (rtol {rtol:g})")
    return ok


def self_test():
    import bridgestan

    ref = load_reference()
    model = bridgestan.StanModel.from_stan_file(str(STAN_FILE), str(ART / "data.json"))
    theta, lp_jac, grad = ref["theta_unc"], ref["lp_jac"], ref["grad_unc"]

    # 1) reference reproduces exactly from the stored params files
    def stan_lp(params):
        t = model.param_unconstrain_json(json.dumps({k: np.asarray(v).tolist() for k, v in params.items()}))
        return model.log_density(t, propto=False, jacobian=False)

    ok = check(stan_lp, rtol=1e-12, label="bridgestan round-trip")

    # 2) stored gradient vs central finite differences along random directions
    rng = np.random.default_rng(0)
    fd_errs = []
    for k in rng.choice(len(lp_jac), size=5, replace=False):
        v = rng.normal(size=theta.shape[1])
        v /= np.linalg.norm(v)
        h = 1e-6
        lp_p = model.log_density(theta[k] + h * v, propto=False, jacobian=True)
        lp_m = model.log_density(theta[k] - h * v, propto=False, jacobian=True)
        fd = (lp_p - lp_m) / (2 * h)
        an = grad[k] @ v
        fd_errs.append(abs(fd - an) / max(abs(an), 1.0))
        print(f"  grad check point {k:3d}: analytic {an:+.8f}  fd {fd:+.8f}")
    ok &= max(fd_errs) < 1e-5
    print(f"gradient finite-difference: max rel err {max(fd_errs):.3e} "
          f"({'PASS' if max(fd_errs) < 1e-5 else 'FAIL'})")
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(__doc__)
