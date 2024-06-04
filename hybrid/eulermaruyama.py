from dataclasses import dataclass
import numpy as np

@dataclass
class EMResult():
    t: np.ndarray
    y: np.ndarray
    nfev: int
    t_events: np.ndarray
    status: int = 0

def dW_f(a, rng):
    return rng.normal(loc=0.0, scale=np.sqrt(a))

def em_solve_ivp(mu_sigma_f, t_span, y0, rng, t_eval=None, events=None, args=None, dt=0.01):
    assert events is None or len(events) == 0

    t, t_end = t_span
    y = y0
    nfev = 0

    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t

        mu, sigma = mu_sigma_f(t, y, dt, dW_f, rng, *args)
        nfev += 1

        dy = mu * dt + sigma

        y = y + dy
        t = t + dt

    result = EMResult(
        np.array([t]),
        np.expand_dims(y, 1),
        nfev,
        t_events=np.array([])
    )
    return result