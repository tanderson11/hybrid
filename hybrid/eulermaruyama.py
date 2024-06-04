import numpy as np

def dW(dt, rng):
    return rng.normal(loc=0.0, scale=np.sqrt(dt))

def em_solve_ivp(mu_f, sigma_f, t_span, y0, rng, t_eval=None, events=None, args=None, dt=0.01):
    assert events is None

    t, t_end = t_span
    y = y0

    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t

        mu = mu_f(t, y)
        sigma = sigma_f(t, y)

        dy = mu * dt + sigma * dW(dt, rng)

        y = y + dy
        t = t + dt

    return t, y