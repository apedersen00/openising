import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT / 'hierarchy'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from ising_hierarchy.problems import hand_dummy
    from lagrange import analog_event, multiplicative_event

    return analog_event, hand_dummy, multiplicative_event, np


@app.cell
def _(np):
    def energy(s, J, h):
        """
        Compute the energy of a configuration s for the Ising model defined by J and h.
        E(s) = -1/2 * sum_{i,j} J_{ij} s_i s_j - sum_i h_i s_i
        """
        return -0.5 * np.sum(J * np.outer(s, s)) - np.dot(h, s)

    def analog_fe(J, h, v, dt, n_steps):
        """
        Step-by-step settle (Forward Euler): many tiny steps.
        """
        v = v.copy()
        for _ in range(n_steps):
            v = np.clip(v + dt * (J @ np.sign(v) + h), -1, 1)
        return np.where(v >= 0, 1.0, -1.0)

    def analog_event_traced(J, h, v):
        """
        Same as lagrange.analog_event in plain Python, printing every event.
        """
        v = v.copy()
        s = np.where(v >= 0, 1.0, -1.0)
        f = J @ s + h                                                           # local fields, constant until a flip
        t = 0.0
        while True:
            moving_to_zero = (v * f < 0) | ((v == 0) & (s * f < 0))             # spins heading for the threshold
            if not moving_to_zero.any():
                print(f'  t={t:.3f}  no spin heads for 0 -> settled, s*f = {s * f}')
                return s
            t_cross = np.where(moving_to_zero, -v / np.where(f == 0, 1, f), np.inf)
            k = int(np.argmin(t_cross))                                         # first spin to cross
            dt = t_cross[k]
            v = np.clip(v + dt * f, -1, 1)                                      # every voltage moves in a straight line
            v[k] = 0.0
            s_old = s[k]
            s[k] = np.sign(f[k])
            f += J[k] * (s[k] - s_old)                                          # only row k of J changes the fields
            t += dt
            print(f'  t={t:.3f}  spin {k} crosses 0: {s_old:.1f} -> {s[k]:.1f}   E = {energy(s, J, h):.1f}')

    return analog_event_traced, analog_fe, energy


@app.cell
def _(
    analog_event,
    analog_event_traced,
    analog_fe,
    energy,
    hand_dummy,
    multiplicative_event,
    np,
):
    J, h = hand_dummy()
    v0 = np.random.default_rng(3).uniform(-1, 1, len(h))
    print('v0   :', np.round(v0, 2))
    print('event trace:')
    s_ev = analog_event_traced(J, h, v0)
    s_fe = analog_fe(J, h, v0, dt=0.001, n_steps=20000)
    s_lib, T, n_events = analog_event(J, h, v0.copy(), 10**6)
    print('event      :', s_ev.astype(int), f'E = {energy(s_ev, J, h):+.1f}')
    print('Euler      :', s_fe.astype(int), f'E = {energy(s_fe, J, h):+.1f}  (20000 steps)')
    print('lagrange.py:', s_lib.astype(int), f'T = {T:.3f}, {n_events} events')

    print('\nsettle, keep best, kick (3 -> 1 spins):')
    best_s, best_E, hist, _ = multiplicative_event(J, h, n_iter=10, init_size=3, end_size=1, seed=0)
    for it in range(10):
        print(f'  it {it}: E this run {hist[it, 0]:+.1f}  best {hist[it, 1]:+.1f}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
