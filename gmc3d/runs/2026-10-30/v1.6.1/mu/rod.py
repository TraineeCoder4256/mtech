import numpy as np
from numba import njit
@njit
def rod(n, R, seed):
    np.random.seed(seed)
    out = np.zeros(2)
    for _ in range(n):
        x = 0.0; u = 1.0 if np.random.random() < 0.5 else -1.0; mu = 0
        first = True
        while True:
            d = -np.log(np.random.random())
            dl = (R - x) if u > 0 else (x + R)
            L = min(d, dl)
            if not first:
                out[0 if mu < 0 else 1] += L
            if dl < d: break
            x += d * u
            nu = 1.0 if np.random.random() < 0.5 else -1.0
            mu = u * nu; u = nu; first = False
    return out / n
print(rod(2_000_000, 3.0, 1))
