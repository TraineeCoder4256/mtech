import numpy as np
from numba import njit
@njit
def iso():
    c = 2.0*np.random.random()-1.0; p = 2*np.pi*np.random.random(); s = np.sqrt(1-c*c)
    return np.array([s*np.cos(p), s*np.sin(p), c])
@njit
def run(n, R, sig, seed):
    np.random.seed(seed)
    out = np.zeros(4); cnt = np.zeros(4)
    for _ in range(n):
        r = np.zeros(3); om = iso(); mu = 9.0
        while True:
            d = -np.log(np.random.random())/sig
            b = r @ om; dl = -b + np.sqrt(b*b - (r@r - R*R))
            L = min(d, dl)
            if mu < 2:
                k = min(int((mu+1)*2), 3); out[k] += L; cnt[k] += 1
            if dl < d: break
            r = r + d*om
            new = iso(); mu = om @ new; om = new
    return out/n, cnt/n
print(run(1_000_000, 6.0, 0.5, 3))
@njit
def lastmu(n, R, sig, seed):
    np.random.seed(seed)
    h = np.zeros(4)
    for _ in range(n):
        r = np.zeros(3); om = iso(); mu = 9.0
        while True:
            d = -np.log(np.random.random())/sig
            b = r @ om; dl = -b + np.sqrt(b*b - (r@r - R*R))
            if dl < d: break
            r = r + d*om
            new = iso(); mu = om @ new; om = new
        if mu < 2: h[min(int((mu+1)*2), 3)] += 1
    return h/h.sum()
print(lastmu(1_000_000, 6.0, 0.5, 4))
