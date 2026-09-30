import math, torch, sys
sys.path.insert(0, ".")
from generators.nflow import SplineFlow, rq_spline
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
net = SplineFlow()
print("params", sum(p.numel() for p in net.parameters()))
with torch.no_grad():
    y = torch.randn(1000, 6); c = torch.randn(1000, 5)
    print("identity at init:", torch.allclose(net.sample(y, c), y),
          torch.allclose(net.log_prob(y, c), -0.5*(y*y).sum(1) - 3*math.log(2*math.pi)))
for std, bstd in ((0.1, 0.3), (0.3, 1.0)):
    torch.manual_seed(1)
    for s in net.steps:
        torch.nn.init.normal_(s.proj_out.weight, std=std); torch.nn.init.normal_(s.proj_out.bias, std=bstd)
    with torch.no_grad():
        z = torch.randn(20000, 6); c = torch.randn(20000, 5); emb = net.embed(c)
        x = net.sample(z, c); zz = x
        for s in net.steps: zz, _ = s(zz, emb)
        err = (zz - z).abs().max(1).values
        print(f"random flow (w std {std}, b std {bstd}): round-trip error median {float(err.median()):.1e}, 99.9% {float(err.quantile(0.999)):.1e}, max {float(err.max()):.1e}")
    # log-det vs autograd Jacobian, one sample at a time
    worst = 0
    for i in range(20):
        ci = c[i:i+1]
        def f(v):
            e = net.embed(ci); out = v[None]
            for s in net.steps: out, _ = s(out, e)
            return out[0]
        J = torch.autograd.functional.jacobian(f, x[i])
        with torch.no_grad():
            lp = net.log_prob(x[i:i+1], ci)[0]; zi = f(x[i])
        ld = lp + 0.5*(zi*zi).sum() + 3*math.log(2*math.pi)
        worst = max(worst, abs(float(ld - torch.slogdet(J)[1])))
    print(f"  log-det vs autograd Jacobian, 20 samples: worst difference {worst:.1e}")
K = 8; torch.manual_seed(2)
uw, uh, ud = torch.randn(K), torch.randn(K), torch.randn(K-1)
g = torch.linspace(-12, 12, 400001)
yv, ld = rq_spline(g, uw.expand(len(g), K), uh.expand(len(g), K), ud.expand(len(g), K-1))
p = torch.exp(-0.5*yv**2)/math.sqrt(2*math.pi)*torch.exp(ld)
print("1D: density integrates to", float(torch.trapezoid(p, g)), " monotone", bool((yv[1:] > yv[:-1]).all()))
