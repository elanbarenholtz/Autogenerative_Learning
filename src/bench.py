"""Quick device/throughput benchmark for the small transformer."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model import FibonacciTransformer, count_parameters

print("params:", count_parameters(FibonacciTransformer(vocab_size=6)))

def bench(dev, bs=32, L=200, iters=20):
    m = FibonacciTransformer(vocab_size=6, d_model=128, nhead=4, num_layers=3,
                             dim_feedforward=512, max_seq_len=L + 10).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.randint(0, 6, (bs, L), device=dev)
    y = torch.randint(0, 6, (bs, L), device=dev)
    for _ in range(3):
        opt.zero_grad(); out = m(x)
        F.cross_entropy(out.reshape(-1, 6), y.reshape(-1)).backward(); opt.step()
    if dev == 'mps':
        torch.mps.synchronize()
    t = time.time()
    for _ in range(iters):
        opt.zero_grad(); out = m(x)
        F.cross_entropy(out.reshape(-1, 6), y.reshape(-1)).backward(); opt.step()
    if dev == 'mps':
        torch.mps.synchronize()
    dt = time.time() - t
    return iters * bs / dt, dt / iters * 1000

for d in ['cpu', 'mps']:
    try:
        sps, mspb = bench(d)
        print(f"{d}: {sps:.0f} seq/s, {mspb:.1f} ms/batch")
    except Exception as e:
        print(d, "ERR", repr(e)[:200])
