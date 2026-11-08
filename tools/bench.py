# Pilot micro-benchmark: ms/step for (channels, batch, device). Used for GATE_1 timings.

import sys, time, torch, numpy as np
from stein.data import Shapes
from stein.models import Encoder, Predictor
from stein.losses import Regulariser
ch, B, steps = int(sys.argv[1]), int(sys.argv[2]), (120 if (len(sys.argv) > 3 and sys.argv[3] == "cpu") else 300)
dev = sys.argv[3] if len(sys.argv) > 3 else "mps"; torch.set_num_threads(1)
S = Shapes(); enc = Encoder(12, ch).to(dev); pred = Predictor(12, 12).to(dev)
opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), 1e-3)
reg = Regulariser("sigreg", 12).to(dev)
for i in range(steps):
    if i == 20: (torch.mps.synchronize() if dev == "mps" else None); t = time.time()
    f1, f2 = S.sample_pair_factors(B, 1.0)
    x = torch.cat([S.images_for(f1, dev), S.images_for(f2, dev)])
    z1, z2 = enc(x).chunk(2)
    l = ((pred(z1) - z2) ** 2).mean() + 0.05 * (reg(z1) + reg(z2))
    opt.zero_grad(); l.backward(); opt.step()
(torch.mps.synchronize() if dev == "mps" else None); print(f"ch={ch} B={B} {(time.time()-t)/(steps-20)*1000:.1f} ms/step", flush=True)
