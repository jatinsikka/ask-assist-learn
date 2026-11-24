"""Bootstrap ensemble of behavior-cloning MLPs. Ensemble disagreement is the 'should I ask?' signal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def mlp(i: int, o: int, h: int = 256) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                         nn.Linear(h, h), nn.GELU(), nn.Linear(h, o), nn.Tanh())


class EnsemblePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, k: int = 5, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.act_dim, self.k, self.hidden = obs_dim, act_dim, k, hidden
        self.members = nn.ModuleList([mlp(obs_dim, act_dim, hidden) for _ in range(k)])
        self.register_buffer("mu", torch.zeros(obs_dim))
        self.register_buffer("sd", torch.ones(obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (k, B, act)
        x = (obs - self.mu) / self.sd
        return torch.stack([m(x) for m in self.members])

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (mean action, uncertainty). Uncertainty = mean ensemble std over action dims."""
        out = self(torch.as_tensor(obs, dtype=torch.float32)[None])[:, 0]
        mean = out.mean(0).numpy()
        unc = float(out.std(0).mean())
        return mean, unc

    # ------------------------------------------------------------------
    def fit(self, obs: np.ndarray, act: np.ndarray, weight: np.ndarray, steps: int = 3000,
            batch: int = 512, lr: float = 1e-3, seed: int = 0, log_every: int = 0) -> float:
        """Weighted MSE. Each member trains on its own bootstrap resample (drawn with the sample weights)."""
        g = torch.Generator().manual_seed(seed)
        O = torch.as_tensor(obs, dtype=torch.float32)
        A = torch.as_tensor(act, dtype=torch.float32)
        w = torch.as_tensor(weight, dtype=torch.float64)
        self.mu.copy_(O.mean(0))
        self.sd.copy_(O.std(0).clamp_min(1e-3))
        p = (w / w.sum()).float()
        boots = [torch.multinomial(p, len(O), replacement=True, generator=g) for _ in range(self.k)]
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        loss_v = 0.0
        for it in range(steps):
            loss = 0.0
            for m, idx in zip(self.members, boots):
                b = idx[torch.randint(len(idx), (batch,), generator=g)]
                pred = m((O[b] - self.mu) / self.sd)
                err = (pred - A[b]) ** 2
                err[:, 4] *= 0.5  # gripper is a sign; looser fit
                loss = loss + err.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_v = float(loss.detach()) / self.k
            if log_every and it % log_every == 0:
                print(f"    fit {it:5d}  loss {loss_v:.4f}")
        return loss_v

    def save(self, path: str | Path):
        torch.save({"state": self.state_dict(), "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                    "k": self.k, "hidden": self.hidden}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EnsemblePolicy":
        ck = torch.load(path, weights_only=False)
        pol = cls(ck["obs_dim"], ck["act_dim"], ck["k"], ck["hidden"])
        pol.load_state_dict(ck["state"])
        return pol
