"""MaskedGAT and Adapter: one GAT layer and a scaled correction head, shared by all three variants
(Decision 7). The three variants (eq. 4, eq. 5) differ only in the mask passed to MaskedGAT."""
import torch
import torch.nn.functional as F
from torch import nn


def masks(nbr: torch.Tensor) -> dict[str, torch.Tensor]:
    """Base (N, N) bool masks for the three variants, before combine_mask adds obs and the identity."""
    N = nbr.shape[-1]
    eye = torch.eye(N, dtype=torch.bool, device=nbr.device)
    return {"mlp": eye, "gat": nbr.bool(), "full": torch.ones(N, N, dtype=torch.bool, device=nbr.device)}


def combine_mask(variant_mask: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """(N, N) variant mask and (B, N) obs -> (B, N, N): the variant mask, ANDed with obs of the
    sender, ORed with the identity (so the diagonal is always true, guide C11)."""
    B, N = obs.shape
    eye = torch.eye(N, dtype=torch.bool, device=obs.device)
    base = variant_mask.unsqueeze(0) & obs.unsqueeze(1)             # mask[b, i, j] requires obs[b, j]
    return base | eye.unsqueeze(0)


class MaskedGAT(nn.Module):
    """One multi-head GAT layer (eq. 4), dense and masked: scores for every (i, j) pair, -inf outside
    the mask, softmax over j, ELU, heads joined."""

    def __init__(self, m: int = 64, heads: int = 4, slope: float = 0.2):
        super().__init__()
        if m % heads != 0:
            raise ValueError(f"m={m} must divide evenly into heads={heads}")
        self.heads, self.head_dim = heads, m // heads
        self.W = nn.ModuleList([nn.Linear(m, self.head_dim, bias=False) for _ in range(heads)])
        self.a1 = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a2 = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a1)
        nn.init.xavier_uniform_(self.a2)
        self.slope = slope

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (B, N, m), mask (B, N, N) bool (mask[b,i,j]: receiver i may read sender j)
        -> (u (B, N, m), alpha (B, K, N, N))."""
        if not mask.diagonal(dim1=-2, dim2=-1).all():
            raise AssertionError("the diagonal must always be in the mask (a receiver always reads itself)")
        outs, alphas = [], []
        for k in range(self.heads):
            Wz = self.W[k](z)                                       # (B, N, head_dim)
            e_i = (Wz * self.a1[k]).sum(-1, keepdim=True)            # (B, N, 1)
            e_j = (Wz * self.a2[k]).sum(-1, keepdim=True)            # (B, N, 1)
            e = F.leaky_relu(e_i + e_j.transpose(-1, -2), self.slope)   # (B, N, N), e[b, i, j]
            e = e.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(e, dim=-1)
            outs.append(F.elu(torch.bmm(alpha, Wz)))
            alphas.append(alpha)
        return torch.cat(outs, dim=-1), torch.stack(alphas, dim=1)


class Adapter(nn.Module):
    """Frozen cached h -> z = W_p h + b_p -> masked GAT -> scaled correction added to y0 (eq. 5).
    forward(h: [B,N,d] fp16, y0: [B,N,H], s: [B,N], mask: [B,N,N] bool) -> (yhat: [B,N,H], alpha)."""

    def __init__(self, d: int = 512, m: int = 64, heads: int = 4, H: int = 12, layers: int = 1):
        super().__init__()
        if layers != 1:
            raise NotImplementedError("guide C11/Decision 7 specifies one GAT layer")
        self.Wp = nn.Linear(d, m)
        self.gat = MaskedGAT(m, heads)
        self.Wg = nn.Linear(m, H)
        nn.init.zeros_(self.Wg.weight)
        nn.init.zeros_(self.Wg.bias)

    def forward(self, h: torch.Tensor, y0: torch.Tensor, s: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.Wp(h.float())
        u, alpha = self.gat(z, mask)
        return y0 + s.unsqueeze(-1) * self.Wg(u), alpha
