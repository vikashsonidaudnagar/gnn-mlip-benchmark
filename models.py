"""
Graph neural network interatomic potentials in pure PyTorch.

  SchNet     - invariant, messages depend only on distances  (Schutt et al. 2018)
  PaiNN      - equivariant, carries vector features per atom (Schutt et al. 2021)
  DimeNet++  - invariant, uses distances AND bond angles     (Gasteiger et al. 2020)
               layers from PyTorch Geometric, periodic triplets built here

All models share PotentialBase: they predict per-atom energies, which are
summed into the total energy; forces are F = -dE/dpositions (autograd).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------
def cosine_cutoff(r, cutoff):
    return 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0) * (r < cutoff).to(r.dtype)


class GaussianRBF(nn.Module):
    def __init__(self, cutoff, n_rbf=32):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = 0.5 / (cutoff / n_rbf) ** 2

    def forward(self, r):
        return torch.exp(-self.gamma * (r[:, None] - self.centers) ** 2)


class SincRBF(nn.Module):
    """sin(n*pi*r/rc)/r basis used by PaiNN."""

    def __init__(self, cutoff, n_rbf=20):
        super().__init__()
        self.register_buffer("freqs", torch.arange(1, n_rbf + 1) * math.pi / cutoff)

    def forward(self, r):
        return torch.sin(self.freqs * r[:, None]) / r[:, None]


class PotentialBase(nn.Module):
    """Handles geometry, energy summation and autograd forces."""

    def __init__(self, cutoff, energy_shift=0.0, energy_scale=1.0):
        super().__init__()
        self.cutoff = cutoff
        self.energy_shift = energy_shift   # mean energy per atom (eV)
        self.energy_scale = energy_scale   # output scale (eV)

    def atomic_energies(self, z, vec, r, src, dst):
        raise NotImplementedError

    def forward(self, b, compute_forces=True, create_graph=True):
        pos = b["pos"]
        if compute_forces:
            pos.requires_grad_(True)
        src, dst = b["edge_src"], b["edge_dst"]
        vec = pos[dst] - pos[src] + b["edge_shift"]       # vector i -> j
        r = vec.norm(dim=-1)

        e_atom = self.atomic_energies(b["z"], vec, r, src, dst)
        e_atom = e_atom * self.energy_scale + self.energy_shift
        energy = torch.zeros(b["n_graphs"], device=pos.device,
                             dtype=e_atom.dtype).index_add_(
            0, b["batch"], e_atom)

        forces = None
        if compute_forces:
            forces = -torch.autograd.grad(
                energy.sum(), pos, create_graph=create_graph)[0]
        return energy, forces


# --------------------------------------------------------------------------
# SchNet
# --------------------------------------------------------------------------
def ssp(x):
    return F.softplus(x) - math.log(2.0)


class SchNetInteraction(nn.Module):
    def __init__(self, n_feat, n_rbf):
        super().__init__()
        self.filter1 = nn.Linear(n_rbf, n_feat)
        self.filter2 = nn.Linear(n_feat, n_feat)
        self.in2f = nn.Linear(n_feat, n_feat, bias=False)
        self.f2out = nn.Linear(n_feat, n_feat)
        self.out = nn.Linear(n_feat, n_feat)

    def forward(self, h, rbf, fcut, src, dst):
        W = self.filter2(ssp(self.filter1(rbf))) * fcut[:, None]
        msg = self.in2f(h)[dst] * W
        agg = torch.zeros_like(h).index_add_(0, src, msg)
        return h + self.out(ssp(self.f2out(agg)))


class SchNet(PotentialBase):
    def __init__(self, cutoff=5.0, n_feat=64, n_rbf=32, n_interactions=3,
                 max_z=100, **kw):
        super().__init__(cutoff, **kw)
        self.embed = nn.Embedding(max_z, n_feat)
        self.rbf = GaussianRBF(cutoff, n_rbf)
        self.blocks = nn.ModuleList(
            [SchNetInteraction(n_feat, n_rbf) for _ in range(n_interactions)])
        self.head = nn.Sequential(nn.Linear(n_feat, n_feat // 2), nn.Softplus(),
                                  nn.Linear(n_feat // 2, 1))

    def atomic_energies(self, z, vec, r, src, dst):
        rbf, fcut = self.rbf(r), cosine_cutoff(r, self.cutoff)
        h = self.embed(z)
        for block in self.blocks:
            h = block(h, rbf, fcut, src, dst)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------
# PaiNN
# --------------------------------------------------------------------------
class PaiNNMessage(nn.Module):
    def __init__(self, n_feat, n_rbf):
        super().__init__()
        self.n_feat = n_feat
        self.phi = nn.Sequential(nn.Linear(n_feat, n_feat), nn.SiLU(),
                                 nn.Linear(n_feat, 3 * n_feat))
        self.W = nn.Linear(n_rbf, 3 * n_feat)

    def forward(self, s, v, rbf, fcut, unit, src, dst):
        x = self.phi(s)[dst] * self.W(rbf) * fcut[:, None]
        ds, dvv, dvs = torch.split(x, self.n_feat, dim=-1)
        dv = v[dst] * dvv[:, None, :] + unit[:, :, None] * dvs[:, None, :]
        s = s + torch.zeros_like(s).index_add_(0, src, ds)
        v = v + torch.zeros_like(v).index_add_(0, src, dv)
        return s, v


class PaiNNUpdate(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.U = nn.Linear(n_feat, n_feat, bias=False)
        self.V = nn.Linear(n_feat, n_feat, bias=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_feat, n_feat), nn.SiLU(),
                                 nn.Linear(n_feat, 3 * n_feat))

    def forward(self, s, v):
        Uv, Vv = self.U(v), self.V(v)                       # (N, 3, F)
        Vv_norm = torch.sqrt((Vv ** 2).sum(dim=1) + 1e-8)   # (N, F)
        a = self.mlp(torch.cat([s, Vv_norm], dim=-1))
        a_vv, a_sv, a_ss = torch.split(a, self.n_feat, dim=-1)
        s = s + a_sv * (Uv * Vv).sum(dim=1) + a_ss
        v = v + a_vv[:, None, :] * Uv
        return s, v


class PaiNN(PotentialBase):
    def __init__(self, cutoff=5.0, n_feat=64, n_rbf=20, n_interactions=3,
                 max_z=100, **kw):
        super().__init__(cutoff, **kw)
        self.n_feat = n_feat
        self.embed = nn.Embedding(max_z, n_feat)
        self.rbf = SincRBF(cutoff, n_rbf)
        self.messages = nn.ModuleList(
            [PaiNNMessage(n_feat, n_rbf) for _ in range(n_interactions)])
        self.updates = nn.ModuleList(
            [PaiNNUpdate(n_feat) for _ in range(n_interactions)])
        self.head = nn.Sequential(nn.Linear(n_feat, n_feat // 2), nn.SiLU(),
                                  nn.Linear(n_feat // 2, 1))

    def atomic_energies(self, z, vec, r, src, dst):
        rbf, fcut = self.rbf(r), cosine_cutoff(r, self.cutoff)
        unit = vec / r[:, None]
        s = self.embed(z)
        v = torch.zeros(s.shape[0], 3, self.n_feat, device=s.device, dtype=s.dtype)
        for msg, upd in zip(self.messages, self.updates):
            s, v = msg(s, v, rbf, fcut, unit, src, dst)
            s, v = upd(s, v)
        return self.head(s).squeeze(-1)


# --------------------------------------------------------------------------
# DimeNet++ (PyTorch Geometric layers, periodic-boundary-safe geometry)
# --------------------------------------------------------------------------
def build_triplets(src, dst, vec, num_nodes):
    """
    Angle triplets k -> j -> i for periodic graphs.
    Edge e goes from centre src[e] (= i) to neighbour dst[e] (= j), vector vec[e].
    For each edge ji we pair every edge leaving j (j -> k), except the edge
    pointing straight back to the same periodic image of i.
    Returns (idx_kj, idx_ji) in PyG DimeNet convention.
    """
    device = src.device
    order = torch.argsort(src, stable=True)
    deg = torch.bincount(src, minlength=num_nodes)
    ptr = torch.cumsum(deg, 0) - deg
    n_per_edge = deg[dst]
    e_ji = torch.arange(src.numel(), device=device).repeat_interleave(n_per_edge)
    start = torch.cumsum(n_per_edge, 0) - n_per_edge
    local = torch.arange(e_ji.numel(), device=device) - start.repeat_interleave(n_per_edge)
    e_kj = order[ptr[dst[e_ji]] + local]
    v = vec.detach()
    reverse = (dst[e_kj] == src[e_ji]) & ((v[e_kj] + v[e_ji]).abs().sum(-1) < 1e-3)
    keep = ~reverse
    return e_kj[keep], e_ji[keep]


class DimeNetPP(PotentialBase):
    def __init__(self, cutoff=5.0, n_feat=128, n_interactions=4, max_z=100, **kw):
        super().__init__(cutoff, **kw)
        from torch_geometric.nn.models import DimeNetPlusPlus
        self.net = DimeNetPlusPlus(
            hidden_channels=n_feat, out_channels=1, num_blocks=n_interactions,
            int_emb_size=64, basis_emb_size=8, out_emb_channels=2 * n_feat,
            num_spherical=7, num_radial=6, cutoff=cutoff)

    def atomic_energies(self, z, vec, r, src, dst):
        net, n = self.net, z.size(0)
        idx_kj, idx_ji = build_triplets(src, dst, vec, n)
        v1, v2 = vec[idx_ji], vec[idx_kj]
        a = (v1 * v2).sum(-1)
        b = torch.sqrt(torch.cross(v1, v2, dim=-1).pow(2).sum(-1) + 1e-12)
        angle = torch.atan2(b, a)

        rbf = net.rbf(r)
        sbf = net.sbf(r, angle, idx_kj)
        x = net.emb(z, rbf, src, dst)
        P = net.output_blocks[0](x, rbf, src, num_nodes=n)
        for inter, out in zip(net.interaction_blocks, net.output_blocks[1:]):
            x = inter(x, rbf, sbf, idx_kj, idx_ji)
            P = P + out(x, rbf, src, num_nodes=n)
        return P.squeeze(-1)


# --------------------------------------------------------------------------
MODELS = {"schnet": SchNet, "painn": PaiNN, "dimenetpp": DimeNetPP}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {list(MODELS)}")
    return MODELS[name](**kwargs)
