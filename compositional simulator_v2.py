"""
1D Compositional Gas-Condensate Reservoir Simulator (Prototype)

Scope
-----
This is a research/thesis-grade prototype for a 1D compositional reservoir
simulation workflow inspired by commercial simulators such as ECLIPSE/CMG,
but intentionally simplified for transparency and extensibility.

Current capabilities
--------------------
- 1D Cartesian grid
- Multi-component overall composition per cell
- Peng-Robinson EOS utilities
- Wilson K-value initialization
- Isothermal two-phase flash (successive-substitution + Rachford-Rice)
- Simple phase-mobility-based component transport
- Pressure diffusion-like update
- Producer well in final grid block under fixed bottom-hole pressure
- Time stepping and history capture

Current limitations
-------------------
- Prototype formulation, not a full fully-implicit simulator
- Simplified transport and pressure coupling
- No capillary pressure
- No gravity
- Approximate viscosity model
- No rock/fluid compressibility coupling beyond simple pore volume handling
- No EOS volume translation / advanced tuning
- No Peaceman well model yet

Recommended next extensions
---------------------------
1. Fully implicit Newton solver
2. Proper transmissibility with upstream weighting
3. Peaceman well model
4. Relative permeability curves
5. Near-well local grid refinement
6. Condensate banking diagnostics
7. History matching / sensitivity workflow

Author: OpenAI assistant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


R = 10.7316  # psia ft^3 / (lbmol R)


# -----------------------------------------------------------------------------
# Component and fluid definitions
# -----------------------------------------------------------------------------

@dataclass
class Component:
    name: str
    Tc: float          # Rankine
    Pc: float          # psia
    omega: float       # acentric factor
    Mw: float          # molecular weight


@dataclass
class FluidModel:
    components: List[Component]
    kij: np.ndarray    # binary interaction coefficients [nc, nc]

    @property
    def nc(self) -> int:
        return len(self.components)

    def critical_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Tc = np.array([c.Tc for c in self.components], dtype=float)
        Pc = np.array([c.Pc for c in self.components], dtype=float)
        omega = np.array([c.omega for c in self.components], dtype=float)
        Mw = np.array([c.Mw for c in self.components], dtype=float)
        return Tc, Pc, omega, Mw


# -----------------------------------------------------------------------------
# Peng-Robinson EOS
# -----------------------------------------------------------------------------

class PengRobinsonEOS:
    def __init__(self, fluid: FluidModel):
        self.fluid = fluid
        self.Tc, self.Pc, self.omega, self.Mw = fluid.critical_arrays()
        self.kij = fluid.kij

    def _ai_bi(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pure-component a_i and b_i for Peng-Robinson EOS."""
        Tr = T / self.Tc
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        alpha = (1.0 + kappa * (1.0 - np.sqrt(Tr))) ** 2
        ai = 0.45724 * (R ** 2) * (self.Tc ** 2) * alpha / self.Pc
        bi = 0.07780 * R * self.Tc / self.Pc
        return ai, bi

    def mixture_params(self, z: np.ndarray, T: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        ai, bi = self._ai_bi(T)
        aij = np.sqrt(np.outer(ai, ai)) * (1.0 - self.kij)
        am = float(z @ aij @ z)
        bm = float(np.dot(z, bi))
        return am, bm, ai, bi

    @staticmethod
    def solve_cubic(A: float, B: float) -> np.ndarray:
        """Solve Peng-Robinson cubic EOS for compressibility factor Z."""
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B ** 2 - 2.0 * B
        c0 = -(A * B - B ** 2 - B ** 3)
        coeffs = [1.0, c2, c1, c0]
        roots = np.roots(coeffs)
        real_roots = np.real(roots[np.isreal(roots)])
        real_roots.sort()
        return real_roots

    def z_factor(self, z: np.ndarray, P: float, T: float, phase: str) -> float:
        am, bm, _, _ = self.mixture_params(z, T)
        A = am * P / (R ** 2 * T ** 2)
        B = bm * P / (R * T)
        roots = self.solve_cubic(A, B)
        if len(roots) == 0:
            raise RuntimeError("No real Z-factor root found.")
        if phase.lower() == "v":
            return float(np.max(roots))
        if phase.lower() == "l":
            return float(np.min(roots))
        raise ValueError("phase must be 'v' or 'l'")

    def fugacity_coefficients(self, x: np.ndarray, P: float, T: float, phase: str) -> np.ndarray:
        am, bm, ai, bi = self.mixture_params(x, T)
        A = am * P / (R ** 2 * T ** 2)
        B = bm * P / (R * T)
        Z = self.z_factor(x, P, T, phase)

        aij = np.sqrt(np.outer(ai, ai)) * (1.0 - self.kij)
        sum_aij = aij @ x

        sqrt2 = math.sqrt(2.0)
        ln_phi = np.zeros_like(x)

        for i in range(len(x)):
            term1 = bi[i] / bm * (Z - 1.0)
            term2 = -math.log(Z - B)
            term3 = A / (2.0 * sqrt2 * B)
            term4 = (2.0 * sum_aij[i] / am) - (bi[i] / bm)
            term5 = math.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
            ln_phi[i] = term1 + term2 - term3 * term4 * term5

        return np.exp(ln_phi)


# -----------------------------------------------------------------------------
# Flash calculation
# -----------------------------------------------------------------------------

class FlashCalculator:
    def __init__(self, eos: PengRobinsonEOS):
        self.eos = eos
        self.fluid = eos.fluid
        self.Tc, self.Pc, self.omega, _ = self.fluid.critical_arrays()

    def wilson_k(self, P: float, T: float) -> np.ndarray:
        return (self.Pc / P) * np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.Tc / T))

    @staticmethod
    def rachford_rice(beta: float, z: np.ndarray, K: np.ndarray) -> float:
        return np.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0)))

    def solve_beta(self, z: np.ndarray, K: np.ndarray) -> float:
        f0 = self.rachford_rice(0.0, z, K)
        f1 = self.rachford_rice(1.0, z, K)

        if f0 < 0.0 and f1 < 0.0:
            return 0.0
        if f0 > 0.0 and f1 > 0.0:
            return 1.0

        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm = self.rachford_rice(mid, z, K)
            if abs(fm) < 1e-10:
                return mid
            if self.rachford_rice(lo, z, K) * fm < 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def flash(self, z: np.ndarray, P: float, T: float, max_iter: int = 50, tol: float = 1e-7) -> Dict[str, np.ndarray | float | str]:
        z = np.clip(np.asarray(z, dtype=float), 1e-12, None)
        z = z / np.sum(z)

        K = np.clip(self.wilson_k(P, T), 1e-6, 1e6)
        beta = self.solve_beta(z, K)

        if beta <= 1e-10:
            return {
                "state": "liquid",
                "beta": 0.0,
                "x": z.copy(),
                "y": z.copy(),
                "K": K,
            }
        if beta >= 1.0 - 1e-10:
            return {
                "state": "vapor",
                "beta": 1.0,
                "x": z.copy(),
                "y": z.copy(),
                "K": K,
            }

        for _ in range(max_iter):
            denom = 1.0 + beta * (K - 1.0)
            x = z / denom
            y = K * x
            x /= np.sum(x)
            y /= np.sum(y)

            phi_l = self.eos.fugacity_coefficients(x, P, T, phase="l")
            phi_v = self.eos.fugacity_coefficients(y, P, T, phase="v")
            K_new = np.clip(phi_l / phi_v, 1e-8, 1e8)

            beta_new = self.solve_beta(z, K_new)

            err = max(np.max(np.abs(K_new - K)), abs(beta_new - beta))
            K = 0.5 * K + 0.5 * K_new
            beta = beta_new

            if err < tol:
                denom = 1.0 + beta * (K - 1.0)
                x = z / denom
                y = K * x
                x /= np.sum(x)
                y /= np.sum(y)
                return {
                    "state": "two_phase",
                    "beta": beta,
                    "x": x,
                    "y": y,
                    "K": K,
                }

        denom = 1.0 + beta * (K - 1.0)
        x = z / denom
        y = K * x
        x /= np.sum(x)
        y /= np.sum(y)
        return {
            "state": "two_phase",
            "beta": beta,
            "x": x,
            "y": y,
            "K": K,
        }


# -----------------------------------------------------------------------------
# Reservoir model definitions
# -----------------------------------------------------------------------------

@dataclass
class Grid1D:
    nx: int
    length_ft: float
    area_ft2: float
    thickness_ft: float = 50.0
    dx_array: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.dx_array is None:
            self.dx_array = np.full(self.nx, self.length_ft / self.nx, dtype=float)
        else:
            self.dx_array = np.asarray(self.dx_array, dtype=float)
            if len(self.dx_array) != self.nx:
                raise ValueError("dx_array length must equal nx")
            total = float(np.sum(self.dx_array))
            if total <= 0.0:
                raise ValueError("dx_array total length must be positive")
            self.dx_array = self.dx_array * (self.length_ft / total)

    @property
    def width_ft(self) -> float:
        return self.area_ft2 / self.thickness_ft

    def cell_width(self, i: int) -> float:
        return float(self.dx_array[i])

    def cell_center(self, i: int) -> float:
        return float(np.sum(self.dx_array[:i]) + 0.5 * self.dx_array[i])

    def bulk_volume(self, i: int) -> float:
        return self.cell_width(i) * self.area_ft2

    def interface_distance(self, i: int, j: int) -> float:
        if abs(i - j) != 1:
            raise ValueError("Only adjacent cells are supported")
        return 0.5 * (self.cell_width(i) + self.cell_width(j))


def build_near_well_lgr(
    nx: int,
    length_ft: float,
    refined_cells: int = 8,
    min_dx_ft: float = 20.0,
    growth: float = 1.35,
) -> np.ndarray:
    """
    Build a simple 1D local-grid-refinement-style spacing with the smallest cells
    placed at the producer side (last cells in the grid).
    """
    refined_cells = max(2, min(refined_cells, nx - 1))
    coarse_cells = nx - refined_cells

    refined = np.array([min_dx_ft * (growth ** i) for i in range(refined_cells)], dtype=float)
    refined = refined[::-1]  # smallest cell at the well side
    refined_total = float(np.sum(refined))

    remaining = max(length_ft - refined_total, 0.25 * length_ft)
    if coarse_cells > 0:
        coarse = np.full(coarse_cells, remaining / coarse_cells, dtype=float)
        dx = np.concatenate([coarse, refined])
    else:
        dx = refined.copy()

    return dx * (length_ft / float(np.sum(dx)))


@dataclass
class Rock:
    porosity: float
    permeability_md: float


@dataclass
class Well:
    cell_index: int
    bhp_psia: float
    productivity_index: float
    rw_ft: float = 0.35
    skin: float = 0.0


@dataclass
class SimulationState:
    pressure: np.ndarray           # [nx]
    z: np.ndarray                  # [nx, nc] overall composition
    nt: np.ndarray                 # [nx] total moles in each cell


@dataclass
class SimulationHistory:
    time_days: List[float] = field(default_factory=list)
    avg_pressure_psia: List[float] = field(default_factory=list)
    well_rate_total: List[float] = field(default_factory=list)
    well_gas_fraction: List[float] = field(default_factory=list)
    well_liquid_fraction: List[float] = field(default_factory=list)
    min_pressure_psia: List[float] = field(default_factory=list)
    well_dropout_indicator: List[float] = field(default_factory=list)
    well_krg: List[float] = field(default_factory=list)
    well_krl: List[float] = field(default_factory=list)
    well_pressure_psia: List[float] = field(default_factory=list)
    well_damage_factor: List[float] = field(default_factory=list)
    well_effective_wi: List[float] = field(default_factory=list)
    well_rate_undamaged: List[float] = field(default_factory=list)
    productivity_loss_fraction: List[float] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Utility correlations
# -----------------------------------------------------------------------------

def phase_viscosity_cp(phase: str, comp: np.ndarray) -> float:
    """
    Very simple placeholder viscosity model.
    Vapor viscosity is low, liquid viscosity is higher and composition dependent.
    """
    if phase == "v":
        return 0.02 + 0.03 * float(np.sum(comp))
    if phase == "l":
        heaviness = np.linspace(0.5, 1.5, len(comp)) @ comp
        return 0.15 + 0.25 * heaviness
    raise ValueError("phase must be 'v' or 'l'")


def corey_relperm(Sg: float, Sgc: float = 0.05, Slr: float = 0.15, ng: float = 2.0, nl: float = 2.0) -> Tuple[float, float]:
    """
    Corey-style relative permeability curves for gas and liquid.
    Returns krg, krl.
    """
    Sg_eff = (Sg - Sgc) / max(1.0 - Sgc - Slr, 1e-10)
    Sg_eff = float(np.clip(Sg_eff, 0.0, 1.0))
    Sl_eff = 1.0 - Sg_eff
    krg = Sg_eff ** ng
    krl = Sl_eff ** nl
    return krg, krl


def liquid_dropout_fraction(z: np.ndarray, beta: float) -> float:
    """
    Diagnostic proxy for condensate dropout risk.
    Higher heavy-end content and lower vapor fraction imply stronger dropout.
    """
    heaviness_weights = np.linspace(0.0, 1.0, len(z))
    heaviness = float(np.dot(heaviness_weights, z))
    return max((1.0 - beta) * heaviness, 0.0)


def condensate_bank_damage_factor(
    dropout: float,
    krl: float,
    damage_strength: float = 8.0,
    critical_dropout: float = 0.03,
) -> float:
    """
    Proxy near-well damage multiplier from condensate banking.
    1.0 means no impairment; lower values imply stronger productivity loss.
    """
    excess = max(dropout - critical_dropout, 0.0)
    liquid_penalty = max(1.0 - krl, 0.0)
    damage = math.exp(-damage_strength * excess * (0.5 + 0.5 * liquid_penalty))
    return float(np.clip(damage, 0.05, 1.0))


def productivity_loss_fraction(q_actual: float, q_undamaged: float) -> float:
    if q_undamaged <= 1e-12:
        return 0.0
    loss = 1.0 - q_actual / q_undamaged
    return float(np.clip(loss, 0.0, 1.0))


def z_to_density_lbmol_ft3(P: float, T: float, Z: float) -> float:
    return P / (Z * R * T)


# -----------------------------------------------------------------------------
# Main simulator
# -----------------------------------------------------------------------------

class CompositionalSimulator1D:
    def spatial_diagnostics(self, state: SimulationState) -> Dict[str, np.ndarray]:
        xcoord = np.array([self.grid.cell_center(i) for i in range(self.grid.nx)], dtype=float)
        beta_arr = np.zeros(self.grid.nx)
        dropout_arr = np.zeros(self.grid.nx)
        krg_arr = np.zeros(self.grid.nx)
        krl_arr = np.zeros(self.grid.nx)
        gas_mob_arr = np.zeros(self.grid.nx)
        liq_mob_arr = np.zeros(self.grid.nx)

        for i in range(self.grid.nx):
            fl = self.cell_flash(state, i)
            beta = float(fl["beta"])
            mob = self.phase_mobility_data(state, i)
            beta_arr[i] = beta
            dropout_arr[i] = liquid_dropout_fraction(state.z[i], beta)
            krg_arr[i] = mob["krg"]
            krl_arr[i] = mob["krl"]
            gas_mob_arr[i] = mob["lam_g"]
            liq_mob_arr[i] = mob["lam_l"]

        return {
            "x_ft": xcoord,
            "pressure_psia": state.pressure.copy(),
            "gas_saturation_proxy": beta_arr,
            "liquid_saturation_proxy": 1.0 - beta_arr,
            "dropout_indicator": dropout_arr,
            "krg": krg_arr,
            "krl": krl_arr,
            "gas_mobility": gas_mob_arr,
            "liquid_mobility": liq_mob_arr,
        }

    def __init__(
        self,
        grid: Grid1D,
        rock: Rock,
        fluid: FluidModel,
        eos: PengRobinsonEOS,
        flash: FlashCalculator,
        well: Well,
        temperature_R: float,
    ):
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.eos = eos
        self.flash = flash
        self.well = well
        self.T = temperature_R
        self.nc = fluid.nc

        self.phi = rock.porosity
        self.k = rock.permeability_md
        self.pv = np.array([self.grid.bulk_volume(i) * self.phi for i in range(self.grid.nx)], dtype=float)

    def initialize_state(self, p_init_psia: float, z_init: np.ndarray) -> SimulationState:
        z_init = np.asarray(z_init, dtype=float)
        z_init = z_init / np.sum(z_init)

        pressure = np.full(self.grid.nx, p_init_psia, dtype=float)
        z = np.tile(z_init, (self.grid.nx, 1))

        # Initialize total moles in-place from vapor EOS as a first estimate.
        Zg = self.eos.z_factor(z_init, p_init_psia, self.T, phase="v")
        molar_density = z_to_density_lbmol_ft3(p_init_psia, self.T, Zg)
        nt = molar_density * self.pv.copy()

        return SimulationState(pressure=pressure, z=z, nt=nt)

    def cell_flash(self, state: SimulationState, i: int) -> Dict[str, np.ndarray | float | str]:
        return self.flash.flash(state.z[i], state.pressure[i], self.T)

    def phase_mobility_data(self, state: SimulationState, i: int) -> Dict[str, float]:
        fl = self.cell_flash(state, i)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])

        mu_l = phase_viscosity_cp("l", x)
        mu_v = phase_viscosity_cp("v", y)

        Sg = float(np.clip(beta, 0.0, 1.0))
        Sl = 1.0 - Sg
        krg, krl = corey_relperm(Sg)

        lam_g = krg / max(mu_v, 1e-8)
        lam_l = krl / max(mu_l, 1e-8)
        lam_t = lam_g + lam_l

        return {
            "Sg": Sg,
            "Sl": Sl,
            "krg": krg,
            "krl": krl,
            "mu_g": mu_v,
            "mu_l": mu_l,
            "lam_g": lam_g,
            "lam_l": lam_l,
            "lam_t": max(lam_t, 1e-10),
        }

    def total_mobility(self, state: SimulationState, i: int) -> float:
        return float(self.phase_mobility_data(state, i)["lam_t"])

    def component_flux_between(self, state: SimulationState, i: int, j: int) -> np.ndarray:
        """
        Simplified upwinded component molar flux from i to j.
        Positive means flow from i -> j.
        """
        p_i = state.pressure[i]
        p_j = state.pressure[j]
        dp = p_i - p_j

        if abs(dp) < 1e-12:
            return np.zeros(self.nc)

        upstream = i if dp >= 0.0 else j
        fl = self.cell_flash(state, upstream)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])

        lam = self.total_mobility(state, upstream)

        distance = self.grid.interface_distance(i, j)
        trans = self.k * self.grid.area_ft2 / distance
        q_total = trans * lam * dp * 1e-5  # scale factor for numerical stability

        comp_frac = beta * y + (1.0 - beta) * x
        return q_total * comp_frac

    def peaceman_well_index(self, i: int) -> float:
        dx = self.grid.cell_width(i)
        dy = self.grid.width_ft
        h = self.grid.thickness_ft
        rw = max(self.well.rw_ft, 1e-4)
        skin = self.well.skin

        re = 0.28 * math.sqrt(dx * dx + dy * dy) / ((dy / dx) ** 0.5 + (dx / dy) ** 0.5)
        re = max(re, 1.01 * rw)

        # Geometric well index; phase mobility is applied separately.
        wi_geom = 0.00708 * self.k * h / max(math.log(re / rw) + skin, 1e-8)
        return self.well.productivity_index * wi_geom

    def well_sink(self, state: SimulationState) -> Tuple[np.ndarray, float, float, float, float, float, float, float, float, float, float]:
        i = self.well.cell_index
        pwf = self.well.bhp_psia
        pr = state.pressure[i]
        dp = max(pr - pwf, 0.0)

        fl = self.cell_flash(state, i)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])

        mob = self.phase_mobility_data(state, i)
        lam_g = mob["lam_g"]
        lam_l = mob["lam_l"]
        lam_t = mob["lam_t"]
        krg = mob["krg"]
        krl = mob["krl"]

        wi_geom = self.peaceman_well_index(i)
        q_undamaged = wi_geom * lam_t * dp * 1e-2

        dropout = liquid_dropout_fraction(state.z[i], beta)
        damage_factor = condensate_bank_damage_factor(dropout, krl)
        wi_eff = wi_geom * damage_factor
        q_total = wi_eff * lam_t * dp * 1e-2

        fg = lam_g / max(lam_t, 1e-12)
        flq = lam_l / max(lam_t, 1e-12)

        qg = q_total * fg
        ql = q_total * flq

        sink = qg * y + ql * x
        ploss = productivity_loss_fraction(q_total, q_undamaged)
        return sink, q_total, fg, flq, dropout, krg, krl, damage_factor, wi_eff, q_undamaged, ploss

    def pressure_update(self, state: SimulationState, dt_days: float) -> np.ndarray:
        """
        Simple explicit diffusion-like pressure update.
        This is intentionally simplified for prototype purposes.
        """
        p = state.pressure.copy()
        p_new = p.copy()
        alpha = 0.15 * dt_days

        for i in range(self.grid.nx):
            left = p[i - 1] if i > 0 else p[i]
            right = p[i + 1] if i < self.grid.nx - 1 else p[i]
            lap = left - 2.0 * p[i] + right
            p_new[i] = p[i] + alpha * lap

        # Well drawdown effect
        wi = self.well.cell_index
        p_new[wi] -= 0.05 * dt_days * max(p[wi] - self.well.bhp_psia, 0.0)

        return np.maximum(p_new, self.well.bhp_psia)

    def transport_update(self, state: SimulationState, dt_days: float) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, float, float]:
        ncomp = state.nt[:, None] * state.z
        ncomp_new = ncomp.copy()

        # Intercell fluxes
        for i in range(self.grid.nx - 1):
            flux = self.component_flux_between(state, i, i + 1)
            ncomp_new[i] -= dt_days * flux
            ncomp_new[i + 1] += dt_days * flux

        # Producer sink
        sink, q_total, gas_frac, liq_frac, dropout, krg, krl, damage_factor, wi_eff, q_undamaged, ploss = self.well_sink(state)
        wi = self.well.cell_index
        ncomp_new[wi] -= dt_days * sink

        ncomp_new = np.clip(ncomp_new, 1e-12, None)
        nt_new = np.sum(ncomp_new, axis=1)
        z_new = ncomp_new / nt_new[:, None]

        return nt_new, z_new, q_total, gas_frac, liq_frac, dropout, krg, krl, damage_factor, wi_eff, q_undamaged, ploss

    def step(self, state: SimulationState, dt_days: float) -> Tuple[SimulationState, Dict[str, float]]:
        p_new = self.pressure_update(state, dt_days)
        state_mid = SimulationState(pressure=p_new, z=state.z.copy(), nt=state.nt.copy())
        nt_new, z_new, q_total, gas_frac, liq_frac, dropout, krg, krl, damage_factor, wi_eff, q_undamaged, ploss = self.transport_update(state_mid, dt_days)

        new_state = SimulationState(pressure=p_new, z=z_new, nt=nt_new)
        summary = {
            "well_rate_total": q_total,
            "well_gas_fraction": gas_frac,
            "well_liquid_fraction": liq_frac,
            "well_dropout_indicator": dropout,
            "well_krg": krg,
            "well_krl": krl,
            "well_damage_factor": damage_factor,
            "well_effective_wi": wi_eff,
            "well_rate_undamaged": q_undamaged,
            "productivity_loss_fraction": ploss,
            "avg_pressure": float(np.mean(p_new)),
            "min_pressure": float(np.min(p_new)),
        }
        return new_state, summary

    def run(self, state0: SimulationState, t_end_days: float, dt_days: float) -> Tuple[SimulationState, SimulationHistory]:
        history = SimulationHistory()
        state = state0
        t = 0.0

        while t <= t_end_days + 1e-12:
            history.time_days.append(t)
            history.avg_pressure_psia.append(float(np.mean(state.pressure)))
            history.min_pressure_psia.append(float(np.min(state.pressure)))

            sink, q_total, gas_frac, liq_frac, dropout, krg, krl, damage_factor, wi_eff, q_undamaged, ploss = self.well_sink(state)
            history.well_rate_total.append(float(q_total))
            history.well_gas_fraction.append(float(gas_frac))
            history.well_liquid_fraction.append(float(liq_frac))
            history.well_dropout_indicator.append(float(dropout))
            history.well_krg.append(float(krg))
            history.well_krl.append(float(krl))
            history.well_pressure_psia.append(float(state.pressure[self.well.cell_index]))
            history.well_damage_factor.append(float(damage_factor))
            history.well_effective_wi.append(float(wi_eff))
            history.well_rate_undamaged.append(float(q_undamaged))
            history.productivity_loss_fraction.append(float(ploss))

            state, _ = self.step(state, dt_days)
            t += dt_days

        return state, history


# -----------------------------------------------------------------------------
# Example case resembling a lean gas-condensate system
# -----------------------------------------------------------------------------

def build_example_fluid() -> FluidModel:
    comps = [
        Component("C1", Tc=343.0, Pc=667.8, omega=0.011, Mw=16.04),
        Component("C3", Tc=665.7, Pc=616.0, omega=0.152, Mw=44.10),
        Component("nC5", Tc=845.4, Pc=488.6, omega=0.251, Mw=72.15),
        Component("C10+", Tc=1110.0, Pc=305.0, omega=0.490, Mw=140.0),
    ]
    kij = np.array(
        [
            [0.0, 0.015, 0.030, 0.060],
            [0.015, 0.0, 0.010, 0.025],
            [0.030, 0.010, 0.0, 0.010],
            [0.060, 0.025, 0.010, 0.0],
        ],
        dtype=float,
    )
    return FluidModel(components=comps, kij=kij)


def export_history_csv(history: SimulationHistory, filepath: str) -> None:
    rows = zip(
        history.time_days,
        history.avg_pressure_psia,
        history.min_pressure_psia,
        history.well_pressure_psia,
        history.well_rate_total,
        history.well_rate_undamaged,
        history.productivity_loss_fraction,
        history.well_damage_factor,
        history.well_effective_wi,
        history.well_gas_fraction,
        history.well_liquid_fraction,
        history.well_dropout_indicator,
        history.well_krg,
        history.well_krl,
    )
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_days",
            "avg_pressure_psia",
            "min_pressure_psia",
            "well_pressure_psia",
            "well_rate_total_lbmol_day",
            "well_rate_undamaged_lbmol_day",
            "productivity_loss_fraction",
            "well_damage_factor",
            "well_effective_wi",
            "well_gas_fraction",
            "well_liquid_fraction",
            "well_dropout_indicator",
            "well_krg",
            "well_krl",
        ])
        writer.writerows(rows)


def export_spatial_csv(diag: Dict[str, np.ndarray], filepath: str) -> None:
    headers = list(diag.keys())
    data = np.column_stack([diag[h] for h in headers])
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


def generate_chapter4_summary(history: SimulationHistory) -> List[List[object]]:
    def peak_with_time(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        idx = int(np.argmax(arr))
        return float(arr[idx]), float(history.time_days[idx])

    def final_value(values: List[float]) -> float:
        return float(values[-1])

    def initial_value(values: List[float]) -> float:
        return float(values[0])

    p_peak, p_peak_t = peak_with_time(history.avg_pressure_psia)
    d_peak, d_peak_t = peak_with_time(history.well_dropout_indicator)
    df_peak, df_peak_t = peak_with_time(history.well_damage_factor)
    pl_peak, pl_peak_t = peak_with_time(history.productivity_loss_fraction)

    return [
        ["Average Pressure (psia)", initial_value(history.avg_pressure_psia), p_peak, p_peak_t, final_value(history.avg_pressure_psia)],
        ["Dropout Indicator (-)", initial_value(history.well_dropout_indicator), d_peak, d_peak_t, final_value(history.well_dropout_indicator)],
        ["Damage Factor (-)", initial_value(history.well_damage_factor), df_peak, df_peak_t, final_value(history.well_damage_factor)],
        ["Productivity Loss Fraction (-)", initial_value(history.productivity_loss_fraction), pl_peak, pl_peak_t, final_value(history.productivity_loss_fraction)],
    ]


def export_chapter4_summary_csv(history: SimulationHistory, filepath: str) -> None:
    rows = generate_chapter4_summary(history)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Initial Value", "Peak Value", "Time of Peak (days)", "Final Value"])
        writer.writerows(rows)


def print_chapter4_summary(history: SimulationHistory) -> None:
    rows = generate_chapter4_summary(history)
    print()
    print("Chapter 4 Summary Table:")
    print(f"{'Metric':35s} {'Initial':>12s} {'Peak':>12s} {'Peak Time':>12s} {'Final':>12s}")
    for metric, initial, peak, peak_time, final in rows:
        print(f"{metric:35s} {initial:12.4f} {peak:12.4f} {peak_time:12.1f} {final:12.4f}")


def plot_history(history: SimulationHistory, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.avg_pressure_psia, label="Average Pressure")
    plt.plot(history.time_days, history.well_pressure_psia, label="Well-Cell Pressure")
    plt.xlabel("Time (days)")
    plt.ylabel("Pressure (psia)")
    plt.title("Pressure Depletion History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pressure_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.well_rate_total)
    plt.xlabel("Time (days)")
    plt.ylabel("Total Well Rate (lbmol/day)")
    plt.title("Producer Total Rate History")
    plt.tight_layout()
    plt.savefig(output_dir / "well_rate_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.well_gas_fraction, label="Gas Fraction")
    plt.plot(history.time_days, history.well_liquid_fraction, label="Liquid Fraction")
    plt.xlabel("Time (days)")
    plt.ylabel("Produced Fraction")
    plt.title("Produced Phase Fractions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "phase_fraction_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.well_dropout_indicator)
    plt.xlabel("Time (days)")
    plt.ylabel("Dropout Indicator")
    plt.title("Near-Well Condensate Dropout Indicator")
    plt.tight_layout()
    plt.savefig(output_dir / "dropout_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.well_rate_total, label="Damaged")
    plt.plot(history.time_days, history.well_rate_undamaged, label="Undamaged")
    plt.xlabel("Time (days)")
    plt.ylabel("Well Rate (lbmol/day)")
    plt.title("Productivity Loss from Condensate Banking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "productivity_loss_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.productivity_loss_fraction)
    plt.xlabel("Time (days)")
    plt.ylabel("Productivity Loss Fraction")
    plt.title("Fractional Deliverability Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "productivity_loss_fraction_history.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.well_damage_factor)
    plt.xlabel("Time (days)")
    plt.ylabel("Damage Factor")
    plt.title("Condensate-Bank Damage Factor")
    plt.tight_layout()
    plt.savefig(output_dir / "damage_factor_history.png", dpi=200)
    plt.close()


def plot_spatial_diagnostics(diag: Dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = diag["x_ft"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["pressure_psia"])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Pressure (psia)")
    plt.title("Final Pressure Profile")
    plt.tight_layout()
    plt.savefig(output_dir / "pressure_profile.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["dropout_indicator"])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Dropout Indicator")
    plt.title("Final Condensate Dropout Profile")
    plt.tight_layout()
    plt.savefig(output_dir / "dropout_profile.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["krg"], label="krg")
    plt.plot(x, diag["krl"], label="krl")
    plt.xlabel("Distance (ft)")
    plt.ylabel("Relative Permeability")
    plt.title("Final Relative Permeability Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "relperm_profile.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["gas_saturation_proxy"], label="Gas Saturation Proxy")
    plt.plot(x, diag["liquid_saturation_proxy"], label="Liquid Saturation Proxy")
    plt.xlabel("Distance (ft)")
    plt.ylabel("Phase Saturation Proxy")
    plt.title("Final Phase Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "phase_distribution_profile.png", dpi=200)
    plt.close()


def run_example() -> Dict[str, object]:
    fluid = build_example_fluid()
    eos = PengRobinsonEOS(fluid)
    flash = FlashCalculator(eos)

    dx_array = build_near_well_lgr(
        nx=40,
        length_ft=4000.0,
        refined_cells=10,
        min_dx_ft=12.0,
        growth=1.45,
    )
    grid = Grid1D(
        nx=40,
        length_ft=4000.0,
        area_ft2=2500.0,
        thickness_ft=50.0,
        dx_array=dx_array,
    )
    rock = Rock(porosity=0.18, permeability_md=8.0)
    well = Well(cell_index=39, bhp_psia=3000.0, productivity_index=1.0, rw_ft=0.35, skin=0.0)

    sim = CompositionalSimulator1D(
        grid=grid,
        rock=rock,
        fluid=fluid,
        eos=eos,
        flash=flash,
        well=well,
        temperature_R=680.0,
    )

    z_init = np.array([0.82, 0.08, 0.06, 0.04], dtype=float)
    state0 = sim.initialize_state(p_init_psia=5600.0, z_init=z_init)

    final_state, history = sim.run(state0, t_end_days=240.0, dt_days=2.0)
    final_diag = sim.spatial_diagnostics(final_state)

    output_dir = Path("simulation_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    history_csv = output_dir / "history.csv"
    spatial_csv = output_dir / "final_spatial_diagnostics.csv"
    summary_csv = output_dir / "chapter4_summary.csv"

    export_history_csv(history, str(history_csv))
    export_spatial_csv(final_diag, str(spatial_csv))
    export_chapter4_summary_csv(history, str(summary_csv))
    plot_history(history, output_dir)
    plot_spatial_diagnostics(final_diag, output_dir)

    summary_rows = generate_chapter4_summary(history)
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Metric", "Initial Value", "Peak Value", "Time of Peak (days)", "Final Value"],
    )

    return {
        "fluid": fluid,
        "grid": grid,
        "well": well,
        "sim": sim,
        "final_state": final_state,
        "history": history,
        "final_diag": final_diag,
        "summary_df": summary_df,
        "output_dir": output_dir,
        "history_csv": history_csv,
        "spatial_csv": spatial_csv,
        "summary_csv": summary_csv,
    }


def display_streamlit_app() -> None:
    st.set_page_config(page_title="Compositional Reservoir Simulator", layout="wide")
    st.title("Compositional Reservoir Simulator")
    st.write(
        "1D compositional gas-condensate prototype with Peng-Robinson EOS, "
        "Peaceman well model, local-grid refinement, condensate-bank damage, "
        "and Chapter 4 summary outputs."
    )

    st.sidebar.header("Model Overview")
    st.sidebar.write("Grid: 1D nonuniform Cartesian")
    st.sidebar.write("Fluid: 4-component lean gas-condensate")
    st.sidebar.write("Well: Peaceman producer with fixed BHP")
    st.sidebar.write("Outputs: CSV exports, summary table, history charts, spatial diagnostics")

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running compositional simulation..."):
            result = run_example()

        history = result["history"]
        final_diag = result["final_diag"]
        summary_df = result["summary_df"]
        fluid = result["fluid"]
        well = result["well"]
        final_state = result["final_state"]
        output_dir = result["output_dir"]
        sim = result["sim"]
        grid = result["grid"]

        st.success("Simulation completed.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Avg Pressure (psia)", f"{history.avg_pressure_psia[-1]:.2f}")
        c2.metric("Final Well Rate (lbmol/day)", f"{history.well_rate_total[-1]:.6f}")
        c3.metric("Productivity Loss Fraction", f"{history.productivity_loss_fraction[-1]:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Dropout Indicator", f"{history.well_dropout_indicator[-1]:.4f}")
        c5.metric("Damage Factor", f"{history.well_damage_factor[-1]:.4f}")
        c6.metric("Effective WI", f"{history.well_effective_wi[-1]:.5f}")

        st.subheader("Chapter 4 Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Well-Cell Final Composition")
        comp_df = pd.DataFrame(
            {
                "Component": [c.name for c in fluid.components],
                "Mole Fraction": final_state.z[well.cell_index],
            }
        )
        st.dataframe(comp_df, use_container_width=True)

        st.subheader("Model Diagnostics")
        st.write(f"Output directory: {output_dir.resolve()}")
        st.write(f"Near-well minimum dx (ft): {grid.cell_width(well.cell_index):.3f}")
        st.write(f"Peaceman WI geom: {sim.peaceman_well_index(well.cell_index):.5f}")

        st.subheader("History Plots")
        fig1 = plt.figure(figsize=(8, 5))
        plt.plot(history.time_days, history.avg_pressure_psia, label="Average Pressure")
        plt.plot(history.time_days, history.well_pressure_psia, label="Well-Cell Pressure")
        plt.xlabel("Time (days)")
        plt.ylabel("Pressure (psia)")
        plt.title("Pressure Depletion History")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8, 5))
        plt.plot(history.time_days, history.well_rate_total, label="Damaged")
        plt.plot(history.time_days, history.well_rate_undamaged, label="Undamaged")
        plt.xlabel("Time (days)")
        plt.ylabel("Well Rate (lbmol/day)")
        plt.title("Productivity Loss from Condensate Banking")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(8, 5))
        plt.plot(history.time_days, history.well_damage_factor, label="Damage Factor")
        plt.plot(history.time_days, history.productivity_loss_fraction, label="Productivity Loss Fraction")
        plt.xlabel("Time (days)")
        plt.ylabel("Dimensionless")
        plt.title("Condensate-Bank Impairment")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("Spatial Diagnostics")
        fig4 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["pressure_psia"])
        plt.xlabel("Distance (ft)")
        plt.ylabel("Pressure (psia)")
        plt.title("Final Pressure Profile")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        fig5 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["dropout_indicator"])
        plt.xlabel("Distance (ft)")
        plt.ylabel("Dropout Indicator")
        plt.title("Final Condensate Dropout Profile")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        fig6 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["krg"], label="krg")
        plt.plot(final_diag["x_ft"], final_diag["krl"], label="krl")
        plt.xlabel("Distance (ft)")
        plt.ylabel("Relative Permeability")
        plt.title("Final Relative Permeability Profile")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        st.subheader("Download Outputs")
        with open(result["history_csv"], "rb") as f:
            st.download_button("Download history.csv", f.read(), file_name="history.csv", mime="text/csv")
        with open(result["spatial_csv"], "rb") as f:
            st.download_button("Download final_spatial_diagnostics.csv", f.read(), file_name="final_spatial_diagnostics.csv", mime="text/csv")
        with open(result["summary_csv"], "rb") as f:
            st.download_button("Download chapter4_summary.csv", f.read(), file_name="chapter4_summary.csv", mime="text/csv")
    else:
        st.info("Click 'Run Simulation' to execute the model and render results.")


display_streamlit_app()

