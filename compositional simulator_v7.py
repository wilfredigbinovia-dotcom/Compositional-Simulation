"""
1D Compositional Gas-Condensate-Water Reservoir Simulator (Prototype)

Scope
-----
Research/thesis-grade 1D prototype for compositional gas-condensate-water
simulation with transparent, extensible physics.

Capabilities
------------
- 1D Cartesian grid with near-well refinement
- Multi-component hydrocarbon composition per cell
- Peng-Robinson EOS utilities
- Wilson K-value initialization
- Isothermal two-phase hydrocarbon flash
- Explicit aqueous phase tracking
- Three-phase Corey-style relative permeability
- Pressure diffusion-like update
- Peaceman producer well
- Natural depletion, gas cycling, and lean gas injection scenarios
- Gas injection input in Mscf/day or MMscf/day
- Streamlit UI with plots and CSV export

Current limitations
-------------------
- Prototype formulation, not fully implicit
- Simplified pressure/transport coupling
- No capillary pressure
- No gravity
- Approximate viscosity model
- No dissolved gas in water
- No salinity/aquifer model
- No advanced EOS tuning or volume translation
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

# Standard conditions for reporting
P_STD_PSIA = 14.7
T_STD_R = 519.67  # 60 F
SCF_PER_LBMOL = R * T_STD_R / P_STD_PSIA
MM_PER_ONE = 1.0e-6

# Liquid reporting assumptions
CONDENSATE_DENSITY_LBFT3 = 50.0
WATER_DENSITY_LBFT3 = 62.4
FT3_PER_STB = 5.614583333333333

MW_WATER = 18.01528  # lbm/lbmol
WATER_VISCOSITY_CP = 0.5


# -----------------------------------------------------------------------------
# Component and fluid definitions
# -----------------------------------------------------------------------------

@dataclass
class Component:
    name: str
    Tc: float
    Pc: float
    omega: float
    Mw: float


@dataclass
class FluidModel:
    components: List[Component]
    kij: np.ndarray

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
            term2 = -math.log(max(Z - B, 1e-12))
            term3 = A / max(2.0 * sqrt2 * B, 1e-12)
            term4 = (2.0 * sum_aij[i] / max(am, 1e-12)) - (bi[i] / max(bm, 1e-12))
            ratio = (Z + (1.0 + sqrt2) * B) / max(Z + (1.0 - sqrt2) * B, 1e-12)
            term5 = math.log(max(ratio, 1e-12))
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
            flo = self.rachford_rice(lo, z, K)
            if flo * fm < 0.0:
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
            return {"state": "liquid", "beta": 0.0, "x": z.copy(), "y": z.copy(), "K": K}
        if beta >= 1.0 - 1e-10:
            return {"state": "vapor", "beta": 1.0, "x": z.copy(), "y": z.copy(), "K": K}

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
                return {"state": "two_phase", "beta": beta, "x": x, "y": y, "K": K}

        denom = 1.0 + beta * (K - 1.0)
        x = z / denom
        y = K * x
        x /= np.sum(x)
        y /= np.sum(y)
        return {"state": "two_phase", "beta": beta, "x": x, "y": y, "K": K}


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


def build_near_well_lgr(nx: int, length_ft: float, refined_cells: int = 8, min_dx_ft: float = 20.0, growth: float = 1.35) -> np.ndarray:
    refined_cells = max(2, min(refined_cells, nx - 1))
    coarse_cells = nx - refined_cells
    refined = np.array([min_dx_ft * (growth ** i) for i in range(refined_cells)], dtype=float)[::-1]
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
class RelPermParams:
    Sgc: float = 0.05
    Swc: float = 0.20
    Sorw: float = 0.20
    Sorg: float = 0.05
    krg0: float = 1.0
    kro0: float = 1.0
    krw0: float = 0.4
    ng: float = 2.0
    no: float = 2.0
    nw: float = 2.0


@dataclass
class Well:
    cell_index: int
    bhp_psia: float
    productivity_index: float
    rw_ft: float = 0.35
    skin: float = 0.0


@dataclass
class InjectorWell:
    cell_index: int
    rate_lbmol_day: float
    injection_composition: np.ndarray
    rate_field_input: float = 0.0
    rate_field_unit: str = "MMscf/day"


@dataclass
class ScenarioConfig:
    name: str
    injector: InjectorWell | None = None


@dataclass
class SimulationState:
    pressure: np.ndarray
    z: np.ndarray
    nt: np.ndarray
    nw: np.ndarray
    sw: np.ndarray


@dataclass
class SimulationHistory:
    time_days: List[float] = field(default_factory=list)
    avg_pressure_psia: List[float] = field(default_factory=list)
    min_pressure_psia: List[float] = field(default_factory=list)
    well_pressure_psia: List[float] = field(default_factory=list)

    well_rate_total: List[float] = field(default_factory=list)
    well_rate_undamaged: List[float] = field(default_factory=list)
    productivity_loss_fraction: List[float] = field(default_factory=list)
    well_damage_factor: List[float] = field(default_factory=list)
    well_effective_wi: List[float] = field(default_factory=list)

    well_gas_fraction: List[float] = field(default_factory=list)
    well_oil_fraction: List[float] = field(default_factory=list)
    well_water_fraction: List[float] = field(default_factory=list)

    well_dropout_indicator: List[float] = field(default_factory=list)

    well_krg: List[float] = field(default_factory=list)
    well_kro: List[float] = field(default_factory=list)
    well_krw: List[float] = field(default_factory=list)

    injector_rate_total_lbmol_day: List[float] = field(default_factory=list)
    injector_rate_input_field: List[float] = field(default_factory=list)

    gas_rate_mmscf_day: List[float] = field(default_factory=list)
    condensate_rate_stb_day: List[float] = field(default_factory=list)
    water_rate_stb_day: List[float] = field(default_factory=list)

    cum_gas_bscf: List[float] = field(default_factory=list)
    cum_condensate_mstb: List[float] = field(default_factory=list)
    cum_water_mstb: List[float] = field(default_factory=list)

    avg_sw: List[float] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Utility correlations
# -----------------------------------------------------------------------------

def phase_viscosity_cp(phase: str, comp: np.ndarray) -> float:
    if phase == "v":
        return 0.02 + 0.03 * float(np.sum(comp))
    if phase == "l":
        heaviness = np.linspace(0.5, 1.5, len(comp)) @ comp
        return 0.15 + 0.25 * heaviness
    raise ValueError("phase must be 'v' or 'l'")


def three_phase_relperm(Sg: float, So: float, Sw: float, params: RelPermParams) -> Tuple[float, float, float]:
    Sg = float(np.clip(Sg, 0.0, 1.0))
    So = float(np.clip(So, 0.0, 1.0))
    Sw = float(np.clip(Sw, 0.0, 1.0))

    total = Sg + So + Sw
    if total > 1e-12:
        Sg /= total
        So /= total
        Sw /= total

    denom_g = max(1.0 - params.Sgc - params.Swc, 1e-10)
    Sg_eff = np.clip((Sg - params.Sgc) / denom_g, 0.0, 1.0)

    denom_w = max(1.0 - params.Swc - params.Sorw, 1e-10)
    Sw_eff = np.clip((Sw - params.Swc) / denom_w, 0.0, 1.0)

    so_mobile_max = max(1.0 - params.Swc - params.Sorg, 1e-10)
    So_eff = np.clip(So / so_mobile_max, 0.0, 1.0)

    krg = params.krg0 * (Sg_eff ** params.ng)
    kro = params.kro0 * (So_eff ** params.no)
    krw = params.krw0 * (Sw_eff ** params.nw)

    return float(krg), float(kro), float(krw)


def liquid_dropout_fraction(z: np.ndarray, beta: float) -> float:
    heaviness_weights = np.linspace(0.0, 1.0, len(z))
    heaviness = float(np.dot(heaviness_weights, z))
    return max((1.0 - beta) * heaviness, 0.0)


def condensate_bank_damage_factor(dropout: float, kro: float, damage_strength: float = 8.0, critical_dropout: float = 0.03) -> float:
    excess = max(dropout - critical_dropout, 0.0)
    liquid_penalty = max(1.0 - kro, 0.0)
    damage = math.exp(-damage_strength * excess * (0.5 + 0.5 * liquid_penalty))
    return float(np.clip(damage, 0.05, 1.0))


def productivity_loss_fraction(q_actual: float, q_undamaged: float) -> float:
    if q_undamaged <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - q_actual / q_undamaged, 0.0, 1.0))


def z_to_density_lbmol_ft3(P: float, T: float, Z: float) -> float:
    return P / (Z * R * T)


def gas_lbmol_day_to_mmscf_day(rate_lbmol_day: float) -> float:
    return rate_lbmol_day * SCF_PER_LBMOL * MM_PER_ONE


def gas_field_rate_to_lbmol_day(rate_value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "mscf/day":
        scf_day = rate_value * 1.0e3
    elif unit == "mmscf/day":
        scf_day = rate_value * 1.0e6
    else:
        raise ValueError(f"Unsupported gas rate unit: {unit}")
    return scf_day / SCF_PER_LBMOL


def liquid_lbmol_day_to_stb_day(rate_lbmol_day: float, liquid_comp: np.ndarray, mw_components: np.ndarray, density_lbft3: float = CONDENSATE_DENSITY_LBFT3) -> float:
    liquid_comp = np.asarray(liquid_comp, dtype=float)
    liquid_comp = liquid_comp / max(np.sum(liquid_comp), 1e-12)
    mw_mix = float(np.dot(liquid_comp, mw_components))
    mass_rate_lbm_day = rate_lbmol_day * mw_mix
    vol_rate_ft3_day = mass_rate_lbm_day / max(density_lbft3, 1e-12)
    return vol_rate_ft3_day / FT3_PER_STB


def water_lbmol_day_to_stb_day(rate_lbmol_day: float) -> float:
    mass_rate_lbm_day = rate_lbmol_day * MW_WATER
    vol_rate_ft3_day = mass_rate_lbm_day / WATER_DENSITY_LBFT3
    return vol_rate_ft3_day / FT3_PER_STB


def water_moles_from_saturation(sw: np.ndarray, pv: np.ndarray) -> np.ndarray:
    water_vol_ft3 = sw * pv
    water_mass_lbm = water_vol_ft3 * WATER_DENSITY_LBFT3
    return water_mass_lbm / MW_WATER


def saturation_from_water_moles(nw: np.ndarray, pv: np.ndarray) -> np.ndarray:
    water_mass_lbm = nw * MW_WATER
    water_vol_ft3 = water_mass_lbm / WATER_DENSITY_LBFT3
    sw = water_vol_ft3 / np.maximum(pv, 1e-12)
    return np.clip(sw, 0.0, 0.95)


def convert_time_to_days(value: float, unit: str) -> float:
    factors = {"days": 1.0, "months": 30.4375, "years": 365.25}
    return value * factors[unit.lower()]


def format_time_axis(time_days: List[float], unit: str) -> np.ndarray:
    factors = {"days": 1.0, "months": 30.4375, "years": 365.25}
    return np.asarray(time_days, dtype=float) / factors[unit.lower()]


# -----------------------------------------------------------------------------
# Main simulator
# -----------------------------------------------------------------------------

class CompositionalSimulator1D:
    def __init__(
        self,
        grid: Grid1D,
        rock: Rock,
        fluid: FluidModel,
        eos: PengRobinsonEOS,
        flash: FlashCalculator,
        well: Well,
        temperature_R: float,
        scenario: ScenarioConfig | None = None,
        relperm: RelPermParams | None = None,
    ):
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.eos = eos
        self.flash = flash
        self.well = well
        self.T = temperature_R
        self.nc = fluid.nc
        self.scenario = scenario or ScenarioConfig(name="natural_depletion", injector=None)
        self.relperm = relperm or RelPermParams()
        self.last_produced_gas_composition = None

        self.phi = rock.porosity
        self.k = rock.permeability_md
        self.pv = np.array([self.grid.bulk_volume(i) * self.phi for i in range(self.grid.nx)], dtype=float)
        _, _, _, self.mw_components = self.fluid.critical_arrays()

    def initialize_state(self, p_init_psia: float, z_init: np.ndarray, sw_init: float = 0.20) -> SimulationState:
        z_init = np.asarray(z_init, dtype=float)
        z_init = z_init / np.sum(z_init)

        pressure = np.full(self.grid.nx, p_init_psia, dtype=float)
        z = np.tile(z_init, (self.grid.nx, 1))

        sw = np.full(self.grid.nx, sw_init, dtype=float)
        sw = np.clip(sw, 0.0, 0.95)

        Zg = self.eos.z_factor(z_init, p_init_psia, self.T, phase="v")
        molar_density = z_to_density_lbmol_ft3(p_init_psia, self.T, Zg)

        hc_pv = self.pv * (1.0 - sw)
        nt = molar_density * hc_pv
        nw = water_moles_from_saturation(sw, self.pv)

        return SimulationState(pressure=pressure, z=z, nt=nt, nw=nw, sw=sw)

    def cell_flash(self, state: SimulationState, i: int) -> Dict[str, np.ndarray | float | str]:
        return self.flash.flash(state.z[i], state.pressure[i], self.T)

    def phase_mobility_data(self, state: SimulationState, i: int) -> Dict[str, float]:
        fl = self.cell_flash(state, i)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])

        mu_o = phase_viscosity_cp("l", x)
        mu_g = phase_viscosity_cp("v", y)
        mu_w = WATER_VISCOSITY_CP

        Sw = float(np.clip(state.sw[i], 0.0, 0.95))
        Shc = max(1.0 - Sw, 0.0)
        Sg = Shc * beta
        So = Shc * (1.0 - beta)

        krg, kro, krw = three_phase_relperm(Sg, So, Sw, self.relperm)

        lam_g = krg / max(mu_g, 1e-8)
        lam_o = kro / max(mu_o, 1e-8)
        lam_w = krw / max(mu_w, 1e-8)
        lam_t = lam_g + lam_o + lam_w

        return {
            "Sg": Sg,
            "So": So,
            "Sw": Sw,
            "krg": krg,
            "kro": kro,
            "krw": krw,
            "mu_g": mu_g,
            "mu_o": mu_o,
            "mu_w": mu_w,
            "lam_g": lam_g,
            "lam_o": lam_o,
            "lam_w": lam_w,
            "lam_hc": lam_g + lam_o,
            "lam_t": max(lam_t, 1e-10),
        }

    def component_flux_between(self, state: SimulationState, i: int, j: int) -> np.ndarray:
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
        mob = self.phase_mobility_data(state, upstream)
        lam = mob["lam_hc"]

        distance = self.grid.interface_distance(i, j)
        trans = self.k * self.grid.area_ft2 / distance
        q_total = trans * lam * dp * 1e-5
        comp_frac = beta * y + (1.0 - beta) * x
        return q_total * comp_frac

    def water_flux_between(self, state: SimulationState, i: int, j: int) -> float:
        p_i = state.pressure[i]
        p_j = state.pressure[j]
        dp = p_i - p_j
        if abs(dp) < 1e-12:
            return 0.0

        upstream = i if dp >= 0.0 else j
        mob = self.phase_mobility_data(state, upstream)
        lam_w = mob["lam_w"]

        distance = self.grid.interface_distance(i, j)
        trans = self.k * self.grid.area_ft2 / distance
        qw = trans * lam_w * dp * 1e-5
        return float(qw)

    def peaceman_well_index(self, i: int) -> float:
        dx = self.grid.cell_width(i)
        dy = self.grid.width_ft
        h = self.grid.thickness_ft
        rw = max(self.well.rw_ft, 1e-4)
        skin = self.well.skin
        re = 0.28 * math.sqrt(dx * dx + dy * dy) / ((dy / dx) ** 0.5 + (dx / dy) ** 0.5)
        re = max(re, 1.01 * rw)
        wi_geom = 0.00708 * self.k * h / max(math.log(re / rw) + skin, 1e-8)
        return self.well.productivity_index * wi_geom

    def active_injector(self) -> InjectorWell | None:
        return None if self.scenario is None else self.scenario.injector

    def get_injection_composition(self, state: SimulationState) -> np.ndarray | None:
        injector = self.active_injector()
        if injector is None:
            return None
        if self.scenario.name == "natural_depletion":
            return None
        if self.scenario.name == "lean_gas_injection":
            zinj = np.asarray(injector.injection_composition, dtype=float)
            return zinj / np.sum(zinj)
        if self.scenario.name == "gas_cycling":
            if self.last_produced_gas_composition is not None:
                zinj = np.asarray(self.last_produced_gas_composition, dtype=float)
                return zinj / np.sum(zinj)
            wi = self.well.cell_index
            fl = self.cell_flash(state, wi)
            y = np.asarray(fl["y"], dtype=float)
            return y / np.sum(y)
        raise ValueError(f"Unknown scenario: {self.scenario.name}")

    def injector_source(self, state: SimulationState) -> tuple[int | None, np.ndarray | None]:
        injector = self.active_injector()
        if injector is None:
            return None, None
        zinj = self.get_injection_composition(state)
        if zinj is None:
            return None, None
        source = injector.rate_lbmol_day * zinj
        return injector.cell_index, source

    def well_sink(
        self, state: SimulationState
    ) -> Tuple[np.ndarray, float, float, float, float, float, float, float, float, float, float, float, float, float]:
        i = self.well.cell_index
        pwf = self.well.bhp_psia
        pr = state.pressure[i]
        dp = max(pr - pwf, 0.0)

        fl = self.cell_flash(state, i)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])
        self.last_produced_gas_composition = y.copy()

        mob = self.phase_mobility_data(state, i)
        lam_g = mob["lam_g"]
        lam_o = mob["lam_o"]
        lam_w = mob["lam_w"]
        lam_t = mob["lam_t"]

        krg = mob["krg"]
        kro = mob["kro"]
        krw = mob["krw"]

        wi_geom = self.peaceman_well_index(i)
        q_undamaged = wi_geom * lam_t * dp * 1e-2

        dropout = liquid_dropout_fraction(state.z[i], beta)
        damage_factor = condensate_bank_damage_factor(dropout, kro)
        wi_eff = wi_geom * damage_factor
        q_total = wi_eff * lam_t * dp * 1e-2

        fg = lam_g / max(lam_t, 1e-12)
        fo = lam_o / max(lam_t, 1e-12)
        fw = lam_w / max(lam_t, 1e-12)

        qg = q_total * fg
        qo = q_total * fo
        qw = q_total * fw

        hc_sink = qg * y + qo * x
        ploss = productivity_loss_fraction(q_total, q_undamaged)

        return hc_sink, qw, q_total, fg, fo, fw, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss

    def pressure_update(self, state: SimulationState, dt_days: float) -> np.ndarray:
        p = state.pressure.copy()
        p_new = p.copy()
        alpha = 0.15 * dt_days

        for i in range(self.grid.nx):
            left = p[i - 1] if i > 0 else p[i]
            right = p[i + 1] if i < self.grid.nx - 1 else p[i]
            lap = left - 2.0 * p[i] + right
            p_new[i] = p[i] + alpha * lap

        wi = self.well.cell_index
        p_new[wi] -= 0.05 * dt_days * max(p[wi] - self.well.bhp_psia, 0.0)

        injector = self.active_injector()
        if injector is not None and self.scenario.name != "natural_depletion":
            ii = injector.cell_index
            p_new[ii] += 0.02 * dt_days * max(injector.rate_lbmol_day, 0.0) / max(self.pv[ii], 1.0)

        return np.maximum(p_new, self.well.bhp_psia)

    def transport_update(
        self, state: SimulationState, dt_days: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, float, float, float, float]:
        ncomp = state.nt[:, None] * state.z
        ncomp_new = ncomp.copy()
        nw_new = state.nw.copy()

        for i in range(self.grid.nx - 1):
            hc_flux = self.component_flux_between(state, i, i + 1)
            w_flux = self.water_flux_between(state, i, i + 1)

            ncomp_new[i] -= dt_days * hc_flux
            ncomp_new[i + 1] += dt_days * hc_flux

            nw_new[i] -= dt_days * w_flux
            nw_new[i + 1] += dt_days * w_flux

        inj_cell, inj_source = self.injector_source(state)
        if inj_cell is not None and inj_source is not None:
            ncomp_new[inj_cell] += dt_days * inj_source

        hc_sink, qw, q_total, gas_frac, oil_frac, water_frac, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss = self.well_sink(state)
        wi = self.well.cell_index
        ncomp_new[wi] -= dt_days * hc_sink
        nw_new[wi] -= dt_days * qw

        ncomp_new = np.clip(ncomp_new, 1e-12, None)
        nw_new = np.clip(nw_new, 1e-12, None)

        nt_new = np.sum(ncomp_new, axis=1)
        z_new = ncomp_new / nt_new[:, None]
        sw_new = saturation_from_water_moles(nw_new, self.pv)

        return (
            nt_new, z_new, nw_new, sw_new, q_total, gas_frac, oil_frac, water_frac,
            dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss
        )

    def step(self, state: SimulationState, dt_days: float) -> Tuple[SimulationState, Dict[str, float]]:
        p_new = self.pressure_update(state, dt_days)
        state_mid = SimulationState(
            pressure=p_new, z=state.z.copy(), nt=state.nt.copy(), nw=state.nw.copy(), sw=state.sw.copy()
        )

        nt_new, z_new, nw_new, sw_new, q_total, gas_frac, oil_frac, water_frac, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss = self.transport_update(state_mid, dt_days)

        new_state = SimulationState(pressure=p_new, z=z_new, nt=nt_new, nw=nw_new, sw=sw_new)
        summary = {
            "well_rate_total": q_total,
            "well_gas_fraction": gas_frac,
            "well_oil_fraction": oil_frac,
            "well_water_fraction": water_frac,
            "well_dropout_indicator": dropout,
            "well_krg": krg,
            "well_kro": kro,
            "well_krw": krw,
            "well_damage_factor": damage_factor,
            "well_effective_wi": wi_eff,
            "well_rate_undamaged": q_undamaged,
            "productivity_loss_fraction": ploss,
            "avg_pressure": float(np.mean(p_new)),
            "min_pressure": float(np.min(p_new)),
            "avg_sw": float(np.mean(sw_new)),
        }
        return new_state, summary

    def run(self, state0: SimulationState, t_end_days: float, dt_days: float) -> Tuple[SimulationState, SimulationHistory]:
        history = SimulationHistory()
        state = state0
        t = 0.0

        cum_gas_scf = 0.0
        cum_condensate_stb = 0.0
        cum_water_stb = 0.0

        while t <= t_end_days + 1e-12:
            history.time_days.append(t)
            history.avg_pressure_psia.append(float(np.mean(state.pressure)))
            history.min_pressure_psia.append(float(np.min(state.pressure)))
            history.avg_sw.append(float(np.mean(state.sw)))

            hc_sink, qw, q_total, gas_frac, oil_frac, water_frac, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss = self.well_sink(state)

            history.well_rate_total.append(float(q_total))
            history.well_gas_fraction.append(float(gas_frac))
            history.well_oil_fraction.append(float(oil_frac))
            history.well_water_fraction.append(float(water_frac))
            history.well_dropout_indicator.append(float(dropout))
            history.well_krg.append(float(krg))
            history.well_kro.append(float(kro))
            history.well_krw.append(float(krw))
            history.well_pressure_psia.append(float(state.pressure[self.well.cell_index]))
            history.well_damage_factor.append(float(damage_factor))
            history.well_effective_wi.append(float(wi_eff))
            history.well_rate_undamaged.append(float(q_undamaged))
            history.productivity_loss_fraction.append(float(ploss))

            injector = self.active_injector()
            if injector is None or self.scenario.name == "natural_depletion":
                history.injector_rate_total_lbmol_day.append(0.0)
                history.injector_rate_input_field.append(0.0)
            else:
                history.injector_rate_total_lbmol_day.append(float(injector.rate_lbmol_day))
                history.injector_rate_input_field.append(float(injector.rate_field_input))

            qg_lbmol_day = q_total * gas_frac
            qo_lbmol_day = q_total * oil_frac
            qw_lbmol_day = q_total * water_frac

            fl = self.cell_flash(state, self.well.cell_index)
            x = np.asarray(fl["x"], dtype=float)

            gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(qg_lbmol_day)
            condensate_rate_stb_day = liquid_lbmol_day_to_stb_day(qo_lbmol_day, x, self.mw_components, CONDENSATE_DENSITY_LBFT3)
            water_rate_stb_day = water_lbmol_day_to_stb_day(qw_lbmol_day)

            history.gas_rate_mmscf_day.append(float(gas_rate_mmscf_day))
            history.condensate_rate_stb_day.append(float(condensate_rate_stb_day))
            history.water_rate_stb_day.append(float(water_rate_stb_day))

            cum_gas_scf += gas_rate_mmscf_day * 1.0e6 * dt_days
            cum_condensate_stb += condensate_rate_stb_day * dt_days
            cum_water_stb += water_rate_stb_day * dt_days

            history.cum_gas_bscf.append(float(cum_gas_scf / 1.0e9))
            history.cum_condensate_mstb.append(float(cum_condensate_stb / 1.0e3))
            history.cum_water_mstb.append(float(cum_water_stb / 1.0e3))

            state, _ = self.step(state, dt_days)
            t += dt_days

        return state, history

    def spatial_diagnostics(self, state: SimulationState) -> Dict[str, np.ndarray]:
        xcoord = np.array([self.grid.cell_center(i) for i in range(self.grid.nx)], dtype=float)

        Sg_arr = np.zeros(self.grid.nx)
        So_arr = np.zeros(self.grid.nx)
        Sw_arr = np.zeros(self.grid.nx)

        dropout_arr = np.zeros(self.grid.nx)
        krg_arr = np.zeros(self.grid.nx)
        kro_arr = np.zeros(self.grid.nx)
        krw_arr = np.zeros(self.grid.nx)

        gas_mob_arr = np.zeros(self.grid.nx)
        oil_mob_arr = np.zeros(self.grid.nx)
        water_mob_arr = np.zeros(self.grid.nx)

        for i in range(self.grid.nx):
            fl = self.cell_flash(state, i)
            beta = float(fl["beta"])
            mob = self.phase_mobility_data(state, i)

            Sg_arr[i] = mob["Sg"]
            So_arr[i] = mob["So"]
            Sw_arr[i] = mob["Sw"]
            dropout_arr[i] = liquid_dropout_fraction(state.z[i], beta)
            krg_arr[i] = mob["krg"]
            kro_arr[i] = mob["kro"]
            krw_arr[i] = mob["krw"]
            gas_mob_arr[i] = mob["lam_g"]
            oil_mob_arr[i] = mob["lam_o"]
            water_mob_arr[i] = mob["lam_w"]

        return {
            "x_ft": xcoord,
            "pressure_psia": state.pressure.copy(),
            "Sg": Sg_arr,
            "So": So_arr,
            "Sw": Sw_arr,
            "dropout_indicator": dropout_arr,
            "krg": krg_arr,
            "kro": kro_arr,
            "krw": krw_arr,
            "gas_mobility": gas_mob_arr,
            "oil_mobility": oil_mob_arr,
            "water_mobility": water_mob_arr,
        }


# -----------------------------------------------------------------------------
# Example fluid
# -----------------------------------------------------------------------------

def build_example_fluid() -> FluidModel:
    comps = [
        Component("H2S", Tc=672.4, Pc=1306.0, omega=0.100, Mw=34.08),
        Component("CO2", Tc=547.6, Pc=1071.0, omega=0.225, Mw=44.01),
        Component("N2", Tc=227.2, Pc=492.3, omega=0.037, Mw=28.01),
        Component("C1", Tc=343.0, Pc=667.8, omega=0.011, Mw=16.04),
        Component("C2", Tc=549.6, Pc=707.8, omega=0.099, Mw=30.07),
        Component("C3", Tc=665.7, Pc=616.0, omega=0.152, Mw=44.10),
        Component("iC4", Tc=734.5, Pc=529.1, omega=0.184, Mw=58.12),
        Component("nC4", Tc=765.3, Pc=550.7, omega=0.200, Mw=58.12),
        Component("C5", Tc=845.4, Pc=488.6, omega=0.251, Mw=72.15),
        Component("C6", Tc=913.4, Pc=436.9, omega=0.301, Mw=86.18),
        Component("C7+", Tc=1030.0, Pc=397.0, omega=0.420, Mw=120.00),
    ]
    nc = len(comps)
    kij = np.zeros((nc, nc), dtype=float)

    for i in range(nc):
        for j in range(nc):
            if i == j:
                kij[i, j] = 0.0
            else:
                if i < 3 and j < 3:
                    kij[i, j] = 0.02
                elif i < 3 or j < 3:
                    kij[i, j] = 0.04
                else:
                    kij[i, j] = 0.005 * abs(i - j)

    return FluidModel(components=comps, kij=kij)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

def export_history_csv(history: SimulationHistory, filepath: str) -> None:
    rows = zip(
        history.time_days,
        history.avg_pressure_psia,
        history.min_pressure_psia,
        history.well_pressure_psia,
        history.avg_sw,
        history.well_rate_total,
        history.well_rate_undamaged,
        history.productivity_loss_fraction,
        history.well_damage_factor,
        history.well_effective_wi,
        history.injector_rate_total_lbmol_day,
        history.injector_rate_input_field,
        history.well_gas_fraction,
        history.well_oil_fraction,
        history.well_water_fraction,
        history.well_dropout_indicator,
        history.well_krg,
        history.well_kro,
        history.well_krw,
        history.gas_rate_mmscf_day,
        history.condensate_rate_stb_day,
        history.water_rate_stb_day,
        history.cum_gas_bscf,
        history.cum_condensate_mstb,
        history.cum_water_mstb,
    )
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_days", "avg_pressure_psia", "min_pressure_psia", "well_pressure_psia", "avg_sw",
            "well_rate_total_lbmol_day", "well_rate_undamaged_lbmol_day", "productivity_loss_fraction",
            "well_damage_factor", "well_effective_wi", "injector_rate_total_lbmol_day",
            "injector_rate_input_field", "well_gas_fraction", "well_oil_fraction", "well_water_fraction",
            "well_dropout_indicator", "well_krg", "well_kro", "well_krw",
            "gas_rate_mmscf_day", "condensate_rate_stb_day", "water_rate_stb_day",
            "cum_gas_bscf", "cum_condensate_mstb", "cum_water_mstb"
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
        ["Average Water Saturation (-)", initial_value(history.avg_sw), max(history.avg_sw), history.time_days[int(np.argmax(history.avg_sw))], final_value(history.avg_sw)],
        ["Dropout Indicator (-)", initial_value(history.well_dropout_indicator), d_peak, d_peak_t, final_value(history.well_dropout_indicator)],
        ["Damage Factor (-)", initial_value(history.well_damage_factor), df_peak, df_peak_t, final_value(history.well_damage_factor)],
        ["Productivity Loss Fraction (-)", initial_value(history.productivity_loss_fraction), pl_peak, pl_peak_t, final_value(history.productivity_loss_fraction)],
        ["Gas Rate (MMscf/day)", initial_value(history.gas_rate_mmscf_day), max(history.gas_rate_mmscf_day), history.time_days[int(np.argmax(history.gas_rate_mmscf_day))], final_value(history.gas_rate_mmscf_day)],
        ["Condensate Rate (STB/day)", initial_value(history.condensate_rate_stb_day), max(history.condensate_rate_stb_day), history.time_days[int(np.argmax(history.condensate_rate_stb_day))], final_value(history.condensate_rate_stb_day)],
        ["Water Rate (STB/day)", initial_value(history.water_rate_stb_day), max(history.water_rate_stb_day), history.time_days[int(np.argmax(history.water_rate_stb_day))], final_value(history.water_rate_stb_day)],
        ["Cumulative Gas (Bscf)", initial_value(history.cum_gas_bscf), max(history.cum_gas_bscf), history.time_days[int(np.argmax(history.cum_gas_bscf))], final_value(history.cum_gas_bscf)],
        ["Cumulative Condensate (MSTB)", initial_value(history.cum_condensate_mstb), max(history.cum_condensate_mstb), history.time_days[int(np.argmax(history.cum_condensate_mstb))], final_value(history.cum_condensate_mstb)],
        ["Cumulative Water (MSTB)", initial_value(history.cum_water_mstb), max(history.cum_water_mstb), history.time_days[int(np.argmax(history.cum_water_mstb))], final_value(history.cum_water_mstb)],
    ]


def export_chapter4_summary_csv(history: SimulationHistory, filepath: str) -> None:
    rows = generate_chapter4_summary(history)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Initial Value", "Peak Value", "Time of Peak (days)", "Final Value"])
        writer.writerows(rows)


def plot_history(history: SimulationHistory, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def save(fig_name: str):
        plt.tight_layout()
        plt.savefig(output_dir / fig_name, dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.avg_pressure_psia, label="Average Pressure")
    plt.plot(history.time_days, history.well_pressure_psia, label="Well-Cell Pressure")
    plt.xlabel("Time (days)")
    plt.ylabel("Pressure (psia)")
    plt.title("Pressure Depletion History")
    plt.legend()
    save("pressure_history.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.gas_rate_mmscf_day, label="Gas Rate")
    plt.xlabel("Time (days)")
    plt.ylabel("Gas Rate (MMscf/day)")
    plt.title("Gas Production Rate History")
    plt.legend()
    save("gas_rate_history.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.condensate_rate_stb_day, label="Condensate Rate")
    plt.plot(history.time_days, history.water_rate_stb_day, label="Water Rate")
    plt.xlabel("Time (days)")
    plt.ylabel("Liquid Rate (STB/day)")
    plt.title("Condensate and Water Production Rates")
    plt.legend()
    save("liquid_rate_history.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.cum_gas_bscf, label="Cumulative Gas")
    plt.xlabel("Time (days)")
    plt.ylabel("Cumulative Gas (Bscf)")
    plt.title("Cumulative Gas Production")
    plt.legend()
    save("cum_gas_history.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.cum_condensate_mstb, label="Cumulative Condensate")
    plt.plot(history.time_days, history.cum_water_mstb, label="Cumulative Water")
    plt.xlabel("Time (days)")
    plt.ylabel("Cumulative Liquid (MSTB)")
    plt.title("Cumulative Condensate and Water Production")
    plt.legend()
    save("cum_liquid_history.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.avg_sw)
    plt.xlabel("Time (days)")
    plt.ylabel("Average Sw")
    plt.title("Average Water Saturation History")
    save("average_sw_history.png")

    plt.figure(figsize=(8, 5))
    plt.plot(history.time_days, history.well_damage_factor, label="Damage Factor")
    plt.plot(history.time_days, history.productivity_loss_fraction, label="Productivity Loss Fraction")
    plt.xlabel("Time (days)")
    plt.ylabel("Dimensionless")
    plt.title("Condensate-Bank Impairment")
    plt.legend()
    save("impairment_history.png")


def plot_spatial_diagnostics(diag: Dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = diag["x_ft"]

    def save(fig_name: str):
        plt.tight_layout()
        plt.savefig(output_dir / fig_name, dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["pressure_psia"])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Pressure (psia)")
    plt.title("Final Pressure Profile")
    save("pressure_profile.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["Sg"], label="Sg")
    plt.plot(x, diag["So"], label="So")
    plt.plot(x, diag["Sw"], label="Sw")
    plt.xlabel("Distance (ft)")
    plt.ylabel("Saturation")
    plt.title("Final Saturation Profile")
    plt.legend()
    save("saturation_profile.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["dropout_indicator"])
    plt.xlabel("Distance (ft)")
    plt.ylabel("Dropout Indicator")
    plt.title("Final Condensate Dropout Profile")
    save("dropout_profile.png")

    plt.figure(figsize=(8, 5))
    plt.plot(x, diag["krg"], label="krg")
    plt.plot(x, diag["kro"], label="kro")
    plt.plot(x, diag["krw"], label="krw")
    plt.xlabel("Distance (ft)")
    plt.ylabel("Relative Permeability")
    plt.title("Final Three-Phase Relative Permeability Profile")
    plt.legend()
    save("relperm_profile.png")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_example(
    scenario_name: str = "natural_depletion",
    injection_rate_value: float = 0.0,
    injection_rate_unit: str = "MMscf/day",
    lean_gas_composition: np.ndarray | None = None,
    nx: int = 40,
    length_ft: float = 4000.0,
    area_ft2: float = 2500.0,
    thickness_ft: float = 50.0,
    refined_cells: int = 10,
    min_dx_ft: float = 12.0,
    growth: float = 1.45,
    porosity: float = 0.18,
    permeability_md: float = 8.0,
    bhp_psia: float = 3000.0,
    productivity_index: float = 1.0,
    rw_ft: float = 0.35,
    skin: float = 0.0,
    temperature_R: float = 680.0,
    p_init_psia: float = 5600.0,
    sw_init: float = 0.20,
    t_end_days: float = 240.0,
    dt_days: float = 2.0,
    z_init: np.ndarray | None = None,
    Sgc: float = 0.05,
    Swc: float = 0.20,
    Sorw: float = 0.20,
    Sorg: float = 0.05,
    krg0: float = 1.0,
    kro0: float = 1.0,
    krw0: float = 0.4,
    ng: float = 2.0,
    no: float = 2.0,
    nw: float = 2.0,
) -> Dict[str, object]:
    fluid = build_example_fluid()
    eos = PengRobinsonEOS(fluid)
    flash = FlashCalculator(eos)

    if z_init is None:
        z_init = np.array([0.005, 0.02, 0.01, 0.78, 0.07, 0.04, 0.015, 0.015, 0.02, 0.015, 0.01], dtype=float)
    else:
        z_init = np.asarray(z_init, dtype=float)
        z_init = z_init / np.sum(z_init)

    if lean_gas_composition is None:
        lean_gas_composition = np.array([0.0, 0.02, 0.01, 0.94, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    lean_gas_composition = np.asarray(lean_gas_composition, dtype=float)
    lean_gas_composition = lean_gas_composition / np.sum(lean_gas_composition)

    refined_cells = max(2, min(refined_cells, nx - 1))
    dx_array = build_near_well_lgr(nx, length_ft, refined_cells, min_dx_ft, growth)

    grid = Grid1D(nx=nx, length_ft=length_ft, area_ft2=area_ft2, thickness_ft=thickness_ft, dx_array=dx_array)
    rock = Rock(porosity=porosity, permeability_md=permeability_md)
    relperm = RelPermParams(Sgc=Sgc, Swc=Swc, Sorw=Sorw, Sorg=Sorg, krg0=krg0, kro0=kro0, krw0=krw0, ng=ng, no=no, nw=nw)
    well = Well(cell_index=nx - 1, bhp_psia=bhp_psia, productivity_index=productivity_index, rw_ft=rw_ft, skin=skin)

    injection_rate_lbmol_day = gas_field_rate_to_lbmol_day(injection_rate_value, injection_rate_unit)

    injector = None
    if scenario_name in ("gas_cycling", "lean_gas_injection"):
        inj_comp = z_init.copy() if scenario_name == "gas_cycling" else lean_gas_composition.copy()
        injector = InjectorWell(
            cell_index=0,
            rate_lbmol_day=injection_rate_lbmol_day,
            injection_composition=inj_comp,
            rate_field_input=injection_rate_value,
            rate_field_unit=injection_rate_unit,
        )

    scenario = ScenarioConfig(name=scenario_name, injector=injector)

    sim = CompositionalSimulator1D(
        grid=grid,
        rock=rock,
        fluid=fluid,
        eos=eos,
        flash=flash,
        well=well,
        temperature_R=temperature_R,
        scenario=scenario,
        relperm=relperm,
    )

    state0 = sim.initialize_state(p_init_psia=p_init_psia, z_init=z_init, sw_init=sw_init)
    final_state, history = sim.run(state0, t_end_days=t_end_days, dt_days=dt_days)
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
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Initial Value", "Peak Value", "Time of Peak (days)", "Final Value"])

    return {
        "fluid": fluid,
        "grid": grid,
        "well": well,
        "scenario": scenario,
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


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def display_streamlit_app() -> None:
    st.set_page_config(page_title="Compositional Reservoir Simulator", layout="wide")
    st.title("Compositional Reservoir Simulator")
    st.write(
        "1D compositional gas-condensate-water prototype with Peng-Robinson EOS, "
        "Peaceman well model, local-grid refinement, three-phase relative permeability, "
        "and scenario-based development analysis."
    )

    st.sidebar.header("Simulation Inputs")

    st.sidebar.subheader("Development Scenario")
    scenario_label = st.sidebar.selectbox("Scenario", ["Natural Depletion", "Gas Cycling", "Lean Gas Injection"], index=0)
    scenario_map = {
        "Natural Depletion": "natural_depletion",
        "Gas Cycling": "gas_cycling",
        "Lean Gas Injection": "lean_gas_injection",
    }
    scenario_name = scenario_map[scenario_label]

    injection_rate_value = st.sidebar.number_input("Gas injection rate", min_value=0.0, max_value=1.0e6, value=5.0, step=0.5)
    injection_rate_unit = st.sidebar.selectbox("Gas injection rate unit", ["Mscf/day", "MMscf/day"], index=1)

    nx = st.sidebar.number_input("Number of grid cells", min_value=10, max_value=200, value=40, step=1)
    length_ft = st.sidebar.number_input("Reservoir length (ft)", min_value=500.0, max_value=20000.0, value=4000.0, step=100.0)
    area_ft2 = st.sidebar.number_input("Cross-sectional area (ft²)", min_value=100.0, max_value=50000.0, value=2500.0, step=100.0)
    thickness_ft = st.sidebar.number_input("Reservoir thickness (ft)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)

    st.sidebar.subheader("Near-Well Refinement")
    refined_cells = st.sidebar.number_input("Refined cells near well", min_value=2, max_value=max(int(nx) - 1, 2), value=min(10, int(nx) - 1), step=1)
    min_dx_ft = st.sidebar.number_input("Minimum near-well cell size (ft)", min_value=1.0, max_value=500.0, value=12.0, step=1.0)
    growth = st.sidebar.number_input("LGR growth factor", min_value=1.05, max_value=3.0, value=1.45, step=0.05)

    st.sidebar.subheader("Rock and Well")
    porosity = st.sidebar.number_input("Porosity", min_value=0.01, max_value=0.50, value=0.18, step=0.01, format="%.2f")
    permeability_md = st.sidebar.number_input("Permeability (mD)", min_value=0.01, max_value=10000.0, value=8.0, step=1.0)
    bhp_psia = st.sidebar.number_input("Bottom-hole pressure (psia)", min_value=100.0, max_value=20000.0, value=3000.0, step=100.0)
    productivity_index = st.sidebar.number_input("Productivity index multiplier", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
    rw_ft = st.sidebar.number_input("Wellbore radius (ft)", min_value=0.05, max_value=5.0, value=0.35, step=0.05)
    skin = st.sidebar.number_input("Skin factor", min_value=-10.0, max_value=50.0, value=0.0, step=0.5)

    st.sidebar.subheader("Three-Phase Relative Permeability")
    Sgc = st.sidebar.number_input("Critical gas saturation, Sgc", 0.0, 0.95, 0.05, 0.01, format="%.2f")
    Swc = st.sidebar.number_input("Connate water saturation, Swc", 0.0, 0.95, 0.20, 0.01, format="%.2f")
    Sorw = st.sidebar.number_input("Residual condensate saturation to water, Sorw", 0.0, 0.95, 0.20, 0.01, format="%.2f")
    Sorg = st.sidebar.number_input("Residual condensate saturation to gas, Sorg", 0.0, 0.95, 0.05, 0.01, format="%.2f")

    krg0 = st.sidebar.number_input("Gas endpoint krg0", 0.0, 2.0, 1.0, 0.05, format="%.2f")
    kro0 = st.sidebar.number_input("Condensate endpoint kro0", 0.0, 2.0, 1.0, 0.05, format="%.2f")
    krw0 = st.sidebar.number_input("Water endpoint krw0", 0.0, 2.0, 0.4, 0.05, format="%.2f")

    ng = st.sidebar.number_input("Gas Corey exponent, ng", 0.1, 10.0, 2.0, 0.1, format="%.1f")
    no = st.sidebar.number_input("Condensate Corey exponent, no", 0.1, 10.0, 2.0, 0.1, format="%.1f")
    nw = st.sidebar.number_input("Water Corey exponent, nw", 0.1, 10.0, 2.0, 0.1, format="%.1f")

    if Sgc + Swc >= 1.0:
        st.error("Invalid inputs: Sgc + Swc must be less than 1.0.")
        return

    st.sidebar.subheader("Fluid and Time")
    temperature_R = st.sidebar.number_input("Temperature (°R)", min_value=400.0, max_value=1200.0, value=680.0, step=10.0)
    p_init_psia = st.sidebar.number_input("Initial reservoir pressure (psia)", min_value=100.0, max_value=20000.0, value=5600.0, step=100.0)
    sw_init = st.sidebar.number_input("Initial water saturation, Swi", min_value=0.0, max_value=0.95, value=0.20, step=0.01, format="%.2f")

    sim_length_value = st.sidebar.number_input("Simulation length", min_value=1.0, max_value=10000.0, value=240.0, step=10.0)
    sim_length_unit = st.sidebar.selectbox("Simulation length unit", ["days", "months", "years"], index=0)

    dt_value = st.sidebar.number_input("Time step", min_value=0.01, max_value=1000.0, value=2.0, step=0.5)
    dt_unit = st.sidebar.selectbox("Time step unit", ["days", "months", "years"], index=0)

    t_end_days = convert_time_to_days(float(sim_length_value), sim_length_unit)
    dt_days = convert_time_to_days(float(dt_value), dt_unit)

    if dt_days > t_end_days:
        st.sidebar.warning("Time step exceeds total simulation length.")

    st.sidebar.caption(f"Internal simulation length = {t_end_days:.3f} days")
    st.sidebar.caption(f"Internal time step = {dt_days:.3f} days")

    st.sidebar.subheader("Initial Hydrocarbon Composition")
    z_h2s = st.sidebar.number_input("H2S mole fraction", min_value=0.0, max_value=1.0, value=0.005, step=0.001, format="%.4f")
    z_co2 = st.sidebar.number_input("CO2 mole fraction", min_value=0.0, max_value=1.0, value=0.020, step=0.001, format="%.4f")
    z_n2 = st.sidebar.number_input("N2 mole fraction", min_value=0.0, max_value=1.0, value=0.010, step=0.001, format="%.4f")
    z_c1 = st.sidebar.number_input("C1 mole fraction", min_value=0.0, max_value=1.0, value=0.780, step=0.001, format="%.4f")
    z_c2 = st.sidebar.number_input("C2 mole fraction", min_value=0.0, max_value=1.0, value=0.070, step=0.001, format="%.4f")
    z_c3 = st.sidebar.number_input("C3 mole fraction", min_value=0.0, max_value=1.0, value=0.040, step=0.001, format="%.4f")
    z_ic4 = st.sidebar.number_input("iC4 mole fraction", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f")
    z_nc4 = st.sidebar.number_input("nC4 mole fraction", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f")
    z_c5 = st.sidebar.number_input("C5 mole fraction", min_value=0.0, max_value=1.0, value=0.020, step=0.001, format="%.4f")
    z_c6 = st.sidebar.number_input("C6 mole fraction", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f")
    z_c7p = st.sidebar.number_input("C7+ mole fraction", min_value=0.0, max_value=1.0, value=0.010, step=0.001, format="%.4f")

    z_init = np.array([z_h2s, z_co2, z_n2, z_c1, z_c2, z_c3, z_ic4, z_nc4, z_c5, z_c6, z_c7p], dtype=float)
    if np.sum(z_init) <= 0.0:
        st.error("Initial composition must have a positive total.")
        return
    z_init = z_init / np.sum(z_init)
    st.sidebar.caption(f"Normalized composition sum: {np.sum(z_init):.4f}")

    lean_gas_composition = np.array([0.0, 0.02, 0.01, 0.94, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    if scenario_name == "lean_gas_injection":
        st.sidebar.subheader("Lean Gas Injection Composition")
        lg_h2s = st.sidebar.number_input("Lean gas H2S", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
        lg_co2 = st.sidebar.number_input("Lean gas CO2", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        lg_n2 = st.sidebar.number_input("Lean gas N2", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        lg_c1 = st.sidebar.number_input("Lean gas C1", min_value=0.0, max_value=1.0, value=0.94, step=0.001, format="%.4f")
        lg_c2 = st.sidebar.number_input("Lean gas C2", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        lg_c3 = st.sidebar.number_input("Lean gas C3", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        lg_ic4 = st.sidebar.number_input("Lean gas iC4", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
        lg_nc4 = st.sidebar.number_input("Lean gas nC4", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
        lg_c5 = st.sidebar.number_input("Lean gas C5", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
        lg_c6 = st.sidebar.number_input("Lean gas C6", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
        lg_c7p = st.sidebar.number_input("Lean gas C7+", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
        lean_gas_composition = np.array([lg_h2s, lg_co2, lg_n2, lg_c1, lg_c2, lg_c3, lg_ic4, lg_nc4, lg_c5, lg_c6, lg_c7p], dtype=float)

    if np.sum(lean_gas_composition) > 0.0:
        lean_gas_composition = lean_gas_composition / np.sum(lean_gas_composition)

    if scenario_name != "natural_depletion" and injection_rate_value <= 0.0:
        st.warning("Injection scenario selected, but gas injection rate is zero.")

    st.sidebar.header("Model Overview")
    st.sidebar.write("Grid: 1D nonuniform Cartesian")
    st.sidebar.write("Fluid: 11-component hydrocarbon compositional + explicit water phase")
    st.sidebar.write("Producer: Peaceman producer with fixed BHP")
    st.sidebar.write("Injector: first grid cell for cycling/injection cases")

    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running compositional simulation..."):
            result = run_example(
                scenario_name=scenario_name,
                injection_rate_value=float(injection_rate_value),
                injection_rate_unit=injection_rate_unit,
                lean_gas_composition=lean_gas_composition,
                nx=int(nx),
                length_ft=float(length_ft),
                area_ft2=float(area_ft2),
                thickness_ft=float(thickness_ft),
                refined_cells=int(refined_cells),
                min_dx_ft=float(min_dx_ft),
                growth=float(growth),
                porosity=float(porosity),
                permeability_md=float(permeability_md),
                bhp_psia=float(bhp_psia),
                productivity_index=float(productivity_index),
                rw_ft=float(rw_ft),
                skin=float(skin),
                temperature_R=float(temperature_R),
                p_init_psia=float(p_init_psia),
                sw_init=float(sw_init),
                t_end_days=float(t_end_days),
                dt_days=float(dt_days),
                z_init=z_init,
                Sgc=float(Sgc),
                Swc=float(Swc),
                Sorw=float(Sorw),
                Sorg=float(Sorg),
                krg0=float(krg0),
                kro0=float(kro0),
                krw0=float(krw0),
                ng=float(ng),
                no=float(no),
                nw=float(nw),
            )

        history = result["history"]
        final_diag = result["final_diag"]
        summary_df = result["summary_df"]
        fluid = result["fluid"]
        well = result["well"]
        final_state = result["final_state"]
        output_dir = result["output_dir"]
        sim = result["sim"]
        grid = result["grid"]

        time_axis = format_time_axis(history.time_days, sim_length_unit)

        st.success("Simulation completed.")
        st.subheader("Scenario")
        st.write(f"Selected development scenario: **{scenario_label}**")
        if scenario_name != "natural_depletion":
            st.write("Injector location: cell 0")
            st.write(f"Gas injection rate: {injection_rate_value:.3f} {injection_rate_unit}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Avg Pressure (psia)", f"{history.avg_pressure_psia[-1]:.2f}")
        c2.metric("Final Gas Rate (MMscf/day)", f"{history.gas_rate_mmscf_day[-1]:.6f}")
        c3.metric("Final Condensate Rate (STB/day)", f"{history.condensate_rate_stb_day[-1]:.3f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Final Water Rate (STB/day)", f"{history.water_rate_stb_day[-1]:.3f}")
        c5.metric("Final Cumulative Gas (Bscf)", f"{history.cum_gas_bscf[-1]:.6f}")
        c6.metric("Final Cumulative Condensate (MSTB)", f"{history.cum_condensate_mstb[-1]:.6f}")

        c7, c8, c9 = st.columns(3)
        c7.metric("Final Cumulative Water (MSTB)", f"{history.cum_water_mstb[-1]:.6f}")
        c8.metric("Average Water Saturation", f"{history.avg_sw[-1]:.4f}")
        c9.metric("Productivity Loss Fraction", f"{history.productivity_loss_fraction[-1]:.4f}")

        st.subheader("Three-Phase Relative Permeability Settings")
        rp1, rp2, rp3, rp4 = st.columns(4)
        rp1.metric("Sgc", f"{Sgc:.2f}")
        rp2.metric("Swc", f"{Swc:.2f}")
        rp3.metric("Sorw", f"{Sorw:.2f}")
        rp4.metric("Sorg", f"{Sorg:.2f}")

        rp5, rp6, rp7, rp8, rp9, rp10 = st.columns(6)
        rp5.metric("krg0", f"{krg0:.2f}")
        rp6.metric("kro0", f"{kro0:.2f}")
        rp7.metric("krw0", f"{krw0:.2f}")
        rp8.metric("ng", f"{ng:.1f}")
        rp9.metric("no", f"{no:.1f}")
        rp10.metric("nw", f"{nw:.1f}")

        st.subheader("Chapter 4 Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Well-Cell Final Hydrocarbon Composition")
        comp_df = pd.DataFrame({
            "Component": [c.name for c in fluid.components],
            "Mole Fraction": final_state.z[well.cell_index]
        })
        st.dataframe(comp_df, use_container_width=True)

        st.subheader("Model Diagnostics")
        st.write(f"Output directory: {output_dir.resolve()}")
        st.write(f"Near-well minimum dx (ft): {grid.cell_width(well.cell_index):.3f}")
        st.write(f"Peaceman WI geom: {sim.peaceman_well_index(well.cell_index):.5f}")
        st.write(
            f"Relperm inputs: Sgc={Sgc:.2f}, Swc={Swc:.2f}, Sorw={Sorw:.2f}, Sorg={Sorg:.2f}, "
            f"krg0={krg0:.2f}, kro0={kro0:.2f}, krw0={krw0:.2f}, ng={ng:.1f}, no={no:.1f}, nw={nw:.1f}"
        )

        st.subheader("History Plots")

        fig1 = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.avg_pressure_psia, label="Average Pressure")
        plt.plot(time_axis, history.well_pressure_psia, label="Well-Cell Pressure")
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Pressure (psia)")
        plt.title("Pressure Depletion History")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.gas_rate_mmscf_day, label="Gas Rate")
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Gas Rate (MMscf/day)")
        plt.title("Gas Production Rate History")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        fig3 = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.condensate_rate_stb_day, label="Condensate Rate")
        plt.plot(time_axis, history.water_rate_stb_day, label="Water Rate")
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Liquid Rate (STB/day)")
        plt.title("Condensate and Water Production Rate History")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        fig4 = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.cum_gas_bscf, label="Cumulative Gas")
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Cumulative Gas (Bscf)")
        plt.title("Cumulative Gas Production")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        fig5 = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.cum_condensate_mstb, label="Cumulative Condensate")
        plt.plot(time_axis, history.cum_water_mstb, label="Cumulative Water")
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Cumulative Liquid (MSTB)")
        plt.title("Cumulative Condensate and Water Production")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        fig6 = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.well_damage_factor, label="Damage Factor")
        plt.plot(time_axis, history.productivity_loss_fraction, label="Productivity Loss Fraction")
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Dimensionless")
        plt.title("Condensate-Bank Impairment")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        fig_sw = plt.figure(figsize=(8, 5))
        plt.plot(time_axis, history.avg_sw)
        plt.xlabel(f"Time ({sim_length_unit})")
        plt.ylabel("Average Sw")
        plt.title("Average Water Saturation History")
        plt.tight_layout()
        st.pyplot(fig_sw)
        plt.close(fig_sw)

        st.subheader("Spatial Diagnostics")

        fig7 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["pressure_psia"])
        plt.xlabel("Distance (ft)")
        plt.ylabel("Pressure (psia)")
        plt.title("Final Pressure Profile")
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

        fig8 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["dropout_indicator"])
        plt.xlabel("Distance (ft)")
        plt.ylabel("Dropout Indicator")
        plt.title("Final Condensate Dropout Profile")
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close(fig8)

        fig9 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["krg"], label="krg")
        plt.plot(final_diag["x_ft"], final_diag["kro"], label="kro")
        plt.plot(final_diag["x_ft"], final_diag["krw"], label="krw")
        plt.xlabel("Distance (ft)")
        plt.ylabel("Relative Permeability")
        plt.title("Final Three-Phase Relative Permeability Profile")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig9)
        plt.close(fig9)

        fig10 = plt.figure(figsize=(8, 5))
        plt.plot(final_diag["x_ft"], final_diag["Sg"], label="Sg")
        plt.plot(final_diag["x_ft"], final_diag["So"], label="So")
        plt.plot(final_diag["x_ft"], final_diag["Sw"], label="Sw")
        plt.xlabel("Distance (ft)")
        plt.ylabel("Saturation")
        plt.title("Final Saturation Profile")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig10)
        plt.close(fig10)

        st.subheader("Download Outputs")
        with open(result["history_csv"], "rb") as f:
            st.download_button("Download history.csv", f.read(), file_name="history.csv", mime="text/csv")
        with open(result["spatial_csv"], "rb") as f:
            st.download_button("Download final_spatial_diagnostics.csv", f.read(), file_name="final_spatial_diagnostics.csv", mime="text/csv")
        with open(result["summary_csv"], "rb") as f:
            st.download_button("Download chapter4_summary.csv", f.read(), file_name="chapter4_summary.csv", mime="text/csv")

        st.caption(
            "Gas rates are reported in MMscf/day using standard conditions of 14.7 psia and 60°F. "
            "Condensate rates are approximate STB/day estimates from liquid hydrocarbon molar rate, "
            "mixture molecular weight, and assumed condensate density of 50 lbm/ft³. "
            "Water is explicitly modeled as a separate immiscible phase in this prototype."
        )
    else:
        st.info("Click 'Run Simulation' to execute the model and render results.")


display_streamlit_app()