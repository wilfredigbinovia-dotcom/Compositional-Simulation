from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal
import math
from datetime import datetime

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

ScenarioName = Literal["natural_depletion", "gas_cycling", "lean_gas_injection"]
WellControlMode = Literal["bhp", "drawdown", "gas_rate", "thp"]


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
            term1 = bi[i] / max(bm, 1e-12) * (Z - 1.0)
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
        return float(np.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0))))

    def solve_beta(self, z: np.ndarray, K: np.ndarray) -> float:
        f0 = self.rachford_rice(0.0, z, K)
        f1 = self.rachford_rice(1.0, z, K)

        if f0 < 0.0 and f1 < 0.0:
            return 0.0
        if f0 > 0.0 and f1 > 0.0:
            return 1.0

        lo, hi = 0.0, 1.0
        flo = f0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            fm = self.rachford_rice(mid, z, K)
            if abs(fm) < 1e-10:
                return mid
            if flo * fm < 0.0:
                hi = mid
            else:
                lo = mid
                flo = fm
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
            K_new = np.clip(phi_l / np.maximum(phi_v, 1e-12), 1e-8, 1e8)
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
    if nx < 2:
        raise ValueError("nx must be at least 2")
    refined_cells = max(2, min(refined_cells, nx - 1))
    if growth <= 1.0:
        raise ValueError("growth must be greater than 1.0")
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

    def __post_init__(self) -> None:
        if self.Sgc + self.Swc >= 1.0:
            raise ValueError("Invalid relperm inputs: Sgc + Swc must be < 1.0")
        if self.Swc + self.Sorw >= 1.0:
            raise ValueError("Invalid relperm inputs: Swc + Sorw must be < 1.0")
        if self.Swc + self.Sorg >= 1.0:
            raise ValueError("Invalid relperm inputs: Swc + Sorg must be < 1.0")
        for value_name in ["krg0", "kro0", "krw0", "ng", "no", "nw"]:
            if getattr(self, value_name) < 0.0:
                raise ValueError(f"{value_name} must be non-negative")


@dataclass
class Well:
    cell_index: int
    control_mode: WellControlMode = "bhp"
    bhp_psia: float = 3000.0
    drawdown_psia: float = 200.0
    min_bhp_psia: float = 50.0
    target_gas_rate_mmscf_day: float = 0.0
    thp_psia: float = 500.0
    tvd_ft: float = 8000.0
    tubing_id_in: float = 2.441
    wellhead_temperature_R: float = 520.0
    thp_friction_coeff: float = 0.02
    productivity_index: float = 1.0
    rw_ft: float = 0.35
    skin: float = 0.0


@dataclass
class InjectorWell:
    cell_index: int
    rate_lbmol_day: float
    injection_composition: np.ndarray
    rate_field_input: float = 0.0
    rate_field_unit: str = "MMscf/day"

    def __post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValueError("Injector cell index must be non-negative")
        if self.rate_lbmol_day < 0.0:
            raise ValueError("Injector rate must be non-negative")
        comp = np.asarray(self.injection_composition, dtype=float)
        if comp.ndim != 1 or comp.size == 0:
            raise ValueError("Injection composition must be a non-empty 1D array")
        if np.sum(comp) <= 0.0:
            raise ValueError("Injection composition must have positive total")
        self.injection_composition = comp / np.sum(comp)


@dataclass
class ScenarioConfig:
    name: ScenarioName
    injector: InjectorWell | None = None

    def __post_init__(self) -> None:
        allowed = {"natural_depletion", "gas_cycling", "lean_gas_injection"}
        if self.name not in allowed:
            raise ValueError(f"Unsupported scenario: {self.name}")


@dataclass
class PVTConfig:
    dew_point_psia: float


@dataclass
class SeparatorConfig:
    pressure_psia: float = 300.0
    temperature_R: float = 520.0


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

    dew_point_psia: List[float] = field(default_factory=list)
    avg_pressure_minus_dewpoint_psia: List[float] = field(default_factory=list)
    well_pressure_minus_dewpoint_psia: List[float] = field(default_factory=list)
    well_below_dewpoint_flag: List[float] = field(default_factory=list)

    hc_mass_balance_error_lbmol: List[float] = field(default_factory=list)
    water_mass_balance_error_lbmol: List[float] = field(default_factory=list)
    accepted_dt_days: List[float] = field(default_factory=list)
    timestep_retries: List[int] = field(default_factory=list)

    well_control_mode: List[str] = field(default_factory=list)
    well_flowing_pwf_psia: List[float] = field(default_factory=list)
    well_estimated_thp_psia: List[float] = field(default_factory=list)
    well_target_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    controlled_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    reported_gas_rate_mmscf_day: List[float] = field(default_factory=list)

    separator_pressure_psia: List[float] = field(default_factory=list)
    separator_temperature_R: List[float] = field(default_factory=list)
    separator_vapor_fraction: List[float] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Utility correlations
# -----------------------------------------------------------------------------

def phase_viscosity_cp(phase: str, comp: np.ndarray, pressure_psia: float | None = None, temperature_R: float | None = None) -> float:
    comp = np.asarray(comp, dtype=float)
    comp = comp / max(np.sum(comp), 1e-12)
    n = len(comp)
    heaviness_weights = np.linspace(0.2, 2.0, n)
    heaviness = float(np.dot(heaviness_weights, comp))

    pterm = 0.0 if pressure_psia is None else max(pressure_psia, 14.7) / 10000.0
    tterm = 0.0 if temperature_R is None else max(temperature_R, 300.0) / 1000.0

    if phase == "v":
        return float(np.clip(0.010 + 0.010 * heaviness + 0.020 * pterm + 0.005 * tterm, 0.008, 0.12))
    if phase == "l":
        return float(np.clip(0.08 + 0.20 * heaviness + 0.03 * pterm - 0.02 * tterm, 0.05, 3.0))
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

    # Soften the gas relative-permeability transition near critical gas saturation.
    # A hard clip at Sgc was causing krg to collapse to exactly zero and remain frozen.
    sgc_soft = max(params.Sgc - 0.02, 0.0)
    Sg_eff_raw = (Sg - sgc_soft) / max(denom_g + (params.Sgc - sgc_soft), 1e-10)
    Sg_eff = float(np.clip(Sg_eff_raw, 0.0, 1.0))

    denom_w = max(1.0 - params.Swc - params.Sorw, 1e-10)
    Sw_eff = np.clip((Sw - params.Swc) / denom_w, 0.0, 1.0)

    so_mobile_max = max(1.0 - params.Swc - params.Sorg, 1e-10)
    So_eff = np.clip(So / so_mobile_max, 0.0, 1.0)

    krg = params.krg0 * (Sg_eff ** params.ng)
    if Sg > params.Sgc and krg < 0.01 * params.krg0:
        krg = 0.01 * params.krg0

    kro = params.kro0 * (So_eff ** params.no)
    krw = params.krw0 * (Sw_eff ** params.nw)

    return float(krg), float(kro), float(krw)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def beta_to_flowing_saturations(
    beta: float,
    Sw: float,
    params: RelPermParams,
    beta_lo: float = 0.02,
    beta_hi: float = 0.88,
    min_gas_sat: float = 0.06,
    beta_weight: float = 0.45,
) -> Tuple[float, float, float]:
    """
    Smoothly map flash vapor fraction beta to flowing gas/oil saturations.

    The key requirement is to keep the mapped gas saturation from sitting
    below critical gas saturation for the entire depletion tail; otherwise
    krg clips to zero and the wellstream gas fraction collapses artificially.
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    Sw = float(np.clip(Sw, 0.0, 0.95))
    Shc = max(1.0 - Sw, 0.0)

    if Shc <= 1e-12:
        return 0.0, 0.0, Sw

    t = smoothstep((beta - beta_lo) / max(beta_hi - beta_lo, 1e-12))
    gas_proxy = (1.0 - beta_weight) * t + beta_weight * beta
    gas_proxy = float(np.clip(gas_proxy, 0.0, 1.0))

    # Static floor from user-entered residual gas saturation.
    sg_floor_base = Shc * min_gas_sat

    # Dynamic floor to keep the flowing gas saturation slightly above Sgc
    # whenever there is a non-trivial flashed gas fraction.
    sg_crit = float(np.clip(params.Sgc + 0.03, 0.0, 0.98))
    sg_floor_dynamic = sg_crit * smoothstep(beta / 0.08) if beta > 1e-8 else 0.0

    sg_floor = min(Shc, max(sg_floor_base, sg_floor_dynamic))

    Sg_raw = sg_floor + (Shc - sg_floor) * gas_proxy
    Sg = float(np.clip(Sg_raw, sg_floor, Shc))
    So = float(np.clip(Shc - Sg, 0.0, Shc))

    total = Sg + So + Sw
    if total > 1e-12:
        Sg /= total
        So /= total
        Sw /= total

    return Sg, So, Sw


def liquid_dropout_fraction(z: np.ndarray, beta: float) -> float:
    heaviness_weights = np.linspace(0.0, 1.0, len(z))
    heaviness = float(np.dot(heaviness_weights, z))
    return max((1.0 - beta) * heaviness, 0.0)


def condensate_bank_damage_factor(dropout: float, kro: float, damage_strength: float = 3.0, critical_dropout: float = 0.05) -> float:
    """
    Smoother, less aggressive condensate-bank productivity penalty.
    Prevents unrealistic late-life mobility collapse/recovery artifacts.
    """
    dropout = float(max(dropout, 0.0))
    kro = float(np.clip(kro, 0.0, 1.0))

    excess = max(dropout - critical_dropout, 0.0)
    liquid_penalty = 1.0 - kro

    damage = math.exp(-damage_strength * excess * (0.35 + 0.65 * liquid_penalty))
    return float(np.clip(damage, 0.20, 1.0))


def productivity_loss_fraction(q_actual: float, q_undamaged: float) -> float:
    if q_undamaged <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - q_actual / q_undamaged, 0.0, 1.0))


def z_to_density_lbmol_ft3(P: float, T: float, Z: float) -> float:
    return P / max(Z * R * T, 1e-12)


def gas_lbmol_day_to_mmscf_day(rate_lbmol_day: float) -> float:
    return rate_lbmol_day * SCF_PER_LBMOL * MM_PER_ONE


def gas_mmscf_day_to_lbmol_day(rate_mmscf_day: float) -> float:
    return rate_mmscf_day * 1.0e6 / SCF_PER_LBMOL


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


def parse_ddmmyyyy(date_text: str) -> datetime:
    return datetime.strptime(date_text.strip(), "%d/%m/%Y")


def days_between_dates(start_date_text: str, end_date_text: str) -> float:
    start_dt = parse_ddmmyyyy(start_date_text)
    end_dt = parse_ddmmyyyy(end_date_text)
    delta_days = (end_dt - start_dt).days
    if delta_days <= 0:
        raise ValueError("End date must be later than start date.")
    return float(delta_days)


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
        pvt: PVTConfig | None = None,
        separator: SeparatorConfig | None = None,
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
        self.pvt = pvt or PVTConfig(dew_point_psia=4000.0)
        self.separator = separator or SeparatorConfig()
        self.last_produced_gas_composition = None
        self._current_time_days = 0.0

        self.phi = rock.porosity
        self.k = rock.permeability_md
        self.pv = np.array([self.grid.bulk_volume(i) * self.phi for i in range(self.grid.nx)], dtype=float)
        _, _, _, self.mw_components = self.fluid.critical_arrays()

        self._cache_state_id = None
        self._flash_cache: Dict[int, Dict[str, np.ndarray | float | str]] = {}
        self._mobility_cache: Dict[int, Dict[str, float]] = {}

        self.initial_hc_lbmol = 0.0
        self.initial_water_lbmol = 0.0
        self.cum_hc_produced_lbmol = 0.0
        self.cum_hc_injected_lbmol = 0.0
        self.cum_water_produced_lbmol = 0.0

    def _initialize_accounting(self, state0: SimulationState) -> None:
        self.initial_hc_lbmol = float(np.sum(state0.nt))
        self.initial_water_lbmol = float(np.sum(state0.nw))
        self.cum_hc_produced_lbmol = 0.0
        self.cum_hc_injected_lbmol = 0.0
        self.cum_water_produced_lbmol = 0.0

    def _invalidate_caches(self) -> None:
        self._flash_cache = {}
        self._mobility_cache = {}

    def _set_cache_state(self, state: SimulationState) -> None:
        self._cache_state_id = id(state)
        self._invalidate_caches()

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

        state0 = SimulationState(pressure=pressure, z=z, nt=nt, nw=nw, sw=sw)
        self._initialize_accounting(state0)
        return state0

    def cell_flash_cached(self, state: SimulationState, i: int) -> Dict[str, np.ndarray | float | str]:
        if self._cache_state_id != id(state):
            self._set_cache_state(state)
        if i not in self._flash_cache:
            self._flash_cache[i] = self.flash.flash(state.z[i], state.pressure[i], self.T)
        return self._flash_cache[i]

    def phase_mobility_data(self, state: SimulationState, i: int) -> Dict[str, float]:
        if self._cache_state_id != id(state):
            self._set_cache_state(state)
        if i in self._mobility_cache:
            return self._mobility_cache[i]

        fl = self.cell_flash_cached(state, i)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])

        mu_o = phase_viscosity_cp("l", x, state.pressure[i], self.T)
        mu_g = phase_viscosity_cp("v", y, state.pressure[i], self.T)
        mu_w = WATER_VISCOSITY_CP

        Sw = float(np.clip(state.sw[i], 0.0, 0.95))
        Sg, So, Sw = beta_to_flowing_saturations(beta, Sw, self.relperm)

        krg, kro, krw = three_phase_relperm(Sg, So, Sw, self.relperm)

        lam_g = krg / max(mu_g, 1e-8)
        lam_o = kro / max(mu_o, 1e-8)
        lam_w = krw / max(mu_w, 1e-8)
        lam_t = lam_g + lam_o + lam_w

        data = {
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
        self._mobility_cache[i] = data
        return data

    def transmissibility(self, state: SimulationState, i: int, j: int) -> float:
        mob_i = self.phase_mobility_data(state, i)
        mob_j = self.phase_mobility_data(state, j)
        lam_i = mob_i["lam_t"]
        lam_j = mob_j["lam_t"]
        lam_face = 2.0 * lam_i * lam_j / max(lam_i + lam_j, 1e-12)
        distance = self.grid.interface_distance(i, j)
        return self.k * self.grid.area_ft2 * lam_face / max(distance, 1e-12)

    def component_flux_between(self, state: SimulationState, i: int, j: int) -> np.ndarray:
        p_i = state.pressure[i]
        p_j = state.pressure[j]
        dp = p_i - p_j
        if abs(dp) < 1e-12:
            return np.zeros(self.nc)

        upstream = i if dp >= 0.0 else j
        fl = self.cell_flash_cached(state, upstream)
        x = np.asarray(fl["x"], dtype=float)
        y = np.asarray(fl["y"], dtype=float)
        mob = self.phase_mobility_data(state, upstream)
        lam_g = float(mob["lam_g"])
        lam_o = float(mob["lam_o"])

        distance = self.grid.interface_distance(i, j)
        trans = self.k * self.grid.area_ft2 / max(distance, 1e-12)

        # Phase-segregated hydrocarbon transport. The previous implementation
        # moved a single blended hydrocarbon composition, which effectively
        # reduced to the cell overall composition z and suppressed reservoir-wide
        # compositional fractionation. That let the producer cell and separator
        # flash lock into a nearly constant split after the first timestep.
        qg_face = trans * lam_g * dp * 1e-5
        qo_face = trans * lam_o * dp * 1e-5
        return qg_face * y + qo_face * x

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
        trans = self.k * self.grid.area_ft2 / max(distance, 1e-12)
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
            fl = self.cell_flash_cached(state, wi)
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

    def well_response_at_pwf(self, state: SimulationState, pwf_psia: float) -> Dict[str, float | np.ndarray]:
        i = self.well.cell_index
        pr = float(state.pressure[i])
        pwf = float(np.clip(pwf_psia, 50.0, pr))
        dp = max(pr - pwf, 0.0)

        # Evaluate the producing-cell flash at flowing bottomhole pressure,
        # not just at the reservoir-cell pressure. This keeps the wellstream
        # split responsive to drawdown and depletion.
        fl = self.flash.flash(state.z[i], pwf, self.T)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])
        self.last_produced_gas_composition = y.copy()

        mu_o = phase_viscosity_cp("l", x, pwf, self.T)
        mu_g = phase_viscosity_cp("v", y, pwf, self.T)
        mu_w = WATER_VISCOSITY_CP

        Sw = float(np.clip(state.sw[i], 0.0, 0.95))
        Sg, So, Sw = beta_to_flowing_saturations(beta, Sw, self.relperm)
        krg, kro, krw = three_phase_relperm(Sg, So, Sw, self.relperm)

        lam_g = krg / max(mu_g, 1e-8)
        lam_o = kro / max(mu_o, 1e-8)
        lam_w = krw / max(mu_w, 1e-8)
        lam_t = max(lam_g + lam_o + lam_w, 1e-8)

        wi_geom = self.peaceman_well_index(i)
        q_undamaged = wi_geom * lam_t * dp * 1e-2

        dropout = liquid_dropout_fraction(state.z[i], beta)
        damage_factor = condensate_bank_damage_factor(dropout, kro)
        wi_eff = wi_geom * damage_factor
        q_total = wi_eff * lam_t * dp * 1e-2

        fw = lam_w / max(lam_t, 1e-12)

        lam_hc = lam_g + lam_o
        hc_frac = max(1.0 - fw, 0.0)
        if lam_hc > 1e-12 and hc_frac > 1e-12:
            fg_hc_mob = lam_g / lam_hc
            fg_hc_flash = float(np.clip(beta, 0.0, 1.0))

            # Keep the hydrocarbon split mainly mobility-led, but anchor it to
            # the flowing-pressure flash so it evolves with drawdown.
            fg_hc_blend = 0.90 * fg_hc_mob + 0.10 * fg_hc_flash
            min_fg_hc = 0.03 * fg_hc_flash if fg_hc_flash > 1e-6 else 0.0
            fg_hc = float(np.clip(max(fg_hc_blend, min_fg_hc), 0.0, 1.0))
            fo_hc = 1.0 - fg_hc
            fg = hc_frac * fg_hc
            fo = hc_frac * fo_hc
        else:
            fg = 0.0
            fo = hc_frac

        qg = q_total * fg
        qo = q_total * fo
        qw = q_total * fw

        hc_sink = qg * y + qo * x
        ploss = productivity_loss_fraction(q_total, q_undamaged)

        return {
            "pwf_psia": pwf,
            "pr_psia": pr,
            "dp_psia": dp,
            "hc_sink": hc_sink,
            "qw_lbmol_day": float(qw),
            "q_total_lbmol_day": float(q_total),
            "qg_lbmol_day": float(qg),
            "qo_lbmol_day": float(qo),
            "gas_frac": float(fg),
            "oil_frac": float(fo),
            "water_frac": float(fw),
            "dropout": float(dropout),
            "krg": float(krg),
            "kro": float(kro),
            "krw": float(krw),
            "damage_factor": float(damage_factor),
            "wi_eff": float(wi_eff),
            "q_undamaged": float(q_undamaged),
            "ploss": float(ploss),
            "x": x,
            "y": y,
        }

    def estimate_mixture_density_lbft3(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> float:
        qg = float(response["qg_lbmol_day"])
        qo = float(response["qo_lbmol_day"])
        qw = float(response["qw_lbmol_day"])

        total = qg + qo + qw
        if total <= 1e-12:
            return 0.1

        gas_density = 2.0
        condensate_density = CONDENSATE_DENSITY_LBFT3
        water_density = WATER_DENSITY_LBFT3

        rho_mix = (qg * gas_density + qo * condensate_density + qw * water_density) / total
        return float(np.clip(rho_mix, 0.1, 62.4))

    def tubing_head_pressure_from_response(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> float:
        pwf = float(response["pwf_psia"])
        qg_lbmol_day = float(response["qg_lbmol_day"])
        qo_lbmol_day = float(response["qo_lbmol_day"])
        qw_lbmol_day = float(response["qw_lbmol_day"])

        gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(qg_lbmol_day)
        liquid_rate_stb_day = (
            liquid_lbmol_day_to_stb_day(qo_lbmol_day, np.asarray(response["x"]), self.mw_components, CONDENSATE_DENSITY_LBFT3)
            + water_lbmol_day_to_stb_day(qw_lbmol_day)
        )

        rho_mix = self.estimate_mixture_density_lbft3(state, response)
        hydrostatic_psi = 0.006944444444444444 * rho_mix * self.well.tvd_ft
        friction_psi = self.well.thp_friction_coeff * (gas_rate_mmscf_day + 0.001 * liquid_rate_stb_day) ** 2

        thp = pwf - hydrostatic_psi - friction_psi
        return float(max(thp, 14.7))

    def solve_bhp_control(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        return self.well_response_at_pwf(state, self.well.bhp_psia)

    def solve_drawdown_control(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        i = self.well.cell_index
        pr = float(state.pressure[i])

        pwf_target = pr - self.well.drawdown_psia
        pwf_target = max(self.well.min_bhp_psia, pwf_target)

        return self.well_response_at_pwf(state, pwf_target)

    def solve_gas_rate_control(self, state: SimulationState, tol_mmscf_day: float = 1e-4, max_iter: int = 60) -> Dict[str, float | np.ndarray]:
        target_qg_lbmol_day = gas_mmscf_day_to_lbmol_day(self.well.target_gas_rate_mmscf_day)

        i = self.well.cell_index
        pr = float(state.pressure[i])
        pwf_lo = 50.0
        pwf_hi = max(pr - 1e-6, pwf_lo + 1e-6)

        resp_lo = self.well_response_at_pwf(state, pwf_lo)
        resp_hi = self.well_response_at_pwf(state, pwf_hi)

        qg_lo = float(resp_lo["qg_lbmol_day"])
        qg_hi = float(resp_hi["qg_lbmol_day"])

        if target_qg_lbmol_day >= qg_lo:
            return resp_lo
        if target_qg_lbmol_day <= qg_hi:
            return resp_hi

        best = resp_hi
        for _ in range(max_iter):
            pwf_mid = 0.5 * (pwf_lo + pwf_hi)
            resp_mid = self.well_response_at_pwf(state, pwf_mid)
            qg_mid = float(resp_mid["qg_lbmol_day"])
            err_mmscf_day = abs(gas_lbmol_day_to_mmscf_day(qg_mid - target_qg_lbmol_day))
            best = resp_mid

            if err_mmscf_day < tol_mmscf_day:
                return resp_mid

            if qg_mid > target_qg_lbmol_day:
                pwf_lo = pwf_mid
            else:
                pwf_hi = pwf_mid

        return best

    def solve_thp_control(self, state: SimulationState, tol_psia: float = 1.0, max_iter: int = 60) -> Dict[str, float | np.ndarray]:
        target_thp = self.well.thp_psia
        i = self.well.cell_index
        pr = float(state.pressure[i])

        pwf_lo = 50.0
        pwf_hi = max(pr - 1e-6, pwf_lo + 1e-6)

        resp_lo = self.well_response_at_pwf(state, pwf_lo)
        resp_hi = self.well_response_at_pwf(state, pwf_hi)

        thp_lo = self.tubing_head_pressure_from_response(state, resp_lo)
        thp_hi = self.tubing_head_pressure_from_response(state, resp_hi)

        if target_thp <= thp_lo:
            resp_lo["estimated_thp_psia"] = thp_lo
            return resp_lo
        if target_thp >= thp_hi:
            resp_hi["estimated_thp_psia"] = thp_hi
            return resp_hi

        best = resp_hi
        for _ in range(max_iter):
            pwf_mid = 0.5 * (pwf_lo + pwf_hi)
            resp_mid = self.well_response_at_pwf(state, pwf_mid)
            thp_mid = self.tubing_head_pressure_from_response(state, resp_mid)
            resp_mid["estimated_thp_psia"] = thp_mid
            best = resp_mid

            if abs(thp_mid - target_thp) < tol_psia:
                return resp_mid

            if thp_mid < target_thp:
                pwf_lo = pwf_mid
            else:
                pwf_hi = pwf_mid

        return best

    def enforce_gas_rate_target(self, response: Dict[str, float | np.ndarray]) -> Dict[str, float | np.ndarray]:
        if self.well.control_mode != "gas_rate":
            return response

        target_qg_lbmol_day = gas_mmscf_day_to_lbmol_day(self.well.target_gas_rate_mmscf_day)
        actual_qg_lbmol_day = float(response["qg_lbmol_day"])

        if target_qg_lbmol_day <= 0.0:
            return response

        if actual_qg_lbmol_day < target_qg_lbmol_day:
            return response

        response = dict(response)
        scale = target_qg_lbmol_day / max(actual_qg_lbmol_day, 1e-12)

        q_total_lbmol_day = float(response["q_total_lbmol_day"]) * scale
        qo_lbmol_day = float(response["qo_lbmol_day"]) * scale
        qw_lbmol_day = float(response["qw_lbmol_day"]) * scale

        y = np.asarray(response["y"], dtype=float)
        x = np.asarray(response["x"], dtype=float)

        response["qg_lbmol_day"] = float(target_qg_lbmol_day)
        response["qo_lbmol_day"] = float(qo_lbmol_day)
        response["qw_lbmol_day"] = float(qw_lbmol_day)
        response["q_total_lbmol_day"] = float(q_total_lbmol_day)
        response["hc_sink"] = target_qg_lbmol_day * y + qo_lbmol_day * x

        qt = max(q_total_lbmol_day, 1e-12)
        response["gas_frac"] = float(target_qg_lbmol_day / qt)
        response["oil_frac"] = float(qo_lbmol_day / qt)
        response["water_frac"] = float(qw_lbmol_day / qt)

        return response

    def solve_well_control(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        mode = self.well.control_mode

        if mode == "bhp":
            response = self.solve_bhp_control(state)
            response["control_mode"] = "bhp"
            response["estimated_thp_psia"] = self.tubing_head_pressure_from_response(state, response)
            return response

        if mode == "drawdown":
            response = self.solve_drawdown_control(state)
            response["control_mode"] = "drawdown"
            response["estimated_thp_psia"] = self.tubing_head_pressure_from_response(state, response)
            return response

        if mode == "gas_rate":
            response = self.solve_gas_rate_control(state)
            response = self.enforce_gas_rate_target(response)
            response["control_mode"] = "gas_rate"
            response["estimated_thp_psia"] = self.tubing_head_pressure_from_response(state, response)
            return response

        if mode == "thp":
            response = self.solve_thp_control(state)
            response["control_mode"] = "thp"
            if "estimated_thp_psia" not in response:
                response["estimated_thp_psia"] = self.tubing_head_pressure_from_response(state, response)
            return response

        raise ValueError(f"Unsupported well control mode: {mode}")

    def produced_hydrocarbon_stream(self, response: Dict[str, float | np.ndarray]) -> Tuple[float, np.ndarray]:
        qg = float(response["qg_lbmol_day"])
        qo = float(response["qo_lbmol_day"])
        y = np.asarray(response["y"], dtype=float)
        x = np.asarray(response["x"], dtype=float)

        qhc = qg + qo
        if qhc <= 1e-12:
            z_prod = 0.5 * (x + y)
            z_prod = z_prod / np.sum(z_prod)
            return 0.0, z_prod

        z_prod = (qg * y + qo * x) / qhc
        z_prod = z_prod / np.sum(z_prod)
        return qhc, z_prod

    def separator_flash(self, response: Dict[str, float | np.ndarray]) -> Dict[str, float | np.ndarray]:
        qhc_lbmol_day, z_prod = self.produced_hydrocarbon_stream(response)

        if qhc_lbmol_day <= 1e-12:
            return {
                "qhc_lbmol_day": 0.0,
                "q_sep_gas_lbmol_day": 0.0,
                "q_sep_liq_lbmol_day": 0.0,
                "beta_sep": 1.0,
                "z_prod": z_prod,
                "x_sep": z_prod.copy(),
                "y_sep": z_prod.copy(),
                "state": "vapor",
            }

        sep = self.flash.flash(z_prod, self.separator.pressure_psia, self.separator.temperature_R)

        beta_sep_flash = float(sep["beta"])
        x_sep = np.asarray(sep["x"], dtype=float)
        y_sep = np.asarray(sep["y"], dtype=float)

        # Use the separator flash result directly so surface reporting remains
        # thermodynamically consistent with the produced hydrocarbon stream.
        beta_sep = float(np.clip(beta_sep_flash, 0.0, 1.0))

        q_sep_gas_lbmol_day = qhc_lbmol_day * beta_sep
        q_sep_liq_lbmol_day = qhc_lbmol_day * (1.0 - beta_sep)

        return {
            "qhc_lbmol_day": float(qhc_lbmol_day),
            "q_sep_gas_lbmol_day": float(q_sep_gas_lbmol_day),
            "q_sep_liq_lbmol_day": float(q_sep_liq_lbmol_day),
            "beta_sep": beta_sep,
            "z_prod": z_prod,
            "x_sep": x_sep,
            "y_sep": y_sep,
            "state": str(sep["state"]),
        }

    def separator_rates(self, response: Dict[str, float | np.ndarray]) -> Dict[str, float]:
        sep = self.separator_flash(response)
        q_sep_gas_lbmol_day = float(sep["q_sep_gas_lbmol_day"])
        q_sep_liq_lbmol_day = float(sep["q_sep_liq_lbmol_day"])
        q_w_lbmol_day = float(response.get("qw_lbmol_day", 0.0))
        x_sep = np.asarray(sep["x_sep"], dtype=float)

        gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(q_sep_gas_lbmol_day)
        condensate_rate_stb_day = liquid_lbmol_day_to_stb_day(q_sep_liq_lbmol_day, x_sep, self.mw_components, CONDENSATE_DENSITY_LBFT3)

        total_surface = max(q_sep_gas_lbmol_day + q_sep_liq_lbmol_day + q_w_lbmol_day, 1e-12)
        surface_gas_fraction = q_sep_gas_lbmol_day / total_surface
        surface_oil_fraction = q_sep_liq_lbmol_day / total_surface
        surface_water_fraction = q_w_lbmol_day / total_surface

        return {
            "gas_rate_mmscf_day": float(gas_rate_mmscf_day),
            "condensate_rate_stb_day": float(condensate_rate_stb_day),
            "separator_vapor_fraction": float(sep["beta_sep"]),
            "surface_gas_fraction": float(surface_gas_fraction),
            "surface_oil_fraction": float(surface_oil_fraction),
            "surface_water_fraction": float(surface_water_fraction),
        }

    def reported_well_response(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        response = self.solve_well_control(state)
        response = dict(response)

        if self.well.control_mode == "gas_rate":
            response["reported_gas_rate_mmscf_day"] = float(self.well.target_gas_rate_mmscf_day)
        else:
            response["reported_gas_rate_mmscf_day"] = gas_lbmol_day_to_mmscf_day(float(response["qg_lbmol_day"]))

        return response

    def well_sink(self, state: SimulationState) -> Tuple[np.ndarray, float, float, float, float, float, float, float, float, float, float, float, float, float]:
        response = self.solve_well_control(state)
        return (
            np.asarray(response["hc_sink"]),
            float(response["qw_lbmol_day"]),
            float(response["q_total_lbmol_day"]),
            float(response["gas_frac"]),
            float(response["oil_frac"]),
            float(response["water_frac"]),
            float(response["dropout"]),
            float(response["krg"]),
            float(response["kro"]),
            float(response["krw"]),
            float(response["damage_factor"]),
            float(response["wi_eff"]),
            float(response["q_undamaged"]),
            float(response["ploss"]),
        )

    def pressure_update(self, state: SimulationState, dt_days: float) -> np.ndarray:
        p_old = state.pressure.copy()
        n = self.grid.nx
        A = np.zeros((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        ct = 1.0e-5
        acc_scale = ct * self.pv / max(dt_days, 1e-12)

        for i in range(n):
            A[i, i] += acc_scale[i]
            b[i] += acc_scale[i] * p_old[i]

        for i in range(n - 1):
            Tij = self.transmissibility(state, i, i + 1) * 1.0e-3
            A[i, i] += Tij
            A[i + 1, i + 1] += Tij
            A[i, i + 1] -= Tij
            A[i + 1, i] -= Tij

        wi = self.well.cell_index
        well_mob = self.phase_mobility_data(state, wi)
        Jw = self.peaceman_well_index(wi) * well_mob["lam_t"] * 2.5e-3
        response = self.solve_well_control(state)
        flowing_pwf = float(response["pwf_psia"])

        A[wi, wi] += Jw
        b[wi] += Jw * flowing_pwf

        injector = self.active_injector()
        if injector is not None and self.scenario.name != "natural_depletion" and injector.rate_lbmol_day > 0.0:
            ii = injector.cell_index
            inj_strength = 1.0e-4 * injector.rate_lbmol_day / max(self.pv[ii], 1.0)
            b[ii] += inj_strength

        try:
            p_solved = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            p_solved = p_old.copy()

        omega_p = 0.15
        p_new = (1.0 - omega_p) * p_old + omega_p * p_solved
        return np.maximum(p_new, 50.0)

    def transport_update(self, state: SimulationState, dt_days: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, float, float, float, float]:
        """
        Transport is sub-cycled so the producer-cell composition and phase state
        are refreshed within a large reporting timestep. A single coarse explicit
        update was letting the well cell jump rapidly into a near-frozen regime.
        """
        # Lightweight subcycling: preserve the idea of refreshing the producer-cell
        # state within the reporting step, but avoid 1-day global substeps that make
        # long forecasts excessively slow and can prevent output generation.
        max_substep_days = 10.0
        n_sub = max(1, min(4, int(math.ceil(dt_days / max_substep_days))))
        dt_sub = dt_days / n_sub

        work_state = SimulationState(
            pressure=state.pressure.copy(),
            z=state.z.copy(),
            nt=state.nt.copy(),
            nw=state.nw.copy(),
            sw=state.sw.copy(),
        )

        q_total = gas_frac = oil_frac = water_frac = 0.0
        dropout = krg = kro = krw = 0.0
        damage_factor = wi_eff = q_undamaged = ploss = 0.0

        for _ in range(n_sub):
            self._set_cache_state(work_state)
            ncomp = work_state.nt[:, None] * work_state.z
            ncomp_new = ncomp.copy()
            nw_new = work_state.nw.copy()

            for i in range(self.grid.nx - 1):
                hc_flux = self.component_flux_between(work_state, i, i + 1)
                w_flux = self.water_flux_between(work_state, i, i + 1)

                ncomp_new[i] -= dt_sub * hc_flux
                ncomp_new[i + 1] += dt_sub * hc_flux

                nw_new[i] -= dt_sub * w_flux
                nw_new[i + 1] += dt_sub * w_flux

            inj_cell, inj_source = self.injector_source(work_state)
            if inj_cell is not None and inj_source is not None:
                ncomp_new[inj_cell] += dt_sub * inj_source

            hc_sink, qw, q_total, gas_frac, oil_frac, water_frac, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss = self.well_sink(work_state)
            wi = self.well.cell_index

            # Limit producer-cell withdrawal to the inventory actually available
            # over the substep. Without this, one coarse withdrawal can drive the
            # well cell to the numerical floor, after which composition/flash can
            # become artificially frozen for the remainder of the run.
            hc_remove = dt_sub * hc_sink
            hc_available = np.maximum(ncomp_new[wi] - 1e-12, 0.0)
            hc_req_total = float(np.sum(hc_remove))
            hc_avail_total = float(np.sum(hc_available))
            hc_scale_total = 1.0 if hc_req_total <= 1e-12 else min(1.0, 0.20 * hc_avail_total / hc_req_total)
            if hc_req_total > 1e-12:
                comp_scales = np.where(hc_remove > 1e-18, hc_available / np.maximum(hc_remove, 1e-18), 1.0)
                hc_scale_comp = float(np.clip(np.min(comp_scales), 0.0, 1.0))
            else:
                hc_scale_comp = 1.0
            hc_scale = min(hc_scale_total, hc_scale_comp)
            hc_remove *= hc_scale

            w_remove = dt_sub * qw
            w_available = max(float(nw_new[wi] - 1e-12), 0.0)
            w_scale = 1.0 if w_remove <= 1e-12 else min(1.0, 0.20 * w_available / w_remove, w_available / max(w_remove, 1e-18))
            w_remove *= w_scale

            sink_scale = min(hc_scale, w_scale if qw > 1e-12 else 1.0)
            q_total *= sink_scale
            q_undamaged *= sink_scale

            ncomp_new[wi] -= hc_remove
            nw_new[wi] -= w_remove

            ncomp_new = np.clip(ncomp_new, 1e-12, None)
            nw_new = np.clip(nw_new, 1e-12, None)

            nt_new = np.sum(ncomp_new, axis=1)
            z_new = ncomp_new / nt_new[:, None]
            sw_new = saturation_from_water_moles(nw_new, self.pv)

            work_state = SimulationState(
                pressure=work_state.pressure.copy(),
                z=z_new,
                nt=nt_new,
                nw=nw_new,
                sw=sw_new,
            )

        return (
            work_state.nt, work_state.z, work_state.nw, work_state.sw,
            q_total, gas_frac, oil_frac, water_frac,
            dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss
        )

    def validate_state(self, state: SimulationState) -> tuple[bool, str]:
        if np.any(~np.isfinite(state.pressure)):
            return False, "Non-finite pressure detected"
        if np.any(~np.isfinite(state.z)) or np.any(~np.isfinite(state.nt)) or np.any(~np.isfinite(state.nw)):
            return False, "Non-finite transport state detected"
        if np.any(state.nt <= 0.0):
            return False, "Non-positive hydrocarbon moles detected"
        if np.any(state.nw < 0.0):
            return False, "Negative water moles detected"
        zsum = np.sum(state.z, axis=1)
        if np.any(np.abs(zsum - 1.0) > 1e-5):
            return False, "Composition normalization failure"
        if np.any(state.sw < -1e-8) or np.any(state.sw > 0.98):
            return False, "Water saturation out of range"
        return True, "ok"

    def step_once(self, state: SimulationState, dt_days: float) -> Tuple[SimulationState, Dict[str, float]]:
        self._set_cache_state(state)
        p_new = self.pressure_update(state, dt_days)
        state_mid = SimulationState(
            pressure=p_new,
            z=state.z.copy(),
            nt=state.nt.copy(),
            nw=state.nw.copy(),
            sw=state.sw.copy(),
        )
        self._set_cache_state(state_mid)

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

    def adaptive_step(self, state: SimulationState, dt_days: float, min_dt_days: float = 1e-3, max_retries: int = 10) -> Tuple[SimulationState, Dict[str, float], float, int]:
        trial_dt = dt_days
        retries = 0
        last_msg = "unknown"
        while retries <= max_retries:
            try:
                new_state, summary = self.step_once(state, trial_dt)
                ok, msg = self.validate_state(new_state)
                if ok:
                    return new_state, summary, trial_dt, retries
                last_msg = msg
            except Exception as exc:
                last_msg = str(exc)

            trial_dt *= 0.5
            retries += 1
            if trial_dt < min_dt_days:
                raise RuntimeError(f"Adaptive timestep failed. Last reason: {last_msg}")

        raise RuntimeError(f"Adaptive timestep failed after maximum retries. Last reason: {last_msg}")

    def record_mass_balance(self, history: SimulationHistory, state: SimulationState, dt_days: float, q_total: float, gas_frac: float, oil_frac: float, water_frac: float, injector_rate_lbmol_day: float) -> None:
        qg_lbmol_day = q_total * gas_frac
        qo_lbmol_day = q_total * oil_frac
        qw_lbmol_day = q_total * water_frac
        hc_prod = (qg_lbmol_day + qo_lbmol_day) * dt_days
        w_prod = qw_lbmol_day * dt_days
        hc_inj = injector_rate_lbmol_day * dt_days

        self.cum_hc_produced_lbmol += hc_prod
        self.cum_water_produced_lbmol += w_prod
        self.cum_hc_injected_lbmol += hc_inj

        hc_current = float(np.sum(state.nt))
        w_current = float(np.sum(state.nw))

        hc_err = self.initial_hc_lbmol + self.cum_hc_injected_lbmol - self.cum_hc_produced_lbmol - hc_current
        w_err = self.initial_water_lbmol - self.cum_water_produced_lbmol - w_current

        history.hc_mass_balance_error_lbmol.append(float(hc_err))
        history.water_mass_balance_error_lbmol.append(float(w_err))

    def run(self, state0: SimulationState, t_end_days: float, dt_days: float) -> Tuple[SimulationState, SimulationHistory]:
        if dt_days <= 0.0:
            raise ValueError("Time step must be positive")

        history = SimulationHistory()
        state = state0
        t = 0.0
        self._initialize_accounting(state0)
        dew_point = self.pvt.dew_point_psia

        cum_gas_scf = 0.0
        cum_condensate_stb = 0.0
        cum_water_stb = 0.0

        while t <= t_end_days + 1e-12:
            self._set_cache_state(state)
            self._current_time_days = t

            response0 = self.reported_well_response(state)
            q_total_0 = float(response0["q_total_lbmol_day"])
            gas_frac_0 = float(response0["gas_frac"])
            oil_frac_0 = float(response0["oil_frac"])
            water_frac_0 = float(response0["water_frac"])
            dropout_0 = float(response0["dropout"])
            krg_0 = float(response0["krg"])
            kro_0 = float(response0["kro"])
            krw_0 = float(response0["krw"])
            damage_factor_0 = float(response0["damage_factor"])
            wi_eff_0 = float(response0["wi_eff"])
            q_undamaged_0 = float(response0["q_undamaged"])
            ploss_0 = float(response0["ploss"])

            injector = self.active_injector()
            injector_rate_lbmol_day = 0.0 if (injector is None or self.scenario.name == "natural_depletion") else float(injector.rate_lbmol_day)

            next_dt = min(dt_days, t_end_days - t) if t < t_end_days else dt_days
            if next_dt <= 1e-12:
                avg_p_minus_dp = float(np.mean(state.pressure) - dew_point)
                well_p_minus_dp = float(state.pressure[self.well.cell_index] - dew_point)
                well_below_dp = 1.0 if state.pressure[self.well.cell_index] < dew_point else 0.0

                history.time_days.append(t)
                history.avg_pressure_psia.append(float(np.mean(state.pressure)))
                history.min_pressure_psia.append(float(np.min(state.pressure)))
                history.well_pressure_psia.append(float(state.pressure[self.well.cell_index]))
                history.avg_sw.append(float(np.mean(state.sw)))

                history.dew_point_psia.append(float(dew_point))
                history.avg_pressure_minus_dewpoint_psia.append(avg_p_minus_dp)
                history.well_pressure_minus_dewpoint_psia.append(well_p_minus_dp)
                history.well_below_dewpoint_flag.append(well_below_dp)

                history.well_control_mode.append(str(response0["control_mode"]))
                history.well_flowing_pwf_psia.append(float(response0["pwf_psia"]))
                history.well_estimated_thp_psia.append(float(response0["estimated_thp_psia"]))
                history.well_target_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
                if self.well.control_mode == "gas_rate":
                    history.controlled_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
                else:
                    history.controlled_gas_rate_mmscf_day.append(float(response0["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE)
                history.reported_gas_rate_mmscf_day.append(
                    float(response0.get("reported_gas_rate_mmscf_day", float(response0["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE))
                )

                if injector is None or self.scenario.name == "natural_depletion":
                    history.injector_rate_total_lbmol_day.append(0.0)
                    history.injector_rate_input_field.append(0.0)
                else:
                    history.injector_rate_total_lbmol_day.append(float(injector.rate_lbmol_day))
                    history.injector_rate_input_field.append(float(injector.rate_field_input))

                q_total = q_total_0
                gas_frac = gas_frac_0
                oil_frac = oil_frac_0
                water_frac = water_frac_0
                dropout = dropout_0
                krg = krg_0
                kro = kro_0
                krw = krw_0
                damage_factor = damage_factor_0
                wi_eff = wi_eff_0
                q_undamaged = q_undamaged_0
                ploss = ploss_0

                sep_rates = self.separator_rates(response0)
                gas_frac_report = float(sep_rates["surface_gas_fraction"])
                oil_frac_report = float(sep_rates["surface_oil_fraction"])
                water_frac_report = float(sep_rates["surface_water_fraction"])

                history.well_rate_total.append(float(q_total))
                history.well_gas_fraction.append(float(gas_frac_report))
                history.well_oil_fraction.append(float(oil_frac_report))
                history.well_water_fraction.append(float(water_frac_report))
                history.well_dropout_indicator.append(float(dropout))
                history.well_krg.append(float(krg))
                history.well_kro.append(float(kro))
                history.well_krw.append(float(krw))
                history.well_damage_factor.append(float(damage_factor))
                history.well_effective_wi.append(float(wi_eff))
                history.well_rate_undamaged.append(float(q_undamaged))
                history.productivity_loss_fraction.append(float(ploss))

                qw_lbmol_day = q_total * water_frac
                gas_rate_mmscf_day = float(sep_rates["gas_rate_mmscf_day"])
                condensate_rate_stb_day = float(sep_rates["condensate_rate_stb_day"])
                water_rate_stb_day = water_lbmol_day_to_stb_day(qw_lbmol_day)

                history.gas_rate_mmscf_day.append(float(gas_rate_mmscf_day))
                history.condensate_rate_stb_day.append(float(condensate_rate_stb_day))
                history.water_rate_stb_day.append(float(water_rate_stb_day))

                history.separator_pressure_psia.append(float(self.separator.pressure_psia))
                history.separator_temperature_R.append(float(self.separator.temperature_R))
                history.separator_vapor_fraction.append(float(sep_rates["separator_vapor_fraction"]))

                history.cum_gas_bscf.append(float(cum_gas_scf / 1.0e9))
                history.cum_condensate_mstb.append(float(cum_condensate_stb / 1.0e3))
                history.cum_water_mstb.append(float(cum_water_stb / 1.0e3))

                self.record_mass_balance(history, state, 0.0, q_total, gas_frac, oil_frac, water_frac, injector_rate_lbmol_day)
                history.accepted_dt_days.append(0.0)
                history.timestep_retries.append(0)
                break

            state_new, _, accepted_dt, retries = self.adaptive_step(state, next_dt)

            t_new = t + accepted_dt
            self._set_cache_state(state_new)
            self._current_time_days = t_new

            response1 = self.reported_well_response(state_new)
            q_total_1 = float(response1["q_total_lbmol_day"])
            gas_frac_1 = float(response1["gas_frac"])
            oil_frac_1 = float(response1["oil_frac"])
            water_frac_1 = float(response1["water_frac"])
            dropout_1 = float(response1["dropout"])
            krg_1 = float(response1["krg"])
            kro_1 = float(response1["kro"])
            krw_1 = float(response1["krw"])
            damage_factor_1 = float(response1["damage_factor"])
            wi_eff_1 = float(response1["wi_eff"])
            q_undamaged_1 = float(response1["q_undamaged"])
            ploss_1 = float(response1["ploss"])

            q_total = 0.5 * (q_total_0 + q_total_1)
            gas_frac = 0.5 * (gas_frac_0 + gas_frac_1)
            oil_frac = 0.5 * (oil_frac_0 + oil_frac_1)
            water_frac = 0.5 * (water_frac_0 + water_frac_1)
            dropout = 0.5 * (dropout_0 + dropout_1)
            krg = 0.5 * (krg_0 + krg_1)
            kro = 0.5 * (kro_0 + kro_1)
            krw = 0.5 * (krw_0 + krw_1)
            damage_factor = 0.5 * (damage_factor_0 + damage_factor_1)
            wi_eff = 0.5 * (wi_eff_0 + wi_eff_1)
            q_undamaged = 0.5 * (q_undamaged_0 + q_undamaged_1)
            ploss = 0.5 * (ploss_0 + ploss_1)

            qg_lbmol_day = 0.5 * (float(response0["qg_lbmol_day"]) + float(response1["qg_lbmol_day"]))
            qo_lbmol_day = 0.5 * (float(response0["qo_lbmol_day"]) + float(response1["qo_lbmol_day"]))
            qw_lbmol_day = q_total * water_frac

            response_avg = dict(response1)
            response_avg["qg_lbmol_day"] = qg_lbmol_day
            response_avg["qo_lbmol_day"] = qo_lbmol_day
            response_avg["x"] = 0.5 * (np.asarray(response0["x"], dtype=float) + np.asarray(response1["x"], dtype=float))
            response_avg["y"] = 0.5 * (np.asarray(response0["y"], dtype=float) + np.asarray(response1["y"], dtype=float))
            response_avg["x"] = np.asarray(response_avg["x"], dtype=float) / np.sum(response_avg["x"])
            response_avg["y"] = np.asarray(response_avg["y"], dtype=float) / np.sum(response_avg["y"])

            sep_rates = self.separator_rates(response_avg)
            gas_rate_mmscf_day = float(sep_rates["gas_rate_mmscf_day"])
            condensate_rate_stb_day = float(sep_rates["condensate_rate_stb_day"])
            water_rate_stb_day = water_lbmol_day_to_stb_day(qw_lbmol_day)

            cum_gas_scf += gas_rate_mmscf_day * 1.0e6 * accepted_dt
            cum_condensate_stb += condensate_rate_stb_day * accepted_dt
            cum_water_stb += water_rate_stb_day * accepted_dt

            avg_p_minus_dp = float(np.mean(state_new.pressure) - dew_point)
            well_p_minus_dp = float(state_new.pressure[self.well.cell_index] - dew_point)
            well_below_dp = 1.0 if state_new.pressure[self.well.cell_index] < dew_point else 0.0

            history.time_days.append(t_new)
            history.avg_pressure_psia.append(float(np.mean(state_new.pressure)))
            history.min_pressure_psia.append(float(np.min(state_new.pressure)))
            history.well_pressure_psia.append(float(state_new.pressure[self.well.cell_index]))
            history.avg_sw.append(float(np.mean(state_new.sw)))

            history.dew_point_psia.append(float(dew_point))
            history.avg_pressure_minus_dewpoint_psia.append(avg_p_minus_dp)
            history.well_pressure_minus_dewpoint_psia.append(well_p_minus_dp)
            history.well_below_dewpoint_flag.append(well_below_dp)

            history.well_control_mode.append(str(response1["control_mode"]))
            history.well_flowing_pwf_psia.append(float(response1["pwf_psia"]))
            history.well_estimated_thp_psia.append(float(response1["estimated_thp_psia"]))
            history.well_target_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
            if self.well.control_mode == "gas_rate":
                history.controlled_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
            else:
                history.controlled_gas_rate_mmscf_day.append(float(response1["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE)
            history.reported_gas_rate_mmscf_day.append(
                float(response1.get("reported_gas_rate_mmscf_day", float(response1["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE))
            )

            if injector is None or self.scenario.name == "natural_depletion":
                history.injector_rate_total_lbmol_day.append(0.0)
                history.injector_rate_input_field.append(0.0)
            else:
                history.injector_rate_total_lbmol_day.append(float(injector.rate_lbmol_day))
                history.injector_rate_input_field.append(float(injector.rate_field_input))

            gas_frac_report = float(sep_rates["surface_gas_fraction"])
            oil_frac_report = float(sep_rates["surface_oil_fraction"])
            water_frac_report = float(sep_rates["surface_water_fraction"])

            history.well_rate_total.append(float(q_total))
            history.well_gas_fraction.append(float(gas_frac_report))
            history.well_oil_fraction.append(float(oil_frac_report))
            history.well_water_fraction.append(float(water_frac_report))
            history.well_dropout_indicator.append(float(dropout))
            history.well_krg.append(float(krg))
            history.well_kro.append(float(kro))
            history.well_krw.append(float(krw))
            history.well_damage_factor.append(float(damage_factor))
            history.well_effective_wi.append(float(wi_eff))
            history.well_rate_undamaged.append(float(q_undamaged))
            history.productivity_loss_fraction.append(float(ploss))

            history.gas_rate_mmscf_day.append(float(gas_rate_mmscf_day))
            history.condensate_rate_stb_day.append(float(condensate_rate_stb_day))
            history.water_rate_stb_day.append(float(water_rate_stb_day))

            history.separator_pressure_psia.append(float(self.separator.pressure_psia))
            history.separator_temperature_R.append(float(self.separator.temperature_R))
            history.separator_vapor_fraction.append(float(sep_rates["separator_vapor_fraction"]))

            history.cum_gas_bscf.append(float(cum_gas_scf / 1.0e9))
            history.cum_condensate_mstb.append(float(cum_condensate_stb / 1.0e3))
            history.cum_water_mstb.append(float(cum_water_stb / 1.0e3))

            self.record_mass_balance(history, state_new, accepted_dt, q_total, gas_frac, oil_frac, water_frac, injector_rate_lbmol_day)
            history.accepted_dt_days.append(float(accepted_dt))
            history.timestep_retries.append(int(retries))

            state = state_new
            t = t_new

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
        dew_point_minus_pressure = np.zeros(self.grid.nx)

        self._set_cache_state(state)
        for i in range(self.grid.nx):
            fl = self.cell_flash_cached(state, i)
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
            dew_point_minus_pressure[i] = self.pvt.dew_point_psia - state.pressure[i]

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
            "dew_point_minus_pressure_psia": dew_point_minus_pressure,
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
        Component("iC5", Tc=829.4, Pc=490.4, omega=0.227, Mw=72.15),
        Component("nC5", Tc=845.4, Pc=488.6, omega=0.251, Mw=72.15),
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
# Outputs and summaries
# -----------------------------------------------------------------------------

def history_dataframe(history: SimulationHistory, start_date_text: str | None = None) -> pd.DataFrame:
    df = pd.DataFrame({
        "time_days": history.time_days,
        "avg_pressure_psia": history.avg_pressure_psia,
        "min_pressure_psia": history.min_pressure_psia,
        "well_pressure_psia": history.well_pressure_psia,
        "avg_sw": history.avg_sw,
        "dew_point_psia": history.dew_point_psia,
        "avg_pressure_minus_dewpoint_psia": history.avg_pressure_minus_dewpoint_psia,
        "well_pressure_minus_dewpoint_psia": history.well_pressure_minus_dewpoint_psia,
        "well_below_dewpoint_flag": history.well_below_dewpoint_flag,
        "well_rate_total_lbmol_day": history.well_rate_total,
        "well_rate_undamaged_lbmol_day": history.well_rate_undamaged,
        "productivity_loss_fraction": history.productivity_loss_fraction,
        "well_damage_factor": history.well_damage_factor,
        "well_effective_wi": history.well_effective_wi,
        "injector_rate_total_lbmol_day": history.injector_rate_total_lbmol_day,
        "injector_rate_input_field": history.injector_rate_input_field,
        "well_gas_fraction": history.well_gas_fraction,
        "well_oil_fraction": history.well_oil_fraction,
        "well_water_fraction": history.well_water_fraction,
        "well_dropout_indicator": history.well_dropout_indicator,
        "well_krg": history.well_krg,
        "well_kro": history.well_kro,
        "well_krw": history.well_krw,
        "gas_rate_mmscf_day": history.gas_rate_mmscf_day,
        "condensate_rate_stb_day": history.condensate_rate_stb_day,
        "water_rate_stb_day": history.water_rate_stb_day,
        "cum_gas_bscf": history.cum_gas_bscf,
        "cum_condensate_mstb": history.cum_condensate_mstb,
        "cum_water_mstb": history.cum_water_mstb,
        "hc_mass_balance_error_lbmol": history.hc_mass_balance_error_lbmol,
        "water_mass_balance_error_lbmol": history.water_mass_balance_error_lbmol,
        "accepted_dt_days": history.accepted_dt_days,
        "timestep_retries": history.timestep_retries,
        "well_control_mode": history.well_control_mode,
        "well_flowing_pwf_psia": history.well_flowing_pwf_psia,
        "well_estimated_thp_psia": history.well_estimated_thp_psia,
        "well_target_gas_rate_mmscf_day": history.well_target_gas_rate_mmscf_day,
        "controlled_gas_rate_mmscf_day": history.controlled_gas_rate_mmscf_day,
        "reported_gas_rate_mmscf_day": history.reported_gas_rate_mmscf_day,
        "separator_pressure_psia": history.separator_pressure_psia,
        "separator_temperature_R": history.separator_temperature_R,
        "separator_vapor_fraction": history.separator_vapor_fraction,
    })
    if start_date_text is not None:
        start_dt = parse_ddmmyyyy(start_date_text)
        df.insert(1, "date", [(start_dt + pd.to_timedelta(t, unit="D")).strftime("%d/%m/%Y") for t in df["time_days"]])
    return df


def spatial_dataframe(diag: Dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(diag)


def generate_chapter4_summary(history: SimulationHistory) -> List[List[object]]:
    def max_with_time(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        idx = int(np.argmax(arr))
        return float(arr[idx]), float(history.time_days[idx])

    def min_with_time(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        idx = int(np.argmin(arr))
        return float(arr[idx]), float(history.time_days[idx])

    def final_value(values: List[float]) -> float:
        return float(values[-1])

    def initial_value(values: List[float]) -> float:
        return float(values[0])

    p_min, p_min_t = min_with_time(history.avg_pressure_psia)
    dp_margin_min, dp_margin_min_t = min_with_time(history.well_pressure_minus_dewpoint_psia)
    d_peak, d_peak_t = max_with_time(history.well_dropout_indicator)
    df_min, df_min_t = min_with_time(history.well_damage_factor)
    pl_peak, pl_peak_t = max_with_time(history.productivity_loss_fraction)
    mb_hc_peak, mb_hc_t = max_with_time(np.abs(np.asarray(history.hc_mass_balance_error_lbmol, dtype=float)).tolist())

    return [
        ["Average Pressure (psia)", initial_value(history.avg_pressure_psia), p_min, p_min_t, final_value(history.avg_pressure_psia)],
        ["Dew Point (psia)", initial_value(history.dew_point_psia), final_value(history.dew_point_psia), history.time_days[-1], final_value(history.dew_point_psia)],
        ["Well Pressure - Dew Point (psia)", initial_value(history.well_pressure_minus_dewpoint_psia), dp_margin_min, dp_margin_min_t, final_value(history.well_pressure_minus_dewpoint_psia)],
        ["Average Water Saturation (-)", initial_value(history.avg_sw), max(history.avg_sw), history.time_days[int(np.argmax(history.avg_sw))], final_value(history.avg_sw)],
        ["Dropout Indicator (-)", initial_value(history.well_dropout_indicator), d_peak, d_peak_t, final_value(history.well_dropout_indicator)],
        ["Damage Factor (-, minimum is worst)", initial_value(history.well_damage_factor), df_min, df_min_t, final_value(history.well_damage_factor)],
        ["Productivity Loss Fraction (-)", initial_value(history.productivity_loss_fraction), pl_peak, pl_peak_t, final_value(history.productivity_loss_fraction)],
        ["Gas Rate (MMscf/day)", initial_value(history.gas_rate_mmscf_day), max(history.gas_rate_mmscf_day), history.time_days[int(np.argmax(history.gas_rate_mmscf_day))], final_value(history.gas_rate_mmscf_day)],
        ["Condensate Rate (STB/day)", initial_value(history.condensate_rate_stb_day), max(history.condensate_rate_stb_day), history.time_days[int(np.argmax(history.condensate_rate_stb_day))], final_value(history.condensate_rate_stb_day)],
        ["Water Rate (STB/day)", initial_value(history.water_rate_stb_day), max(history.water_rate_stb_day), history.time_days[int(np.argmax(history.water_rate_stb_day))], final_value(history.water_rate_stb_day)],
        ["Cumulative Gas (Bscf)", initial_value(history.cum_gas_bscf), max(history.cum_gas_bscf), history.time_days[int(np.argmax(history.cum_gas_bscf))], final_value(history.cum_gas_bscf)],
        ["Cumulative Condensate (MSTB)", initial_value(history.cum_condensate_mstb), max(history.cum_condensate_mstb), history.time_days[int(np.argmax(history.cum_condensate_mstb))], final_value(history.cum_condensate_mstb)],
        ["Cumulative Water (MSTB)", initial_value(history.cum_water_mstb), max(history.cum_water_mstb), history.time_days[int(np.argmax(history.cum_water_mstb))], final_value(history.cum_water_mstb)],
        ["|HC Mass Balance Error| (lbmol)", abs(initial_value(history.hc_mass_balance_error_lbmol)), mb_hc_peak, mb_hc_t, abs(final_value(history.hc_mass_balance_error_lbmol))],
    ]


def make_line_figure(x: np.ndarray, series: List[Tuple[np.ndarray, str]], xlabel: str, ylabel: str, title: str):
    fig = plt.figure(figsize=(8, 5))
    for y, label in series:
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_example(
    start_date_text: str = "01/01/2025",
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
    well_control_mode: str = "bhp",
    bhp_psia: float = 3000.0,
    drawdown_psia: float = 200.0,
    min_bhp_psia: float = 50.0,
    target_gas_rate_mmscf_day: float = 5.0,
    thp_psia: float = 500.0,
    tvd_ft: float = 8000.0,
    tubing_id_in: float = 2.441,
    wellhead_temperature_R: float = 520.0,
    thp_friction_coeff: float = 0.02,
    productivity_index: float = 1.0,
    rw_ft: float = 0.35,
    skin: float = 0.0,
    temperature_R: float = 680.0,
    p_init_psia: float = 5600.0,
    dew_point_psia: float = 4200.0,
    sw_init: float = 0.20,
    separator_pressure_psia: float = 300.0,
    separator_temperature_R: float = 520.0,
    t_end_days: float = 240.0,
    dt_days: float = 0.25,
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
        z_init = np.array([0.005, 0.02, 0.01, 0.76, 0.07, 0.04, 0.015, 0.015, 0.01, 0.01, 0.015, 0.01], dtype=float)
    else:
        z_init = np.asarray(z_init, dtype=float)
        if len(z_init) != fluid.nc:
            raise ValueError("Initial composition length does not match number of components")
        z_init = z_init / np.sum(z_init)

    if lean_gas_composition is None:
        lean_gas_composition = np.array([0.0, 0.02, 0.01, 0.94, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    lean_gas_composition = np.asarray(lean_gas_composition, dtype=float)
    if len(lean_gas_composition) != fluid.nc:
        raise ValueError("Lean gas composition length does not match number of components")
    lean_gas_composition = lean_gas_composition / np.sum(lean_gas_composition)

    refined_cells = max(2, min(refined_cells, nx - 1))
    dx_array = build_near_well_lgr(nx, length_ft, refined_cells, min_dx_ft, growth)

    grid = Grid1D(nx=nx, length_ft=length_ft, area_ft2=area_ft2, thickness_ft=thickness_ft, dx_array=dx_array)
    rock = Rock(porosity=porosity, permeability_md=permeability_md)
    relperm = RelPermParams(Sgc=Sgc, Swc=Swc, Sorw=Sorw, Sorg=Sorg, krg0=krg0, kro0=kro0, krw0=krw0, ng=ng, no=no, nw=nw)
    pvt = PVTConfig(dew_point_psia=dew_point_psia)
    separator = SeparatorConfig(pressure_psia=separator_pressure_psia, temperature_R=separator_temperature_R)
    well = Well(
        cell_index=nx - 1,
        control_mode=well_control_mode,
        bhp_psia=bhp_psia,
        drawdown_psia=drawdown_psia,
        min_bhp_psia=min_bhp_psia,
        target_gas_rate_mmscf_day=target_gas_rate_mmscf_day,
        thp_psia=thp_psia,
        tvd_ft=tvd_ft,
        tubing_id_in=tubing_id_in,
        wellhead_temperature_R=wellhead_temperature_R,
        thp_friction_coeff=thp_friction_coeff,
        productivity_index=productivity_index,
        rw_ft=rw_ft,
        skin=skin,
    )

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
        pvt=pvt,
        separator=separator,
    )

    state0 = sim.initialize_state(p_init_psia=p_init_psia, z_init=z_init, sw_init=sw_init)
    final_state, history = sim.run(state0, t_end_days=t_end_days, dt_days=dt_days)
    final_diag = sim.spatial_diagnostics(final_state)

    summary_rows = generate_chapter4_summary(history)
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Initial Value", "Peak / Critical Value", "Time of Peak / Critical (days)", "Final Value"])
    history_df = history_dataframe(history, start_date_text=start_date_text)
    spatial_df = spatial_dataframe(final_diag)

    return {
        "fluid": fluid,
        "grid": grid,
        "well": well,
        "scenario": scenario,
        "sim": sim,
        "final_state": final_state,
        "history": history,
        "history_df": history_df,
        "final_diag": final_diag,
        "spatial_df": spatial_df,
        "summary_df": summary_df,
    }


# -----------------------------------------------------------------------------
# Streamlit UI with tabs
# -----------------------------------------------------------------------------

def display_streamlit_app() -> None:
    st.set_page_config(page_title="Compositional Reservoir Simulator", layout="wide")
    st.title("Compositional Reservoir Simulator")
    st.write(
        "1D compositional gas-condensate-water prototype with Peng-Robinson EOS, semi-implicit pressure update, adaptive timestepping, "
        "Peaceman well model, local-grid refinement, three-phase relative permeability, dew-point diagnostics, scenario-based development analysis, "
        "producer control by fixed BHP, fixed drawdown, fixed gas rate, or simplified fixed THP, and one-stage surface separator flash for reporting gas and condensate rates."
    )

    input_tabs = st.tabs([
        "Scenario & Grid",
        "Rock, Well & RelPerm",
        "PVT, Separator & Schedule",
        "Compositions",
        "Run Model",
    ])

    with input_tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            scenario_label = st.selectbox("Scenario", ["Natural Depletion", "Gas Cycling", "Lean Gas Injection"], index=0)
            scenario_map = {
                "Natural Depletion": "natural_depletion",
                "Gas Cycling": "gas_cycling",
                "Lean Gas Injection": "lean_gas_injection",
            }
            scenario_name = scenario_map[scenario_label]
            injection_rate_value = st.number_input(
                "Gas injection rate",
                min_value=0.0,
                max_value=1.0e6,
                value=5.0,
                step=0.5,
                disabled=(scenario_name == "natural_depletion"),
            )
            injection_rate_unit = st.selectbox("Gas injection rate unit", ["Mscf/day", "MMscf/day"], index=1)

        with col2:
            nx = st.number_input("Number of grid cells", min_value=10, max_value=200, value=40, step=1)
            length_ft = st.number_input("Reservoir length (ft)", min_value=500.0, max_value=20000.0, value=4000.0, step=100.0)

            radius_ft = st.number_input(
                "Equivalent drainage radius (ft)",
                min_value=1.0,
                max_value=100000.0,
                value=100.0,
                step=10.0,
            )

            thickness_ft = st.number_input("Reservoir thickness (ft)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)

            equivalent_width_ft = 2.0 * radius_ft
            area_ft2 = thickness_ft * equivalent_width_ft

            st.caption(f"Equivalent model width = {equivalent_width_ft:,.2f} ft")
            st.caption(f"Equivalent cross-sectional area = {area_ft2:,.2f} ft²")

        with col3:
            refined_cells = st.number_input("Refined cells near well", min_value=2, max_value=max(int(nx) - 1, 2), value=min(10, int(nx) - 1), step=1)
            min_dx_ft = st.number_input("Minimum near-well cell size (ft)", min_value=1.0, max_value=500.0, value=12.0, step=1.0)
            growth = st.number_input("LGR growth factor", min_value=1.05, max_value=3.0, value=1.45, step=0.05)

        st.info(
            "Grid: 1D nonuniform Cartesian with near-well refinement. "
            "Radius is interpreted as an equivalent drainage half-width for this Cartesian model, "
            "so the simulator uses width = 2 × radius and cross-sectional area = thickness × width. "
            "Producer is in the last cell; injector, when active, is in the first cell."
        )

    with input_tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            porosity = st.number_input("Porosity", min_value=0.01, max_value=0.50, value=0.18, step=0.01, format="%.2f")
            permeability_md = st.number_input("Permeability (mD)", min_value=0.01, max_value=10000.0, value=8.0, step=1.0)

        with c2:
            st.subheader("Producer Well Control")
            well_control_label = st.selectbox(
                "Producer control mode",
                ["Fixed BHP", "Fixed Drawdown", "Fixed Gas Rate", "Fixed THP"],
                index=0,
            )
            control_map = {
                "Fixed BHP": "bhp",
                "Fixed Drawdown": "drawdown",
                "Fixed Gas Rate": "gas_rate",
                "Fixed THP": "thp",
            }
            well_control_mode = control_map[well_control_label]

            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1:
                bhp_psia = st.number_input(
                    "Bottom-hole pressure target (psia)",
                    min_value=100.0,
                    max_value=20000.0,
                    value=3000.0,
                    step=100.0,
                    disabled=(well_control_mode != "bhp"),
                )
            with wc2:
                drawdown_psia = st.number_input(
                    "Pressure drawdown target (psi)",
                    min_value=1.0,
                    max_value=10000.0,
                    value=200.0,
                    step=10.0,
                    disabled=(well_control_mode != "drawdown"),
                )
            with wc3:
                target_gas_rate_mmscf_day = st.number_input(
                    "Target gas rate (MMscf/day)",
                    min_value=0.0,
                    max_value=5000.0,
                    value=5.0,
                    step=0.5,
                    disabled=(well_control_mode != "gas_rate"),
                )
            with wc4:
                thp_psia = st.number_input(
                    "Tubing head pressure target (psia)",
                    min_value=14.7,
                    max_value=10000.0,
                    value=500.0,
                    step=25.0,
                    disabled=(well_control_mode != "thp"),
                )

            min_bhp_psia = st.number_input(
                "Minimum flowing BHP allowed (psia)",
                min_value=50.0,
                max_value=5000.0,
                value=50.0,
                step=25.0,
                disabled=(well_control_mode != "drawdown"),
            )

        with c3:
            productivity_index = st.number_input("Productivity index multiplier", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
            rw_ft = st.number_input("Wellbore radius (ft)", min_value=0.05, max_value=5.0, value=0.35, step=0.05)
            skin = st.number_input("Skin factor", min_value=-10.0, max_value=50.0, value=0.0, step=0.5)

        st.subheader("Tubing / Lift Inputs")
        st.caption("Used for THP calculation/reporting; only constraining under Fixed THP mode.")
        thp1, thp2, thp3, thp4 = st.columns(4)
        with thp1:
            tvd_ft = st.number_input(
                "True vertical depth, TVD (ft)",
                min_value=100.0,
                max_value=30000.0,
                value=8000.0,
                step=100.0,
            )
        with thp2:
            tubing_id_in = st.number_input(
                "Tubing ID (in)",
                min_value=1.0,
                max_value=6.0,
                value=2.441,
                step=0.1,
            )
        with thp3:
            wellhead_temperature_R = st.number_input(
                "Wellhead temperature (°R)",
                min_value=400.0,
                max_value=800.0,
                value=520.0,
                step=5.0,
            )
        with thp4:
            thp_friction_coeff = st.number_input(
                "THP friction coefficient",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.005,
            )

        r1, r2, r3 = st.columns(3)
        with r1:
            Sgc = st.number_input("Critical gas saturation, Sgc", 0.0, 0.95, 0.05, 0.01, format="%.2f")
            Swc = st.number_input("Connate water saturation, Swc", 0.0, 0.95, 0.20, 0.01, format="%.2f")
            Sorw = st.number_input("Residual condensate saturation to water, Sorw", 0.0, 0.95, 0.20, 0.01, format="%.2f")
            Sorg = st.number_input("Residual condensate saturation to gas, Sorg", 0.0, 0.95, 0.05, 0.01, format="%.2f")
        with r2:
            krg0 = st.number_input("Gas endpoint krg0", 0.0, 2.0, 1.0, 0.05, format="%.2f")
            kro0 = st.number_input("Condensate endpoint kro0", 0.0, 2.0, 1.0, 0.05, format="%.2f")
            krw0 = st.number_input("Water endpoint krw0", 0.0, 2.0, 0.4, 0.05, format="%.2f")
        with r3:
            ng = st.number_input("Gas Corey exponent, ng", 0.1, 10.0, 2.0, 0.1, format="%.1f")
            no = st.number_input("Condensate Corey exponent, no", 0.1, 10.0, 2.0, 0.1, format="%.1f")
            nw = st.number_input("Water Corey exponent, nw", 0.1, 10.0, 2.0, 0.1, format="%.1f")

        if Sgc + Swc >= 1.0 or Swc + Sorw >= 1.0 or Swc + Sorg >= 1.0:
            st.error("Invalid relative permeability inputs. Check saturation constraints.")
            return

    with input_tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature_R = st.number_input("Temperature (°R)", min_value=400.0, max_value=1200.0, value=680.0, step=10.0)
            p_init_psia = st.number_input("Initial reservoir pressure (psia)", min_value=100.0, max_value=20000.0, value=5600.0, step=100.0)
            dew_point_psia = st.number_input("Dew-point pressure (psia)", min_value=100.0, max_value=20000.0, value=4200.0, step=100.0)
            sw_init = st.number_input("Initial water saturation, Swi", min_value=0.0, max_value=0.95, value=0.20, step=0.01, format="%.2f")

        with c2:
            st.subheader("Surface Separator")
            separator_pressure_psia = st.number_input("Separator pressure (psia)", min_value=14.7, max_value=5000.0, value=300.0, step=25.0)
            separator_temperature_R = st.number_input("Separator temperature (°R)", min_value=450.0, max_value=800.0, value=520.0, step=5.0)

        with c3:
            start_date_text = st.text_input("Start date (dd/mm/yyyy)", value="01/01/2025")
            end_date_text = st.text_input("End date (dd/mm/yyyy)", value="29/08/2025")
            try:
                t_end_days = days_between_dates(start_date_text, end_date_text)
                st.caption(f"Simulation length = {t_end_days:.0f} days")
            except ValueError as e:
                st.error(str(e))
                return
            dt_days = st.number_input("Requested time step (days)", min_value=0.005, max_value=1000.0, value=0.05, step=0.01)
            st.write("Adaptive timestep retry is enabled.")

    with input_tabs[3]:
        st.subheader("Initial Hydrocarbon Composition")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            z_h2s = st.number_input("H2S mole fraction", min_value=0.0, max_value=1.0, value=0.005, step=0.001, format="%.4f")
            z_co2 = st.number_input("CO2 mole fraction", min_value=0.0, max_value=1.0, value=0.020, step=0.001, format="%.4f")
            z_n2 = st.number_input("N2 mole fraction", min_value=0.0, max_value=1.0, value=0.010, step=0.001, format="%.4f")
        with c2:
            z_c1 = st.number_input("C1 mole fraction", min_value=0.0, max_value=1.0, value=0.780, step=0.001, format="%.4f")
            z_c2 = st.number_input("C2 mole fraction", min_value=0.0, max_value=1.0, value=0.070, step=0.001, format="%.4f")
            z_c3 = st.number_input("C3 mole fraction", min_value=0.0, max_value=1.0, value=0.040, step=0.001, format="%.4f")
        with c3:
            z_ic4 = st.number_input("iC4 mole fraction", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f")
            z_nc4 = st.number_input("nC4 mole fraction", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f")
            z_ic5 = st.number_input("iC5 mole fraction", min_value=0.0, max_value=1.0, value=0.010, step=0.001, format="%.4f")
        with c4:
            z_nc5 = st.number_input("nC5 mole fraction", min_value=0.0, max_value=1.0, value=0.010, step=0.001, format="%.4f")
            z_c6 = st.number_input("C6 mole fraction", min_value=0.0, max_value=1.0, value=0.015, step=0.001, format="%.4f")
            z_c7p = st.number_input("C7+ mole fraction", min_value=0.0, max_value=1.0, value=0.010, step=0.001, format="%.4f")

        z_init = np.array([z_h2s, z_co2, z_n2, z_c1, z_c2, z_c3, z_ic4, z_nc4, z_ic5, z_nc5, z_c6, z_c7p], dtype=float)
        if np.sum(z_init) <= 0.0:
            st.error("Initial composition must have a positive total.")
            return
        z_init = z_init / np.sum(z_init)
        st.caption(f"Normalized initial composition sum: {np.sum(z_init):.4f}")

        lean_gas_composition = np.array([0.0, 0.02, 0.01, 0.94, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        if scenario_name == "lean_gas_injection":
            st.subheader("Lean Gas Injection Composition")
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                lg_h2s = st.number_input("Lean gas H2S", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
                lg_co2 = st.number_input("Lean gas CO2", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
                lg_n2 = st.number_input("Lean gas N2", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
            with lc2:
                lg_c1 = st.number_input("Lean gas C1", min_value=0.0, max_value=1.0, value=0.94, step=0.001, format="%.4f")
                lg_c2 = st.number_input("Lean gas C2", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
                lg_c3 = st.number_input("Lean gas C3", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
            with lc3:
                lg_ic4 = st.number_input("Lean gas iC4", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
                lg_nc4 = st.number_input("Lean gas nC4", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
                lg_ic5 = st.number_input("Lean gas iC5", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
            with lc4:
                lg_nc5 = st.number_input("Lean gas nC5", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
                lg_c6 = st.number_input("Lean gas C6", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
                lg_c7p = st.number_input("Lean gas C7+", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.4f")
            lean_gas_composition = np.array([lg_h2s, lg_co2, lg_n2, lg_c1, lg_c2, lg_c3, lg_ic4, lg_nc4, lg_ic5, lg_nc5, lg_c6, lg_c7p], dtype=float)

        lean_gas_composition = lean_gas_composition / np.sum(lean_gas_composition)

    with input_tabs[4]:
        st.write("Producer supports fixed BHP, fixed drawdown, fixed gas rate, and simplified fixed THP control.")
        st.write("Gas and condensate rates are reported after a one-stage separator flash at the selected separator conditions.")
        run_clicked = st.button("Run Simulation", type="primary")

    if not run_clicked:
        st.info("Configure the inputs in the tabs above, then click 'Run Simulation'.")
        return

    with st.spinner("Running compositional simulation..."):
        try:
            result = run_example(
                start_date_text=start_date_text,
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
                well_control_mode=well_control_mode,
                bhp_psia=float(bhp_psia),
                drawdown_psia=float(drawdown_psia),
                min_bhp_psia=float(min_bhp_psia),
                target_gas_rate_mmscf_day=float(target_gas_rate_mmscf_day),
                thp_psia=float(thp_psia),
                tvd_ft=float(tvd_ft),
                tubing_id_in=float(tubing_id_in),
                wellhead_temperature_R=float(wellhead_temperature_R),
                thp_friction_coeff=float(thp_friction_coeff),
                productivity_index=float(productivity_index),
                rw_ft=float(rw_ft),
                skin=float(skin),
                temperature_R=float(temperature_R),
                p_init_psia=float(p_init_psia),
                dew_point_psia=float(dew_point_psia),
                sw_init=float(sw_init),
                separator_pressure_psia=float(separator_pressure_psia),
                separator_temperature_R=float(separator_temperature_R),
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
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

    history = result["history"]
    history_df = result["history_df"]
    final_diag = result["final_diag"]
    spatial_df = result["spatial_df"]
    summary_df = result["summary_df"]
    fluid = result["fluid"]
    well = result["well"]
    final_state = result["final_state"]
    sim = result["sim"]
    grid = result["grid"]

    time_axis = pd.to_datetime(history_df["date"], format="%d/%m/%Y")

    raw_gas_rate = np.asarray(history.gas_rate_mmscf_day, dtype=float)
    if well.control_mode == "gas_rate":
        gas_rate_plot = raw_gas_rate
    else:
        gas_rate_plot = (
            pd.Series(raw_gas_rate)
            .ewm(span=20, adjust=False)
            .mean()
            .to_numpy()
        )

    st.success("Simulation completed.")

    result_tabs = st.tabs(["Summary", "History Plots", "Spatial Diagnostics", "Tables", "Downloads"])

    with result_tabs[0]:
        st.subheader("Scenario")
        st.write(f"Selected development scenario: **{scenario_label}**")
        st.write(f"Simulation period: **{start_date_text}** to **{end_date_text}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Avg Pressure (psia)", f"{history.avg_pressure_psia[-1]:.2f}")
        c2.metric("Dew Point (psia)", f"{history.dew_point_psia[-1]:.2f}")
        c3.metric("Well P - Dew Point (psia)", f"{history.well_pressure_minus_dewpoint_psia[-1]:.2f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Final Gas Rate (MMscf/day)", f"{history.gas_rate_mmscf_day[-1]:.6f}")
        c5.metric("Final Condensate Rate (STB/day)", f"{history.condensate_rate_stb_day[-1]:.3f}")
        c6.metric("Final Water Rate (STB/day)", f"{history.water_rate_stb_day[-1]:.3f}")

        c7, c8, c9 = st.columns(3)
        c7.metric("Final Cumulative Gas (Bscf)", f"{history.cum_gas_bscf[-1]:.6f}")
        c8.metric("Final Cumulative Condensate (MSTB)", f"{history.cum_condensate_mstb[-1]:.6f}")
        c9.metric("Final Cumulative Water (MSTB)", f"{history.cum_water_mstb[-1]:.6f}")

        c10, c11, c12 = st.columns(3)
        c10.metric("Average Water Saturation", f"{history.avg_sw[-1]:.4f}")
        c11.metric("Well Below Dew Point", "Yes" if history.well_below_dewpoint_flag[-1] > 0.5 else "No")
        c12.metric("Productivity Loss Fraction", f"{history.productivity_loss_fraction[-1]:.4f}")

        c13, c14, c15 = st.columns(3)
        c13.metric("Final HC Mass Balance Error (lbmol)", f"{history.hc_mass_balance_error_lbmol[-1]:.3e}")
        c14.metric("Final Water Mass Balance Error (lbmol)", f"{history.water_mass_balance_error_lbmol[-1]:.3e}")
        c15.metric("Last Accepted Δt (days)", f"{history.accepted_dt_days[-1]:.4f}")

        st.subheader("Well Control Diagnostics")
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Control Mode", history.well_control_mode[-1])
        w2.metric("Flowing Pwf (psia)", f"{history.well_flowing_pwf_psia[-1]:.2f}")
        w3.metric("Estimated THP (psia)", f"{history.well_estimated_thp_psia[-1]:.2f}")
        if history.well_control_mode[-1] == "bhp":
            w4.metric("Active Target", f"BHP = {bhp_psia:.2f} psia")
        elif history.well_control_mode[-1] == "gas_rate":
            w4.metric("Active Target", f"Gas Rate = {target_gas_rate_mmscf_day:.3f} MMscf/day")
            st.caption("Gas Production History is shown as a single reported gas-rate curve consistent with the active well control mode.")
        else:
            w4.metric("Active Target", f"THP = {thp_psia:.2f} psia")

        st.subheader("Separator Diagnostics")
        s1, s2, s3 = st.columns(3)
        s1.metric("Separator Pressure (psia)", f"{history.separator_pressure_psia[-1]:.2f}")
        s2.metric("Separator Temperature (°R)", f"{history.separator_temperature_R[-1]:.2f}")
        s3.metric("Separator Vapor Fraction", f"{history.separator_vapor_fraction[-1]:.4f}")

        st.subheader("Chapter 4 Summary")
        st.dataframe(summary_df, use_container_width=True)

        comp_df = pd.DataFrame({"Component": [c.name for c in fluid.components], "Mole Fraction": final_state.z[well.cell_index]})
        st.subheader("Well-Cell Final Hydrocarbon Composition")
        st.dataframe(comp_df, use_container_width=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.write(f"Equivalent drainage radius (ft): {radius_ft:,.2f}")
        d2.write(f"Equivalent model width (ft): {equivalent_width_ft:,.2f}")
        d3.write(f"Equivalent cross-sectional area (ft²): {area_ft2:,.2f}")
        d4.write(f"Near-well minimum dx (ft): {grid.cell_width(well.cell_index):.3f}")

        d5 = st.columns(1)[0]
        d5.write(f"Peaceman WI geom: {sim.peaceman_well_index(well.cell_index):.5f}")

    with result_tabs[1]:
        figs = [
            make_line_figure(time_axis, [(np.asarray(history.avg_pressure_psia), "Average Pressure"), (np.asarray(history.well_pressure_psia), "Well-Cell Pressure"), (np.asarray(history.dew_point_psia), "Dew Point")], "Date", "Pressure (psia)", "Pressure and Dew-Point History"),
            make_line_figure(time_axis, [(np.asarray(history.well_pressure_minus_dewpoint_psia), "Well Pressure - Dew Point"), (np.asarray(history.avg_pressure_minus_dewpoint_psia), "Average Pressure - Dew Point")], "Date", "Pressure Margin (psia)", "Pressure Margin to Dew Point"),
            make_line_figure(
                time_axis,
                [(gas_rate_plot, "Gas Production History")],
                "Date",
                "Gas Rate (MMscf/day)",
                "Gas Production History",
            ),
            make_line_figure(time_axis, [(np.asarray(history.condensate_rate_stb_day), "Condensate Rate"), (np.asarray(history.water_rate_stb_day), "Water Rate")], "Date", "Liquid Rate (STB/day)", "Condensate and Water Production Rate History"),
            make_line_figure(time_axis, [(np.asarray(history.cum_gas_bscf), "Cumulative Gas")], "Date", "Cumulative Gas (Bscf)", "Cumulative Gas Production"),
            make_line_figure(time_axis, [(np.asarray(history.cum_condensate_mstb), "Cumulative Condensate"), (np.asarray(history.cum_water_mstb), "Cumulative Water")], "Date", "Cumulative Liquid (MSTB)", "Cumulative Condensate and Water Production"),
            make_line_figure(time_axis, [(np.asarray(history.well_damage_factor), "Damage Factor"), (np.asarray(history.productivity_loss_fraction), "Productivity Loss Fraction")], "Date", "Dimensionless", "Condensate-Bank Impairment"),
            make_line_figure(time_axis, [(np.asarray(history.avg_sw), "Average Sw")], "Date", "Average Sw", "Average Water Saturation History"),
            make_line_figure(time_axis, [(np.abs(np.asarray(history.hc_mass_balance_error_lbmol)), "|HC Mass Balance Error|"), (np.abs(np.asarray(history.water_mass_balance_error_lbmol)), "|Water Mass Balance Error|")], "Date", "Absolute Error (lbmol)", "Mass Balance Error History"),
            make_line_figure(time_axis, [(np.asarray(history.well_flowing_pwf_psia), "Flowing Pwf")], "Date", "Pwf (psia)", "Flowing Bottom-Hole Pressure History"),
            make_line_figure(time_axis, [(np.asarray(history.well_estimated_thp_psia), "Estimated THP")], "Date", "THP (psia)", "Tubing Head Pressure History"),
            make_line_figure(time_axis, [(np.asarray(history.separator_vapor_fraction), "Separator Vapor Fraction")], "Date", "Vapor Fraction", "Surface Separator Flash Vapor Fraction History"),
        ]
        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)

    with result_tabs[2]:
        figs = [
            make_line_figure(np.asarray(final_diag["x_ft"]), [(np.asarray(final_diag["pressure_psia"]), "Pressure")], "Distance (ft)", "Pressure (psia)", "Final Pressure Profile"),
            make_line_figure(np.asarray(final_diag["x_ft"]), [(np.asarray(final_diag["dropout_indicator"]), "Dropout Indicator")], "Distance (ft)", "Dropout Indicator", "Final Condensate Dropout Profile"),
            make_line_figure(np.asarray(final_diag["x_ft"]), [(np.asarray(final_diag["krg"]), "krg"), (np.asarray(final_diag["kro"]), "kro"), (np.asarray(final_diag["krw"]), "krw")], "Distance (ft)", "Relative Permeability", "Final Three-Phase Relative Permeability Profile"),
            make_line_figure(np.asarray(final_diag["x_ft"]), [(np.asarray(final_diag["Sg"]), "Sg"), (np.asarray(final_diag["So"]), "So"), (np.asarray(final_diag["Sw"]), "Sw")], "Distance (ft)", "Saturation", "Final Saturation Profile"),
            make_line_figure(np.asarray(final_diag["x_ft"]), [(np.asarray(final_diag["dew_point_minus_pressure_psia"]), "Dew Point - Pressure")], "Distance (ft)", "Dew Point - Pressure (psia)", "Final Dew-Point Margin Profile"),
        ]
        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)

    with result_tabs[3]:
        st.subheader("History Table")
        st.dataframe(history_df, use_container_width=True)
        st.subheader("Spatial Diagnostics Table")
        st.dataframe(spatial_df, use_container_width=True)

    with result_tabs[4]:
        st.download_button("Download history.csv", history_df.to_csv(index=False).encode("utf-8"), file_name="history.csv", mime="text/csv")
        st.download_button("Download final_spatial_diagnostics.csv", spatial_df.to_csv(index=False).encode("utf-8"), file_name="final_spatial_diagnostics.csv", mime="text/csv")
        st.download_button("Download chapter4_summary.csv", summary_df.to_csv(index=False).encode("utf-8"), file_name="chapter4_summary.csv", mime="text/csv")
        st.caption(
            "Gas rates are reported in MMscf/day using standard conditions of 14.7 psia and 60°F. Condensate rates are separator-liquid estimates from a one-stage surface separator flash. "
            "Producer control may be fixed BHP, fixed gas rate, or simplified fixed THP. The THP mode uses an approximate tubing-loss relation and should be treated as a prototype rather than a full VLP implementation."
        )


display_streamlit_app()