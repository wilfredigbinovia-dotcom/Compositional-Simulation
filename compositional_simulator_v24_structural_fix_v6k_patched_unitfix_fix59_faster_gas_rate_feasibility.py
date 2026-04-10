from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal
import math
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import streamlit as st
import re


R = 10.7316  # psia ft^3 / (lbmol R)

# Standard conditions for reporting
P_STD_PSIA = 14.7
T_STD_R = 519.67  # 60 F
SCF_PER_LBMOL = R * T_STD_R / P_STD_PSIA
MM_PER_ONE = 1.0e-6

# Liquid reporting assumptions
CONDENSATE_DENSITY_LBFT3 = 50.0
WATER_DENSITY_LBFT3 = 62.4
AIR_DENSITY_STD_LBFT3 = 0.0764
FT3_PER_STB = 5.614583333333333

MW_WATER = 18.01528  # lbm/lbmol
WATER_VISCOSITY_CP = 0.5


def water_lbmol_per_stb() -> float:
    return WATER_DENSITY_LBFT3 * FT3_PER_STB / MW_WATER


def water_lbmol_per_mstb() -> float:
    return 1000.0 * water_lbmol_per_stb()

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
class CapillaryPressureParams:
    enabled: bool = True
    pcow_entry_psia: float = 15.0
    pcog_entry_psia: float = 8.0
    lambda_w: float = 2.0
    lambda_g: float = 2.0
    swir: float = 0.20
    sorw: float = 0.20
    sgc: float = 0.05
    sorg: float = 0.05

    def __post_init__(self) -> None:
        for value_name in ["pcow_entry_psia", "pcog_entry_psia", "lambda_w", "lambda_g"]:
            if getattr(self, value_name) < 0.0:
                raise ValueError(f"{value_name} must be non-negative")
        for value_name in ["swir", "sorw", "sgc", "sorg"]:
            value = getattr(self, value_name)
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{value_name} must be in [0, 1)")
        if self.swir + self.sorw >= 1.0:
            raise ValueError("Invalid capillary inputs: swir + sorw must be < 1.0")
        if self.sgc + self.sorg >= 1.0:
            raise ValueError("Invalid capillary inputs: sgc + sorg must be < 1.0")


@dataclass
class HysteresisParams:
    enabled: bool = True
    reversal_tolerance: float = 0.01
    gas_trapping_strength: float = 0.60
    imbibition_krg_reduction: float = 0.75
    imbibition_kro_reduction: float = 0.15

    def __post_init__(self) -> None:
        for value_name in ["reversal_tolerance", "gas_trapping_strength", "imbibition_krg_reduction", "imbibition_kro_reduction"]:
            value = getattr(self, value_name)
            if value < 0.0:
                raise ValueError(f"{value_name} must be non-negative")


@dataclass
class TransportParams:
    enabled: bool = True
    phase_split_advection: bool = True
    dispersivity_ft: float = 15.0
    molecular_diffusion_ft2_day: float = 0.15
    max_dispersive_fraction: float = 0.35

    def __post_init__(self) -> None:
        for value_name in ["dispersivity_ft", "molecular_diffusion_ft2_day", "max_dispersive_fraction"]:
            value = getattr(self, value_name)
            if value < 0.0:
                raise ValueError(f"{value_name} must be non-negative")
        self.max_dispersive_fraction = float(min(max(self.max_dispersive_fraction, 0.0), 1.0))


@dataclass
class BoundaryConditionConfig:
    left_mode: str = "closed"
    right_mode: str = "closed"
    left_pressure_psia: float = 5600.0
    right_pressure_psia: float = 3000.0
    left_transmissibility_multiplier: float = 1.0
    right_transmissibility_multiplier: float = 1.0

    def __post_init__(self) -> None:
        allowed = {"closed", "constant_pressure"}
        if self.left_mode not in allowed:
            raise ValueError("left_mode must be 'closed' or 'constant_pressure'")
        if self.right_mode not in allowed:
            raise ValueError("right_mode must be 'closed' or 'constant_pressure'")
        self.left_transmissibility_multiplier = max(float(self.left_transmissibility_multiplier), 0.0)
        self.right_transmissibility_multiplier = max(float(self.right_transmissibility_multiplier), 0.0)


@dataclass
class AquiferConfig:
    enabled: bool = False
    side: str = "left"
    initial_pressure_psia: float = 5600.0
    productivity_index_lbmol_day_psi: float = 500.0
    total_capacity_lbmol_per_psi: float = 5.0e5
    water_influx_fraction: float = 1.0
    allow_backflow: bool = False

    def __post_init__(self) -> None:
        allowed = {"left", "right"}
        if self.side not in allowed:
            raise ValueError("Aquifer side must be 'left' or 'right'")
        self.productivity_index_lbmol_day_psi = max(float(self.productivity_index_lbmol_day_psi), 0.0)
        self.total_capacity_lbmol_per_psi = max(float(self.total_capacity_lbmol_per_psi), 1.0)
        self.water_influx_fraction = float(min(max(self.water_influx_fraction, 0.0), 1.0))


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
    tubing_roughness_in: float = 0.0006
    tubing_calibration_factor: float = 1.0
    tubing_model: str = "mechanistic"
    tubing_segments: int = 25
    min_lift_thp_psia: float = 100.0
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
    start_day: float = 0.0
    end_day: float = float("inf")
    control_mode: str = "simple"
    injection_pressure_psia: float = 7000.0
    max_bhp_psia: float = 9000.0
    injectivity_index_lbmol_day_psi: float = 500.0

    def __post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValueError("Injector cell index must be non-negative")
        if self.rate_lbmol_day < 0.0:
            raise ValueError("Injector rate must be non-negative")
        if self.end_day < self.start_day:
            raise ValueError("Injection end day must be greater than or equal to start day")
        if self.control_mode not in {"simple", "enhanced"}:
            raise ValueError("Injector control mode must be 'simple' or 'enhanced'")
        comp = np.asarray(self.injection_composition, dtype=float)
        if comp.ndim != 1 or comp.size == 0:
            raise ValueError("Injection composition must be a non-empty 1D array")
        if np.sum(comp) <= 0.0:
            raise ValueError("Injection composition must have positive total")
        self.injection_composition = comp / np.sum(comp)

    def is_active(self, time_days: float) -> bool:
        return (time_days >= self.start_day) and (time_days <= self.end_day)


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
    table: PVTTable | None = None


@dataclass
class SeparatorConfig:
    pressure_psia: float = 300.0
    temperature_R: float = 520.0
    stages: int = 1
    second_stage_pressure_psia: float = 100.0
    second_stage_temperature_R: float = 520.0
    third_stage_pressure_psia: float = 50.0
    third_stage_temperature_R: float = 520.0
    stock_tank_pressure_psia: float = 14.7
    stock_tank_temperature_R: float = 519.67


@dataclass
class PVTTable:
    pressure_psia: np.ndarray
    dew_point_psia: np.ndarray | None = None
    reservoir_cgr_stb_mmscf: np.ndarray | None = None
    vaporized_cgr_stb_mmscf: np.ndarray | None = None
    z_factor: np.ndarray | None = None
    gas_fvf_ft3_scf: np.ndarray | None = None
    gas_viscosity_cp: np.ndarray | None = None
    oil_fvf_rb_stb: np.ndarray | None = None
    oil_viscosity_cp: np.ndarray | None = None
    gas_oil_ratio_scf_stb: np.ndarray | None = None
    gas_density_lbft3: np.ndarray | None = None
    water_fvf_rb_stb: np.ndarray | None = None
    water_viscosity_cp: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pressure_psia = np.asarray(self.pressure_psia, dtype=float)
        order = np.argsort(self.pressure_psia)
        self.pressure_psia = self.pressure_psia[order]
        for name in (
            'dew_point_psia', 'reservoir_cgr_stb_mmscf', 'vaporized_cgr_stb_mmscf',
            'z_factor', 'gas_fvf_ft3_scf', 'gas_viscosity_cp', 'oil_fvf_rb_stb',
            'oil_viscosity_cp', 'gas_oil_ratio_scf_stb', 'gas_density_lbft3',
            'water_fvf_rb_stb', 'water_viscosity_cp',
        ):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if len(arr) != len(self.pressure_psia):
                raise ValueError(f'PVT column {name} must have same length as pressure')
            setattr(self, name, arr[order])

    def has(self, field_name: str) -> bool:
        return getattr(self, field_name, None) is not None

    def interp(self, pressure_psia: float, field_name: str, default: float | None = None) -> float | None:
        arr = getattr(self, field_name, None)
        if arr is None:
            return default
        return float(np.interp(float(pressure_psia), self.pressure_psia, arr, left=arr[0], right=arr[-1]))


def _normalize_pvt_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')


def load_pvt_table_from_dataframe(df: pd.DataFrame) -> PVTTable:
    df = df.copy()
    df.columns = [_normalize_pvt_column_name(c) for c in df.columns]

    alias_map = {
        'pressure': 'pressure_psig',
        'pressure_psia': 'pressure_psig',
        'dew_point': 'dew_point_psig',
        'dewpoint': 'dew_point_psig',
        'reservoir_cgr': 'reservoir_cgr_stb_mmscf',
        'vapour_cgr': 'vaporized_cgr_stb_mmscf',
        'vaporized_cgr': 'vaporized_cgr_stb_mmscf',
        'gas_fvf': 'gas_fvf_ft3_scf',
        'oil_fvf': 'oil_fvf_rb_stb',
        'solution_gor': 'gas_oil_ratio_scf_stb',
        'gas_oil_ratio': 'gas_oil_ratio_scf_stb',
        'gor': 'gas_oil_ratio_scf_stb',
        'gas_density': 'gas_density_lbft3',
        'gas_z_factor': 'z_factor',
        'zfactor': 'z_factor',
        'water_fvf': 'water_fvf_rb_stb',
        'water_viscosity': 'water_viscosity_cp',
        'gas_viscosity': 'gas_viscosity_cp',
        'oil_viscosity': 'oil_viscosity_cp',
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if 'pressure_psig' not in df.columns:
        raise ValueError('PVT CSV must contain a Pressure column (psig).')

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['pressure_psig']).copy()
    if df.empty:
        raise ValueError('No numeric pressure rows found in uploaded PVT CSV.')

    pressure_psia = df['pressure_psig'].to_numpy(dtype=float) + 14.7

    def arr(name: str) -> np.ndarray | None:
        if name not in df.columns:
            return None
        vals = df[name].to_numpy(dtype=float)
        good = np.isfinite(vals)
        if not np.any(good):
            return None
        # fill isolated missing values by linear interpolation over valid points
        if not np.all(good):
            x = np.flatnonzero(good)
            vals = np.interp(np.arange(len(vals)), x, vals[good])
        return vals

    dew = arr('dew_point_psig')
    if dew is not None:
        dew = dew + 14.7

    return PVTTable(
        pressure_psia=pressure_psia,
        dew_point_psia=dew,
        reservoir_cgr_stb_mmscf=arr('reservoir_cgr_stb_mmscf'),
        vaporized_cgr_stb_mmscf=arr('vaporized_cgr_stb_mmscf'),
        z_factor=arr('z_factor'),
        gas_fvf_ft3_scf=arr('gas_fvf_ft3_scf'),
        gas_viscosity_cp=arr('gas_viscosity_cp'),
        oil_fvf_rb_stb=arr('oil_fvf_rb_stb'),
        oil_viscosity_cp=arr('oil_viscosity_cp'),
        gas_oil_ratio_scf_stb=arr('gas_oil_ratio_scf_stb'),
        gas_density_lbft3=arr('gas_density_lbft3'),
        water_fvf_rb_stb=arr('water_fvf_rb_stb'),
        water_viscosity_cp=arr('water_viscosity_cp'),
    )


@dataclass
class ReportingConfig:
    gas_specific_gravity: float = 0.65
    condensate_api_gravity: float = 50.0

    def __post_init__(self) -> None:
        if self.gas_specific_gravity <= 0.0:
            raise ValueError("Gas specific gravity must be positive")
        if self.condensate_api_gravity <= -131.0:
            raise ValueError("Condensate API gravity must be greater than -131")

    @property
    def gas_density_lbft3(self) -> float:
        return gas_specific_gravity_to_density_lbft3(self.gas_specific_gravity)

    @property
    def condensate_density_lbft3(self) -> float:
        return api_gravity_to_density_lbft3(self.condensate_api_gravity)


@dataclass
class SimulationState:
    pressure: np.ndarray
    z: np.ndarray
    nt: np.ndarray
    nw: np.ndarray
    sw: np.ndarray
    sg_max: np.ndarray


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
    well_hysteresis_trap_fraction: List[float] = field(default_factory=list)
    well_hysteresis_imbibition_flag: List[float] = field(default_factory=list)

    injector_rate_total_lbmol_day: List[float] = field(default_factory=list)
    injector_rate_input_field: List[float] = field(default_factory=list)
    injector_rate_target_lbmol_day: List[float] = field(default_factory=list)
    injector_rate_achievable_lbmol_day: List[float] = field(default_factory=list)
    injector_cell_pressure_psia: List[float] = field(default_factory=list)
    injector_pressure_delta_psia: List[float] = field(default_factory=list)
    injector_effective_bhp_psia: List[float] = field(default_factory=list)
    injector_active_flag: List[float] = field(default_factory=list)

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
    well_raw_estimated_thp_psia: List[float] = field(default_factory=list)
    well_thp_clipped_to_atmospheric: List[float] = field(default_factory=list)
    well_lift_feasible: List[float] = field(default_factory=list)
    well_tubing_hydrostatic_psi: List[float] = field(default_factory=list)
    well_tubing_friction_psi: List[float] = field(default_factory=list)
    well_tubing_acceleration_psi: List[float] = field(default_factory=list)
    well_tubing_mixture_velocity_ft_s: List[float] = field(default_factory=list)
    well_tubing_reynolds_number: List[float] = field(default_factory=list)
    well_tubing_friction_factor: List[float] = field(default_factory=list)
    well_target_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    controlled_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    reported_gas_rate_mmscf_day: List[float] = field(default_factory=list)

    transport_advective_flux_lbmol_day: List[float] = field(default_factory=list)
    transport_dispersive_flux_lbmol_day: List[float] = field(default_factory=list)
    transport_total_flux_lbmol_day: List[float] = field(default_factory=list)
    aquifer_rate_lbmol_day: List[float] = field(default_factory=list)
    aquifer_cumulative_mlbmol: List[float] = field(default_factory=list)
    aquifer_pressure_psia: List[float] = field(default_factory=list)
    left_boundary_pressure_psia: List[float] = field(default_factory=list)
    right_boundary_pressure_psia: List[float] = field(default_factory=list)

    separator_pressure_psia: List[float] = field(default_factory=list)
    separator_temperature_R: List[float] = field(default_factory=list)
    separator_vapor_fraction: List[float] = field(default_factory=list)
    separator_stage_count: List[float] = field(default_factory=list)
    separator_total_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    separator_stage1_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    separator_stage2_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    separator_stage3_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    separator_stock_tank_gas_rate_mmscf_day: List[float] = field(default_factory=list)
    separator_stock_tank_liquid_rate_stb_day: List[float] = field(default_factory=list)


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


def darcy_friction_factor_goudar_sonnad(reynolds: float, rel_roughness: float) -> float:
    """Darcy friction factor using laminar flow and the Goudar-Sonnad explicit turbulent approximation."""
    Re = max(float(reynolds), 0.0)
    rr = max(float(rel_roughness), 0.0)

    if Re < 1e-12:
        return 0.0
    if Re < 2100.0:
        return 64.0 / Re

    ln10 = math.log(10.0)
    a = 2.0 / ln10
    b = rr / 3.7
    d = ln10 * Re / 5.02
    d = max(d, 1e-30)
    s = max(b * d + math.log(d), 1e-30)
    q = max(s ** (s / (s + 1.0)), 1e-30)
    g = max(b * d + math.log(d / q), 1e-30)
    z = math.log(q / g)
    dla = z * g / (g + 1.0)
    dcfa = dla * (1.0 + (z / 2.0) / (((g + 1.0) ** 2) + (z / 3.0) * (2.0 * g - 1.0)))
    inv_sqrt_f = max(a * (math.log(d / q) + dcfa), 1e-30)
    return max(1.0 / (inv_sqrt_f ** 2), 1e-12)


def gas_density_lbft3_from_eos(eos: PengRobinsonEOS, comp: np.ndarray, pressure_psia: float, temperature_R: float) -> float:
    comp = np.asarray(comp, dtype=float)
    comp = comp / max(np.sum(comp), 1e-12)
    mw = float(np.dot(comp, eos.Mw))
    try:
        zfac = max(float(eos.z_factor(comp, max(pressure_psia, 14.7), temperature_R, phase="v")), 0.05)
    except Exception:
        zfac = 1.0
    rho_lbmol_ft3 = max(pressure_psia, 14.7) / max(zfac * R * temperature_R, 1e-12)
    return float(np.clip(rho_lbmol_ft3 * mw, 0.02, 25.0))


def hydrocarbon_gas_rate_ft3_day(lbmol_day: float, comp: np.ndarray, pressure_psia: float, temperature_R: float, eos: PengRobinsonEOS) -> float:
    comp = np.asarray(comp, dtype=float)
    comp = comp / max(np.sum(comp), 1e-12)
    try:
        zfac = max(float(eos.z_factor(comp, max(pressure_psia, 14.7), temperature_R, phase="v")), 0.05)
    except Exception:
        zfac = 1.0
    return float(max(lbmol_day, 0.0) * zfac * R * temperature_R / max(pressure_psia, 14.7))


def classify_vertical_flow_regime(vsg_ft_s: float, vsl_ft_s: float, diameter_ft: float, rho_l_lbft3: float, rho_g_lbft3: float) -> str:
    vm = max(vsg_ft_s + vsl_ft_s, 1e-9)
    lambda_l = float(np.clip(vsl_ft_s / vm, 0.0, 1.0))
    froude_m = vm / max(math.sqrt(32.174 * max(diameter_ft, 1e-9)), 1e-9)
    density_contrast = max((rho_l_lbft3 - rho_g_lbft3) / max(rho_l_lbft3, 1e-12), 0.0)

    if vsg_ft_s < 0.5 and lambda_l >= 0.4:
        return "bubble"
    if lambda_l >= 0.25 and froude_m < 4.0:
        return "slug"
    if froude_m < 10.0 and density_contrast > 0.1:
        return "churn"
    return "annular"


def regime_liquid_holdup(vsg_ft_s: float, vsl_ft_s: float, diameter_ft: float, rho_l_lbft3: float, rho_g_lbft3: float) -> Tuple[float, str]:
    vm = max(vsg_ft_s + vsl_ft_s, 1e-9)
    lambda_l = float(np.clip(vsl_ft_s / vm, 0.0, 1.0))
    regime = classify_vertical_flow_regime(vsg_ft_s, vsl_ft_s, diameter_ft, rho_l_lbft3, rho_g_lbft3)
    slip_scale = math.sqrt(max(32.174 * diameter_ft * max((rho_l_lbft3 - rho_g_lbft3) / max(rho_l_lbft3, 1e-12), 0.0), 0.0))

    if regime == "bubble":
        c0 = 1.05
        vd = 0.35 * slip_scale
        hg = np.clip(vsg_ft_s / max(c0 * vm + vd, 1e-9), 0.0, 1.0)
        hl = max(1.0 - hg, lambda_l)
    elif regime == "slug":
        c0 = 1.20
        vd = 0.80 * slip_scale
        hg = np.clip(vsg_ft_s / max(c0 * vm + vd, 1e-9), 0.0, 1.0)
        hl = max(1.0 - hg, min(lambda_l + 0.15, 0.98))
    elif regime == "churn":
        c0 = 1.35
        vd = 1.20 * slip_scale
        hg = np.clip(vsg_ft_s / max(c0 * vm + vd, 1e-9), 0.0, 1.0)
        hl = max(1.0 - hg, min(0.08 + 0.85 * lambda_l, 0.95))
    else:
        film_holdup = 0.04 + 0.20 * lambda_l + 0.015 * max(diameter_ft * 12.0 - 2.0, 0.0)
        hl = max(min(film_holdup, 0.35), 0.02)
        hl = max(min(hl, 0.98), min(lambda_l + 0.03, 0.98))

    return float(np.clip(hl, 0.0, 0.98)), regime


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




def normalized_saturation(value: float, s_min: float, s_max: float) -> float:
    if s_max <= s_min:
        return 0.0
    return float(np.clip((value - s_min) / max(s_max - s_min, 1e-10), 0.0, 1.0))


def capillary_pressure_pcow_psia(Sw: float, params: CapillaryPressureParams) -> float:
    if (not params.enabled) or params.pcow_entry_psia <= 0.0:
        return 0.0
    sw_eff = normalized_saturation(Sw, params.swir, 1.0 - params.sorw)
    return float(params.pcow_entry_psia * max(1.0 - sw_eff, 0.0) ** max(params.lambda_w, 1e-8))


def capillary_pressure_pcog_psia(Sg: float, params: CapillaryPressureParams) -> float:
    if (not params.enabled) or params.pcog_entry_psia <= 0.0:
        return 0.0
    sg_eff = normalized_saturation(Sg, params.sgc, 1.0 - params.sorg)
    return float(params.pcog_entry_psia * max(1.0 - sg_eff, 0.0) ** max(params.lambda_g, 1e-8))


def hysteresis_gas_trap_fraction(Sg: float, Sg_max: float, params: HysteresisParams) -> tuple[float, bool]:
    if not params.enabled:
        return 0.0, False
    Sg = float(np.clip(Sg, 0.0, 1.0))
    Sg_max = float(np.clip(Sg_max, 0.0, 1.0))
    if Sg_max <= max(params.reversal_tolerance, 1e-8):
        return 0.0, False
    if Sg >= Sg_max - params.reversal_tolerance:
        return 0.0, False

    retreat = max(Sg_max - Sg, 0.0)
    retreat_norm = retreat / max(Sg_max, 1e-8)
    trap_fraction = float(np.clip(params.gas_trapping_strength * retreat_norm, 0.0, 0.95))
    return trap_fraction, True


def apply_relperm_hysteresis(
    krg: float,
    kro: float,
    Sg: float,
    Sg_max: float,
    params: HysteresisParams,
) -> tuple[float, float, float, float]:
    trap_fraction, imbibition = hysteresis_gas_trap_fraction(Sg, Sg_max, params)
    if imbibition:
        krg *= max(1.0 - params.imbibition_krg_reduction * trap_fraction, 0.02)
        kro *= max(1.0 - params.imbibition_kro_reduction * trap_fraction, 0.05)
    return float(krg), float(kro), float(trap_fraction), float(1.0 if imbibition else 0.0)


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


def injectivity_mmscf_day_psi_to_lbmol_day_psi(injectivity_mmscf_day_psi: float) -> float:
    return gas_mmscf_day_to_lbmol_day(injectivity_mmscf_day_psi)


def gas_field_rate_to_lbmol_day(rate_value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "mscf/day":
        scf_day = rate_value * 1.0e3
    elif unit == "mmscf/day":
        scf_day = rate_value * 1.0e6
    else:
        raise ValueError(f"Unsupported gas rate unit: {unit}")
    return scf_day / SCF_PER_LBMOL


def gas_specific_gravity_to_density_lbft3(gas_specific_gravity: float) -> float:
    gas_specific_gravity = float(max(gas_specific_gravity, 1e-6))
    return gas_specific_gravity * AIR_DENSITY_STD_LBFT3


def api_gravity_to_density_lbft3(api_gravity: float) -> float:
    api_gravity = float(api_gravity)
    oil_specific_gravity = 141.5 / max(api_gravity + 131.5, 1e-6)
    return oil_specific_gravity * WATER_DENSITY_LBFT3


def fahrenheit_to_rankine(temperature_f: float) -> float:
    return float(temperature_f) + 459.67


def rankine_to_fahrenheit(temperature_r: float) -> float:
    return float(temperature_r) - 459.67


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


def days_between_dates(start_date_text: str, end_date_text: str, *, allow_same_day: bool = False) -> float:
    start_dt = parse_ddmmyyyy(start_date_text)
    end_dt = parse_ddmmyyyy(end_date_text)
    delta_days = (end_dt - start_dt).days
    min_delta = 0 if allow_same_day else 1
    if delta_days < min_delta:
        if allow_same_day:
            raise ValueError("End date must be on or after start date.")
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
        capillary: CapillaryPressureParams | None = None,
        hysteresis: HysteresisParams | None = None,
        transport: TransportParams | None = None,
        boundary: BoundaryConditionConfig | None = None,
        aquifer: AquiferConfig | None = None,
        pvt: PVTConfig | None = None,
        separator: SeparatorConfig | None = None,
        reporting: ReportingConfig | None = None,
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
        self.capillary = capillary or CapillaryPressureParams(
            swir=self.relperm.Swc,
            sorw=self.relperm.Sorw,
            sgc=self.relperm.Sgc,
            sorg=self.relperm.Sorg,
        )
        self.hysteresis = hysteresis or HysteresisParams()
        self.transport = transport or TransportParams()
        self.boundary = boundary or BoundaryConditionConfig()
        self.aquifer = aquifer or AquiferConfig(enabled=False, initial_pressure_psia=4000.0)
        self.pvt = pvt or PVTConfig(dew_point_psia=4000.0)
        self.pvt_table = self.pvt.table
        self.separator = separator or SeparatorConfig()
        self.reporting = reporting or ReportingConfig()
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
        self.cum_aquifer_water_lbmol = 0.0

    def _initialize_accounting(self, state0: SimulationState) -> None:
        self.initial_hc_lbmol = float(np.sum(state0.nt))
        self.initial_water_lbmol = float(np.sum(state0.nw))
        self.cum_hc_produced_lbmol = 0.0
        self.cum_hc_injected_lbmol = 0.0
        self.cum_water_produced_lbmol = 0.0
        self.cum_aquifer_water_lbmol = 0.0

    def _invalidate_caches(self) -> None:
        self._flash_cache = {}
        self._mobility_cache = {}

    def _set_cache_state(self, state: SimulationState) -> None:
        self._cache_state_id = id(state)
        self._invalidate_caches()

    def pvt_lookup(self, pressure_psia: float) -> Dict[str, float | None]:
        if self.pvt_table is None:
            return {}
        return {
            'dew_point_psia': self.pvt_table.interp(pressure_psia, 'dew_point_psia', self.pvt.dew_point_psia),
            'z_factor': self.pvt_table.interp(pressure_psia, 'z_factor', None),
            'gas_viscosity_cp': self.pvt_table.interp(pressure_psia, 'gas_viscosity_cp', None),
            'oil_viscosity_cp': self.pvt_table.interp(pressure_psia, 'oil_viscosity_cp', None),
            'gas_density_lbft3': self.pvt_table.interp(pressure_psia, 'gas_density_lbft3', None),
            'water_viscosity_cp': self.pvt_table.interp(pressure_psia, 'water_viscosity_cp', None),
            'gas_fvf_ft3_scf': self.pvt_table.interp(pressure_psia, 'gas_fvf_ft3_scf', None),
            'oil_fvf_rb_stb': self.pvt_table.interp(pressure_psia, 'oil_fvf_rb_stb', None),
            'reservoir_cgr_stb_mmscf': self.pvt_table.interp(pressure_psia, 'reservoir_cgr_stb_mmscf', None),
            'vaporized_cgr_stb_mmscf': self.pvt_table.interp(pressure_psia, 'vaporized_cgr_stb_mmscf', None),
            'gas_oil_ratio_scf_stb': self.pvt_table.interp(pressure_psia, 'gas_oil_ratio_scf_stb', None),
        }

    def dew_point_at_pressure(self, pressure_psia: float) -> float:
        if self.pvt_table is not None and self.pvt_table.has('dew_point_psia'):
            return float(self.pvt_table.interp(pressure_psia, 'dew_point_psia', self.pvt.dew_point_psia))
        return float(self.pvt.dew_point_psia)

    def initialize_state(self, p_init_psia: float, z_init: np.ndarray, sw_init: float = 0.20) -> SimulationState:
        z_init = np.asarray(z_init, dtype=float)
        z_init = z_init / np.sum(z_init)

        pressure = np.full(self.grid.nx, p_init_psia, dtype=float)
        z = np.tile(z_init, (self.grid.nx, 1))

        sw = np.full(self.grid.nx, sw_init, dtype=float)
        sw = np.clip(sw, 0.0, 0.95)

        pvt_props_init = self.pvt_lookup(p_init_psia)
        Zg = pvt_props_init.get('z_factor') if pvt_props_init.get('z_factor') is not None else self.eos.z_factor(z_init, p_init_psia, self.T, phase="v")
        molar_density = z_to_density_lbmol_ft3(p_init_psia, self.T, Zg)

        hc_pv = self.pv * (1.0 - sw)
        nt = molar_density * hc_pv
        nw = water_moles_from_saturation(sw, self.pv)

        state0_base = SimulationState(
            pressure=pressure,
            z=z,
            nt=nt,
            nw=nw,
            sw=sw,
            sg_max=np.zeros_like(sw),
        )
        sg_profile = self.compute_flowing_sg_profile(state0_base)
        state0 = SimulationState(
            pressure=pressure,
            z=z,
            nt=nt,
            nw=nw,
            sw=sw,
            sg_max=sg_profile.copy(),
        )
        self._initialize_accounting(state0)
        return state0

    def cell_flash_cached(self, state: SimulationState, i: int) -> Dict[str, np.ndarray | float | str]:
        if self._cache_state_id != id(state):
            self._set_cache_state(state)
        if i not in self._flash_cache:
            self._flash_cache[i] = self.flash.flash(state.z[i], state.pressure[i], self.T)
        return self._flash_cache[i]

    def compute_flowing_sg_profile(self, state: SimulationState) -> np.ndarray:
        sg_profile = np.zeros(self.grid.nx, dtype=float)
        self._set_cache_state(state)
        for i in range(self.grid.nx):
            fl = self.cell_flash_cached(state, i)
            beta = float(fl["beta"])
            Sw = float(np.clip(state.sw[i], 0.0, 0.95))
            Sg, _, _ = beta_to_flowing_saturations(beta, Sw, self.relperm)
            sg_profile[i] = Sg
        return sg_profile

    def phase_mobility_data(self, state: SimulationState, i: int) -> Dict[str, float]:
        if self._cache_state_id != id(state):
            self._set_cache_state(state)
        if i in self._mobility_cache:
            return self._mobility_cache[i]

        fl = self.cell_flash_cached(state, i)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])

        pvt_props = self.pvt_lookup(state.pressure[i])
        mu_o = pvt_props.get('oil_viscosity_cp') if pvt_props.get('oil_viscosity_cp') is not None else phase_viscosity_cp("l", x, state.pressure[i], self.T)
        mu_g = pvt_props.get('gas_viscosity_cp') if pvt_props.get('gas_viscosity_cp') is not None else phase_viscosity_cp("v", y, state.pressure[i], self.T)
        mu_w = pvt_props.get('water_viscosity_cp') if pvt_props.get('water_viscosity_cp') is not None else WATER_VISCOSITY_CP

        Sw = float(np.clip(state.sw[i], 0.0, 0.95))
        Sg, So, Sw = beta_to_flowing_saturations(beta, Sw, self.relperm)

        krg, kro, krw = three_phase_relperm(Sg, So, Sw, self.relperm)
        sg_max_cell = float(np.clip(state.sg_max[i] if hasattr(state, "sg_max") else Sg, 0.0, 1.0))
        krg, kro, trap_fraction, imbibition_flag = apply_relperm_hysteresis(
            krg, kro, Sg, sg_max_cell, self.hysteresis
        )

        lam_g = krg / max(mu_g, 1e-8)
        lam_o = kro / max(mu_o, 1e-8)
        lam_w = krw / max(mu_w, 1e-8)
        lam_t = lam_g + lam_o + lam_w

        pcog = capillary_pressure_pcog_psia(Sg, self.capillary)
        pcow = capillary_pressure_pcow_psia(Sw, self.capillary)

        data = {
            "Sg": Sg,
            "So": So,
            "Sw": Sw,
            "pcog_psia": pcog,
            "pcow_psia": pcow,
            "krg": krg,
            "kro": kro,
            "krw": krw,
            "sg_max": sg_max_cell,
            "hysteresis_trap_fraction": trap_fraction,
            "hysteresis_imbibition_flag": imbibition_flag,
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

    def hydrocarbon_potential_psia(self, state: SimulationState, i: int) -> float:
        mob = self.phase_mobility_data(state, i)
        fl = self.cell_flash_cached(state, i)
        beta = float(fl["beta"])
        return float(state.pressure[i] - (1.0 - beta) * mob["pcog_psia"])

    def water_potential_psia(self, state: SimulationState, i: int) -> float:
        mob = self.phase_mobility_data(state, i)
        return float(state.pressure[i] - mob["pcog_psia"] - mob["pcow_psia"])

    def transmissibility(self, state: SimulationState, i: int, j: int) -> float:
        mob_i = self.phase_mobility_data(state, i)
        mob_j = self.phase_mobility_data(state, j)
        lam_i = mob_i["lam_t"]
        lam_j = mob_j["lam_t"]
        lam_face = 2.0 * lam_i * lam_j / max(lam_i + lam_j, 1e-12)
        distance = self.grid.interface_distance(i, j)
        return self.k * self.grid.area_ft2 * lam_face / max(distance, 1e-12)

    def hydrocarbon_velocity_ft_day(self, state: SimulationState, i: int, j: int) -> float:
        p_i = self.hydrocarbon_potential_psia(state, i)
        p_j = self.hydrocarbon_potential_psia(state, j)
        dp = p_i - p_j
        distance = self.grid.interface_distance(i, j)
        if distance <= 1e-12:
            return 0.0
        trans = self.k * self.grid.area_ft2 / max(distance, 1e-12)
        mob_i = self.phase_mobility_data(state, i)
        mob_j = self.phase_mobility_data(state, j)
        lam_hc_i = max(mob_i["lam_hc"], 0.0)
        lam_hc_j = max(mob_j["lam_hc"], 0.0)
        lam_face = 2.0 * lam_hc_i * lam_hc_j / max(lam_hc_i + lam_hc_j, 1e-12)
        q_hc = trans * lam_face * dp * 1e-5
        return float(q_hc / max(self.grid.area_ft2 * self.phi, 1e-12))

    def component_flux_breakdown_between(self, state: SimulationState, i: int, j: int) -> Dict[str, np.ndarray | float]:
        p_i = self.hydrocarbon_potential_psia(state, i)
        p_j = self.hydrocarbon_potential_psia(state, j)
        dp = p_i - p_j
        distance = self.grid.interface_distance(i, j)
        if abs(dp) < 1e-12 or distance <= 1e-12:
            zeros = np.zeros(self.nc)
            return {
                "advective_total": zeros.copy(),
                "dispersive_total": zeros.copy(),
                "gas_advective": zeros.copy(),
                "oil_advective": zeros.copy(),
                "total": zeros.copy(),
                "upstream": i,
                "dispersion_coeff_ft2_day": 0.0,
            }

        upstream = i if dp >= 0.0 else j
        downstream = j if upstream == i else i
        fl_up = self.cell_flash_cached(state, upstream)
        fl_dn = self.cell_flash_cached(state, downstream)
        mob_up = self.phase_mobility_data(state, upstream)

        beta_up = float(fl_up["beta"])
        x_up = np.asarray(fl_up["x"], dtype=float)
        y_up = np.asarray(fl_up["y"], dtype=float)

        trans = self.k * self.grid.area_ft2 / max(distance, 1e-12)
        lam_g = max(mob_up["lam_g"], 0.0)
        lam_o = max(mob_up["lam_o"], 0.0)

        gas_advective = np.zeros(self.nc)
        oil_advective = np.zeros(self.nc)
        if self.transport.phase_split_advection:
            qg = trans * lam_g * dp * 1e-5
            qo = trans * lam_o * dp * 1e-5
            gas_advective = qg * beta_up * y_up
            oil_advective = qo * (1.0 - beta_up) * x_up
        else:
            q_total = trans * (lam_g + lam_o) * dp * 1e-5
            comp_frac = beta_up * y_up + (1.0 - beta_up) * x_up
            gas_advective = q_total * comp_frac

        advective_total = gas_advective + oil_advective

        dispersive_total = np.zeros(self.nc)
        dispersion_coeff = 0.0
        if self.transport.enabled and (self.transport.dispersivity_ft > 0.0 or self.transport.molecular_diffusion_ft2_day > 0.0):
            v_hc = abs(self.hydrocarbon_velocity_ft_day(state, i, j))
            dispersion_coeff = float(self.transport.molecular_diffusion_ft2_day + self.transport.dispersivity_ft * v_hc)
            z_up = np.asarray(state.z[upstream], dtype=float)
            z_dn = np.asarray(state.z[downstream], dtype=float)
            z_face = 0.5 * (z_up + z_dn)
            grad_z = (z_dn - z_up) / max(distance, 1e-12)
            dispersive_total = -dispersion_coeff * self.phi * self.grid.area_ft2 * grad_z * float(np.mean(z_face)) * 0.02
            adv_mag = float(np.sum(np.abs(advective_total)))
            disp_mag = float(np.sum(np.abs(dispersive_total)))
            limit = self.transport.max_dispersive_fraction * adv_mag
            if adv_mag > 1e-12 and disp_mag > limit > 0.0:
                dispersive_total *= limit / disp_mag

        total = advective_total + dispersive_total
        return {
            "advective_total": advective_total,
            "dispersive_total": dispersive_total,
            "gas_advective": gas_advective,
            "oil_advective": oil_advective,
            "total": total,
            "upstream": upstream,
            "dispersion_coeff_ft2_day": dispersion_coeff,
        }

    def component_flux_between(self, state: SimulationState, i: int, j: int) -> np.ndarray:
        fluxes = self.component_flux_breakdown_between(state, i, j)
        return np.asarray(fluxes["total"], dtype=float)

    def water_flux_between(self, state: SimulationState, i: int, j: int) -> float:
        p_i = self.water_potential_psia(state, i)
        p_j = self.water_potential_psia(state, j)
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

    def transport_diagnostics(self, state: SimulationState) -> Dict[str, np.ndarray]:
        adv_mag = np.zeros(self.grid.nx)
        disp_mag = np.zeros(self.grid.nx)
        total_mag = np.zeros(self.grid.nx)
        dispersion_coeff = np.zeros(self.grid.nx)
        for i in range(self.grid.nx - 1):
            fluxes = self.component_flux_breakdown_between(state, i, i + 1)
            adv = float(np.sum(np.abs(np.asarray(fluxes["advective_total"], dtype=float))))
            disp = float(np.sum(np.abs(np.asarray(fluxes["dispersive_total"], dtype=float))))
            tot = float(np.sum(np.abs(np.asarray(fluxes["total"], dtype=float))))
            coeff = float(fluxes.get("dispersion_coeff_ft2_day", 0.0))
            adv_mag[i] += 0.5 * adv
            adv_mag[i + 1] += 0.5 * adv
            disp_mag[i] += 0.5 * disp
            disp_mag[i + 1] += 0.5 * disp
            total_mag[i] += 0.5 * tot
            total_mag[i + 1] += 0.5 * tot
            dispersion_coeff[i] = max(dispersion_coeff[i], coeff)
            dispersion_coeff[i + 1] = max(dispersion_coeff[i + 1], coeff)
        return {
            "transport_advective_flux_lbmol_day": adv_mag,
            "transport_dispersive_flux_lbmol_day": disp_mag,
            "transport_total_flux_lbmol_day": total_mag,
            "transport_dispersion_coeff_ft2_day": dispersion_coeff,
        }

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
        zinj = np.asarray(injector.injection_composition, dtype=float)
        if np.sum(zinj) <= 0.0:
            return None
        return zinj / np.sum(zinj)

    def injection_performance(self, state: SimulationState) -> Dict[str, float | bool | np.ndarray | None]:
        injector = self.active_injector()
        if injector is None or self.scenario.name == "natural_depletion":
            return {
                "active": False,
                "target_rate_lbmol_day": 0.0,
                "achievable_rate_lbmol_day": 0.0,
                "actual_rate_lbmol_day": 0.0,
                "cell_pressure_psia": float(state.pressure[0]),
                "effective_bhp_psia": 0.0,
                "delta_p_psia": 0.0,
                "composition": None,
            }
        zinj = self.get_injection_composition(state)
        if zinj is None:
            return {
                "active": False,
                "target_rate_lbmol_day": float(injector.rate_lbmol_day),
                "achievable_rate_lbmol_day": 0.0,
                "actual_rate_lbmol_day": 0.0,
                "cell_pressure_psia": float(state.pressure[injector.cell_index]),
                "effective_bhp_psia": 0.0,
                "delta_p_psia": 0.0,
                "composition": None,
            }
        cell_pressure = float(state.pressure[injector.cell_index])
        target_rate = float(injector.rate_lbmol_day)
        if not injector.is_active(self._current_time_days):
            return {
                "active": False,
                "target_rate_lbmol_day": target_rate,
                "achievable_rate_lbmol_day": 0.0,
                "actual_rate_lbmol_day": 0.0,
                "cell_pressure_psia": cell_pressure,
                "effective_bhp_psia": 0.0,
                "delta_p_psia": 0.0,
                "composition": zinj,
            }
        if injector.control_mode == "enhanced":
            effective_bhp = float(min(injector.injection_pressure_psia, injector.max_bhp_psia))
            delta_p = max(effective_bhp - cell_pressure, 0.0)
            achievable_rate = max(injector.injectivity_index_lbmol_day_psi * delta_p, 0.0)
            actual_rate = min(target_rate, achievable_rate)
        else:
            effective_bhp = float(injector.injection_pressure_psia)
            delta_p = max(effective_bhp - cell_pressure, 0.0)
            achievable_rate = target_rate
            actual_rate = target_rate
        return {
            "active": actual_rate > 0.0,
            "target_rate_lbmol_day": target_rate,
            "achievable_rate_lbmol_day": float(achievable_rate),
            "actual_rate_lbmol_day": float(actual_rate),
            "cell_pressure_psia": cell_pressure,
            "effective_bhp_psia": float(effective_bhp),
            "delta_p_psia": float(delta_p),
            "composition": zinj,
        }

    def injector_source(self, state: SimulationState) -> tuple[int | None, np.ndarray | None]:
        injector = self.active_injector()
        if injector is None:
            return None, None
        perf = self.injection_performance(state)
        zinj = perf.get("composition")
        actual_rate = float(perf.get("actual_rate_lbmol_day", 0.0))
        if zinj is None or actual_rate <= 0.0:
            return None, None
        source = actual_rate * np.asarray(zinj, dtype=float)
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
        sg_max_cell = float(np.clip(state.sg_max[i] if hasattr(state, "sg_max") else Sg, 0.0, 1.0))
        krg, kro, trap_fraction, imbibition_flag = apply_relperm_hysteresis(
            krg, kro, Sg, sg_max_cell, self.hysteresis
        )

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
            "hysteresis_trap_fraction": float(trap_fraction),
            "hysteresis_imbibition_flag": float(imbibition_flag),
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

        pvt_props = self.pvt_lookup(float(state.pressure[self.well.cell_index]))
        gas_density = pvt_props.get('gas_density_lbft3') if pvt_props.get('gas_density_lbft3') is not None else self.reporting.gas_density_lbft3
        condensate_density = self.reporting.condensate_density_lbft3
        water_density = WATER_DENSITY_LBFT3

        rho_mix = (qg * gas_density + qo * condensate_density + qw * water_density) / total
        return float(np.clip(rho_mix, 0.1, 62.4))

    def tubing_pressure_profile_from_response(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> Dict[str, float]:
        pwf = float(response["pwf_psia"])
        qg_lbmol_day = float(response["qg_lbmol_day"])
        qo_lbmol_day = float(response["qo_lbmol_day"])
        qw_lbmol_day = float(response["qw_lbmol_day"])

        gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(qg_lbmol_day)
        liquid_rate_stb_day = (
            liquid_lbmol_day_to_stb_day(qo_lbmol_day, np.asarray(response["x"]), self.mw_components, self.reporting.condensate_density_lbft3)
            + water_lbmol_day_to_stb_day(qw_lbmol_day)
        )

        if self.well.tubing_model == "simple":
            rho_mix = self.estimate_mixture_density_lbft3(state, response)
            hydrostatic_psi = 0.006944444444444444 * rho_mix * self.well.tvd_ft
            friction_psi = self.well.thp_friction_coeff * (gas_rate_mmscf_day + 0.001 * liquid_rate_stb_day) ** 2
            accel_psi = 0.0
            velocity_ft_s = 0.0
            reynolds = 0.0
            friction_factor = max(self.well.thp_friction_coeff, 0.0)
            thp = pwf - hydrostatic_psi - friction_psi - accel_psi
            clipped_thp = float(max(thp, 14.7))
            return {
                "raw_estimated_thp_psia": float(thp),
                "estimated_thp_psia": clipped_thp,
                "thp_clipped_to_atmospheric": float(thp < 14.7),
                "lift_feasible": float(thp >= self.well.min_lift_thp_psia),
                "tubing_hydrostatic_psi": float(max(hydrostatic_psi, 0.0)),
                "tubing_friction_psi": float(max(friction_psi, 0.0)),
                "tubing_acceleration_psi": float(max(accel_psi, 0.0)),
                "tubing_mixture_velocity_ft_s": float(max(velocity_ft_s, 0.0)),
                "tubing_reynolds_number": float(max(reynolds, 0.0)),
                "tubing_friction_factor": float(max(friction_factor, 0.0)),
            }

        area_ft2 = math.pi * (max(self.well.tubing_id_in, 0.25) / 12.0) ** 2 / 4.0
        diameter_ft = max(self.well.tubing_id_in / 12.0, 1e-6)
        roughness_ft = max(self.well.tubing_roughness_in, 0.0) / 12.0
        rel_roughness = roughness_ft / max(diameter_ft, 1e-12)
        nseg = max(int(getattr(self.well, "tubing_segments", 25)), 2)
        seg_len_ft = max(self.well.tvd_ft / nseg, 1e-9)

        hc_total_lbmol_day = max(qg_lbmol_day + qo_lbmol_day, 0.0)
        if hc_total_lbmol_day > 1e-12:
            z_hc = np.asarray(response["hc_sink"], dtype=float) / hc_total_lbmol_day
        else:
            z_hc = np.asarray(state.z[self.well.cell_index], dtype=float)
        z_hc = z_hc / max(np.sum(z_hc), 1e-12)

        def segment_state_props(p_seg_psia: float, t_seg_R: float) -> Dict[str, float | np.ndarray | str]:
            try:
                fl_seg = self.flash.flash(z_hc, p_seg_psia, t_seg_R)
                beta_seg = float(np.clip(fl_seg["beta"], 0.0, 1.0))
                x_seg = np.asarray(fl_seg["x"], dtype=float)
                y_seg = np.asarray(fl_seg["y"], dtype=float)
            except Exception:
                beta_seg = 1.0 if qg_lbmol_day >= qo_lbmol_day else 0.0
                x_seg = z_hc.copy()
                y_seg = z_hc.copy()

            # Carry forward the sandface phase rates through the tubing march.
            # Local pressure/temperature update phase properties and in-situ volumes,
            # but they do not repartition the entire hydrocarbon stream in each segment.
            qg_seg_lbmol_day = max(qg_lbmol_day, 0.0)
            qo_seg_lbmol_day = max(qo_lbmol_day, 0.0)

            rho_g = gas_density_lbft3_from_eos(self.eos, y_seg, p_seg_psia, t_seg_R) if qg_seg_lbmol_day > 1e-12 else 0.05
            rho_o = self.reporting.condensate_density_lbft3
            rho_w = WATER_DENSITY_LBFT3
            ql_seg_lbmol_day = qo_seg_lbmol_day + qw_lbmol_day
            rho_l = (qo_seg_lbmol_day * rho_o + qw_lbmol_day * rho_w) / max(ql_seg_lbmol_day, 1e-12) if ql_seg_lbmol_day > 1e-12 else rho_o

            gas_vol_ft3_day = hydrocarbon_gas_rate_ft3_day(qg_seg_lbmol_day, y_seg, p_seg_psia, t_seg_R, self.eos)
            oil_vol_ft3_day = liquid_lbmol_day_to_stb_day(qo_seg_lbmol_day, x_seg, self.mw_components, rho_o) * FT3_PER_STB
            water_vol_ft3_day = water_lbmol_day_to_stb_day(qw_lbmol_day) * FT3_PER_STB
            liquid_vol_ft3_day = oil_vol_ft3_day + water_vol_ft3_day

            vsg = gas_vol_ft3_day / max(area_ft2 * 86400.0, 1e-12)
            vsl = liquid_vol_ft3_day / max(area_ft2 * 86400.0, 1e-12)
            vm = vsg + vsl
            hl, regime = regime_liquid_holdup(vsg, vsl, diameter_ft, rho_l, rho_g)
            rho_mix = hl * rho_l + (1.0 - hl) * rho_g

            mu_g_cp = phase_viscosity_cp("v", y_seg, p_seg_psia, t_seg_R) if qg_seg_lbmol_day > 1e-12 else 0.01
            mu_o_cp = phase_viscosity_cp("l", x_seg, p_seg_psia, t_seg_R) if qo_seg_lbmol_day > 1e-12 else 0.10
            mu_l_cp = (qo_seg_lbmol_day * mu_o_cp + qw_lbmol_day * WATER_VISCOSITY_CP) / max(ql_seg_lbmol_day, 1e-12) if ql_seg_lbmol_day > 1e-12 else mu_o_cp
            mu_mix_cp = max(hl * mu_l_cp + (1.0 - hl) * mu_g_cp, 1e-6)
            mu_mix_lb_ft_s = mu_mix_cp * 0.000671968975

            reynolds = rho_mix * max(vm, 0.0) * diameter_ft / max(mu_mix_lb_ft_s, 1e-12)
            friction_factor = darcy_friction_factor_goudar_sonnad(reynolds, rel_roughness)
            mass_flux = rho_mix * max(vm, 0.0)
            specific_volume = 1.0 / max(rho_mix, 1e-12)

            hydro_grad_psi_ft = 0.006944444444444444 * rho_mix
            friction_grad_psi_ft = self.well.tubing_calibration_factor * friction_factor * mass_flux * max(vm, 0.0) / max(2.0 * 32.174 * diameter_ft * 144.0, 1e-12)

            return {
                "beta": beta_seg,
                "x": x_seg,
                "y": y_seg,
                "vsg": float(vsg),
                "vsl": float(vsl),
                "vm": float(vm),
                "hl": float(hl),
                "regime": regime,
                "rho_mix": float(rho_mix),
                "specific_volume": float(specific_volume),
                "reynolds": float(reynolds),
                "friction_factor": float(friction_factor),
                "hydro_grad_psi_ft": float(hydro_grad_psi_ft),
                "friction_grad_psi_ft": float(friction_grad_psi_ft),
                "mass_flux": float(mass_flux),
            }

        p_local = max(pwf, 14.7)
        hydrostatic_psi = 0.0
        friction_psi = 0.0
        accel_psi = 0.0
        velocity_samples: List[float] = []
        reynolds_samples: List[float] = []
        friction_factor_samples: List[float] = []

        for seg in range(nseg):
            frac = (seg + 0.5) / nseg
            t_local = self.T + frac * (self.well.wellhead_temperature_R - self.T)
            t_local = float(np.clip(t_local, min(self.T, self.well.wellhead_temperature_R), max(self.T, self.well.wellhead_temperature_R)))

            props_here = segment_state_props(p_local, t_local)
            segment_hydro_psi = props_here["hydro_grad_psi_ft"] * seg_len_ft
            segment_friction_psi = props_here["friction_grad_psi_ft"] * seg_len_ft
            p_prov = max(p_local - segment_hydro_psi - segment_friction_psi, 14.7)

            t_next = self.T + min((seg + 1.0) / nseg, 1.0) * (self.well.wellhead_temperature_R - self.T)
            t_next = float(np.clip(t_next, min(self.T, self.well.wellhead_temperature_R), max(self.T, self.well.wellhead_temperature_R)))
            props_next = segment_state_props(p_prov, t_next)

            g_mass = 0.5 * (props_here["mass_flux"] + props_next["mass_flux"])
            d_specific_volume_dL = (props_next["specific_volume"] - props_here["specific_volume"]) / seg_len_ft
            accel_grad_psi_ft = (g_mass ** 2) * d_specific_volume_dL / max(32.174 * 144.0, 1e-12)
            segment_accel_psi = accel_grad_psi_ft * seg_len_ft

            p_next = max(p_local - segment_hydro_psi - segment_friction_psi - segment_accel_psi, 14.7)

            hydrostatic_psi += segment_hydro_psi
            friction_psi += segment_friction_psi
            accel_psi += segment_accel_psi
            velocity_samples.append(float(props_here["vm"]))
            reynolds_samples.append(float(props_here["reynolds"]))
            friction_factor_samples.append(float(props_here["friction_factor"]))

            p_local = p_next

        thp = p_local
        raw_thp = pwf - hydrostatic_psi - friction_psi - accel_psi
        velocity_ft_s = float(np.mean(velocity_samples)) if velocity_samples else 0.0
        reynolds = float(np.mean(reynolds_samples)) if reynolds_samples else 0.0
        friction_factor = float(np.mean(friction_factor_samples)) if friction_factor_samples else 0.0
        clipped_thp = float(max(raw_thp, 14.7))
        return {
            "raw_estimated_thp_psia": float(raw_thp),
            "estimated_thp_psia": clipped_thp,
            "thp_clipped_to_atmospheric": float(raw_thp < 14.7),
            "lift_feasible": float(raw_thp >= self.well.min_lift_thp_psia),
            "tubing_hydrostatic_psi": float(max(hydrostatic_psi, 0.0)),
            "tubing_friction_psi": float(max(friction_psi, 0.0)),
            "tubing_acceleration_psi": float(accel_psi),
            "tubing_mixture_velocity_ft_s": float(max(velocity_ft_s, 0.0)),
            "tubing_reynolds_number": float(max(reynolds, 0.0)),
            "tubing_friction_factor": float(max(friction_factor, 0.0)),
        }

    def tubing_head_pressure_from_response(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> float:
        return float(self.tubing_pressure_profile_from_response(state, response)["estimated_thp_psia"])

    def _lift_profile_ok(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> Tuple[bool, Dict[str, float | np.ndarray]]:
        profile = self.tubing_pressure_profile_from_response(state, response)
        raw_thp = float(profile.get("raw_estimated_thp_psia", profile.get("estimated_thp_psia", 14.7)))
        ok = raw_thp >= float(self.well.min_lift_thp_psia)
        return ok, profile

    def solve_bhp_control(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        return self.well_response_at_pwf(state, self.well.bhp_psia)

    def solve_drawdown_control(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        i = self.well.cell_index
        pr = float(state.pressure[i])

        pwf_target = pr - self.well.drawdown_psia
        pwf_target = max(self.well.min_bhp_psia, pwf_target)

        return self.well_response_at_pwf(state, pwf_target)

    def solve_gas_rate_control(self, state: SimulationState, tol_mmscf_day: float = 1e-3, max_iter: int = 24) -> Dict[str, float | np.ndarray]:
        target_qg_lbmol_day = gas_mmscf_day_to_lbmol_day(self.well.target_gas_rate_mmscf_day)

        i = self.well.cell_index
        pr = float(state.pressure[i])
        pwf_lo = 50.0
        pwf_hi = max(pr - 1e-6, pwf_lo + 1e-6)

        response_cache: Dict[float, Dict[str, float | np.ndarray]] = {}
        profile_cache: Dict[float, Tuple[bool, Dict[str, float | np.ndarray]]] = {}

        def cached_response(pwf: float) -> Dict[str, float | np.ndarray]:
            key = round(float(pwf), 6)
            if key not in response_cache:
                response_cache[key] = self.well_response_at_pwf(state, float(pwf))
            return response_cache[key]

        def cached_lift_check(resp: Dict[str, float | np.ndarray]) -> Tuple[bool, Dict[str, float | np.ndarray]]:
            key = round(float(resp["pwf_psia"]), 6)
            if key not in profile_cache:
                profile_cache[key] = self._lift_profile_ok(state, resp)
            return profile_cache[key]

        resp_lo = cached_response(pwf_lo)
        resp_hi = cached_response(pwf_hi)

        qg_lo = float(resp_lo["qg_lbmol_day"])
        qg_hi = float(resp_hi["qg_lbmol_day"])

        if target_qg_lbmol_day >= qg_lo:
            candidate = resp_lo
        elif target_qg_lbmol_day <= qg_hi:
            candidate = resp_hi
        else:
            candidate = resp_hi
            for _ in range(max_iter):
                pwf_mid = 0.5 * (pwf_lo + pwf_hi)
                resp_mid = cached_response(pwf_mid)
                qg_mid = float(resp_mid["qg_lbmol_day"])
                err_mmscf_day = abs(gas_lbmol_day_to_mmscf_day(qg_mid - target_qg_lbmol_day))
                candidate = resp_mid

                if err_mmscf_day < tol_mmscf_day or abs(pwf_hi - pwf_lo) < 0.25:
                    break

                if qg_mid > target_qg_lbmol_day:
                    pwf_lo = pwf_mid
                else:
                    pwf_hi = pwf_mid

        feasible, profile = cached_lift_check(candidate)
        if feasible:
            candidate = dict(candidate)
            candidate.update(profile)
            candidate["lift_rate_feasible"] = 1.0
            candidate["requested_gas_rate_mmscf_day"] = float(self.well.target_gas_rate_mmscf_day)
            return candidate

        # Target rate is not lift-feasible. Fall back to a limited-rate solution
        # with a bounded number of tubing evaluations.
        pwf_low = float(candidate["pwf_psia"])
        pwf_high = max(pr - 1e-6, pwf_low + 1e-6)
        resp_high = cached_response(pwf_high)
        feasible_high, profile_high = cached_lift_check(resp_high)

        if not feasible_high:
            fallback = dict(resp_high)
            fallback.update(profile_high)
            fallback["lift_rate_feasible"] = 0.0
            fallback["requested_gas_rate_mmscf_day"] = float(self.well.target_gas_rate_mmscf_day)
            fallback["control_mode"] = "gas_rate_infeasible"
            return fallback

        best_resp = resp_high
        best_profile = profile_high
        for _ in range(min(max_iter, 16)):
            pwf_mid = 0.5 * (pwf_low + pwf_high)
            resp_mid = cached_response(pwf_mid)
            feasible_mid, profile_mid = cached_lift_check(resp_mid)
            if feasible_mid:
                best_resp = resp_mid
                best_profile = profile_mid
                pwf_high = pwf_mid
            else:
                pwf_low = pwf_mid
            if abs(pwf_high - pwf_low) < 1.0:
                break

        fallback = dict(best_resp)
        fallback.update(best_profile)
        fallback["lift_rate_feasible"] = 0.0
        fallback["requested_gas_rate_mmscf_day"] = float(self.well.target_gas_rate_mmscf_day)
        fallback["control_mode"] = "gas_rate_limited"
        return fallback

    def solve_thp_control(self, state: SimulationState, tol_psia: float = 1.0, max_iter: int = 60) -> Dict[str, float | np.ndarray]:
        target_thp = self.well.thp_psia
        i = self.well.cell_index
        pr = float(state.pressure[i])

        pwf_lo = 50.0
        pwf_hi = max(pr - 1e-6, pwf_lo + 1e-6)

        resp_lo = self.well_response_at_pwf(state, pwf_lo)
        resp_hi = self.well_response_at_pwf(state, pwf_hi)

        profile_lo = self.tubing_pressure_profile_from_response(state, resp_lo)
        profile_hi = self.tubing_pressure_profile_from_response(state, resp_hi)
        thp_lo = float(profile_lo["estimated_thp_psia"])
        thp_hi = float(profile_hi["estimated_thp_psia"])

        if target_thp <= thp_lo:
            resp_lo.update(profile_lo)
            return resp_lo
        if target_thp >= thp_hi:
            resp_hi.update(profile_hi)
            return resp_hi

        best = resp_hi
        for _ in range(max_iter):
            pwf_mid = 0.5 * (pwf_lo + pwf_hi)
            resp_mid = self.well_response_at_pwf(state, pwf_mid)
            profile_mid = self.tubing_pressure_profile_from_response(state, resp_mid)
            thp_mid = float(profile_mid["estimated_thp_psia"])
            resp_mid.update(profile_mid)
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
        if float(response.get("lift_rate_feasible", 1.0)) < 0.5:
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
            response.update(self.tubing_pressure_profile_from_response(state, response))
            return response

        if mode == "drawdown":
            response = self.solve_drawdown_control(state)
            response["control_mode"] = "drawdown"
            response.update(self.tubing_pressure_profile_from_response(state, response))
            return response

        if mode == "gas_rate":
            response = self.solve_gas_rate_control(state)
            response = self.enforce_gas_rate_target(response)
            response.setdefault("control_mode", "gas_rate")
            if "estimated_thp_psia" not in response:
                response.update(self.tubing_pressure_profile_from_response(state, response))
            return response

        if mode == "thp":
            response = self.solve_thp_control(state)
            response["control_mode"] = "thp"
            if "estimated_thp_psia" not in response:
                response.update(self.tubing_pressure_profile_from_response(state, response))
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
                "stages": [],
                "total_separator_gas_lbmol_day": 0.0,
                "stock_tank_liquid_lbmol_day": 0.0,
                "stage_count": 1,
            }

        stage_specs = [
            (float(self.separator.pressure_psia), float(self.separator.temperature_R), "stage1"),
        ]
        if getattr(self.separator, "stages", 1) >= 2:
            stage_specs.append((float(self.separator.second_stage_pressure_psia), float(self.separator.second_stage_temperature_R), "stage2"))
        if getattr(self.separator, "stages", 1) >= 3:
            stage_specs.append((float(self.separator.third_stage_pressure_psia), float(self.separator.third_stage_temperature_R), "stage3"))
        if getattr(self.separator, "stages", 1) >= 4:
            stage_specs.append((float(self.separator.stock_tank_pressure_psia), float(self.separator.stock_tank_temperature_R), "stock_tank"))

        q_in = float(qhc_lbmol_day)
        z_in = np.asarray(z_prod, dtype=float)
        stage_results = []
        total_gas_lbmol_day = 0.0
        final_liquid_lbmol_day = q_in
        final_x = z_in.copy()
        final_y = z_in.copy()
        final_beta = 1.0
        final_state = "vapor"

        for p_stage, t_stage, label in stage_specs:
            if q_in <= 1e-12:
                stage_results.append({
                    "label": label,
                    "pressure_psia": p_stage,
                    "temperature_R": t_stage,
                    "beta": 1.0,
                    "q_gas_lbmol_day": 0.0,
                    "q_liq_lbmol_day": 0.0,
                    "x": z_in.copy(),
                    "y": z_in.copy(),
                    "state": "vapor",
                })
                final_liquid_lbmol_day = 0.0
                continue

            sep = self.flash.flash(z_in, p_stage, t_stage)
            beta = float(np.clip(float(sep["beta"]), 0.0, 1.0))
            x = np.asarray(sep["x"], dtype=float)
            y = np.asarray(sep["y"], dtype=float)
            q_gas = q_in * beta
            q_liq = q_in * (1.0 - beta)
            stage_results.append({
                "label": label,
                "pressure_psia": p_stage,
                "temperature_R": t_stage,
                "beta": beta,
                "q_gas_lbmol_day": float(q_gas),
                "q_liq_lbmol_day": float(q_liq),
                "x": x,
                "y": y,
                "state": str(sep["state"]),
            })
            total_gas_lbmol_day += float(q_gas)
            q_in = float(q_liq)
            z_in = x.copy() / max(np.sum(x), 1e-12)
            final_liquid_lbmol_day = float(q_liq)
            final_x = x
            final_y = y
            final_beta = beta
            final_state = str(sep["state"])

        return {
            "qhc_lbmol_day": float(qhc_lbmol_day),
            "q_sep_gas_lbmol_day": float(total_gas_lbmol_day),
            "q_sep_liq_lbmol_day": float(final_liquid_lbmol_day),
            "beta_sep": float(stage_results[0]["beta"] if stage_results else 1.0),
            "z_prod": z_prod,
            "x_sep": final_x,
            "y_sep": final_y,
            "state": final_state,
            "stages": stage_results,
            "total_separator_gas_lbmol_day": float(total_gas_lbmol_day),
            "stock_tank_liquid_lbmol_day": float(final_liquid_lbmol_day),
            "stage_count": int(len(stage_results) if stage_results else 1),
        }

    def separator_rates(self, response: Dict[str, float | np.ndarray]) -> Dict[str, float]:
        sep = self.separator_flash(response)
        q_sep_gas_lbmol_day = float(sep["total_separator_gas_lbmol_day"])
        q_sep_liq_lbmol_day = float(sep["stock_tank_liquid_lbmol_day"])
        q_w_lbmol_day = float(response.get("qw_lbmol_day", 0.0))
        x_sep = np.asarray(sep["x_sep"], dtype=float)

        qg_report_lbmol_day = max(float(response.get("qg_lbmol_day", q_sep_gas_lbmol_day)), 0.0)
        qo_report_lbmol_day = max(float(response.get("qo_lbmol_day", q_sep_liq_lbmol_day)), 0.0)
        x_report = np.asarray(response.get("x", x_sep), dtype=float)
        if x_report.ndim != 1 or x_report.size == 0 or np.sum(x_report) <= 0.0:
            x_report = x_sep.copy()
        x_report = x_report / np.sum(x_report)

        gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(qg_report_lbmol_day)
        condensate_from_wellstream_stb_day = liquid_lbmol_day_to_stb_day(
            qo_report_lbmol_day,
            x_report,
            self.mw_components,
            self.reporting.condensate_density_lbft3,
        )
        stock_tank_liquid_rate_stb_day = liquid_lbmol_day_to_stb_day(
            q_sep_liq_lbmol_day,
            x_sep,
            self.mw_components,
            self.reporting.condensate_density_lbft3,
        )
        derived_cgr_stb_mmscf = condensate_from_wellstream_stb_day / gas_rate_mmscf_day if gas_rate_mmscf_day > 1e-12 else 0.0

        pressure_for_cgr = float(response.get("pr_psia", response.get("pwf_psia", self.separator.pressure_psia)))
        pvt_props = self.pvt_lookup(pressure_for_cgr)
        cgr_stb_mmscf = pvt_props.get("reservoir_cgr_stb_mmscf", None)
        if cgr_stb_mmscf is None:
            cgr_stb_mmscf = pvt_props.get("vaporized_cgr_stb_mmscf", None)

        cgr_user = float(cgr_stb_mmscf) if cgr_stb_mmscf is not None and np.isfinite(cgr_stb_mmscf) else float("nan")
        user_cgr_valid = np.isfinite(cgr_user) and cgr_user > 0.0
        derived_cgr_valid = np.isfinite(derived_cgr_stb_mmscf) and derived_cgr_stb_mmscf > 0.0

        if gas_rate_mmscf_day <= 1e-12:
            condensate_rate_stb_day = 0.0
            effective_cgr_stb_mmscf = 0.0
            condensate_reporting_basis = "zero_gas"
        elif user_cgr_valid and derived_cgr_valid:
            effective_cgr_stb_mmscf = 0.60 * cgr_user + 0.20 * derived_cgr_stb_mmscf + 0.20 * (stock_tank_liquid_rate_stb_day / max(gas_rate_mmscf_day, 1e-12))
            condensate_rate_stb_day = effective_cgr_stb_mmscf * gas_rate_mmscf_day
            condensate_reporting_basis = "blended_pvt_wellstream_surface"
        elif user_cgr_valid:
            effective_cgr_stb_mmscf = cgr_user
            condensate_rate_stb_day = effective_cgr_stb_mmscf * gas_rate_mmscf_day
            condensate_reporting_basis = "pvt_cgr"
        else:
            effective_cgr_stb_mmscf = stock_tank_liquid_rate_stb_day / gas_rate_mmscf_day if gas_rate_mmscf_day > 1e-12 else 0.0
            condensate_rate_stb_day = stock_tank_liquid_rate_stb_day
            condensate_reporting_basis = "surface_train_liquid"

        total_surface = max(qg_report_lbmol_day + qo_report_lbmol_day + q_w_lbmol_day, 1e-12)
        surface_gas_fraction = qg_report_lbmol_day / total_surface
        surface_oil_fraction = qo_report_lbmol_day / total_surface
        surface_water_fraction = q_w_lbmol_day / total_surface

        stages = sep.get("stages", [])
        stage_gas_rates = {s.get("label", f"stage{i+1}"): gas_lbmol_day_to_mmscf_day(float(s.get("q_gas_lbmol_day", 0.0))) for i, s in enumerate(stages)}

        return {
            "gas_rate_mmscf_day": float(gas_rate_mmscf_day),
            "condensate_rate_stb_day": float(condensate_rate_stb_day),
            "separator_vapor_fraction": float(sep["beta_sep"]),
            "surface_gas_fraction": float(surface_gas_fraction),
            "surface_oil_fraction": float(surface_oil_fraction),
            "surface_water_fraction": float(surface_water_fraction),
            "raw_separator_gas_rate_mmscf_day": float(gas_lbmol_day_to_mmscf_day(q_sep_gas_lbmol_day)),
            "raw_separator_condensate_rate_stb_day": float(stock_tank_liquid_rate_stb_day),
            "wellstream_condensate_rate_stb_day": float(condensate_from_wellstream_stb_day),
            "stock_tank_condensate_rate_stb_day": float(stock_tank_liquid_rate_stb_day),
            "total_separator_gas_rate_mmscf_day": float(gas_lbmol_day_to_mmscf_day(q_sep_gas_lbmol_day)),
            "separator_stage1_gas_rate_mmscf_day": float(stage_gas_rates.get("stage1", 0.0)),
            "separator_stage2_gas_rate_mmscf_day": float(stage_gas_rates.get("stage2", 0.0)),
            "separator_stage3_gas_rate_mmscf_day": float(stage_gas_rates.get("stage3", 0.0)),
            "separator_stock_tank_gas_rate_mmscf_day": float(stage_gas_rates.get("stock_tank", 0.0)),
            "separator_stage_count": float(sep.get("stage_count", 1)),
            "cgr_stb_mmscf": float(cgr_user) if np.isfinite(cgr_user) else float("nan"),
            "effective_cgr_stb_mmscf": float(effective_cgr_stb_mmscf),
            "derived_cgr_stb_mmscf": float(derived_cgr_stb_mmscf),
            "condensate_reporting_basis": condensate_reporting_basis,
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

    def boundary_cell_index(self, side: str) -> int:
        return 0 if side == "left" else self.grid.nx - 1

    def boundary_transmissibility(self, side: str) -> float:
        idx = self.boundary_cell_index(side)
        dx = max(0.5 * self.grid.cell_width(idx), 1e-12)
        base = self.k * self.grid.area_ft2 / dx * 1.0e-3
        mult = self.boundary.left_transmissibility_multiplier if side == "left" else self.boundary.right_transmissibility_multiplier
        return float(max(base * mult, 0.0))

    def boundary_pressure_target(self, side: str) -> float | None:
        mode = self.boundary.left_mode if side == "left" else self.boundary.right_mode
        if mode != "constant_pressure":
            return None
        return float(self.boundary.left_pressure_psia if side == "left" else self.boundary.right_pressure_psia)

    def aquifer_current_pressure_psia(self) -> float:
        if not self.aquifer.enabled:
            return float(self.aquifer.initial_pressure_psia)
        depletion = self.cum_aquifer_water_lbmol / max(self.aquifer.total_capacity_lbmol_per_psi, 1e-12)
        return float(max(self.aquifer.initial_pressure_psia - depletion, 14.7))

    def aquifer_water_influx_lbmol_day(self, state: SimulationState) -> tuple[int | None, float, float]:
        if not self.aquifer.enabled:
            return None, 0.0, float(self.aquifer.initial_pressure_psia)
        idx = self.boundary_cell_index(self.aquifer.side)
        p_cell = float(state.pressure[idx])
        p_aq = self.aquifer_current_pressure_psia()
        dp = p_aq - p_cell
        if (not self.aquifer.allow_backflow) and dp <= 0.0:
            return idx, 0.0, p_aq
        q = self.aquifer.productivity_index_lbmol_day_psi * dp * self.aquifer.water_influx_fraction
        return idx, float(q), p_aq

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

        for side in ("left", "right"):
            p_ext = self.boundary_pressure_target(side)
            if p_ext is not None:
                bi = self.boundary_cell_index(side)
                Tb = self.boundary_transmissibility(side)
                A[bi, bi] += Tb
                b[bi] += Tb * p_ext

        if self.aquifer.enabled:
            ai = self.boundary_cell_index(self.aquifer.side)
            p_aq = self.aquifer_current_pressure_psia()
            Jaq = self.aquifer.productivity_index_lbmol_day_psi * 2.0e-4
            A[ai, ai] += Jaq
            b[ai] += Jaq * p_aq

        wi = self.well.cell_index
        well_mob = self.phase_mobility_data(state, wi)
        Jw = self.peaceman_well_index(wi) * well_mob["lam_t"] * 2.5e-3
        response = self.solve_well_control(state)
        flowing_pwf = float(response["pwf_psia"])

        A[wi, wi] += Jw
        b[wi] += Jw * flowing_pwf

        injector = self.active_injector()
        injector_perf = self.injection_performance(state)
        actual_inj_rate = float(injector_perf.get("actual_rate_lbmol_day", 0.0))
        effective_inj_bhp = float(injector_perf.get("effective_bhp_psia", 0.0))
        if injector is not None and self.scenario.name != "natural_depletion" and actual_inj_rate > 0.0:
            ii = injector.cell_index

            # Couple the injector to the pressure solve using the actual
            # pressure-controlled injection rate and the user-entered maximum
            # bottom-hole injection pressure. This gives stronger, more direct
            # pressure support than the previous damped dew-point-based cap.
            inj_mob = self.phase_mobility_data(state, ii)
            Jin = self.peaceman_well_index(ii) * inj_mob["lam_t"] * 5.0e-3

            rate_implied_bhp = p_old[ii] + actual_inj_rate / max(Jin, 1e-12)
            pinj_equiv = float(np.clip(rate_implied_bhp, p_old[ii], max(effective_inj_bhp, p_old[ii])))

            A[ii, ii] += Jin
            b[ii] += Jin * pinj_equiv

        try:
            p_solved = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            p_solved = p_old.copy()

        omega_p = 0.5
        p_new = (1.0 - omega_p) * p_old + omega_p * p_solved
        return np.maximum(p_new, 50.0)

    def transport_update(self, state: SimulationState, dt_days: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float]:
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
            sg_max=state.sg_max.copy(),
        )

        q_total = gas_frac = oil_frac = water_frac = 0.0
        dropout = krg = kro = krw = 0.0
        damage_factor = wi_eff = q_undamaged = ploss = 0.0
        injector_actual_rate_lbmol_day = 0.0

        for _ in range(n_sub):
            self._set_cache_state(work_state)
            ncomp = work_state.nt[:, None] * work_state.z
            ncomp_new = ncomp.copy()
            nw_new = work_state.nw.copy()

            transport_advective_total = 0.0
            transport_dispersive_total = 0.0
            transport_total_flux = 0.0
            for i in range(self.grid.nx - 1):
                fluxes = self.component_flux_breakdown_between(work_state, i, i + 1)
                hc_flux = np.asarray(fluxes["total"], dtype=float)
                w_flux = self.water_flux_between(work_state, i, i + 1)

                ncomp_new[i] -= dt_sub * hc_flux
                ncomp_new[i + 1] += dt_sub * hc_flux

                nw_new[i] -= dt_sub * w_flux
                nw_new[i + 1] += dt_sub * w_flux

                transport_advective_total += float(np.sum(np.abs(np.asarray(fluxes["advective_total"], dtype=float))))
                transport_dispersive_total += float(np.sum(np.abs(np.asarray(fluxes["dispersive_total"], dtype=float))))
                transport_total_flux += float(np.sum(np.abs(hc_flux)))

            inj_perf = self.injection_performance(work_state)
            injector_actual_rate_lbmol_day = float(inj_perf.get("actual_rate_lbmol_day", 0.0))
            inj_cell, inj_source = self.injector_source(work_state)
            if inj_cell is not None and inj_source is not None:
                ncomp_new[inj_cell] += dt_sub * inj_source

            aquifer_cell, aquifer_rate_lbmol_day, aquifer_pressure_psia = self.aquifer_water_influx_lbmol_day(work_state)
            if aquifer_cell is not None and abs(aquifer_rate_lbmol_day) > 0.0:
                nw_new[aquifer_cell] += dt_sub * aquifer_rate_lbmol_day
                self.cum_aquifer_water_lbmol += max(aquifer_rate_lbmol_day, 0.0) * dt_sub

            hc_sink, qw, q_total, gas_frac, oil_frac, water_frac, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss = self.well_sink(work_state)
            wi = self.well.cell_index
            ncomp_new[wi] -= dt_sub * hc_sink
            nw_new[wi] -= dt_sub * qw

            ncomp_new = np.clip(ncomp_new, 1e-12, None)
            nw_new = np.clip(nw_new, 1e-12, None)

            nt_new = np.sum(ncomp_new, axis=1)
            z_new = ncomp_new / nt_new[:, None]
            sw_new = saturation_from_water_moles(nw_new, self.pv)

            provisional_state = SimulationState(
                pressure=work_state.pressure.copy(),
                z=z_new,
                nt=nt_new,
                nw=nw_new,
                sw=sw_new,
                sg_max=work_state.sg_max.copy(),
            )
            sg_profile = self.compute_flowing_sg_profile(provisional_state)
            work_state = SimulationState(
                pressure=provisional_state.pressure.copy(),
                z=provisional_state.z,
                nt=provisional_state.nt,
                nw=provisional_state.nw,
                sw=provisional_state.sw,
                sg_max=np.maximum(work_state.sg_max, sg_profile),
            )

        return (
            work_state.nt, work_state.z, work_state.nw, work_state.sw,
            q_total, gas_frac, oil_frac, water_frac,
            dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss,
            injector_actual_rate_lbmol_day,
            float(transport_advective_total), float(transport_dispersive_total), float(transport_total_flux),
            float(aquifer_rate_lbmol_day if "aquifer_rate_lbmol_day" in locals() else 0.0),
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
        p_pred = self.pressure_update(state, dt_days)
        state_mid = SimulationState(
            pressure=p_pred,
            z=state.z.copy(),
            nt=state.nt.copy(),
            nw=state.nw.copy(),
            sw=state.sw.copy(),
            sg_max=state.sg_max.copy(),
        )
        self._set_cache_state(state_mid)

        nt_new, z_new, nw_new, sw_new, q_total, gas_frac, oil_frac, water_frac, dropout, krg, kro, krw, damage_factor, wi_eff, q_undamaged, ploss, injector_actual_rate_lbmol_day, transport_advective_total, transport_dispersive_total, transport_total_flux, aquifer_rate_lbmol_day = self.transport_update(state_mid, dt_days)

        state_post = SimulationState(
            pressure=p_pred.copy(),
            z=z_new,
            nt=nt_new,
            nw=nw_new,
            sw=sw_new,
            sg_max=state_mid.sg_max.copy(),
        )
        self._set_cache_state(state_post)
        p_corr = self.pressure_update(state_post, dt_days)

        new_state = SimulationState(
            pressure=p_corr,
            z=z_new,
            nt=nt_new,
            nw=nw_new,
            sw=sw_new,
            sg_max=state_post.sg_max.copy(),
        )
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
            "avg_pressure": float(np.mean(p_corr)),
            "min_pressure": float(np.min(p_corr)),
            "avg_sw": float(np.mean(sw_new)),
            "injector_actual_rate_lbmol_day": float(injector_actual_rate_lbmol_day),
            "transport_advective_flux_lbmol_day": float(transport_advective_total),
            "transport_dispersive_flux_lbmol_day": float(transport_dispersive_total),
            "transport_total_flux_lbmol_day": float(transport_total_flux),
            "aquifer_rate_lbmol_day": float(aquifer_rate_lbmol_day),
            "aquifer_pressure_psia": float(self.aquifer_current_pressure_psia()),
            "left_boundary_pressure_psia": float(self.boundary.left_pressure_psia if self.boundary.left_mode == "constant_pressure" else p_corr[0]),
            "right_boundary_pressure_psia": float(self.boundary.right_pressure_psia if self.boundary.right_mode == "constant_pressure" else p_corr[-1]),
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
            injector_perf_0 = self.injection_performance(state)
            injector_rate_lbmol_day = float(injector_perf_0.get("actual_rate_lbmol_day", 0.0))

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
                history.well_raw_estimated_thp_psia.append(float(response0.get("raw_estimated_thp_psia", response0["estimated_thp_psia"])))
                history.well_thp_clipped_to_atmospheric.append(float(response0.get("thp_clipped_to_atmospheric", 0.0)))
                history.well_lift_feasible.append(float(response0.get("lift_feasible", 1.0)))
                history.well_tubing_hydrostatic_psi.append(float(response0.get("tubing_hydrostatic_psi", 0.0)))
                history.well_tubing_friction_psi.append(float(response0.get("tubing_friction_psi", 0.0)))
                history.well_tubing_acceleration_psi.append(float(response0.get("tubing_acceleration_psi", 0.0)))
                history.well_tubing_mixture_velocity_ft_s.append(float(response0.get("tubing_mixture_velocity_ft_s", 0.0)))
                history.well_tubing_reynolds_number.append(float(response0.get("tubing_reynolds_number", 0.0)))
                history.well_tubing_friction_factor.append(float(response0.get("tubing_friction_factor", 0.0)))
                history.well_target_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
                if self.well.control_mode == "gas_rate":
                    history.controlled_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
                else:
                    history.controlled_gas_rate_mmscf_day.append(float(response0["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE)
                history.reported_gas_rate_mmscf_day.append(
                    float(response0.get("reported_gas_rate_mmscf_day", float(response0["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE))
                )
                history.transport_advective_flux_lbmol_day.append(0.0)
                history.transport_dispersive_flux_lbmol_day.append(0.0)
                history.transport_total_flux_lbmol_day.append(0.0)
                history.aquifer_rate_lbmol_day.append(0.0)
                history.aquifer_cumulative_mlbmol.append(float(self.cum_aquifer_water_lbmol / 1.0e6))
                history.aquifer_pressure_psia.append(float(self.aquifer_current_pressure_psia()))
                history.left_boundary_pressure_psia.append(float(self.boundary.left_pressure_psia if self.boundary.left_mode == "constant_pressure" else state.pressure[0]))
                history.right_boundary_pressure_psia.append(float(self.boundary.right_pressure_psia if self.boundary.right_mode == "constant_pressure" else state.pressure[-1]))

                if injector is None or self.scenario.name == "natural_depletion":
                    history.injector_rate_total_lbmol_day.append(0.0)
                    history.injector_rate_input_field.append(0.0)
                    history.injector_rate_target_lbmol_day.append(0.0)
                    history.injector_rate_achievable_lbmol_day.append(0.0)
                    history.injector_cell_pressure_psia.append(float(state.pressure[0]))
                    history.injector_pressure_delta_psia.append(0.0)
                    history.injector_effective_bhp_psia.append(0.0)
                    history.injector_active_flag.append(0.0)
                else:
                    history.injector_rate_total_lbmol_day.append(float(injector_rate_lbmol_day))
                    history.injector_rate_input_field.append(float(injector.rate_field_input))
                    history.injector_rate_target_lbmol_day.append(float(injector_perf_0.get("target_rate_lbmol_day", 0.0)))
                    history.injector_rate_achievable_lbmol_day.append(float(injector_perf_0.get("achievable_rate_lbmol_day", 0.0)))
                    history.injector_cell_pressure_psia.append(float(injector_perf_0.get("cell_pressure_psia", state.pressure[injector.cell_index])))
                    history.injector_pressure_delta_psia.append(float(injector_perf_0.get("delta_p_psia", 0.0)))
                    history.injector_effective_bhp_psia.append(float(injector_perf_0.get("effective_bhp_psia", 0.0)))
                    history.injector_active_flag.append(1.0 if injector_perf_0.get("active", False) else 0.0)

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
                history.well_hysteresis_trap_fraction.append(float(response0.get("hysteresis_trap_fraction", 0.0)))
                history.well_hysteresis_imbibition_flag.append(float(response0.get("hysteresis_imbibition_flag", 0.0)))
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
                history.separator_stage_count.append(float(sep_rates.get("separator_stage_count", getattr(self.separator, "stages", 1))))
                history.separator_total_gas_rate_mmscf_day.append(float(sep_rates.get("total_separator_gas_rate_mmscf_day", 0.0)))
                history.separator_stage1_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stage1_gas_rate_mmscf_day", 0.0)))
                history.separator_stage2_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stage2_gas_rate_mmscf_day", 0.0)))
                history.separator_stage3_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stage3_gas_rate_mmscf_day", 0.0)))
                history.separator_stock_tank_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stock_tank_gas_rate_mmscf_day", 0.0)))
                history.separator_stock_tank_liquid_rate_stb_day.append(float(sep_rates.get("stock_tank_condensate_rate_stb_day", 0.0)))

                history.cum_gas_bscf.append(float(cum_gas_scf / 1.0e9))
                history.cum_condensate_mstb.append(float(cum_condensate_stb / 1.0e3))
                history.cum_water_mstb.append(float(cum_water_stb / 1.0e3))

                self.record_mass_balance(history, state, 0.0, q_total, gas_frac, oil_frac, water_frac, injector_rate_lbmol_day)
                history.accepted_dt_days.append(0.0)
                history.timestep_retries.append(0)
                break

            state_new, summary, accepted_dt, retries = self.adaptive_step(state, next_dt)

            t_new = t + accepted_dt
            self._set_cache_state(state_new)
            self._current_time_days = t_new

            response1 = self.reported_well_response(state_new)
            injector_perf_1 = self.injection_performance(state_new)
            actual_inj_rate_lbmol_day = float(injector_perf_1.get("actual_rate_lbmol_day", 0.0))
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
            history.well_raw_estimated_thp_psia.append(float(response1.get("raw_estimated_thp_psia", response1["estimated_thp_psia"])))
            history.well_thp_clipped_to_atmospheric.append(float(response1.get("thp_clipped_to_atmospheric", 0.0)))
            history.well_lift_feasible.append(float(response1.get("lift_feasible", 1.0)))
            history.well_tubing_hydrostatic_psi.append(float(response1.get("tubing_hydrostatic_psi", 0.0)))
            history.well_tubing_friction_psi.append(float(response1.get("tubing_friction_psi", 0.0)))
            history.well_tubing_acceleration_psi.append(float(response1.get("tubing_acceleration_psi", 0.0)))
            history.well_tubing_mixture_velocity_ft_s.append(float(response1.get("tubing_mixture_velocity_ft_s", 0.0)))
            history.well_tubing_reynolds_number.append(float(response1.get("tubing_reynolds_number", 0.0)))
            history.well_tubing_friction_factor.append(float(response1.get("tubing_friction_factor", 0.0)))
            history.well_target_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
            if self.well.control_mode == "gas_rate":
                history.controlled_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
            else:
                history.controlled_gas_rate_mmscf_day.append(float(response1["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE)
            history.reported_gas_rate_mmscf_day.append(
                float(response1.get("reported_gas_rate_mmscf_day", float(response1["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE))
            )
            history.transport_advective_flux_lbmol_day.append(float(summary.get("transport_advective_flux_lbmol_day", 0.0)))
            history.transport_dispersive_flux_lbmol_day.append(float(summary.get("transport_dispersive_flux_lbmol_day", 0.0)))
            history.transport_total_flux_lbmol_day.append(float(summary.get("transport_total_flux_lbmol_day", 0.0)))
            history.aquifer_rate_lbmol_day.append(float(summary.get("aquifer_rate_lbmol_day", 0.0)))
            history.aquifer_cumulative_mlbmol.append(float(self.cum_aquifer_water_lbmol / 1.0e6))
            history.aquifer_pressure_psia.append(float(summary.get("aquifer_pressure_psia", self.aquifer_current_pressure_psia())))
            history.left_boundary_pressure_psia.append(float(summary.get("left_boundary_pressure_psia", state_new.pressure[0])))
            history.right_boundary_pressure_psia.append(float(summary.get("right_boundary_pressure_psia", state_new.pressure[-1])))

            if injector is None or self.scenario.name == "natural_depletion":
                history.injector_rate_total_lbmol_day.append(0.0)
                history.injector_rate_input_field.append(0.0)
                history.injector_rate_target_lbmol_day.append(0.0)
                history.injector_rate_achievable_lbmol_day.append(0.0)
                history.injector_cell_pressure_psia.append(float(state_new.pressure[0]))
                history.injector_pressure_delta_psia.append(0.0)
                history.injector_effective_bhp_psia.append(0.0)
                history.injector_active_flag.append(0.0)
            else:
                history.injector_rate_total_lbmol_day.append(float(actual_inj_rate_lbmol_day))
                history.injector_rate_input_field.append(float(injector.rate_field_input))
                history.injector_rate_target_lbmol_day.append(float(injector_perf_1.get("target_rate_lbmol_day", 0.0)))
                history.injector_rate_achievable_lbmol_day.append(float(injector_perf_1.get("achievable_rate_lbmol_day", 0.0)))
                history.injector_cell_pressure_psia.append(float(injector_perf_1.get("cell_pressure_psia", state_new.pressure[injector.cell_index])))
                history.injector_pressure_delta_psia.append(float(injector_perf_1.get("delta_p_psia", 0.0)))
                history.injector_effective_bhp_psia.append(float(injector_perf_1.get("effective_bhp_psia", 0.0)))
                history.injector_active_flag.append(1.0 if injector_perf_1.get("active", False) else 0.0)

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
            history.well_hysteresis_trap_fraction.append(float(response1.get("hysteresis_trap_fraction", 0.0)))
            history.well_hysteresis_imbibition_flag.append(float(response1.get("hysteresis_imbibition_flag", 0.0)))
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
            history.separator_stage_count.append(float(sep_rates.get("separator_stage_count", getattr(self.separator, "stages", 1))))
            history.separator_total_gas_rate_mmscf_day.append(float(sep_rates.get("total_separator_gas_rate_mmscf_day", 0.0)))
            history.separator_stage1_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stage1_gas_rate_mmscf_day", 0.0)))
            history.separator_stage2_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stage2_gas_rate_mmscf_day", 0.0)))
            history.separator_stage3_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stage3_gas_rate_mmscf_day", 0.0)))
            history.separator_stock_tank_gas_rate_mmscf_day.append(float(sep_rates.get("separator_stock_tank_gas_rate_mmscf_day", 0.0)))
            history.separator_stock_tank_liquid_rate_stb_day.append(float(sep_rates.get("stock_tank_condensate_rate_stb_day", 0.0)))

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
        sg_max_arr = np.zeros(self.grid.nx)
        hysteresis_trap_arr = np.zeros(self.grid.nx)
        hysteresis_imbibition_arr = np.zeros(self.grid.nx)
        gas_mob_arr = np.zeros(self.grid.nx)
        oil_mob_arr = np.zeros(self.grid.nx)
        water_mob_arr = np.zeros(self.grid.nx)
        pcog_arr = np.zeros(self.grid.nx)
        pcow_arr = np.zeros(self.grid.nx)
        hydrocarbon_potential_arr = np.zeros(self.grid.nx)
        water_potential_arr = np.zeros(self.grid.nx)
        dew_point_minus_pressure = np.zeros(self.grid.nx)
        left_boundary_pressure_arr = np.full(self.grid.nx, self.boundary.left_pressure_psia if self.boundary.left_mode == "constant_pressure" else state.pressure[0])
        right_boundary_pressure_arr = np.full(self.grid.nx, self.boundary.right_pressure_psia if self.boundary.right_mode == "constant_pressure" else state.pressure[-1])
        aquifer_pressure_arr = np.full(self.grid.nx, self.aquifer_current_pressure_psia() if self.aquifer.enabled else np.nan)

        self._set_cache_state(state)
        transport_diag = self.transport_diagnostics(state)
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
            sg_max_arr[i] = mob["sg_max"]
            hysteresis_trap_arr[i] = mob["hysteresis_trap_fraction"]
            hysteresis_imbibition_arr[i] = mob["hysteresis_imbibition_flag"]
            gas_mob_arr[i] = mob["lam_g"]
            oil_mob_arr[i] = mob["lam_o"]
            water_mob_arr[i] = mob["lam_w"]
            pcog_arr[i] = mob["pcog_psia"]
            pcow_arr[i] = mob["pcow_psia"]
            hydrocarbon_potential_arr[i] = self.hydrocarbon_potential_psia(state, i)
            water_potential_arr[i] = self.water_potential_psia(state, i)
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
            "sg_max": sg_max_arr,
            "hysteresis_trap_fraction": hysteresis_trap_arr,
            "hysteresis_imbibition_flag": hysteresis_imbibition_arr,
            "gas_mobility": gas_mob_arr,
            "oil_mobility": oil_mob_arr,
            "water_mobility": water_mob_arr,
            "pcog_psia": pcog_arr,
            "pcow_psia": pcow_arr,
            "hydrocarbon_potential_psia": hydrocarbon_potential_arr,
            "water_potential_psia": water_potential_arr,
            "dew_point_minus_pressure_psia": dew_point_minus_pressure,
            "left_boundary_pressure_psia": left_boundary_pressure_arr,
            "right_boundary_pressure_psia": right_boundary_pressure_arr,
            "aquifer_pressure_psia": aquifer_pressure_arr,
            "transport_advective_flux_lbmol_day": transport_diag["transport_advective_flux_lbmol_day"],
            "transport_dispersive_flux_lbmol_day": transport_diag["transport_dispersive_flux_lbmol_day"],
            "transport_total_flux_lbmol_day": transport_diag["transport_total_flux_lbmol_day"],
            "transport_dispersion_coeff_ft2_day": transport_diag["transport_dispersion_coeff_ft2_day"],
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
        "injector_rate_target_lbmol_day": history.injector_rate_target_lbmol_day,
        "injector_rate_achievable_lbmol_day": history.injector_rate_achievable_lbmol_day,
        "injector_cell_pressure_psia": history.injector_cell_pressure_psia,
        "injector_pressure_delta_psia": history.injector_pressure_delta_psia,
        "injector_effective_bhp_psia": history.injector_effective_bhp_psia,
        "injector_active_flag": history.injector_active_flag,
        "well_gas_fraction": history.well_gas_fraction,
        "well_oil_fraction": history.well_oil_fraction,
        "well_water_fraction": history.well_water_fraction,
        "well_dropout_indicator": history.well_dropout_indicator,
        "well_krg": history.well_krg,
        "well_kro": history.well_kro,
        "well_krw": history.well_krw,
        "well_hysteresis_trap_fraction": history.well_hysteresis_trap_fraction,
        "well_hysteresis_imbibition_flag": history.well_hysteresis_imbibition_flag,
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
        "well_raw_estimated_thp_psia": getattr(history, "well_raw_estimated_thp_psia", [np.nan] * len(history.well_estimated_thp_psia)),
        "well_thp_clipped_to_atmospheric": getattr(history, "well_thp_clipped_to_atmospheric", [np.nan] * len(history.well_estimated_thp_psia)),
        "well_lift_feasible": getattr(history, "well_lift_feasible", [np.nan] * len(history.well_estimated_thp_psia)),
        "well_tubing_hydrostatic_psi": history.well_tubing_hydrostatic_psi,
        "well_tubing_friction_psi": history.well_tubing_friction_psi,
        "well_tubing_acceleration_psi": history.well_tubing_acceleration_psi,
        "well_tubing_mixture_velocity_ft_s": history.well_tubing_mixture_velocity_ft_s,
        "well_tubing_reynolds_number": history.well_tubing_reynolds_number,
        "well_tubing_friction_factor": history.well_tubing_friction_factor,
        "well_target_gas_rate_mmscf_day": history.well_target_gas_rate_mmscf_day,
        "controlled_gas_rate_mmscf_day": history.controlled_gas_rate_mmscf_day,
        "reported_gas_rate_mmscf_day": history.reported_gas_rate_mmscf_day,
        "transport_advective_flux_lbmol_day": history.transport_advective_flux_lbmol_day,
        "transport_dispersive_flux_lbmol_day": history.transport_dispersive_flux_lbmol_day,
        "transport_total_flux_lbmol_day": history.transport_total_flux_lbmol_day,
        "separator_pressure_psia": history.separator_pressure_psia,
        "separator_temperature_R": history.separator_temperature_R,
        "separator_vapor_fraction": history.separator_vapor_fraction,
        "separator_stage_count": history.separator_stage_count,
        "separator_total_gas_rate_mmscf_day": history.separator_total_gas_rate_mmscf_day,
        "separator_stage1_gas_rate_mmscf_day": history.separator_stage1_gas_rate_mmscf_day,
        "separator_stage2_gas_rate_mmscf_day": history.separator_stage2_gas_rate_mmscf_day,
        "separator_stage3_gas_rate_mmscf_day": history.separator_stage3_gas_rate_mmscf_day,
        "separator_stock_tank_gas_rate_mmscf_day": history.separator_stock_tank_gas_rate_mmscf_day,
        "separator_stock_tank_liquid_rate_stb_day": history.separator_stock_tank_liquid_rate_stb_day,
    })
    if start_date_text is not None:
        start_dt = parse_ddmmyyyy(start_date_text)
        actual_dates = pd.to_datetime(start_dt) + pd.to_timedelta(df["time_days"], unit="D")
        df.insert(1, "year", actual_dates.dt.year.astype(int))
        df.insert(2, "date", actual_dates.dt.strftime("%d/%m/%Y"))

        # Report one snapshot per calendar year in the history table.
        # We keep the last simulated record in each year so pressures,
        # saturations, compositions, and cumulative volumes reflect the
        # year-end state rather than every internal/reporting timestep.
        df = (
            df.sort_values("time_days")
              .groupby("year", as_index=False, sort=True)
              .tail(1)
              .reset_index(drop=True)
        )
        df = df.drop(columns=["date"])
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
    hyst_peak, hyst_peak_t = max_with_time(history.well_hysteresis_trap_fraction)
    pl_peak, pl_peak_t = max_with_time(history.productivity_loss_fraction)
    mb_hc_peak, mb_hc_t = max_with_time(np.abs(np.asarray(history.hc_mass_balance_error_lbmol, dtype=float)).tolist())

    return [
        ["Average Pressure (psia)", initial_value(history.avg_pressure_psia), p_min, p_min_t, final_value(history.avg_pressure_psia)],
        ["Dew Point (psia)", initial_value(history.dew_point_psia), final_value(history.dew_point_psia), history.time_days[-1], final_value(history.dew_point_psia)],
        ["Well Pressure - Dew Point (psia)", initial_value(history.well_pressure_minus_dewpoint_psia), dp_margin_min, dp_margin_min_t, final_value(history.well_pressure_minus_dewpoint_psia)],
        ["Average Water Saturation (-)", initial_value(history.avg_sw), max(history.avg_sw), history.time_days[int(np.argmax(history.avg_sw))], final_value(history.avg_sw)],
        ["Dropout Indicator (-)", initial_value(history.well_dropout_indicator), d_peak, d_peak_t, final_value(history.well_dropout_indicator)],
        ["Damage Factor (-, minimum is worst)", initial_value(history.well_damage_factor), df_min, df_min_t, final_value(history.well_damage_factor)],
        ["Well Hysteresis Trap Fraction (-)", initial_value(history.well_hysteresis_trap_fraction), hyst_peak, hyst_peak_t, final_value(history.well_hysteresis_trap_fraction)],
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
    ax = plt.gca()
    for y, label in series:
        ax.plot(x, y, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    x_arr = np.asarray(x)
    is_datetime_axis = np.issubdtype(x_arr.dtype, np.datetime64)

    if is_datetime_axis:
        x_pd = pd.to_datetime(x_arr)
        if len(x_pd) >= 2:
            span_days = max((x_pd.max() - x_pd.min()).days, 1)
        else:
            span_days = 365

        span_years = span_days / 365.25
        if span_years <= 8:
            year_step = 1
        elif span_years <= 20:
            year_step = 2
        elif span_years <= 40:
            year_step = 5
        else:
            year_step = 10

        ax.xaxis.set_major_locator(mdates.YearLocator(base=year_step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', labelrotation=0)
    else:
        xfmt = ScalarFormatter(useOffset=False)
        xfmt.set_scientific(False)
        ax.xaxis.set_major_formatter(xfmt)

    yfmt = ScalarFormatter(useOffset=False)
    yfmt.set_scientific(False)
    ax.yaxis.set_major_formatter(yfmt)

    if len(series) > 1:
        ax.legend()
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
    injection_start_date_text: str | None = None,
    injection_end_date_text: str | None = None,
    injection_control_mode: str = "simple",
    injection_pressure_psia: float = 7000.0,
    max_injection_bhp_psia: float = 9000.0,
    injectivity_index_lbmol_day_psi: float = 500.0,
    injected_gas_composition: np.ndarray | None = None,
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
    tubing_roughness_in: float = 0.0006,
    tubing_calibration_factor: float = 1.0,
    tubing_model: str = "mechanistic",
    tubing_segments: int = 25,
    min_lift_thp_psia: float = 100.0,
    productivity_index: float = 1.0,
    rw_ft: float = 0.35,
    skin: float = 0.0,
    temperature_R: float = 680.0,
    p_init_psia: float = 5600.0,
    dew_point_psia: float = 4200.0,
    sw_init: float = 0.20,
    separator_pressure_psia: float = 300.0,
    separator_temperature_R: float = 520.0,
    separator_stages: int = 1,
    separator_second_stage_pressure_psia: float = 100.0,
    separator_second_stage_temperature_R: float = 520.0,
    separator_third_stage_pressure_psia: float = 50.0,
    separator_third_stage_temperature_R: float = 520.0,
    stock_tank_pressure_psia: float = 14.7,
    stock_tank_temperature_R: float = 519.67,
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
    capillary_enabled: bool = True,
    pcow_entry_psia: float = 15.0,
    pcog_entry_psia: float = 8.0,
    pc_lambda_w: float = 2.0,
    pc_lambda_g: float = 2.0,
    hysteresis_enabled: bool = True,
    hysteresis_reversal_tolerance: float = 0.01,
    hysteresis_gas_trapping_strength: float = 0.60,
    hysteresis_imbibition_krg_reduction: float = 0.75,
    hysteresis_imbibition_kro_reduction: float = 0.15,
    transport_enabled: bool = True,
    transport_phase_split_advection: bool = True,
    transport_dispersivity_ft: float = 15.0,
    transport_molecular_diffusion_ft2_day: float = 0.15,
    transport_max_dispersive_fraction: float = 0.35,
    left_boundary_mode: str = "closed",
    right_boundary_mode: str = "closed",
    left_boundary_pressure_psia: float = 5600.0,
    right_boundary_pressure_psia: float = 3000.0,
    left_boundary_transmissibility_multiplier: float = 1.0,
    right_boundary_transmissibility_multiplier: float = 1.0,
    aquifer_enabled: bool = False,
    aquifer_side: str = "left",
    aquifer_initial_pressure_psia: float = 5600.0,
    aquifer_productivity_index_lbmol_day_psi: float = 500.0,
    aquifer_productivity_index_stb_day_psi: float | None = None,
    aquifer_total_capacity_lbmol_per_psi: float = 5.0e5,
    aquifer_total_capacity_mstb_per_psi: float | None = None,
    aquifer_water_influx_fraction: float = 1.0,
    aquifer_allow_backflow: bool = False,
    gas_specific_gravity: float = 0.65,
    condensate_api_gravity: float = 50.0,
    pvt_table: PVTTable | None = None,
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

    if injected_gas_composition is None:
        injected_gas_composition = z_init.copy()
    injected_gas_composition = np.asarray(injected_gas_composition, dtype=float)
    if len(injected_gas_composition) != fluid.nc:
        raise ValueError("Injected gas composition length does not match number of components")
    injected_gas_composition = injected_gas_composition / np.sum(injected_gas_composition)

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
    capillary = CapillaryPressureParams(
        enabled=capillary_enabled,
        pcow_entry_psia=pcow_entry_psia,
        pcog_entry_psia=pcog_entry_psia,
        lambda_w=pc_lambda_w,
        lambda_g=pc_lambda_g,
        swir=Swc,
        sorw=Sorw,
        sgc=Sgc,
        sorg=Sorg,
    )
    hysteresis = HysteresisParams(
        enabled=hysteresis_enabled,
        reversal_tolerance=hysteresis_reversal_tolerance,
        gas_trapping_strength=hysteresis_gas_trapping_strength,
        imbibition_krg_reduction=hysteresis_imbibition_krg_reduction,
        imbibition_kro_reduction=hysteresis_imbibition_kro_reduction,
    )
    transport = TransportParams(
        enabled=transport_enabled,
        phase_split_advection=transport_phase_split_advection,
        dispersivity_ft=transport_dispersivity_ft,
        molecular_diffusion_ft2_day=transport_molecular_diffusion_ft2_day,
        max_dispersive_fraction=transport_max_dispersive_fraction,
    )
    boundary = BoundaryConditionConfig(
        left_mode=left_boundary_mode,
        right_mode=right_boundary_mode,
        left_pressure_psia=left_boundary_pressure_psia,
        right_pressure_psia=right_boundary_pressure_psia,
        left_transmissibility_multiplier=left_boundary_transmissibility_multiplier,
        right_transmissibility_multiplier=right_boundary_transmissibility_multiplier,
    )
    if aquifer_productivity_index_stb_day_psi is not None:
        aquifer_productivity_index_lbmol_day_psi = float(aquifer_productivity_index_stb_day_psi) * water_lbmol_per_stb()
    if aquifer_total_capacity_mstb_per_psi is not None:
        aquifer_total_capacity_lbmol_per_psi = float(aquifer_total_capacity_mstb_per_psi) * water_lbmol_per_mstb()

    aquifer = AquiferConfig(
        enabled=aquifer_enabled,
        side=aquifer_side,
        initial_pressure_psia=aquifer_initial_pressure_psia,
        productivity_index_lbmol_day_psi=aquifer_productivity_index_lbmol_day_psi,
        total_capacity_lbmol_per_psi=aquifer_total_capacity_lbmol_per_psi,
        water_influx_fraction=aquifer_water_influx_fraction,
        allow_backflow=aquifer_allow_backflow,
    )
    if pvt_table is not None and pvt_table.has('dew_point_psia'):
        dew_point_psia = float(pvt_table.interp(p_init_psia, 'dew_point_psia', dew_point_psia))
    pvt = PVTConfig(dew_point_psia=dew_point_psia, table=pvt_table)
    separator = SeparatorConfig(
        pressure_psia=separator_pressure_psia,
        temperature_R=separator_temperature_R,
        stages=separator_stages,
        second_stage_pressure_psia=separator_second_stage_pressure_psia,
        second_stage_temperature_R=separator_second_stage_temperature_R,
        third_stage_pressure_psia=separator_third_stage_pressure_psia,
        third_stage_temperature_R=separator_third_stage_temperature_R,
        stock_tank_pressure_psia=stock_tank_pressure_psia,
        stock_tank_temperature_R=stock_tank_temperature_R,
    )
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
        tubing_roughness_in=tubing_roughness_in,
        tubing_calibration_factor=tubing_calibration_factor,
        tubing_model=tubing_model,
        tubing_segments=tubing_segments,
        min_lift_thp_psia=min_lift_thp_psia,
        productivity_index=productivity_index,
        rw_ft=rw_ft,
        skin=skin,
    )

    injection_rate_lbmol_day = gas_field_rate_to_lbmol_day(injection_rate_value, injection_rate_unit)

    injector = None
    if scenario_name in ("gas_cycling", "lean_gas_injection"):
        if injection_start_date_text is None:
            injection_start_date_text = start_date_text
        if injection_end_date_text is None:
            injection_end_date_text = start_date_text
        inj_start_day = days_between_dates(start_date_text, injection_start_date_text, allow_same_day=True)
        inj_end_day = days_between_dates(start_date_text, injection_end_date_text, allow_same_day=True)
        inj_comp = injected_gas_composition.copy() if scenario_name in ("gas_cycling", "lean_gas_injection") else lean_gas_composition.copy()
        injector = InjectorWell(
            cell_index=0,
            rate_lbmol_day=injection_rate_lbmol_day,
            injection_composition=inj_comp,
            rate_field_input=injection_rate_value,
            rate_field_unit=injection_rate_unit,
            start_day=inj_start_day,
            end_day=inj_end_day,
            control_mode=injection_control_mode,
            injection_pressure_psia=injection_pressure_psia,
            max_bhp_psia=max_injection_bhp_psia,
            injectivity_index_lbmol_day_psi=injectivity_index_lbmol_day_psi,
        )

    scenario = ScenarioConfig(name=scenario_name, injector=injector)
    reporting = ReportingConfig(
        gas_specific_gravity=gas_specific_gravity,
        condensate_api_gravity=condensate_api_gravity,
    )

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
        capillary=capillary,
        hysteresis=hysteresis,
        transport=transport,
        boundary=boundary,
        aquifer=aquifer,
        pvt=pvt,
        separator=separator,
        reporting=reporting,
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



def parse_date_text(date_text: str) -> datetime:
    try:
        return datetime.strptime(date_text.strip(), "%d/%m/%Y")
    except Exception as exc:
        raise ValueError(f"Invalid date format: {date_text}. Use dd/mm/yyyy.") from exc


def composition_editor(label: str, component_names: List[str], default_values: np.ndarray, key_prefix: str) -> tuple[np.ndarray, float, str]:
    default_values = np.asarray(default_values, dtype=float)
    unit_key = f"{key_prefix}_unit_mode"
    values_key = f"{key_prefix}_values_fraction"
    prev_unit_key = f"{key_prefix}_prev_unit_mode"
    table_key = f"{key_prefix}_table_df"
    render_key = f"{key_prefix}_render_nonce"

    unit_mode = st.radio(
        f"{label} input mode",
        ["Fraction", "Percentage"],
        horizontal=True,
        key=unit_key,
    )

    if values_key not in st.session_state:
        st.session_state[values_key] = default_values.astype(float).copy()

    try:
        stored_values = np.asarray(st.session_state[values_key], dtype=float)
    except Exception:
        stored_values = default_values.astype(float).copy()
    if stored_values.shape != default_values.shape or not np.all(np.isfinite(stored_values)):
        stored_values = default_values.astype(float).copy()
        st.session_state[values_key] = stored_values.copy()

    previous_unit = st.session_state.get(prev_unit_key)
    if previous_unit is None:
        st.session_state[prev_unit_key] = unit_mode
    elif previous_unit != unit_mode:
        st.session_state[prev_unit_key] = unit_mode
        st.session_state[render_key] = int(st.session_state.get(render_key, 0)) + 1
        if table_key in st.session_state:
            del st.session_state[table_key]

    display_scale = 100.0 if unit_mode == "Percentage" else 1.0
    default_table_df = pd.DataFrame({
        "Component": component_names,
        "Value": np.round(stored_values * display_scale, 6),
    })

    current_table_df = st.session_state.get(table_key)
    needs_reset = True
    if isinstance(current_table_df, pd.DataFrame):
        same_columns = list(current_table_df.columns) == ["Component", "Value"]
        same_rows = len(current_table_df) == len(component_names)
        same_components = same_rows and current_table_df["Component"].astype(str).tolist() == list(component_names)
        if same_columns and same_components:
            needs_reset = False
    if needs_reset:
        st.session_state[table_key] = default_table_df.copy()

    widget_key = f"{key_prefix}_editor_{int(st.session_state.get(render_key, 0))}"
    edited_df = st.data_editor(
        st.session_state[table_key],
        key=widget_key,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["Component"],
        column_config={
            "Component": st.column_config.TextColumn("Component", disabled=True),
            "Value": st.column_config.NumberColumn("Value", format="%.6f"),
        },
    )

    if not isinstance(edited_df, pd.DataFrame) or edited_df.empty or list(edited_df.columns) != ["Component", "Value"]:
        edited_df = default_table_df.copy()
    elif len(edited_df) != len(component_names):
        edited_df = default_table_df.copy()
    else:
        edited_df = edited_df[["Component", "Value"]].copy()
        edited_df["Component"] = component_names

    st.session_state[table_key] = edited_df.copy()

    values_display = pd.to_numeric(edited_df["Value"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_sum = float(np.sum(values_display))
    comp = values_display / 100.0 if unit_mode == "Percentage" else values_display.copy()
    total = float(np.sum(comp))
    if total <= 0.0:
        raise ValueError(f"{label} must have a positive total.")
    comp = comp / total
    st.session_state[values_key] = comp.copy()
    return comp, raw_sum, unit_mode

# -----------------------------------------------------------------------------
# Streamlit UI with grouped sidebar inputs
# -----------------------------------------------------------------------------

def _normalize_history_match_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')


def load_history_match_dataframe(df: pd.DataFrame, start_date_text: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_history_match_column_name(c) for c in df.columns]

    alias_map = {
        'time': 'time_days',
        'day': 'time_days',
        'days': 'time_days',
        'date_time': 'date',
        'datetime': 'date',
        'avg_pressure': 'avg_pressure_psia',
        'average_pressure': 'avg_pressure_psia',
        'well_pressure': 'well_pressure_psia',
        'gas_rate': 'gas_rate_mmscf_day',
        'condensate_rate': 'condensate_rate_stb_day',
        'water_rate': 'water_rate_stb_day',
        'cumulative_gas': 'cum_gas_bscf',
        'cumulative_condensate': 'cum_condensate_mstb',
        'cumulative_water': 'cum_water_mstb',
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if 'time_days' not in df.columns:
        if 'date' not in df.columns:
            raise ValueError('History-match file must contain either a Time/Days column or a Date column.')
        if not start_date_text:
            raise ValueError('A simulation start date is required when matching on Date.')
        start_dt = parse_ddmmyyyy(start_date_text)
        obs_dates = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        if obs_dates.isna().all():
            raise ValueError('Could not parse any Date values in the history-match file.')
        df['time_days'] = (obs_dates - pd.Timestamp(start_dt)).dt.total_seconds() / 86400.0

    df['time_days'] = pd.to_numeric(df['time_days'], errors='coerce')
    df = df[np.isfinite(df['time_days'])].copy()
    if df.empty:
        raise ValueError('History-match file does not contain any valid time rows.')

    metric_cols = [
        'avg_pressure_psia', 'well_pressure_psia', 'gas_rate_mmscf_day',
        'condensate_rate_stb_day', 'water_rate_stb_day', 'cum_gas_bscf',
        'cum_condensate_mstb', 'cum_water_mstb',
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    keep_cols = ['time_days'] + [c for c in metric_cols if c in df.columns]
    if len(keep_cols) <= 1:
        raise ValueError('History-match file must contain at least one observed metric column.')

    df = df[keep_cols].sort_values('time_days').drop_duplicates(subset=['time_days'], keep='last').reset_index(drop=True)
    return df


def _safe_interp(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.full_like(x_new, np.nan, dtype=float)
    if x.size == 1:
        return np.full_like(x_new, y[0], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return np.interp(x_new, x, y, left=y[0], right=y[-1])


def compute_history_match(history: SimulationHistory, observed_df: pd.DataFrame, weights: dict[str, float] | None = None) -> dict[str, object]:
    sim_df = history_dataframe(history, start_date_text=None)
    sim_time = np.asarray(sim_df['time_days'], dtype=float)
    obs_time = np.asarray(observed_df['time_days'], dtype=float)

    default_weights = {
        'avg_pressure_psia': 1.0,
        'well_pressure_psia': 1.0,
        'gas_rate_mmscf_day': 1.0,
        'condensate_rate_stb_day': 1.0,
        'water_rate_stb_day': 1.0,
        'cum_gas_bscf': 0.75,
        'cum_condensate_mstb': 0.75,
        'cum_water_mstb': 0.75,
    }
    if weights:
        default_weights.update(weights)

    comparison = pd.DataFrame({'time_days': obs_time})
    metrics_rows = []
    objective = 0.0
    total_weight = 0.0

    for metric, weight in default_weights.items():
        if metric not in observed_df.columns or metric not in sim_df.columns:
            continue
        obs = pd.to_numeric(observed_df[metric], errors='coerce').to_numpy(dtype=float)
        sim = _safe_interp(obs_time, sim_time, np.asarray(sim_df[metric], dtype=float))
        valid = np.isfinite(obs) & np.isfinite(sim)
        comparison[f'obs_{metric}'] = obs
        comparison[f'sim_{metric}'] = sim
        comparison[f'resid_{metric}'] = sim - obs
        if np.count_nonzero(valid) < 2:
            continue
        obs_v = obs[valid]
        sim_v = sim[valid]
        resid = sim_v - obs_v
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        denom = float(np.nanmax(obs_v) - np.nanmin(obs_v))
        if denom <= 1e-12:
            denom = max(float(np.mean(np.abs(obs_v))), 1.0)
        nrmse = float(rmse / denom)
        mape_mask = np.abs(obs_v) > 1e-12
        mape = float(np.mean(np.abs(resid[mape_mask] / obs_v[mape_mask])) * 100.0) if np.any(mape_mask) else np.nan
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((obs_v - np.mean(obs_v)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan
        weighted = float(weight * nrmse)
        objective += weighted
        total_weight += float(weight)
        metrics_rows.append({
            'Metric': metric,
            'Points Matched': int(np.count_nonzero(valid)),
            'Weight': float(weight),
            'RMSE': rmse,
            'MAE': mae,
            'NRMSE': nrmse,
            'MAPE_percent': mape,
            'R2': r2,
            'Weighted Objective Contribution': weighted,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    objective_normalized = float(objective / total_weight) if total_weight > 0.0 else np.nan
    return {
        'metrics_df': metrics_df,
        'comparison_df': comparison,
        'objective_raw': float(objective),
        'objective_normalized': objective_normalized,
        'matched_metrics': int(len(metrics_rows)),
    }


def _monte_carlo_parameter_map(injection_rate_unit: str) -> Dict[str, Tuple[str, str]]:
    return {
        "Permeability (md)": ("permeability_md", "Permeability (md)"),
        "Porosity": ("porosity", "Porosity"),
        "Initial Reservoir Pressure (psia)": ("p_init_psia", "Initial Reservoir Pressure (psia)"),
        "Dew Point Pressure (psia)": ("dew_point_psia", "Dew Point Pressure (psia)"),
        "Producer BHP (psia)": ("bhp_psia", "Producer BHP (psia)"),
        "Gas Injection Rate": ("injection_rate_value", f"Gas Injection Rate ({injection_rate_unit})"),
        "Transport Dispersivity (ft)": ("transport_dispersivity_ft", "Transport Dispersivity (ft)"),
    }


def _sample_parameter_value(param_key: str, base_value: float, low_pct: float, high_pct: float, rng: np.random.Generator, scenario_name: str) -> float:
    low_mult = 1.0 + low_pct / 100.0
    high_mult = 1.0 + high_pct / 100.0
    lo = min(low_mult, high_mult)
    hi = max(low_mult, high_mult)
    if abs(hi - lo) < 1e-12:
        factor = lo
    else:
        factor = float(rng.triangular(lo, 1.0, hi))
    value = float(base_value * factor)
    if param_key == 'porosity':
        return float(np.clip(value, 0.01, 0.95))
    if param_key in {'bhp_psia', 'p_init_psia', 'dew_point_psia'}:
        return max(14.7, value)
    if param_key == 'injection_rate_value':
        if scenario_name == 'natural_depletion':
            return 0.0
        return max(0.0, value)
    if param_key == 'transport_dispersivity_ft':
        return max(0.0, value)
    return max(1e-9, value)


def _percentile_summary(values: np.ndarray) -> Tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    return tuple(float(x) for x in np.percentile(finite, [10, 50, 90]))


def display_streamlit_app() -> None:
    st.set_page_config(page_title="Compositional Reservoir Simulator", layout="wide")
    st.title("Compositional Reservoir Simulator")
    st.write(
        "1D compositional gas-condensate-water prototype with Peng-Robinson EOS, semi-implicit pressure update, adaptive timestepping, "
        "Peaceman well model, local-grid refinement, three-phase relative permeability, capillary-pressure support, dew-point diagnostics, scenario-based development analysis, "
        "producer control by fixed BHP, fixed drawdown, fixed gas rate, or simplified fixed THP, a configurable Stage 1 / Stage 2 / Stage 3 / stock-tank surface separator train for reporting gas and condensate rates, and phase-split/dispersive compositional transport between cells."
    )

    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            display: flex;
            flex-direction: column !important;
            gap: 0.35rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] details {
            border-radius: 0.75rem;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] .streamlit-expanderContent {
            padding-top: 0.25rem;
        }
        section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stTextInput,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stDateInput,
        section[data-testid="stSidebar"] .stCheckbox,
        section[data-testid="stSidebar"] .stButton,
        section[data-testid="stSidebar"] .stFileUploader,
        section[data-testid="stSidebar"] .stDataEditor {
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    section_options = [
        "Case Setup",
        "Reservoir / Grid",
        "Components / Composition",
        "PVT / Separator",
        "Rock / Rock-Fluid",
        "Initial Conditions",
        "Producer Well",
        "Injector Well",
        "Aquifer / Boundaries",
        "Advanced Physics",
        "Analysis Tools",
        "Results",
    ]
    st.sidebar.title("Input Sections")
    st.sidebar.caption("All sections stay mounted below as tabs so values persist while you move around.")
    tab_case_setup, tab_reservoir_grid, tab_components_composition, tab_pvt_separator, tab_rock_rockfluid, tab_initial_conditions, tab_producer_well, tab_injector_well, tab_aquifer_boundaries, tab_advanced_physics, tab_analysis_tools, tab_results = st.tabs(section_options)

    input_validation_errors: List[str] = []
    pvt_table = None

    def _ss(key: str, default):
        return st.session_state.get(key, default)

    # Persist canonical input values across page switches. In page-navigation mode,
    # widgets may be temporarily unmounted; if the values are not anchored in
    # session_state first, Streamlit can recreate them from literal defaults.
    persistent_defaults = {
        "scenario_label": "Natural Depletion",
        "start_date_text": "01/01/2025",
        "end_date_text": "29/08/2025",
        "dt_days": 0.05,
        "nx": 40,
        "length_ft": 4000.0,
        "radius_ft": 100.0,
        "thickness_ft": 50.0,
        "refined_cells": 10,
        "min_dx_ft": 12.0,
        "growth": 1.45,
        "temperature_F": 220.0,
        "dew_point_psia": 4200.0,
        "separator_stages": 4,
        "separator_pressure_psia": 300.0,
        "separator_temperature_F": 100.0,
        "separator_second_stage_pressure_psia": 100.0,
        "separator_second_stage_temperature_F": 80.0,
        "separator_third_stage_pressure_psia": 50.0,
        "separator_third_stage_temperature_F": 70.0,
        "stock_tank_pressure_psia": 14.7,
        "stock_tank_temperature_F": 60.0,
        "porosity": 0.18,
        "permeability_md": 15.0,
        "Sgc": 0.05,
        "Swc": 0.20,
        "Sorw": 0.20,
        "Sorg": 0.05,
        "krg0": 1.0,
        "kro0": 1.0,
        "krw0": 0.4,
        "ng": 2.0,
        "no": 2.0,
        "nw": 2.0,
        "gas_specific_gravity": 0.65,
        "condensate_api_gravity": 50.0,
        "p_init_psia": 5600.0,
        "sw_init": 0.20,
        "well_control_label": "Bottom-hole pressure (BHP)",
        "bhp_psia": 3000.0,
        "drawdown_psia": 200.0,
        "min_bhp_psia": 50.0,
        "target_gas_rate_mmscf_day": 5.0,
        "thp_psia": 500.0,
        "tvd_ft": 8000.0,
        "tubing_id_in": 2.441,
        "wellhead_temperature_F": 60.0,
        "tubing_roughness_in": 0.0006,
        "tubing_calibration_factor": 1.0,
        "tubing_model": "Segmented mechanistic multiphase",
        "tubing_segments": 25,
        "min_lift_thp_psia": 100.0,
        "thp_friction_coeff": 0.02,
        "productivity_index": 1.0,
        "rw_ft": 0.35,
        "skin": 0.0,
        "injection_rate_value": 5.0,
        "injection_rate_unit": "MMscf/day",
        "injection_control_label": "Enhanced pressure-controlled",
        "injection_pressure_psia": 7000.0,
        "max_injection_bhp_psia": 9000.0,
        "injectivity_index_mmscf_day_psi": gas_lbmol_day_to_mmscf_day(500.0),
        "injection_start_date_text": "01/01/2025",
        "injection_end_date_text": "29/08/2025",
        "left_boundary_label": "Closed",
        "left_boundary_pressure_psia": 5600.0,
        "left_boundary_transmissibility_multiplier": 1.0,
        "right_boundary_label": "Closed",
        "right_boundary_pressure_psia": 3000.0,
        "right_boundary_transmissibility_multiplier": 1.0,
        "aquifer_enabled": False,
        "aquifer_side": "left",
        "aquifer_initial_pressure_psia": 5600.0,
        "aquifer_water_influx_fraction": 1.0,
        "aquifer_allow_backflow": False,
        "capillary_enabled": True,
        "pcow_entry_psia": 15.0,
        "pcog_entry_psia": 8.0,
        "pc_lambda_w": 2.0,
        "pc_lambda_g": 2.0,
        "hysteresis_enabled": True,
        "hysteresis_reversal_tolerance": 0.01,
        "hysteresis_gas_trapping_strength": 0.60,
        "hysteresis_imbibition_krg_reduction": 0.75,
        "hysteresis_imbibition_kro_reduction": 0.15,
        "transport_enabled": True,
        "transport_phase_split_advection": True,
        "transport_dispersivity_ft": 15.0,
        "transport_molecular_diffusion_ft2_day": 0.15,
        "transport_max_dispersive_fraction": 0.35,
        "history_match_enabled": False,
        "enable_scenario_comparison": False,
        "comparison_scenarios": ["Natural Depletion", "Gas Cycling", "Lean Gas Injection"],
        "enable_sensitivity": False,
        "sensitivity_param_label": "Permeability",
        "sensitivity_low_pct": -20.0,
        "sensitivity_high_pct": 20.0,
        "enable_monte_carlo": False,
        "monte_carlo_runs": 50,
        "monte_carlo_low_pct": -15.0,
        "monte_carlo_high_pct": 15.0,
        "monte_carlo_seed": 42,
    }
    _aq_pi_default_stb_day_psi = 500.0 / water_lbmol_per_stb()
    _aq_cap_default_mstb_psi = 5.0e5 / water_lbmol_per_mstb()
    persistent_defaults.setdefault("aquifer_productivity_index_stb_day_psi", _aq_pi_default_stb_day_psi)
    persistent_defaults.setdefault("aquifer_total_capacity_mstb_per_psi", _aq_cap_default_mstb_psi)
    for _key, _default in persistent_defaults.items():
        shadow_key = f"persist__{_key}"
        if _key in st.session_state:
            st.session_state[shadow_key] = st.session_state[_key]
        elif shadow_key in st.session_state:
            st.session_state[_key] = st.session_state[shadow_key]
        else:
            st.session_state[_key] = _default
            st.session_state[shadow_key] = _default

    # Stable defaults so page-style navigation can render any section independently.
    scenario_label = _ss("scenario_label", "Natural Depletion")
    scenario_map = {
        "Natural Depletion": "natural_depletion",
        "Gas Cycling": "gas_cycling",
        "Lean Gas Injection": "lean_gas_injection",
    }
    scenario_name = scenario_map.get(scenario_label, "natural_depletion")
    start_date_text = _ss("start_date_text", "01/01/2025")
    end_date_text = _ss("end_date_text", "29/08/2025")
    try:
        t_end_days = days_between_dates(start_date_text, end_date_text)
    except Exception:
        t_end_days = None
    dt_days = float(_ss("dt_days", 0.05))

    nx = int(_ss("nx", 40))
    length_ft = float(_ss("length_ft", 4000.0))
    radius_ft = float(_ss("radius_ft", 100.0))
    thickness_ft = float(_ss("thickness_ft", 50.0))
    equivalent_width_ft = 2.0 * radius_ft
    area_ft2 = thickness_ft * equivalent_width_ft
    refined_cells = int(_ss("refined_cells", min(10, max(nx - 1, 2))))
    min_dx_ft = float(_ss("min_dx_ft", 12.0))
    growth = float(_ss("growth", 1.45))

    default_initial = np.array([0.005, 0.020, 0.010, 0.780, 0.070, 0.040, 0.015, 0.015, 0.010, 0.010, 0.015, 0.010], dtype=float)
    z_init = np.asarray(_ss("initial_composition_values", default_initial), dtype=float)
    if z_init.size != default_initial.size or np.sum(z_init) <= 0:
        z_init = default_initial.copy()
    z_init = z_init / np.sum(z_init)
    injected_gas_composition = np.asarray(_ss("injected_gas_composition_values", z_init.copy()), dtype=float)
    if injected_gas_composition.size != z_init.size or np.sum(injected_gas_composition) <= 0:
        injected_gas_composition = z_init.copy()
    injected_gas_composition = injected_gas_composition / np.sum(injected_gas_composition)
    lean_gas_composition = injected_gas_composition.copy()

    temperature_F = float(_ss("temperature_F", 220.0))
    temperature_R = fahrenheit_to_rankine(temperature_F)
    dew_point_psia = float(_ss("dew_point_psia", 4200.0))
    separator_stages = int(_ss("separator_stages", 4))
    separator_pressure_psia = float(_ss("separator_pressure_psia", 300.0))
    separator_temperature_F = float(_ss("separator_temperature_F", 100.0))
    separator_temperature_R = fahrenheit_to_rankine(separator_temperature_F)
    separator_second_stage_pressure_psia = float(_ss("separator_second_stage_pressure_psia", 100.0))
    separator_second_stage_temperature_F = float(_ss("separator_second_stage_temperature_F", 80.0))
    separator_second_stage_temperature_R = fahrenheit_to_rankine(separator_second_stage_temperature_F)
    separator_third_stage_pressure_psia = float(_ss("separator_third_stage_pressure_psia", 50.0))
    separator_third_stage_temperature_F = float(_ss("separator_third_stage_temperature_F", 70.0))
    separator_third_stage_temperature_R = fahrenheit_to_rankine(separator_third_stage_temperature_F)
    stock_tank_pressure_psia = float(_ss("stock_tank_pressure_psia", 14.7))
    stock_tank_temperature_F = float(_ss("stock_tank_temperature_F", 60.0))
    stock_tank_temperature_R = fahrenheit_to_rankine(stock_tank_temperature_F)

    porosity = float(_ss("porosity", 0.18))
    permeability_md = float(_ss("permeability_md", 15.0))
    Sgc = float(_ss("Sgc", 0.05))
    Swc = float(_ss("Swc", 0.20))
    Sorw = float(_ss("Sorw", 0.20))
    Sorg = float(_ss("Sorg", 0.05))
    krg0 = float(_ss("krg0", 1.0))
    kro0 = float(_ss("kro0", 1.0))
    krw0 = float(_ss("krw0", 0.4))
    ng = float(_ss("ng", 2.0))
    no = float(_ss("no", 2.0))
    nw = float(_ss("nw", 2.0))
    relperm_input_valid = (Sgc + Swc < 1.0) and (Swc + Sorw < 1.0) and (Swc + Sorg < 1.0)
    gas_specific_gravity = float(_ss("gas_specific_gravity", 0.65))
    condensate_api_gravity = float(_ss("condensate_api_gravity", 50.0))

    p_init_psia = float(_ss("p_init_psia", 5600.0))
    sw_init = float(_ss("sw_init", 0.20))

    well_control_label = _ss("well_control_label", "Bottom-hole pressure (BHP)")
    well_control_mode_map = {
        "Bottom-hole pressure (BHP)": "bhp",
        "Pressure drawdown": "drawdown",
        "Target gas rate": "gas_rate",
        "Tubing head pressure (THP)": "thp",
    }
    well_control_mode = well_control_mode_map.get(well_control_label, "bhp")
    bhp_psia = float(_ss("bhp_psia", 3000.0))
    drawdown_psia = float(_ss("drawdown_psia", 200.0))
    min_bhp_psia = float(_ss("min_bhp_psia", 50.0))
    target_gas_rate_mmscf_day = float(_ss("target_gas_rate_mmscf_day", 5.0))
    thp_psia = float(_ss("thp_psia", 500.0))
    tvd_ft = float(_ss("tvd_ft", 8000.0))
    tubing_id_in = float(_ss("tubing_id_in", 2.441))
    wellhead_temperature_F = float(_ss("wellhead_temperature_F", 60.0))
    wellhead_temperature_R = fahrenheit_to_rankine(wellhead_temperature_F)
    tubing_roughness_in = float(_ss("tubing_roughness_in", 0.0006))
    tubing_calibration_factor = float(_ss("tubing_calibration_factor", 1.0))
    tubing_model = _ss("tubing_model", "Segmented mechanistic multiphase")
    tubing_segments = int(_ss("tubing_segments", 25))
    min_lift_thp_psia = float(_ss("min_lift_thp_psia", 100.0))
    thp_friction_coeff = float(_ss("thp_friction_coeff", 0.02))
    productivity_index = float(_ss("productivity_index", 1.0))
    rw_ft = float(_ss("rw_ft", 0.35))
    skin = float(_ss("skin", 0.0))

    injection_rate_value = float(_ss("injection_rate_value", 5.0))
    injection_rate_unit = _ss("injection_rate_unit", "MMscf/day")
    injection_control_label = _ss("injection_control_label", "Enhanced pressure-controlled")
    injection_control_mode = "enhanced" if injection_control_label == "Enhanced pressure-controlled" else "simple"
    injection_pressure_psia = float(_ss("injection_pressure_psia", 7000.0))
    max_injection_bhp_psia = float(_ss("max_injection_bhp_psia", 9000.0))
    injectivity_index_mmscf_day_psi = float(_ss("injectivity_index_mmscf_day_psi", gas_lbmol_day_to_mmscf_day(500.0)))
    injectivity_index_lbmol_day_psi = injectivity_mmscf_day_psi_to_lbmol_day_psi(injectivity_index_mmscf_day_psi)
    injection_start_date_text = _ss("injection_start_date_text", start_date_text)
    injection_end_date_text = _ss("injection_end_date_text", end_date_text)

    left_boundary_label = _ss("left_boundary_label", "Closed")
    left_boundary_mode = "closed" if left_boundary_label == "Closed" else "constant_pressure"
    left_boundary_pressure_psia = float(_ss("left_boundary_pressure_psia", 5600.0))
    left_boundary_transmissibility_multiplier = float(_ss("left_boundary_transmissibility_multiplier", 1.0))
    right_boundary_label = _ss("right_boundary_label", "Closed")
    right_boundary_mode = "closed" if right_boundary_label == "Closed" else "constant_pressure"
    right_boundary_pressure_psia = float(_ss("right_boundary_pressure_psia", 3000.0))
    right_boundary_transmissibility_multiplier = float(_ss("right_boundary_transmissibility_multiplier", 1.0))
    aquifer_enabled = bool(_ss("aquifer_enabled", False))
    aquifer_side = _ss("aquifer_side", "left")
    aquifer_initial_pressure_psia = float(_ss("aquifer_initial_pressure_psia", 5600.0))
    _aq_pi_default_stb_day_psi = 500.0 / water_lbmol_per_stb()
    _aq_cap_default_mstb_psi = 5.0e5 / water_lbmol_per_mstb()
    aquifer_productivity_index_stb_day_psi = float(_ss("aquifer_productivity_index_stb_day_psi", _aq_pi_default_stb_day_psi))
    aquifer_total_capacity_mstb_per_psi = float(_ss("aquifer_total_capacity_mstb_per_psi", _aq_cap_default_mstb_psi))
    aquifer_water_influx_fraction = float(_ss("aquifer_water_influx_fraction", 1.0))
    aquifer_allow_backflow = bool(_ss("aquifer_allow_backflow", False))

    capillary_enabled = bool(_ss("capillary_enabled", True))
    pcow_entry_psia = float(_ss("pcow_entry_psia", 15.0))
    pcog_entry_psia = float(_ss("pcog_entry_psia", 8.0))
    pc_lambda_w = float(_ss("pc_lambda_w", 2.0))
    pc_lambda_g = float(_ss("pc_lambda_g", 2.0))
    hysteresis_enabled = bool(_ss("hysteresis_enabled", True))
    hysteresis_reversal_tolerance = float(_ss("hysteresis_reversal_tolerance", 0.01))
    hysteresis_gas_trapping_strength = float(_ss("hysteresis_gas_trapping_strength", 0.60))
    hysteresis_imbibition_krg_reduction = float(_ss("hysteresis_imbibition_krg_reduction", 0.75))
    hysteresis_imbibition_kro_reduction = float(_ss("hysteresis_imbibition_kro_reduction", 0.15))
    transport_enabled = bool(_ss("transport_enabled", True))
    transport_phase_split_advection = bool(_ss("transport_phase_split_advection", True))
    transport_dispersivity_ft = float(_ss("transport_dispersivity_ft", 15.0))
    transport_molecular_diffusion_ft2_day = float(_ss("transport_molecular_diffusion_ft2_day", 0.15))
    transport_max_dispersive_fraction = float(_ss("transport_max_dispersive_fraction", 0.35))

    history_match_enabled = bool(_ss("history_match_enabled", False))
    history_match_file = None
    history_match_weights = {}
    history_match_observed_df = None
    history_match_error = None
    enable_scenario_comparison = bool(_ss("enable_scenario_comparison", False))
    comparison_scenarios = _ss("comparison_scenarios", ["Natural Depletion", "Gas Cycling", "Lean Gas Injection"])
    enable_sensitivity = bool(_ss("enable_sensitivity", False))
    sensitivity_param_label = _ss("sensitivity_param_label", "Permeability")
    sensitivity_low_pct = float(_ss("sensitivity_low_pct", -20.0))
    sensitivity_high_pct = float(_ss("sensitivity_high_pct", 20.0))
    enable_monte_carlo = bool(_ss("enable_monte_carlo", False))
    monte_carlo_runs = int(_ss("monte_carlo_runs", 50))
    monte_carlo_low_pct = float(_ss("monte_carlo_low_pct", -15.0))
    monte_carlo_high_pct = float(_ss("monte_carlo_high_pct", 15.0))
    monte_carlo_seed = int(_ss("monte_carlo_seed", 42))

    with tab_case_setup:
        st.subheader("Case Setup")
        scenario_label = st.selectbox("Scenario", ["Natural Depletion", "Gas Cycling", "Lean Gas Injection"], index=0, key="scenario_label")
        scenario_map = {
            "Natural Depletion": "natural_depletion",
            "Gas Cycling": "gas_cycling",
            "Lean Gas Injection": "lean_gas_injection",
        }
        scenario_name = scenario_map[scenario_label]
        start_date_text = st.text_input("Start date (dd/mm/yyyy)", value="01/01/2025", key="start_date_text")
        end_date_text = st.text_input("End date (dd/mm/yyyy)", value="29/08/2025", key="end_date_text")
        try:
            t_end_days = days_between_dates(start_date_text, end_date_text)
            st.caption(f"Simulation length = {t_end_days:.0f} days")
        except ValueError as e:
            t_end_days = None
            st.error(str(e))
            input_validation_errors.append(str(e))
        dt_days = st.number_input("Requested time step (days)", min_value=0.005, max_value=1000.0, value=0.05, step=0.01, key="dt_days")
        st.caption("Adaptive timestep retry is enabled.")

    with tab_reservoir_grid:
        st.subheader("Reservoir / Grid")
        nx = st.number_input("Number of grid cells", min_value=10, max_value=200, value=40, step=1, key="nx")
        length_ft = st.number_input("Reservoir length (ft)", min_value=500.0, max_value=20000.0, value=4000.0, step=100.0, key="length_ft")
        radius_ft = st.number_input("Equivalent drainage radius (ft)", min_value=1.0, max_value=100000.0, value=100.0, step=10.0, key="radius_ft")
        thickness_ft = st.number_input("Reservoir thickness (ft)", min_value=5.0, max_value=500.0, value=50.0, step=5.0, key="thickness_ft")
        equivalent_width_ft = 2.0 * radius_ft
        area_ft2 = thickness_ft * equivalent_width_ft
        refined_cells = st.number_input("Refined cells near well", min_value=2, max_value=max(int(nx) - 1, 2), value=min(10, int(nx) - 1), step=1, key="refined_cells")
        min_dx_ft = st.number_input("Minimum near-well cell size (ft)", min_value=1.0, max_value=500.0, value=12.0, step=1.0, key="min_dx_ft")
        growth = st.number_input("LGR growth factor", min_value=1.05, max_value=3.0, value=1.45, step=0.05, key="growth")
        st.caption(f"Equivalent model width = {equivalent_width_ft:,.2f} ft")
        st.caption(f"Equivalent cross-sectional area = {area_ft2:,.2f} ft²")
        st.info(
            "Grid: 1D nonuniform Cartesian with near-well refinement. "
            "Radius is interpreted as an equivalent drainage half-width for this Cartesian model, "
            "so the simulator uses width = 2 × radius and cross-sectional area = thickness × width. "
            "Producer is in the last cell; injector, when active, is in the first cell."
        )

    with tab_components_composition:
        st.subheader("Components / Composition")
        fluid_components = build_example_fluid().components
        component_names = [c.name for c in fluid_components]

        st.subheader("Initial Hydrocarbon Composition")
        default_initial = np.array([0.005, 0.020, 0.010, 0.780, 0.070, 0.040, 0.015, 0.015, 0.010, 0.010, 0.015, 0.010], dtype=float)
        try:
            z_init, z_input_sum, z_unit_mode = composition_editor(
                label="Initial hydrocarbon composition",
                component_names=component_names,
                default_values=default_initial,
                key_prefix="initial_composition",
            )
        except ValueError as e:
            st.error(str(e))
            input_validation_errors.append(str(e))
            z_init = default_initial / np.sum(default_initial)
            z_input_sum = float(np.sum(default_initial))
            z_unit_mode = "fraction"
        st.caption(f"Input total ({z_unit_mode.lower()} basis): {z_input_sum:.6f} | Normalized sum: {np.sum(z_init):.4f}")

        injected_gas_composition = z_init.copy()
        if scenario_name in ("gas_cycling", "lean_gas_injection"):
            st.subheader("Injected Gas Composition")
            default_injected = np.array([0.0, 0.02, 0.01, 0.94, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            try:
                injected_gas_composition, inj_input_sum, inj_unit_mode = composition_editor(
                    label="Injected gas composition",
                    component_names=component_names,
                    default_values=default_injected,
                    key_prefix="injected_gas_composition",
                )
            except ValueError as e:
                st.error(str(e))
                input_validation_errors.append(str(e))
                injected_gas_composition = default_injected / np.sum(default_injected)
                inj_input_sum = float(np.sum(default_injected))
                inj_unit_mode = "fraction"
            st.caption(f"Input total ({inj_unit_mode.lower()} basis): {inj_input_sum:.6f} | Normalized sum: {np.sum(injected_gas_composition):.4f}")
        lean_gas_composition = injected_gas_composition.copy()

    with tab_pvt_separator:
        st.subheader("PVT / Separator")
        temperature_F = st.number_input("Reservoir temperature (°F)", min_value=-60.0, max_value=740.0, value=220.0, step=10.0, key="temperature_F")
        temperature_R = fahrenheit_to_rankine(temperature_F)
        dew_point_psia = st.number_input("Dew-point pressure (psia)", min_value=100.0, max_value=20000.0, value=4200.0, step=100.0, key="dew_point_psia")
        separator_stages = int(st.selectbox("Separator train", options=[1, 2, 3, 4], index=3, format_func=lambda x: {1: "1-stage separator", 2: "2-stage separator", 3: "3-stage separator", 4: "Stage 1 + Stage 2 + Stage 3 + Stock Tank"}[x], key="separator_stages"))
        separator_pressure_psia = st.number_input("Stage 1 separator pressure (psia)", min_value=14.7, max_value=5000.0, value=300.0, step=25.0, key="separator_pressure_psia")
        separator_temperature_F = st.number_input("Stage 1 separator temperature (°F)", min_value=-10.0, max_value=340.0, value=60.0, step=5.0, key="separator_temperature_F")
        separator_temperature_R = fahrenheit_to_rankine(separator_temperature_F)
        separator_second_stage_pressure_psia = st.number_input("Stage 2 separator pressure (psia)", min_value=14.7, max_value=5000.0, value=100.0, step=10.0, disabled=separator_stages < 2, key="separator_second_stage_pressure_psia")
        separator_second_stage_temperature_F = st.number_input("Stage 2 separator temperature (°F)", min_value=-10.0, max_value=340.0, value=80.0, step=5.0, disabled=separator_stages < 2, key="separator_second_stage_temperature_F")
        separator_second_stage_temperature_R = fahrenheit_to_rankine(separator_second_stage_temperature_F)
        separator_third_stage_pressure_psia = st.number_input("Stage 3 separator pressure (psia)", min_value=14.7, max_value=5000.0, value=50.0, step=5.0, disabled=separator_stages < 3, key="separator_third_stage_pressure_psia")
        separator_third_stage_temperature_F = st.number_input("Stage 3 separator temperature (°F)", min_value=-10.0, max_value=340.0, value=70.0, step=5.0, disabled=separator_stages < 3, key="separator_third_stage_temperature_F")
        separator_third_stage_temperature_R = fahrenheit_to_rankine(separator_third_stage_temperature_F)
        stock_tank_pressure_psia = st.number_input("Stock tank pressure (psia)", min_value=14.7, max_value=100.0, value=14.7, step=1.0, disabled=separator_stages < 4, key="stock_tank_pressure_psia")
        stock_tank_temperature_F = st.number_input("Stock tank temperature (°F)", min_value=-10.0, max_value=340.0, value=60.0, step=5.0, disabled=separator_stages < 4, key="stock_tank_temperature_F")
        stock_tank_temperature_R = fahrenheit_to_rankine(stock_tank_temperature_F)
        gas_specific_gravity = st.number_input("Gas specific gravity (air = 1)", min_value=0.1, max_value=3.0, value=0.65, step=0.01, format="%.2f", key="gas_specific_gravity")
        condensate_api_gravity = st.number_input("Condensate API gravity", min_value=1.0, max_value=100.0, value=50.0, step=1.0, format="%.1f", key="condensate_api_gravity")

        st.subheader("Flash PVT Table")
        st.caption("Paste pressure-dependent PVT data below. Pressure should be in psig and gas density in lb/ft³ before pasting. If Reservoir CGR or Vapour CGR is provided, the simulator will use that table to report condensate rate versus pressure.")
        pvt_editor_key = "flash_pvt_table_editor"
        pvt_editor_seed_key = "flash_pvt_table_editor_seed"
        if pvt_editor_seed_key not in st.session_state:
            st.session_state[pvt_editor_seed_key] = pd.DataFrame([
                {"Pressure": "", "Gas FVF": "", "Gas Viscosity": "", "Gas Z Factor": "", "Gas Density": "",
                 "Oil FVF": "", "Oil Viscosity": "", "Solution GOR": "", "Vapour CGR": "",
                 "Reservoir CGR": "", "Dew Point": ""}
                for _ in range(8)
            ])
        pvt_df = st.data_editor(
            st.session_state[pvt_editor_seed_key],
            key=pvt_editor_key,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )
        try:
            pressure_series = pd.to_numeric(pvt_df.get("Pressure", pd.Series(dtype=float)), errors="coerce")
            has_pvt_rows = bool(pressure_series.notna().any())
        except Exception:
            has_pvt_rows = False

        if has_pvt_rows:
            try:
                pvt_table = load_pvt_table_from_dataframe(pvt_df)
                st.success(f"Loaded Flash PVT table with {len(pvt_table.pressure_psia)} pressure rows.")
                preview_cols = [c for c in [
                    'pressure_psia', 'dew_point_psia', 'reservoir_cgr_stb_mmscf', 'vaporized_cgr_stb_mmscf',
                    'z_factor', 'gas_fvf_ft3_scf', 'gas_viscosity_cp', 'oil_fvf_rb_stb', 'oil_viscosity_cp',
                    'gas_oil_ratio_scf_stb', 'gas_density_lbft3'
                ] if getattr(pvt_table, c, None) is not None]
                preview_df = pd.DataFrame({c: getattr(pvt_table, c) for c in preview_cols}).head(10)
                st.dataframe(preview_df, use_container_width=True)
            except Exception as e:
                msg = f"Failed to read Flash PVT table: {e}"
                st.error(msg)
                input_validation_errors.append(msg)
        else:
            st.caption("Leave the table blank to run without a Flash PVT table.")

    with tab_rock_rockfluid:
        st.subheader("Rock / Rock-Fluid")
        porosity = st.number_input("Porosity", min_value=0.01, max_value=0.50, value=0.18, step=0.01, format="%.2f", key="porosity")
        permeability_md = st.number_input("Permeability (mD)", min_value=0.01, max_value=10000.0, value=8.0, step=1.0, key="permeability_md")
        Sgc = st.number_input("Critical gas saturation, Sgc", 0.0, 0.95, 0.05, 0.01, format="%.2f", key="Sgc")
        Swc = st.number_input("Connate water saturation, Swc", 0.0, 0.95, 0.20, 0.01, format="%.2f", key="Swc")
        Sorw = st.number_input("Residual condensate saturation to water, Sorw", 0.0, 0.95, 0.20, 0.01, format="%.2f", key="Sorw")
        Sorg = st.number_input("Residual condensate saturation to gas, Sorg", 0.0, 0.95, 0.05, 0.01, format="%.2f", key="Sorg")
        krg0 = st.number_input("Gas endpoint krg0", 0.0, 2.0, 1.0, 0.05, format="%.2f", key="krg0")
        kro0 = st.number_input("Condensate endpoint kro0", 0.0, 2.0, 1.0, 0.05, format="%.2f", key="kro0")
        krw0 = st.number_input("Water endpoint krw0", 0.0, 2.0, 0.4, 0.05, format="%.2f", key="krw0")
        ng = st.number_input("Gas Corey exponent, ng", 0.1, 10.0, 2.0, 0.1, format="%.1f", key="ng")
        no = st.number_input("Condensate Corey exponent, no", 0.1, 10.0, 2.0, 0.1, format="%.1f", key="no")
        nw = st.number_input("Water Corey exponent, nw", 0.1, 10.0, 2.0, 0.1, format="%.1f", key="nw")

        relperm_input_valid = True
        if Sgc + Swc >= 1.0 or Swc + Sorw >= 1.0 or Swc + Sorg >= 1.0:
            relperm_input_valid = False
            st.error("Invalid relative permeability inputs. Check saturation constraints.")

    with tab_initial_conditions:
        st.subheader("Initial Conditions")
        p_init_psia = st.number_input("Initial reservoir pressure (psia)", min_value=100.0, max_value=20000.0, value=5600.0, step=100.0, key="p_init_psia")
        sw_init = st.number_input("Initial water saturation, Swi", min_value=0.0, max_value=0.95, value=0.20, step=0.01, format="%.2f", key="sw_init")
        st.caption("Initial hydrocarbon composition is taken from the Components / Composition section.")

    with tab_producer_well:
        st.subheader("Producer Well")
        well_control_label = st.selectbox("Producer control mode", ["Fixed BHP", "Fixed Drawdown", "Fixed Gas Rate", "Fixed THP"], index=0, key="well_control_label")
        control_map = {
            "Fixed BHP": "bhp",
            "Fixed Drawdown": "drawdown",
            "Fixed Gas Rate": "gas_rate",
            "Fixed THP": "thp",
        }
        well_control_mode = control_map[well_control_label]
        bhp_psia = st.number_input("Bottom-hole pressure target (psia)", min_value=100.0, max_value=20000.0, value=3000.0, step=100.0, disabled=(well_control_mode != "bhp"), key="bhp_psia")
        drawdown_psia = st.number_input("Pressure drawdown target (psi)", min_value=1.0, max_value=10000.0, value=200.0, step=10.0, disabled=(well_control_mode != "drawdown"), key="drawdown_psia")
        target_gas_rate_mmscf_day = st.number_input("Target gas rate (MMscf/day)", min_value=0.0, max_value=5000.0, value=5.0, step=0.5, disabled=(well_control_mode != "gas_rate"), key="target_gas_rate_mmscf_day")
        thp_psia = st.number_input("Tubing head pressure target (psia)", min_value=14.7, max_value=10000.0, value=500.0, step=25.0, disabled=(well_control_mode != "thp"), key="thp_psia")
        min_bhp_psia = st.number_input("Minimum flowing BHP allowed (psia)", min_value=50.0, max_value=5000.0, value=50.0, step=25.0, disabled=(well_control_mode != "drawdown"), key="min_bhp_psia")
        productivity_index = st.number_input("Productivity index multiplier", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="productivity_index")
        rw_ft = st.number_input("Wellbore radius (ft)", min_value=0.05, max_value=5.0, value=0.35, step=0.05, key="rw_ft")
        skin = st.number_input("Skin factor", min_value=-10.0, max_value=50.0, value=0.0, step=0.5, key="skin")
        tvd_ft = st.number_input("True vertical depth, TVD (ft)", min_value=100.0, max_value=30000.0, value=8000.0, step=100.0, key="tvd_ft")
        tubing_id_in = st.number_input("Tubing ID (in)", min_value=1.0, max_value=6.0, value=2.441, step=0.1, key="tubing_id_in")
        wellhead_temperature_F = st.number_input("Wellhead temperature (°F)", min_value=-60.0, max_value=340.0, value=60.0, step=5.0, key="wellhead_temperature_F")
        wellhead_temperature_R = fahrenheit_to_rankine(wellhead_temperature_F)
        tubing_model_label = st.selectbox("Tubing hydraulics model", ["Segmented mechanistic multiphase", "Simple coefficient"], index=0, key="tubing_model_label")
        tubing_model = "mechanistic" if tubing_model_label == "Segmented mechanistic multiphase" else "simple"
        tubing_segments = st.number_input("Tubing depth segments", min_value=5, max_value=200, value=25, step=5, disabled=(tubing_model != "mechanistic"), key="tubing_segments")
        thp_friction_coeff = st.number_input("THP friction coefficient", min_value=0.0, max_value=1.0, value=0.02, step=0.005, help="Used only under Simple coefficient mode. The segmented mechanistic model now computes distributed hydrostatic, friction, and acceleration losses directly.", key="thp_friction_coeff")
        tubing_roughness_in = st.number_input("Tubing roughness (in)", min_value=0.0, max_value=0.05, value=0.0006, step=0.0002, format="%.4f", disabled=(tubing_model != "mechanistic"), key="tubing_roughness_in")
        tubing_calibration_factor = st.number_input("Tubing calibration factor", min_value=0.1, max_value=10.0, value=1.0, step=0.1, disabled=(tubing_model != "mechanistic"), key="tubing_calibration_factor")

    with tab_injector_well:
        st.subheader("Injector Well")
        injection_rate_value = st.number_input("Gas injection rate", min_value=0.0, max_value=1.0e6, value=5.0, step=0.5, disabled=(scenario_name == "natural_depletion"), key="injection_rate_value")
        injection_rate_unit = st.selectbox("Gas injection rate unit", ["Mscf/day", "MMscf/day"], index=1, key="injection_rate_unit")
        injection_control_label = st.selectbox("Injection mode", ["Simple schedule-driven", "Enhanced pressure-controlled"], index=1, disabled=(scenario_name == "natural_depletion"), key="injection_control_label")
        injection_control_mode = "enhanced" if injection_control_label == "Enhanced pressure-controlled" else "simple"
        injection_pressure_psia = st.number_input("Injection pressure (psia)", min_value=14.7, max_value=20000.0, value=7000.0, step=100.0, disabled=(scenario_name == "natural_depletion"), key="injection_pressure_psia")
        max_injection_bhp_psia = st.number_input("Maximum bottom-hole injection pressure (psia)", min_value=14.7, max_value=25000.0, value=9000.0, step=100.0, disabled=(scenario_name == "natural_depletion" or injection_control_mode != "enhanced"), key="max_injection_bhp_psia")
        injectivity_index_mmscf_day_psi = st.number_input("Injectivity index (MMscf/day/psi)", min_value=0.0, max_value=1.0e4, value=gas_lbmol_day_to_mmscf_day(500.0), step=0.01, format="%.4f", disabled=(scenario_name == "natural_depletion" or injection_control_mode != "enhanced"), key="injectivity_index_mmscf_day_psi")
        injectivity_index_lbmol_day_psi = injectivity_mmscf_day_psi_to_lbmol_day_psi(injectivity_index_mmscf_day_psi)
        injection_start_date_text = start_date_text
        injection_end_date_text = end_date_text
        if scenario_name in ("gas_cycling", "lean_gas_injection"):
            injection_start_date_text = st.text_input("Injection start date (dd/mm/yyyy)", value=start_date_text, key="injection_start_date_text")
            injection_end_date_text = st.text_input("Injection end date (dd/mm/yyyy)", value=end_date_text, key="injection_end_date_text")
            try:
                inj_start_day = days_between_dates(start_date_text, injection_start_date_text, allow_same_day=True)
                inj_end_day = days_between_dates(start_date_text, injection_end_date_text, allow_same_day=True)
                if inj_end_day < inj_start_day:
                    msg = "Injection end date must be on or after the injection start date."
                    st.error(msg)
                    input_validation_errors.append(msg)
                else:
                    st.caption(f"Injection window = day {inj_start_day:.0f} to day {inj_end_day:.0f} of the simulation")
            except ValueError as e:
                st.error(str(e))
                input_validation_errors.append(str(e))
        else:
            st.caption("Injector settings are only active for gas-cycling and lean-gas-injection cases.")

    with tab_aquifer_boundaries:
        st.subheader("Aquifer / Boundaries")
        left_boundary_label = st.selectbox("Left boundary", ["Closed", "Constant pressure"], index=0, key="left_boundary_label")
        left_boundary_mode = "closed" if left_boundary_label == "Closed" else "constant_pressure"
        left_boundary_pressure_psia = st.number_input("Left boundary pressure (psia)", min_value=14.7, max_value=20000.0, value=5600.0, step=100.0, disabled=(left_boundary_mode != "constant_pressure"), key="left_boundary_pressure_psia")
        left_boundary_transmissibility_multiplier = st.number_input("Left boundary transmissibility multiplier", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="left_boundary_transmissibility_multiplier")
        right_boundary_label = st.selectbox("Right boundary", ["Closed", "Constant pressure"], index=0, key="right_boundary_label")
        right_boundary_mode = "closed" if right_boundary_label == "Closed" else "constant_pressure"
        right_boundary_pressure_psia = st.number_input("Right boundary pressure (psia)", min_value=14.7, max_value=20000.0, value=3000.0, step=100.0, disabled=(right_boundary_mode != "constant_pressure"), key="right_boundary_pressure_psia")
        right_boundary_transmissibility_multiplier = st.number_input("Right boundary transmissibility multiplier", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="right_boundary_transmissibility_multiplier")
        aquifer_enabled = st.checkbox("Enable analytic aquifer", value=False, key="aquifer_enabled")
        aquifer_side = st.selectbox("Aquifer side", ["left", "right"], index=0, disabled=(not aquifer_enabled), key="aquifer_side")
        aquifer_initial_pressure_psia = st.number_input("Aquifer initial pressure (psia)", min_value=14.7, max_value=20000.0, value=5600.0, step=100.0, disabled=(not aquifer_enabled), key="aquifer_initial_pressure_psia")
        _aq_pi_default_stb_day_psi = 500.0 / water_lbmol_per_stb()
        _aq_cap_default_mstb_psi = 5.0e5 / water_lbmol_per_mstb()
        aquifer_productivity_index_stb_day_psi = st.number_input("Aquifer productivity index (STB/day/psi)", min_value=0.0, max_value=1.0e7, value=float(_aq_pi_default_stb_day_psi), step=1.0, format="%.2f", disabled=(not aquifer_enabled), key="aquifer_productivity_index_stb_day_psi")
        aquifer_total_capacity_mstb_per_psi = st.number_input("Aquifer total capacity (MSTB/psi)", min_value=0.001, max_value=1.0e7, value=float(_aq_cap_default_mstb_psi), step=1.0, format="%.2f", disabled=(not aquifer_enabled), key="aquifer_total_capacity_mstb_per_psi")
        aquifer_water_influx_fraction = st.number_input("Aquifer influx fraction", min_value=0.0, max_value=1.0, value=1.0, step=0.05, format="%.2f", disabled=(not aquifer_enabled), key="aquifer_water_influx_fraction")
        aquifer_allow_backflow = st.checkbox("Allow aquifer backflow", value=False, disabled=(not aquifer_enabled), key="aquifer_allow_backflow")
        st.caption("Constant-pressure boundaries add external pressure support at the grid edge. The analytic aquifer adds pressure support and water influx using a simple productivity-capacity model.")

    with tab_advanced_physics:
        st.subheader("Advanced Physics")
        capillary_enabled = st.checkbox("Enable capillary pressure", value=True, key="capillary_enabled")
        pcow_entry_psia = st.number_input("Pcow entry pressure (psia)", min_value=0.0, max_value=500.0, value=15.0, step=1.0, key="pcow_entry_psia")
        pcog_entry_psia = st.number_input("Pcog entry pressure (psia)", min_value=0.0, max_value=500.0, value=8.0, step=1.0, key="pcog_entry_psia")
        pc_lambda_w = st.number_input("Pcow exponent", min_value=0.1, max_value=10.0, value=2.0, step=0.1, format="%.1f", key="pc_lambda_w")
        pc_lambda_g = st.number_input("Pcog exponent", min_value=0.1, max_value=10.0, value=2.0, step=0.1, format="%.1f", key="pc_lambda_g")
        st.caption(f"Using Swir = {Swc:.2f}, Sorw = {Sorw:.2f}, Sgc = {Sgc:.2f}, Sorg = {Sorg:.2f}")
        hysteresis_enabled = st.checkbox("Enable gas-relperm hysteresis", value=True, key="hysteresis_enabled")
        hysteresis_reversal_tolerance = st.number_input("Reversal tolerance in Sg", min_value=0.0, max_value=0.30, value=0.01, step=0.005, format="%.3f", key="hysteresis_reversal_tolerance")
        hysteresis_gas_trapping_strength = st.number_input("Gas trapping strength", min_value=0.0, max_value=1.0, value=0.60, step=0.05, format="%.2f", key="hysteresis_gas_trapping_strength")
        hysteresis_imbibition_krg_reduction = st.number_input("Imbibition krg reduction multiplier", min_value=0.0, max_value=1.5, value=0.75, step=0.05, format="%.2f", key="hysteresis_imbibition_krg_reduction")
        hysteresis_imbibition_kro_reduction = st.number_input("Imbibition kro reduction multiplier", min_value=0.0, max_value=1.0, value=0.15, step=0.05, format="%.2f", key="hysteresis_imbibition_kro_reduction")
        transport_enabled = st.checkbox("Enable dispersive transport", value=True, key="transport_enabled")
        transport_phase_split_advection = st.checkbox("Use phase-split hydrocarbon advection", value=True, key="transport_phase_split_advection")
        transport_dispersivity_ft = st.number_input("Longitudinal dispersivity (ft)", min_value=0.0, max_value=5000.0, value=15.0, step=1.0, key="transport_dispersivity_ft")
        transport_molecular_diffusion_ft2_day = st.number_input("Molecular diffusion (ft²/day)", min_value=0.0, max_value=100.0, value=0.15, step=0.05, format="%.3f", key="transport_molecular_diffusion_ft2_day")
        transport_max_dispersive_fraction = st.number_input("Max dispersive/advective flux ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.05, format="%.2f", key="transport_max_dispersive_fraction")


    with tab_analysis_tools:
        st.subheader("Analysis Tools")
        st.write("Producer supports fixed BHP, fixed drawdown, fixed gas rate, and simplified fixed THP control.")
        st.write("Gas and condensate rates are reported through a configurable 1-stage, 2-stage, 3-stage, or Stage 1/Stage 2/Stage 3/stock-tank separator train. Hydrocarbon transport now supports phase-split advection and optional dispersion.")

        st.subheader("History Matching / Calibration")
        history_match_enabled = st.checkbox("Enable history-matching diagnostics", value=False, key="history_match_enabled")
        history_match_file = None
        history_match_weights = {}
        history_match_observed_df = None
        history_match_error = None
        if history_match_enabled:
            st.caption("Upload observed field history as CSV. Include either Time/Days or Date, plus one or more observed columns such as Avg Pressure, Well Pressure, Gas Rate, Condensate Rate, Water Rate, Cumulative Gas, Cumulative Condensate, or Cumulative Water.")
            history_match_file = st.file_uploader("Observed history CSV", type=["csv"], key="history_match_csv")
            hm1, hm2, hm3, hm4 = st.columns(4)
            with hm1:
                history_match_weights['avg_pressure_psia'] = st.number_input("Weight: Avg Pressure", min_value=0.0, max_value=100.0, value=1.0, step=0.25, disabled=not history_match_enabled, key="history_match_weights_avg_pressure_psia")
            with hm2:
                history_match_weights['well_pressure_psia'] = st.number_input("Weight: Well Pressure", min_value=0.0, max_value=100.0, value=1.0, step=0.25, disabled=not history_match_enabled, key="history_match_weights_well_pressure_psia")
            with hm3:
                history_match_weights['gas_rate_mmscf_day'] = st.number_input("Weight: Gas Rate", min_value=0.0, max_value=100.0, value=1.0, step=0.25, disabled=not history_match_enabled, key="history_match_weights_gas_rate_mmscf_day")
            with hm4:
                history_match_weights['condensate_rate_stb_day'] = st.number_input("Weight: Condensate Rate", min_value=0.0, max_value=100.0, value=1.0, step=0.25, disabled=not history_match_enabled, key="history_match_weights_condensate_rate_stb_day")
            hm5, hm6, hm7, hm8 = st.columns(4)
            with hm5:
                history_match_weights['water_rate_stb_day'] = st.number_input("Weight: Water Rate", min_value=0.0, max_value=100.0, value=1.0, step=0.25, disabled=not history_match_enabled, key="history_match_weights_water_rate_stb_day")
            with hm6:
                history_match_weights['cum_gas_bscf'] = st.number_input("Weight: Cum Gas", min_value=0.0, max_value=100.0, value=0.75, step=0.25, disabled=not history_match_enabled, key="history_match_weights_cum_gas_bscf")
            with hm7:
                history_match_weights['cum_condensate_mstb'] = st.number_input("Weight: Cum Condensate", min_value=0.0, max_value=100.0, value=0.75, step=0.25, disabled=not history_match_enabled, key="history_match_weights_cum_condensate_mstb")
            with hm8:
                history_match_weights['cum_water_mstb'] = st.number_input("Weight: Cum Water", min_value=0.0, max_value=100.0, value=0.75, step=0.25, disabled=not history_match_enabled, key="history_match_weights_cum_water_mstb")
            if history_match_file is not None:
                try:
                    raw_match_df = pd.read_csv(history_match_file)
                    history_match_observed_df = load_history_match_dataframe(raw_match_df, start_date_text=start_date_text)
                    st.success(f"Loaded history-match data with {len(history_match_observed_df)} rows.")
                    st.dataframe(history_match_observed_df.head(10), use_container_width=True)
                except Exception as e:
                    history_match_error = f"Failed to read history-match CSV: {e}"
                    st.error(history_match_error)
                    input_validation_errors.append(history_match_error)

        st.subheader("Scenario Comparison")
        enable_scenario_comparison = st.checkbox("Enable side-by-side scenario comparison", value=False, key="enable_scenario_comparison")
        comparison_options = ["Natural Depletion", "Gas Cycling", "Lean Gas Injection"]
        comparison_labels = st.multiselect(
            "Scenarios to compare",
            comparison_options,
            default=comparison_options,
            disabled=not enable_scenario_comparison,
        )
        st.caption("Uses the same rock, well, transport, separator, and schedule settings for each selected scenario. Natural depletion is run with zero injection automatically.")

        st.subheader("One-Way Sensitivity Analysis")
        enable_sensitivity = st.checkbox("Enable one-way sensitivity analysis", value=False, key="enable_sensitivity")
        sensitivity_parameter_label = st.selectbox(
            "Sensitivity parameter",
            [
                "Permeability (md)",
                "Porosity",
                "Initial Reservoir Pressure (psia)",
                "Dew Point Pressure (psia)",
                "Producer BHP (psia)",
                "Gas Injection Rate",
                "Transport Dispersivity (ft)",
            ],
            index=0,
            disabled=not enable_sensitivity,
        )
        sens1, sens2, sens3 = st.columns(3)
        with sens1:
            sensitivity_low_pct = st.number_input("Low case change (%)", min_value=-95.0, max_value=0.0, value=-20.0, step=5.0, disabled=not enable_sensitivity, key="sensitivity_low_pct")
        with sens2:
            sensitivity_base_pct = st.number_input("Base case change (%)", min_value=-95.0, max_value=500.0, value=0.0, step=5.0, disabled=not enable_sensitivity, key="sensitivity_base_pct")
        with sens3:
            sensitivity_high_pct = st.number_input("High case change (%)", min_value=-95.0, max_value=500.0, value=20.0, step=5.0, disabled=not enable_sensitivity, key="sensitivity_high_pct")
        st.caption("Runs low, base, and high cases around the current input set for the selected parameter.")

        st.subheader("Uncertainty / Monte Carlo Analysis")
        enable_monte_carlo = st.checkbox("Enable Monte Carlo uncertainty analysis", value=False, key="enable_monte_carlo")
        monte_carlo_parameter_options = [
            "Permeability (md)",
            "Porosity",
            "Initial Reservoir Pressure (psia)",
            "Dew Point Pressure (psia)",
            "Producer BHP (psia)",
            "Gas Injection Rate",
            "Transport Dispersivity (ft)",
        ]
        monte_carlo_selected_labels = st.multiselect(
            "Uncertain parameters",
            monte_carlo_parameter_options,
            default=["Permeability (md)", "Dew Point Pressure (psia)", "Gas Injection Rate"],
            disabled=not enable_monte_carlo,
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            monte_carlo_runs = int(st.number_input("Monte Carlo runs", min_value=10, max_value=500, value=50, step=10, disabled=not enable_monte_carlo, key="monte_carlo_runs"))
        with mc2:
            monte_carlo_low_pct = st.number_input("Low bound change (%)", min_value=-95.0, max_value=0.0, value=-15.0, step=5.0, disabled=not enable_monte_carlo, key="monte_carlo_low_pct")
        with mc3:
            monte_carlo_high_pct = st.number_input("High bound change (%)", min_value=0.0, max_value=500.0, value=15.0, step=5.0, disabled=not enable_monte_carlo, key="monte_carlo_high_pct")
        with mc4:
            monte_carlo_seed = int(st.number_input("Random seed", min_value=0, max_value=1000000, value=42, step=1, disabled=not enable_monte_carlo, key="monte_carlo_seed"))
        st.caption("Samples selected inputs independently using a triangular distribution bounded by the low/high percentage changes around the current base values, then reruns the simulator for each realization.")

    # Refresh shadow persistence after widgets render. Update only the shadow
    # keys here; writing back to a widget-owned key after instantiation causes
    # StreamlitAPIException.
    for _key in persistent_defaults:
        if _key in st.session_state:
            st.session_state[f"persist__{_key}"] = st.session_state[_key]

    with tab_results:
        st.subheader("Results")
        st.caption("Simulation results, plots, tables, and downloads appear on this page after you click Run Simulation.")

    st.sidebar.divider()
    run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Select a section in the sidebar to configure inputs, then click 'Run Simulation'.")
        return

    if not relperm_input_valid:
        input_validation_errors.append("Invalid relative permeability inputs. Check saturation constraints.")

    if input_validation_errors:
        st.error("Please fix the input issues shown in the tabs before running the simulation.")
        return

    run_kwargs = dict(
        start_date_text=start_date_text,
        scenario_name=scenario_name,
        injection_rate_value=float(injection_rate_value),
        injection_rate_unit=injection_rate_unit,
        injection_start_date_text=injection_start_date_text,
        injection_end_date_text=injection_end_date_text,
        injection_control_mode=injection_control_mode,
        injection_pressure_psia=injection_pressure_psia,
        max_injection_bhp_psia=max_injection_bhp_psia,
        injectivity_index_lbmol_day_psi=injectivity_index_lbmol_day_psi,
        injected_gas_composition=injected_gas_composition,
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
        tubing_roughness_in=float(tubing_roughness_in),
        tubing_calibration_factor=float(tubing_calibration_factor),
        tubing_model=str(tubing_model),
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
        separator_stages=int(separator_stages),
        separator_second_stage_pressure_psia=float(separator_second_stage_pressure_psia),
        separator_second_stage_temperature_R=float(separator_second_stage_temperature_R),
        separator_third_stage_pressure_psia=float(separator_third_stage_pressure_psia),
        separator_third_stage_temperature_R=float(separator_third_stage_temperature_R),
        stock_tank_pressure_psia=float(stock_tank_pressure_psia),
        stock_tank_temperature_R=float(stock_tank_temperature_R),
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
        capillary_enabled=bool(capillary_enabled),
        pcow_entry_psia=float(pcow_entry_psia),
        pcog_entry_psia=float(pcog_entry_psia),
        pc_lambda_w=float(pc_lambda_w),
        pc_lambda_g=float(pc_lambda_g),
        hysteresis_enabled=bool(hysteresis_enabled),
        hysteresis_reversal_tolerance=float(hysteresis_reversal_tolerance),
        hysteresis_gas_trapping_strength=float(hysteresis_gas_trapping_strength),
        hysteresis_imbibition_krg_reduction=float(hysteresis_imbibition_krg_reduction),
        hysteresis_imbibition_kro_reduction=float(hysteresis_imbibition_kro_reduction),
        transport_enabled=bool(transport_enabled),
        transport_phase_split_advection=bool(transport_phase_split_advection),
        transport_dispersivity_ft=float(transport_dispersivity_ft),
        transport_molecular_diffusion_ft2_day=float(transport_molecular_diffusion_ft2_day),
        transport_max_dispersive_fraction=float(transport_max_dispersive_fraction),
        left_boundary_mode=str(left_boundary_mode),
        right_boundary_mode=str(right_boundary_mode),
        left_boundary_pressure_psia=float(left_boundary_pressure_psia),
        right_boundary_pressure_psia=float(right_boundary_pressure_psia),
        left_boundary_transmissibility_multiplier=float(left_boundary_transmissibility_multiplier),
        right_boundary_transmissibility_multiplier=float(right_boundary_transmissibility_multiplier),
        aquifer_enabled=bool(aquifer_enabled),
        aquifer_side=str(aquifer_side),
        aquifer_initial_pressure_psia=float(aquifer_initial_pressure_psia),
        aquifer_productivity_index_stb_day_psi=float(aquifer_productivity_index_stb_day_psi),
        aquifer_total_capacity_mstb_per_psi=float(aquifer_total_capacity_mstb_per_psi),
        aquifer_water_influx_fraction=float(aquifer_water_influx_fraction),
        aquifer_allow_backflow=bool(aquifer_allow_backflow),
        gas_specific_gravity=float(gas_specific_gravity),
        condensate_api_gravity=float(condensate_api_gravity),
        pvt_table=pvt_table,
    )

    with st.spinner("Running compositional simulation..."):
        try:
            result = run_example(**run_kwargs)
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

    # Plot the main history charts using annual snapshots, but always include
    # the simulation start point so the first calendar year is visible on the
    # x-axis even when the history table keeps year-end snapshots.
    full_history_df = history_dataframe(history, None).sort_values("time_days").reset_index(drop=True)
    annual_history_df = full_history_df.copy()
    if start_date_text:
        start_dt = parse_ddmmyyyy(start_date_text)
        actual_dates = pd.to_datetime(start_dt) + pd.to_timedelta(full_history_df["time_days"].to_numpy(dtype=float), unit="D")
        annual_history_df = full_history_df.copy()
        annual_history_df.insert(1, "date_actual", actual_dates)
        annual_history_df.insert(2, "year", actual_dates.year.astype(int))

        annual_rows = []
        if not annual_history_df.empty:
            # Always include the simulation start point.
            annual_rows.append(annual_history_df.iloc[[0]])

            unique_years = annual_history_df["year"].drop_duplicates().tolist()
            # Use the last simulated record in each intermediate calendar year.
            for year in unique_years[:-1]:
                year_rows = annual_history_df[annual_history_df["year"] == year]
                if not year_rows.empty:
                    annual_rows.append(year_rows.tail(1))
            # Always include the final simulated record.
            annual_rows.append(annual_history_df.tail(1))

        if annual_rows:
            annual_history_df = (
                pd.concat(annual_rows, ignore_index=False)
                .sort_values("time_days")
                .drop_duplicates(subset=["time_days"], keep="last")
                .reset_index(drop=True)
            )

        time_axis = annual_history_df["date_actual"].to_numpy(dtype="datetime64[ns]")
        time_axis_label = "Date"
    else:
        annual_rows = []
        if not annual_history_df.empty:
            annual_rows.append(annual_history_df.iloc[[0]])
            whole_year_index = np.floor(annual_history_df["time_days"].to_numpy(dtype=float) / 365.25)
            for whole_year in np.unique(whole_year_index.astype(int))[:-1]:
                yr_rows = annual_history_df[whole_year_index.astype(int) == whole_year]
                if not yr_rows.empty:
                    annual_rows.append(yr_rows.tail(1))
            annual_rows.append(annual_history_df.tail(1))
            annual_history_df = (
                pd.concat(annual_rows, ignore_index=False)
                .sort_values("time_days")
                .drop_duplicates(subset=["time_days"], keep="last")
                .reset_index(drop=True)
            )
        time_axis = annual_history_df["time_days"].to_numpy(dtype=float) / 365.25
        time_axis_label = "Time (years)"

    raw_gas_rate = annual_history_df["gas_rate_mmscf_day"].to_numpy(dtype=float)
    gas_rate_plot = raw_gas_rate

    history_match_result = None
    if history_match_enabled and history_match_observed_df is not None and history_match_error is None:
        try:
            history_match_result = compute_history_match(history, history_match_observed_df, weights=history_match_weights)
        except Exception as e:
            st.warning(f"History-matching diagnostics could not be computed: {e}")

    comparison_bundle = None
    if enable_scenario_comparison:
        comparison_map = {
            "Natural Depletion": "natural_depletion",
            "Gas Cycling": "gas_cycling",
            "Lean Gas Injection": "lean_gas_injection",
        }
        selected_comparison_labels = [label for label in comparison_labels if label in comparison_map]
        if len(selected_comparison_labels) >= 2:
            comparison_runs = []
            comparison_histories = []
            comparison_errors = []
            for label in selected_comparison_labels:
                compare_kwargs = dict(run_kwargs)
                compare_kwargs["scenario_name"] = comparison_map[label]
                if comparison_map[label] == "natural_depletion":
                    compare_kwargs["injection_rate_value"] = 0.0
                try:
                    comp_result = run_example(**compare_kwargs)
                    comp_history = comp_result["history"]
                    comparison_runs.append({
                        "Scenario": label,
                        "Final Avg Pressure (psia)": float(comp_history.avg_pressure_psia[-1]),
                        "Final Well Pressure (psia)": float(comp_history.well_pressure_psia[-1]),
                        "Final Gas Rate (MMscf/day)": float(comp_history.gas_rate_mmscf_day[-1]),
                        "Final Condensate Rate (STB/day)": float(comp_history.condensate_rate_stb_day[-1]),
                        "Final Water Rate (STB/day)": float(comp_history.water_rate_stb_day[-1]),
                        "Cumulative Gas (Bscf)": float(comp_history.cum_gas_bscf[-1]),
                        "Cumulative Condensate (MSTB)": float(comp_history.cum_condensate_mstb[-1]),
                        "Cumulative Water (MSTB)": float(comp_history.cum_water_mstb[-1]),
                        "Final Well P - Dew Point (psia)": float(comp_history.well_pressure_minus_dewpoint_psia[-1]),
                        "Final Productivity Loss Fraction": float(comp_history.productivity_loss_fraction[-1]),
                    })
                    comparison_histories.append((label, comp_history))
                except Exception as e:
                    comparison_errors.append(f"{label}: {e}")
            if comparison_runs:
                comparison_bundle = {
                    "summary_df": pd.DataFrame(comparison_runs),
                    "histories": comparison_histories,
                    "errors": comparison_errors,
                }
            elif comparison_errors:
                st.warning("Scenario comparison could not be computed. " + "; ".join(comparison_errors))
        elif selected_comparison_labels:
            st.info("Select at least two scenarios to generate a comparison table.")

    sensitivity_bundle = None
    if enable_sensitivity:
        sensitivity_param_map = {
            "Permeability (md)": ("permeability_md", "Permeability (md)"),
            "Porosity": ("porosity", "Porosity"),
            "Initial Reservoir Pressure (psia)": ("p_init_psia", "Initial Reservoir Pressure (psia)"),
            "Dew Point Pressure (psia)": ("dew_point_psia", "Dew Point Pressure (psia)"),
            "Producer BHP (psia)": ("bhp_psia", "Producer BHP (psia)"),
            "Gas Injection Rate": ("injection_rate_value", f"Gas Injection Rate ({injection_rate_unit})"),
            "Transport Dispersivity (ft)": ("transport_dispersivity_ft", "Transport Dispersivity (ft)"),
        }
        param_key, param_title = sensitivity_param_map[sensitivity_parameter_label]
        pct_cases = [("Low", float(sensitivity_low_pct)), ("Base", float(sensitivity_base_pct)), ("High", float(sensitivity_high_pct))]
        seen_case_keys = set()
        sensitivity_rows = []
        sensitivity_histories = []
        sensitivity_errors = []
        base_value = float(run_kwargs.get(param_key, 0.0))
        for case_label, pct_change in pct_cases:
            case_key = round(pct_change, 8)
            if case_key in seen_case_keys:
                continue
            seen_case_keys.add(case_key)
            factor = 1.0 + pct_change / 100.0
            case_kwargs = dict(run_kwargs)
            if param_key == 'porosity':
                varied_value = float(np.clip(base_value * factor, 0.01, 0.95))
            elif param_key in {'bhp_psia', 'p_init_psia', 'dew_point_psia'}:
                varied_value = max(14.7, float(base_value * factor))
            elif param_key == 'injection_rate_value':
                varied_value = max(0.0, float(base_value * factor))
                if case_kwargs.get('scenario_name') == 'natural_depletion':
                    varied_value = 0.0
            elif param_key == 'transport_dispersivity_ft':
                varied_value = max(0.0, float(base_value * factor))
            else:
                varied_value = max(1e-9, float(base_value * factor))
            case_kwargs[param_key] = varied_value
            try:
                sens_result = run_example(**case_kwargs)
                sens_history = sens_result['history']
                sensitivity_rows.append({
                    'Case': case_label,
                    'Change (%)': pct_change,
                    'Parameter': param_title,
                    'Varied Value': varied_value,
                    'Final Avg Pressure (psia)': float(sens_history.avg_pressure_psia[-1]),
                    'Final Well Pressure (psia)': float(sens_history.well_pressure_psia[-1]),
                    'Final Gas Rate (MMscf/day)': float(sens_history.gas_rate_mmscf_day[-1]),
                    'Final Condensate Rate (STB/day)': float(sens_history.condensate_rate_stb_day[-1]),
                    'Cumulative Gas (Bscf)': float(sens_history.cum_gas_bscf[-1]),
                    'Cumulative Condensate (MSTB)': float(sens_history.cum_condensate_mstb[-1]),
                    'Final Productivity Loss Fraction': float(sens_history.productivity_loss_fraction[-1]),
                    'Final Well P - Dew Point (psia)': float(sens_history.well_pressure_minus_dewpoint_psia[-1]),
                })
                sensitivity_histories.append((case_label, sens_history, varied_value, pct_change))
            except Exception as e:
                sensitivity_errors.append(f"{case_label}: {e}")
        if sensitivity_rows:
            sensitivity_df = pd.DataFrame(sensitivity_rows)
            tornado_rows = []
            base_mask = sensitivity_df['Case'] == 'Base'
            if base_mask.any():
                base_row = sensitivity_df.loc[base_mask].iloc[0]
                for metric in ['Final Avg Pressure (psia)', 'Final Condensate Rate (STB/day)', 'Cumulative Condensate (MSTB)', 'Final Well P - Dew Point (psia)']:
                    low_vals = sensitivity_df.loc[sensitivity_df['Case'] == 'Low', metric]
                    high_vals = sensitivity_df.loc[sensitivity_df['Case'] == 'High', metric]
                    tornado_rows.append({
                        'Metric': metric,
                        'Low Delta vs Base': float(low_vals.iloc[0] - base_row[metric]) if len(low_vals) else np.nan,
                        'High Delta vs Base': float(high_vals.iloc[0] - base_row[metric]) if len(high_vals) else np.nan,
                    })
            sensitivity_bundle = {
                'parameter_title': param_title,
                'base_value': base_value,
                'summary_df': sensitivity_df,
                'histories': sensitivity_histories,
                'tornado_df': pd.DataFrame(tornado_rows),
                'errors': sensitivity_errors,
            }
        elif sensitivity_errors:
            st.warning("Sensitivity analysis could not be computed. " + "; ".join(sensitivity_errors))

    monte_carlo_bundle = None
    if enable_monte_carlo:
        mc_param_map = _monte_carlo_parameter_map(injection_rate_unit)
        selected_mc = [label for label in monte_carlo_selected_labels if label in mc_param_map]
        if selected_mc:
            rng = np.random.default_rng(int(monte_carlo_seed))
            mc_rows = []
            mc_errors = []
            for run_idx in range(int(monte_carlo_runs)):
                case_kwargs = dict(run_kwargs)
                sampled_inputs = {}
                for label in selected_mc:
                    param_key, param_title = mc_param_map[label]
                    base_value = float(run_kwargs.get(param_key, 0.0))
                    sampled_value = _sample_parameter_value(param_key, base_value, float(monte_carlo_low_pct), float(monte_carlo_high_pct), rng, str(case_kwargs.get('scenario_name', '')))
                    case_kwargs[param_key] = sampled_value
                    sampled_inputs[param_title] = sampled_value
                try:
                    mc_result = run_example(**case_kwargs)
                    mc_history = mc_result['history']
                    row = {
                        'Run': run_idx + 1,
                        'Final Avg Pressure (psia)': float(mc_history.avg_pressure_psia[-1]),
                        'Final Well Pressure (psia)': float(mc_history.well_pressure_psia[-1]),
                        'Final Gas Rate (MMscf/day)': float(mc_history.gas_rate_mmscf_day[-1]),
                        'Final Condensate Rate (STB/day)': float(mc_history.condensate_rate_stb_day[-1]),
                        'Cumulative Gas (Bscf)': float(mc_history.cum_gas_bscf[-1]),
                        'Cumulative Condensate (MSTB)': float(mc_history.cum_condensate_mstb[-1]),
                        'Cumulative Water (MSTB)': float(mc_history.cum_water_mstb[-1]),
                        'Final Well P - Dew Point (psia)': float(mc_history.well_pressure_minus_dewpoint_psia[-1]),
                        'Final Productivity Loss Fraction': float(mc_history.productivity_loss_fraction[-1]),
                    }
                    row.update(sampled_inputs)
                    mc_rows.append(row)
                except Exception as e:
                    mc_errors.append(f"Run {run_idx + 1}: {e}")
            if mc_rows:
                mc_df = pd.DataFrame(mc_rows)
                summary_rows = []
                for metric in [
                    'Final Avg Pressure (psia)',
                    'Final Well Pressure (psia)',
                    'Final Gas Rate (MMscf/day)',
                    'Final Condensate Rate (STB/day)',
                    'Cumulative Gas (Bscf)',
                    'Cumulative Condensate (MSTB)',
                    'Final Well P - Dew Point (psia)',
                    'Final Productivity Loss Fraction',
                ]:
                    vals = pd.to_numeric(mc_df[metric], errors='coerce').to_numpy(dtype=float)
                    p10, p50, p90 = _percentile_summary(vals)
                    summary_rows.append({
                        'Metric': metric,
                        'Mean': float(np.nanmean(vals)),
                        'Std Dev': float(np.nanstd(vals)),
                        'P10': p10,
                        'P50': p50,
                        'P90': p90,
                        'Min': float(np.nanmin(vals)),
                        'Max': float(np.nanmax(vals)),
                    })
                monte_carlo_bundle = {
                    'runs_df': mc_df,
                    'summary_df': pd.DataFrame(summary_rows),
                    'selected_labels': selected_mc,
                    'errors': mc_errors,
                    'successful_runs': len(mc_df),
                    'requested_runs': int(monte_carlo_runs),
                }
            elif mc_errors:
                st.warning("Monte Carlo analysis could not be computed. " + "; ".join(mc_errors[:5]))
        else:
            st.info("Select at least one uncertain parameter to run Monte Carlo analysis.")

    st.success("Simulation completed.")

    st.header("Results")

    tab_names = ["Summary", "History Plots", "Spatial Diagnostics", "Tables"]
    if comparison_bundle is not None:
        tab_names.append("Scenario Comparison")
    if sensitivity_bundle is not None:
        tab_names.append("Sensitivity")
    if history_match_result is not None:
        tab_names.append("History Match")
    if monte_carlo_bundle is not None:
        tab_names.append("Monte Carlo")
    tab_names.append("Downloads")
    result_tabs = st.tabs(tab_names)

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

        if scenario_name in ("gas_cycling", "lean_gas_injection"):
            i1, i2, i3 = st.columns(3)
            i1.metric("Final Actual Injection Rate (lbmol/day)", f"{history.injector_rate_total_lbmol_day[-1]:.2f}")
            i2.metric("Final Injection ΔP (psi)", f"{history.injector_pressure_delta_psia[-1]:.2f}")
            i3.metric("Injector Active", "Yes" if history.injector_active_flag[-1] > 0.5 else "No")

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
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Stage 1 Separator Pressure (psia)", f"{history.separator_pressure_psia[-1]:.2f}")
        s2.metric("Stage 1 Separator Temperature (°F)", f"{rankine_to_fahrenheit(history.separator_temperature_R[-1]):.2f}")
        s3.metric("Stage 1 Vapor Fraction", f"{history.separator_vapor_fraction[-1]:.4f}")
        s4.metric("Separator Stages", f"{history.separator_stage_count[-1]:.0f}")

        st.subheader("Chapter 4 Summary")
        st.dataframe(summary_df, use_container_width=True)

        if history_match_result is not None:
            hm_a, hm_b, hm_c = st.columns(3)
            hm_a.metric("History-Match Objective", f"{history_match_result['objective_normalized']:.4f}" if np.isfinite(history_match_result['objective_normalized']) else "NA")
            hm_b.metric("Matched Metrics", f"{history_match_result['matched_metrics']}")
            hm_c.metric("Observed Rows", f"{len(history_match_observed_df)}")

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
            make_line_figure(time_axis, [(annual_history_df["avg_pressure_psia"].to_numpy(dtype=float), "Average Pressure"), (annual_history_df["well_pressure_psia"].to_numpy(dtype=float), "Well-Cell Pressure"), (annual_history_df["dew_point_psia"].to_numpy(dtype=float), "Dew Point")], time_axis_label, "Pressure (psia)", "Pressure and Dew-Point History"),
            make_line_figure(time_axis, [(annual_history_df["well_pressure_minus_dewpoint_psia"].to_numpy(dtype=float), "Well Pressure - Dew Point"), (annual_history_df["avg_pressure_minus_dewpoint_psia"].to_numpy(dtype=float), "Average Pressure - Dew Point")], time_axis_label, "Pressure Margin (psia)", "Pressure Margin to Dew Point"),
            make_line_figure(
                time_axis,
                [(gas_rate_plot, "Gas Production History")],
                time_axis_label,
                "Gas Rate (MMscf/day)",
                "Gas Production History",
            ),
            make_line_figure(time_axis, [(annual_history_df["condensate_rate_stb_day"].to_numpy(dtype=float), "Condensate Rate"), (annual_history_df["water_rate_stb_day"].to_numpy(dtype=float), "Water Rate")], time_axis_label, "Liquid Rate (STB/day)", "Condensate and Water Production Rate History"),
            make_line_figure(time_axis, [(annual_history_df["cum_gas_bscf"].to_numpy(dtype=float), "Cumulative Gas")], time_axis_label, "Cumulative Gas (Bscf)", "Cumulative Gas Production"),
            make_line_figure(time_axis, [(annual_history_df["cum_condensate_mstb"].to_numpy(dtype=float), "Cumulative Condensate"), (annual_history_df["cum_water_mstb"].to_numpy(dtype=float), "Cumulative Water")], time_axis_label, "Cumulative Liquid (MSTB)", "Cumulative Condensate and Water Production"),
            make_line_figure(time_axis, [(annual_history_df["well_damage_factor"].to_numpy(dtype=float), "Damage Factor"), (annual_history_df["productivity_loss_fraction"].to_numpy(dtype=float), "Productivity Loss Fraction")], time_axis_label, "Dimensionless", "Condensate-Bank Impairment"),
            make_line_figure(time_axis, [(annual_history_df["avg_sw"].to_numpy(dtype=float), "Average Sw")], time_axis_label, "Average Sw", "Average Water Saturation History"),
            make_line_figure(time_axis, [(np.abs(annual_history_df["hc_mass_balance_error_lbmol"].to_numpy(dtype=float)), "|HC Mass Balance Error|"), (np.abs(annual_history_df["water_mass_balance_error_lbmol"].to_numpy(dtype=float)), "|Water Mass Balance Error|")], time_axis_label, "Absolute Error (lbmol)", "Mass Balance Error History"),
            make_line_figure(time_axis, [(annual_history_df["well_flowing_pwf_psia"].to_numpy(dtype=float), "Flowing Pwf")], time_axis_label, "Pwf (psia)", "Flowing Bottom-Hole Pressure History"),
            make_line_figure(time_axis, [(annual_history_df["well_estimated_thp_psia"].to_numpy(dtype=float), "Estimated THP")], time_axis_label, "THP (psia)", "Tubing Head Pressure History"),
            make_line_figure(time_axis, [(annual_history_df["separator_vapor_fraction"].to_numpy(dtype=float), "Stage 1 Vapor Fraction")], time_axis_label, "Vapor Fraction", "Surface Separator Flash Vapor Fraction History"),
            make_line_figure(time_axis, [
                (annual_history_df["separator_total_gas_rate_mmscf_day"].to_numpy(dtype=float), "Total Separator Gas"),
                (annual_history_df["separator_stage1_gas_rate_mmscf_day"].to_numpy(dtype=float), "Stage 1 Gas"),
                (annual_history_df["separator_stage2_gas_rate_mmscf_day"].to_numpy(dtype=float), "Stage 2 Gas"),
                (annual_history_df["separator_stage3_gas_rate_mmscf_day"].to_numpy(dtype=float), "Stage 3 Gas"),
                (annual_history_df["separator_stock_tank_gas_rate_mmscf_day"].to_numpy(dtype=float), "Stock Tank Gas"),
            ], time_axis_label, "Gas Rate (MMscf/day)", "Surface Train Gas Release by Stage"),
            make_line_figure(time_axis, [(annual_history_df["separator_stock_tank_liquid_rate_stb_day"].to_numpy(dtype=float), "Stock Tank Liquid")], time_axis_label, "Liquid Rate (STB/day)", "Stock Tank Liquid Rate History"),
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

    next_tab_idx = 4
    if comparison_bundle is not None:
        with result_tabs[next_tab_idx]:
            st.subheader("Scenario Comparison Summary")
            st.dataframe(comparison_bundle['summary_df'], use_container_width=True)
            if comparison_bundle.get('errors'):
                st.warning("Some comparison runs failed: " + "; ".join(comparison_bundle['errors']))

            fig_pressure = make_line_figure(
                time_axis,
                [(np.asarray(comp_history.avg_pressure_psia, dtype=float), label) for label, comp_history in comparison_bundle['histories']],
                time_axis_label,
                "Average Pressure (psia)",
                "Scenario Comparison: Average Pressure",
            )
            st.pyplot(fig_pressure)
            plt.close(fig_pressure)

            fig_cond = make_line_figure(
                time_axis,
                [(np.asarray(comp_history.condensate_rate_stb_day, dtype=float), label) for label, comp_history in comparison_bundle['histories']],
                time_axis_label,
                "Condensate Rate (STB/day)",
                "Scenario Comparison: Condensate Rate",
            )
            st.pyplot(fig_cond)
            plt.close(fig_cond)

            fig_cum_cond = make_line_figure(
                time_axis,
                [(np.asarray(comp_history.cum_condensate_mstb, dtype=float), label) for label, comp_history in comparison_bundle['histories']],
                time_axis_label,
                "Cumulative Condensate (MSTB)",
                "Scenario Comparison: Cumulative Condensate",
            )
            st.pyplot(fig_cum_cond)
            plt.close(fig_cum_cond)
        next_tab_idx += 1

    if sensitivity_bundle is not None:
        with result_tabs[next_tab_idx]:
            st.subheader(f"One-Way Sensitivity: {sensitivity_bundle['parameter_title']}")
            st.write(f"Base value used for the sensitivity runs: **{sensitivity_bundle['base_value']:.6g}**")
            st.dataframe(sensitivity_bundle['summary_df'], use_container_width=True)
            if not sensitivity_bundle['tornado_df'].empty:
                st.subheader("Delta vs Base")
                st.dataframe(sensitivity_bundle['tornado_df'], use_container_width=True)
            if sensitivity_bundle.get('errors'):
                st.warning("Some sensitivity runs failed: " + "; ".join(sensitivity_bundle['errors']))

            sensitivity_histories = sensitivity_bundle['histories']
            fig_sens_pressure = make_line_figure(
                time_axis,
                [(np.asarray(s_hist.avg_pressure_psia, dtype=float), f"{label} ({pct:+.0f}%)") for label, s_hist, _, pct in sensitivity_histories],
                time_axis_label,
                "Average Pressure (psia)",
                "Sensitivity: Average Pressure",
            )
            st.pyplot(fig_sens_pressure)
            plt.close(fig_sens_pressure)

            fig_sens_cond = make_line_figure(
                time_axis,
                [(np.asarray(s_hist.condensate_rate_stb_day, dtype=float), f"{label} ({pct:+.0f}%)") for label, s_hist, _, pct in sensitivity_histories],
                time_axis_label,
                "Condensate Rate (STB/day)",
                "Sensitivity: Condensate Rate",
            )
            st.pyplot(fig_sens_cond)
            plt.close(fig_sens_cond)

            fig_sens_cum = make_line_figure(
                time_axis,
                [(np.asarray(s_hist.cum_condensate_mstb, dtype=float), f"{label} ({pct:+.0f}%)") for label, s_hist, _, pct in sensitivity_histories],
                time_axis_label,
                "Cumulative Condensate (MSTB)",
                "Sensitivity: Cumulative Condensate",
            )
            st.pyplot(fig_sens_cum)
            plt.close(fig_sens_cum)
        next_tab_idx += 1

    if history_match_result is not None:
        with result_tabs[next_tab_idx]:
            st.subheader("History-Match Metrics")
            st.dataframe(history_match_result['metrics_df'], use_container_width=True)
            st.subheader("Observed vs Simulated Comparison")
            st.dataframe(history_match_result['comparison_df'], use_container_width=True)

            comparison_df = history_match_result['comparison_df']
            match_figs = []
            metric_labels = {
                'avg_pressure_psia': 'Average Pressure',
                'well_pressure_psia': 'Well Pressure',
                'gas_rate_mmscf_day': 'Gas Rate',
                'condensate_rate_stb_day': 'Condensate Rate',
                'water_rate_stb_day': 'Water Rate',
                'cum_gas_bscf': 'Cumulative Gas',
                'cum_condensate_mstb': 'Cumulative Condensate',
                'cum_water_mstb': 'Cumulative Water',
            }
            x_match = np.asarray(comparison_df['time_days'], dtype=float)
            for metric, label in metric_labels.items():
                obs_col = f'obs_{metric}'
                sim_col = f'sim_{metric}'
                if obs_col in comparison_df.columns and sim_col in comparison_df.columns:
                    obs_vals = pd.to_numeric(comparison_df[obs_col], errors='coerce').to_numpy(dtype=float)
                    sim_vals = pd.to_numeric(comparison_df[sim_col], errors='coerce').to_numpy(dtype=float)
                    if np.isfinite(obs_vals).any() and np.isfinite(sim_vals).any():
                        match_figs.append(make_line_figure(x_match, [(obs_vals, f'Observed {label}'), (sim_vals, f'Simulated {label}')], 'Time (days)', label, f'Observed vs Simulated {label}'))
            for fig in match_figs:
                st.pyplot(fig)
                plt.close(fig)
        next_tab_idx += 1

    if monte_carlo_bundle is not None:
        with result_tabs[next_tab_idx]:
            st.subheader("Monte Carlo Uncertainty Summary")
            mc_a, mc_b = st.columns(2)
            mc_a.metric("Successful Runs", f"{monte_carlo_bundle['successful_runs']}/{monte_carlo_bundle['requested_runs']}")
            mc_b.metric("Uncertain Inputs", ", ".join(monte_carlo_bundle['selected_labels']))
            st.dataframe(monte_carlo_bundle['summary_df'], use_container_width=True)
            if monte_carlo_bundle.get('errors'):
                st.warning("Some Monte Carlo runs failed: " + "; ".join(monte_carlo_bundle['errors'][:5]))

            runs_df = monte_carlo_bundle['runs_df']
            for metric in ['Final Avg Pressure (psia)', 'Cumulative Condensate (MSTB)', 'Final Well P - Dew Point (psia)']:
                vals = pd.to_numeric(runs_df[metric], errors='coerce').to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    fig_hist, ax_hist = plt.subplots(figsize=(8, 4.5))
                    ax_hist.hist(vals, bins=min(20, max(8, int(np.sqrt(vals.size)))))
                    ax_hist.set_xlabel(metric)
                    ax_hist.set_ylabel('Frequency')
                    ax_hist.set_title(f'Monte Carlo Distribution: {metric}')
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
        next_tab_idx += 1

    with result_tabs[next_tab_idx]:
        st.download_button("Download history.csv", history_df.to_csv(index=False).encode("utf-8"), file_name="history.csv", mime="text/csv")
        st.download_button("Download final_spatial_diagnostics.csv", spatial_df.to_csv(index=False).encode("utf-8"), file_name="final_spatial_diagnostics.csv", mime="text/csv")
        st.download_button("Download chapter4_summary.csv", summary_df.to_csv(index=False).encode("utf-8"), file_name="chapter4_summary.csv", mime="text/csv")
        if comparison_bundle is not None:
            st.download_button("Download scenario_comparison_summary.csv", comparison_bundle['summary_df'].to_csv(index=False).encode("utf-8"), file_name="scenario_comparison_summary.csv", mime="text/csv")
        if sensitivity_bundle is not None:
            st.download_button("Download sensitivity_summary.csv", sensitivity_bundle['summary_df'].to_csv(index=False).encode("utf-8"), file_name="sensitivity_summary.csv", mime="text/csv")
            if not sensitivity_bundle['tornado_df'].empty:
                st.download_button("Download sensitivity_delta_vs_base.csv", sensitivity_bundle['tornado_df'].to_csv(index=False).encode("utf-8"), file_name="sensitivity_delta_vs_base.csv", mime="text/csv")
        if history_match_result is not None:
            st.download_button("Download history_match_metrics.csv", history_match_result['metrics_df'].to_csv(index=False).encode("utf-8"), file_name="history_match_metrics.csv", mime="text/csv")
            st.download_button("Download history_match_comparison.csv", history_match_result['comparison_df'].to_csv(index=False).encode("utf-8"), file_name="history_match_comparison.csv", mime="text/csv")
        if monte_carlo_bundle is not None:
            st.download_button("Download monte_carlo_summary.csv", monte_carlo_bundle['summary_df'].to_csv(index=False).encode("utf-8"), file_name="monte_carlo_summary.csv", mime="text/csv")
            st.download_button("Download monte_carlo_runs.csv", monte_carlo_bundle['runs_df'].to_csv(index=False).encode("utf-8"), file_name="monte_carlo_runs.csv", mime="text/csv")
        st.caption(
            "Gas rates are reported in MMscf/day using standard conditions of 14.7 psia and 60°F. Condensate rates are reported from a configurable surface train with Stage 1, Stage 2, Stage 3, and optional stock-tank gas release plus stock-tank liquid stabilization, while the stage-1 separator vapor fraction is retained as a diagnostic. "
            "Producer control may be fixed BHP, fixed gas rate, or simplified fixed THP. Condensate reporting uses the producer wellstream and blends with the user PVT CGR table when available. The THP mode uses an approximate tubing-loss relation and should be treated as a prototype rather than a full VLP implementation. History-matching diagnostics interpolate simulated results to observed timestamps and report weighted NRMSE-based objective contributions for quick calibration screening. One-way sensitivity analysis runs low/base/high perturbations around a selected input and reports delta-versus-base screening metrics. Monte Carlo analysis independently samples selected uncertain inputs using bounded triangular distributions around the current base case and reports probabilistic ranges such as P10, P50, and P90 for key outcomes."
        )


display_streamlit_app()
