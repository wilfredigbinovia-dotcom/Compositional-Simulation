from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal, NamedTuple
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import streamlit as st
import re
from scipy.linalg import solve_banded


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


def water_lbmol_per_stb() -> float:
    return WATER_DENSITY_LBFT3 * FT3_PER_STB / MW_WATER


def water_lbmol_per_mstb() -> float:
    return 1000.0 * water_lbmol_per_stb()

ScenarioName = Literal["natural_depletion", "gas_cycling", "lean_gas_injection"]
WellControlMode = Literal["bhp", "drawdown", "gas_rate", "thp"]


# NamedTuples to replace unwieldy positional return tuples
class TransportResult(NamedTuple):
    nt: np.ndarray
    z: np.ndarray
    nw: np.ndarray
    sw: np.ndarray
    q_total: float
    gas_frac: float
    oil_frac: float
    water_frac: float
    dropout: float
    krg: float
    kro: float
    krw: float
    damage_factor: float
    wi_eff: float
    q_undamaged: float
    ploss: float
    injector_actual_rate_lbmol_day: float
    transport_advective_total: float
    transport_dispersive_total: float
    transport_total_flux: float
    aquifer_rate_lbmol_day: float
    hc_injected_lbmol: float   # exact moles added by injector across all subcycles


class WellSinkResult(NamedTuple):
    hc_sink: np.ndarray
    qw_lbmol_day: float
    q_total_lbmol_day: float
    gas_frac: float
    oil_frac: float
    water_frac: float
    dropout: float
    krg: float
    kro: float
    krw: float
    damage_factor: float
    wi_eff: float
    q_undamaged: float
    ploss: float


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
        # bi is independent of temperature — cache it permanently.
        self._bi = 0.07780 * R * self.Tc / self.Pc
        # kappa is also temperature-independent.
        self._kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        # Cache (T, ai, aij) so repeated calls at the same T (which is always
        # the case within one simulation) skip the O(nc²) outer-product.
        self._cached_T: float | None = None
        self._cached_ai: np.ndarray | None = None
        self._cached_aij: np.ndarray | None = None

    def _ai_bi(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        if self._cached_T is None or abs(T - self._cached_T) > 1e-8:
            Tr = T / self.Tc
            alpha = (1.0 + self._kappa * (1.0 - np.sqrt(Tr))) ** 2
            self._cached_ai = 0.45724 * (R ** 2) * (self.Tc ** 2) * alpha / self.Pc
            self._cached_aij = np.sqrt(np.outer(self._cached_ai, self._cached_ai)) * (1.0 - self.kij)
            self._cached_T = T
        return self._cached_ai, self._bi

    def mixture_params(self, z: np.ndarray, T: float):
        ai, bi = self._ai_bi(T)
        sum_aij = self._cached_aij @ z   # O(nc²) — computed once, reused by fugacity
        am = float(z @ sum_aij)
        bm = float(np.dot(z, bi))
        return am, bm, ai, bi, sum_aij

    @staticmethod
    def solve_cubic(A: float, B: float) -> np.ndarray:
        """Solve the PR-EOS cubic Z³ + c2·Z² + c1·Z + c0 = 0 analytically.

        Uses Cardano's method via the depressed cubic substitution.  This is
        ~50× faster than numpy.roots which computes eigenvalues of the 3×3
        companion matrix.
        """
        c2 = -(1.0 - B)
        c1 = A - 3.0 * B ** 2 - 2.0 * B
        c0 = -(A * B - B ** 2 - B ** 3)

        # Depress: substitute Z = t - c2/3
        p = c1 - c2 ** 2 / 3.0
        q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0
        disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

        shift = -c2 / 3.0

        if disc > 0.0:
            # One real root
            sqrt_disc = math.sqrt(disc)
            u_arg = -q / 2.0 + sqrt_disc
            v_arg = -q / 2.0 - sqrt_disc
            u = math.copysign(abs(u_arg) ** (1.0 / 3.0), u_arg)
            v = math.copysign(abs(v_arg) ** (1.0 / 3.0), v_arg)
            roots = np.array([u + v + shift])
        else:
            # Three real roots (casus irreducibilis)
            r   = math.sqrt(max(-(p / 3.0) ** 3, 0.0))
            cos_arg = float(np.clip(-q / (2.0 * max(r, 1e-300)), -1.0, 1.0))
            theta = math.acos(cos_arg)
            m = 2.0 * r ** (1.0 / 3.0)
            roots = np.array([
                m * math.cos((theta          ) / 3.0) + shift,
                m * math.cos((theta + 2.0 * math.pi) / 3.0) + shift,
                m * math.cos((theta + 4.0 * math.pi) / 3.0) + shift,
            ])

        # Reject unphysical roots: Z must exceed B (from EOS volume constraint)
        real_roots = roots[roots > B]
        real_roots.sort()
        return real_roots

    def z_factor(self, z: np.ndarray, P: float, T: float, phase: str) -> float:
        am, bm, _, _, _ = self.mixture_params(z, T)
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
        am, bm, ai, bi, sum_aij = self.mixture_params(x, T)
        A = am * P / (R ** 2 * T ** 2)
        B = bm * P / (R * T)
        # Solve cubic directly — avoids a second mixture_params call that z_factor would make
        roots = self.solve_cubic(A, B)
        if len(roots) == 0:
            roots = np.array([max(B + 1e-6, 1.0)])
        Z = float(np.max(roots)) if phase.lower() == "v" else float(np.min(roots))

        sqrt2 = math.sqrt(2.0)
        bm_safe = max(bm, 1e-12)
        am_safe = max(am, 1e-12)
        term1 = (bi / bm_safe) * (Z - 1.0)
        term2 = -math.log(max(Z - B, 1e-12))
        term3 = A / max(2.0 * sqrt2 * B, 1e-12)
        term4 = 2.0 * sum_aij / am_safe - bi / bm_safe
        ratio = (Z + (1.0 + sqrt2) * B) / max(Z + (1.0 - sqrt2) * B, 1e-12)
        term5 = math.log(max(ratio, 1e-12))
        ln_phi = term1 + term2 - term3 * term4 * term5

        return np.exp(ln_phi)


# -----------------------------------------------------------------------------
# Flash calculation
# -----------------------------------------------------------------------------

class FlashCalculator:
    def __init__(self, eos: PengRobinsonEOS):
        self.eos = eos
        self.fluid = eos.fluid
        self.Tc, self.Pc, self.omega, _ = self.fluid.critical_arrays()
        # K-value warm-start cache: maps a composition hash to the last
        # converged K-vector at a nearby (P, T).  Using the converged K from
        # the previous timestep as the starting point typically halves the
        # number of successive-substitution iterations near phase boundaries
        # and eliminates wrong-root convergence near the critical point.
        self._K_cache: Dict[int, np.ndarray] = {}  # hash(z_bytes) → K

    def wilson_k(self, P: float, T: float) -> np.ndarray:
        return (self.Pc / P) * np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.Tc / T))

    @staticmethod
    def rachford_rice(beta: float, z: np.ndarray, K: np.ndarray) -> float:
        # Pure NumPy — avoids Python-level sum loop
        Km1 = K - 1.0
        return float(np.dot(z, Km1 / (1.0 + beta * Km1)))

    def solve_beta(self, z: np.ndarray, K: np.ndarray) -> float:
        """Solve the Rachford-Rice equation for vapour fraction β ∈ (0, 1).

        Uses Newton-Raphson with a safe bracket [lo, hi] that avoids poles.
        Falls back to bisection if Newton steps leave the bracket.
        This is ~5× faster than scipy.brentq for this specific equation because
        the derivative is cheap and the function is monotone on the bracket.
        """
        f0 = self.rachford_rice(0.0, z, K)
        f1 = self.rachford_rice(1.0, z, K)
        if f0 < 0.0 and f1 < 0.0:
            return 0.0
        if f0 > 0.0 and f1 > 0.0:
            return 1.0

        # Safe bracket: avoid poles at β = -1/(Ki-1)
        Km1 = K - 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            poles = np.where(np.abs(Km1) > 1e-10, -1.0 / Km1, np.inf)
        lo = float(np.max(poles[poles < 0.0], initial=-1e10)) + 1e-8
        hi = float(np.min(poles[poles > 0.0], initial=1e10 + 1.0)) - 1e-8
        lo = max(lo, 0.0)
        hi = min(hi, 1.0)
        if lo >= hi:
            lo, hi = 0.0, 1.0

        beta = 0.5 * (lo + hi)
        for _ in range(50):
            denom = 1.0 + beta * Km1
            f  = float(np.sum(z * Km1 / denom))
            df = -float(np.sum(z * Km1 ** 2 / denom ** 2))
            if abs(f) < 1e-10:
                break
            if abs(df) > 1e-14:
                step = -f / df
                beta_new = beta + step
                if lo < beta_new < hi:
                    beta = beta_new
                else:
                    # Newton left the bracket — bisect instead
                    if f > 0.0:
                        lo = beta
                    else:
                        hi = beta
                    beta = 0.5 * (lo + hi)
            else:
                beta = 0.5 * (lo + hi)
                if f > 0.0:
                    lo = beta
                else:
                    hi = beta
        return float(np.clip(beta, 0.0, 1.0))

    def flash(self, z: np.ndarray, P: float, T: float, max_iter: int = 50, tol: float = 1e-7) -> Dict[str, np.ndarray | float | str]:
        z = np.clip(np.asarray(z, dtype=float), 1e-12, None)
        z = z / np.sum(z)

        # Warm-start: use cached K-values from previous flash at this composition
        # if available, otherwise fall back to Wilson K-values.
        z_key = hash(z.tobytes())
        K_wilson = np.clip(self.wilson_k(P, T), 1e-6, 1e6)
        if z_key in self._K_cache:
            K_prev = self._K_cache[z_key]
            # Only use the cached K if the Rachford-Rice function still brackets
            # zero — if the pressure has moved far enough that the old K puts us
            # outside the two-phase region, fall back to Wilson.
            f0 = self.rachford_rice(0.0, z, K_prev)
            f1 = self.rachford_rice(1.0, z, K_prev)
            K = K_prev if (f0 > 0.0 and f1 < 0.0) else K_wilson
        else:
            K = K_wilson
        beta = self.solve_beta(z, K)

        if beta <= 1e-10:
            # Single-phase liquid — do not update K cache; Wilson K is not representative
            return {"state": "liquid", "beta": 0.0, "x": z.copy(), "y": z.copy(), "K": K}
        if beta >= 1.0 - 1e-10:
            # Single-phase vapour — do not update K cache
            return {"state": "vapor", "beta": 1.0, "x": z.copy(), "y": z.copy(), "K": K}

        converged = False
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
                converged = True
                break

        if not converged:
            warnings.warn(
                f"Flash did not converge after {max_iter} iterations at P={P:.1f} psia, T={T:.1f} R. "
                "Results may be inaccurate near critical point.",
                RuntimeWarning,
                stacklevel=2,
            )

        denom = 1.0 + beta * (K - 1.0)
        x = z / denom
        y = K * x
        x /= np.sum(x)
        y /= np.sum(y)
        # Store converged K for warm-starting future flashes at this composition.
        # Cap cache size to avoid unbounded memory growth across many cells/steps.
        if len(self._K_cache) > 2000:
            self._K_cache.clear()
        self._K_cache[z_key] = K.copy()
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
    compressibility_1_psi: float = 3.0e-6  # typical sandstone value; used in pressure solve


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
    # Condensate bank impairment parameters
    condensate_damage_strength: float = 3.0      # exponential decay rate; higher = more damage
    condensate_critical_dropout: float = 0.05    # molar liquid fraction threshold below which no damage
    condensate_min_damage_factor: float = 0.20   # floor: well never loses more than (1 - floor) of PI

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
        if not (0.0 <= self.condensate_min_damage_factor <= 1.0):
            raise ValueError("condensate_min_damage_factor must be in [0, 1]")
        if self.condensate_damage_strength < 0.0:
            raise ValueError("condensate_damage_strength must be non-negative")
        if not (0.0 <= self.condensate_critical_dropout <= 1.0):
            raise ValueError("condensate_critical_dropout must be in [0, 1]")


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
    productivity_index: float = 1.0
    rw_ft: float = 0.35
    skin: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.cell_index < 0:
            raise ValueError("Well cell_index must be non-negative")
        if self.min_bhp_psia < 14.7:
            raise ValueError("min_bhp_psia must be ≥ 14.7 psia (atmospheric)")
        if self.bhp_psia < self.min_bhp_psia:
            raise ValueError(f"bhp_psia ({self.bhp_psia:.1f}) must be ≥ min_bhp_psia ({self.min_bhp_psia:.1f})")
        if self.target_gas_rate_mmscf_day < 0.0:
            raise ValueError("target_gas_rate_mmscf_day must be non-negative")
        if self.thp_psia < 14.7:
            raise ValueError("thp_psia must be ≥ 14.7 psia")
        if self.tvd_ft < 0.0:
            raise ValueError("tvd_ft must be non-negative")
        if self.tubing_id_in <= 0.0:
            raise ValueError("tubing_id_in must be positive")
        if self.rw_ft <= 0.0:
            raise ValueError("rw_ft must be positive")
        if self.control_mode not in {"bhp", "drawdown", "gas_rate", "thp"}:
            raise ValueError(f"control_mode '{self.control_mode}' not recognised; "
                             "must be one of bhp, drawdown, gas_rate, thp")


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
    condensate_api_gravity: float = 50.0

    def __post_init__(self) -> None:
        if self.condensate_api_gravity <= -131.0:
            raise ValueError("Condensate API gravity must be greater than -131")

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

def phase_viscosity_cp(
    phase: str,
    comp: np.ndarray,
    pressure_psia: float | None = None,
    temperature_R: float | None = None,
    Tc_R: np.ndarray | None = None,
    Pc_psia: np.ndarray | None = None,
    Mw_lbmol: np.ndarray | None = None,
) -> float:
    """Lohrenz-Bray-Clark (LBC) viscosity correlation.

    Implements LBC (Lohrenz et al., JPT 1964) in its original CGS units:
    Tc in K, Pc in atm, Mw in g/mol, Vc in cm^3/mol, rho in mol/cm^3.

    The previous version used field units for xi (Tc in R, Pc in psia),
    which made the Stiel-Thodos parameter ~5x too small, inflating dilute-gas
    viscosity by the same factor.
    """
    comp = np.asarray(comp, dtype=float)
    total = np.sum(comp)
    if total <= 1e-12:
        return 0.01 if phase == "v" else 0.3
    comp = comp / total
    n = len(comp)

    T_R    = float(temperature_R) if temperature_R is not None else 600.0
    P_psia = float(pressure_psia) if pressure_psia is not None else 3000.0

    Tc_r = Tc_R     if Tc_R     is not None else np.linspace(343.1, 1100.0, n)
    Pc_p = Pc_psia  if Pc_psia  is not None else np.linspace(667.8,  250.0, n)
    Mw   = Mw_lbmol if Mw_lbmol is not None else np.linspace( 16.04, 200.0, n)

    # Convert to CGS
    Tc_K   = Tc_r / 1.8
    Pc_atm = Pc_p / 14.6959
    T_K    = T_R  / 1.8
    P_atm  = max(P_psia, 14.7) / 14.6959

    # Stiel-Thodos xi and dilute-gas viscosity in CGS
    xi = Tc_K**(1.0/6.0) / (Mw**0.5 * np.maximum(Pc_atm, 1e-6)**(2.0/3.0))
    Tr = T_K / np.maximum(Tc_K, 1e-6)
    mu_stiel = np.where(
        Tr <= 1.5,
        (34e-5 * Tr**0.94) / np.maximum(xi, 1e-12),
        (17.78e-5 * np.maximum(4.58*Tr - 1.67, 0.0)**0.625) / np.maximum(xi, 1e-12),
    )
    mu_stiel = np.clip(mu_stiel, 1e-5, 0.10)

    sqrt_mw      = np.sqrt(Mw)
    denom_hz     = float(np.sum(comp * sqrt_mw))
    mu_mix_dilute = float(np.sum(comp * mu_stiel * sqrt_mw) / max(denom_hz, 1e-12))
    xi_mix       = float(np.sum(comp * xi))

    # Pseudo-critical volume (CGS): Vc_i = Zc * R_cgs * Tc_K / Pc_atm  [cm^3/mol]
    Zc_st   = 0.2901
    R_cgs   = 82.06
    Vc_i    = Zc_st * R_cgs * Tc_K / np.maximum(Pc_atm, 1e-6)
    Vc_mix  = float(np.dot(comp, Vc_i))
    rho_pc  = 1.0 / max(Vc_mix, 1e-6)

    a = [0.1023, 0.023364, 0.058533, -0.040758, 0.0093324]

    if phase == "v":
        rho_g  = P_atm / max(R_cgs * T_K, 1e-6)      # mol/cm^3, Z=1 approx
        rho_r  = float(np.clip(rho_g / rho_pc, 0.0, 0.95))
        bracket = sum(a[k] * rho_r**k for k in range(5))
        mu = mu_mix_dilute + (bracket**4 - 1e-4) / max(xi_mix, 1e-12)
        return float(np.clip(mu, 0.005, 0.10))

    if phase == "l":
        rho_l  = 2.0 / max(Vc_mix, 1e-6)             # mol/cm^3 (Vl ~ Vc/2)
        rho_r  = float(np.clip(rho_l / rho_pc, 0.0, 2.5))
        bracket = sum(a[k] * rho_r**k for k in range(5))
        mu = mu_mix_dilute + (bracket**4 - 1e-4) / max(xi_mix, 1e-12)
        return float(np.clip(mu, 0.02, 5.0))

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



def beta_to_flowing_saturations(
    beta: float,
    Sw: float,
    params: RelPermParams,
) -> Tuple[float, float, float]:
    """Map flash vapour fraction β to flowing phase saturations.

    Physical basis
    --------------
    The molar vapour fraction β from the PR-EOS flash is used directly as the
    gas saturation proxy for the hydrocarbon pore space:

        Sg = β × Shc      (gas occupies β of the HC pore space)
        So = (1−β) × Shc  (liquid occupies the remainder)

    This is exact for a system in thermodynamic equilibrium where the grid
    cell tracks total HC molar inventory and the flash gives the vapour split.

    The previous implementation used a smoothstep blend with several empirical
    parameters (beta_lo, beta_hi, beta_weight, min_gas_sat) and dynamic floors
    that had no physical justification and produced non-zero kro — and therefore
    non-zero qo — even when the reservoir was single-phase gas (β=1).  This
    inflated condensate rate estimates by treating the artificial kro as a proxy
    for surface liquid yield.

    Critical gas saturation handling
    ---------------------------------
    Below the critical gas saturation Sgc the gas phase is immobile (trapped).
    Rather than a hard cliff (krg drops to zero at Sg=Sgc), a soft transition
    is applied: for Sg in (0, Sgc], krg is set to zero and the gas is fully
    trapped.  This is consistent with the Brooks-Corey / Stone model used in
    three_phase_relperm.

    Water saturation is returned unchanged — it is governed by the water
    transport equation, not by the flash.
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    Sw   = float(np.clip(Sw, 0.0, 0.95))
    Shc  = max(1.0 - Sw, 0.0)

    if Shc <= 1e-12:
        return 0.0, 0.0, Sw

    Sg_raw = beta * Shc
    So_raw = (1.0 - beta) * Shc

    # Normalise to ensure Sg + So + Sw = 1 exactly
    total = Sg_raw + So_raw + Sw
    if total > 1e-12:
        Sg = Sg_raw / total
        So = So_raw / total
        Sw_out = Sw / total
    else:
        Sg, So, Sw_out = 0.0, 0.0, Sw

    return float(Sg), float(So), float(Sw_out)


def liquid_dropout_fraction(beta: float) -> float:
    """Molar liquid fraction from the flash.

    The flash vapour fraction beta is already the molar fraction that is gas;
    (1 - beta) is therefore the molar liquid fraction.  This is the physically
    consistent definition of liquid dropout for a flash at a given pressure and
    is what a constant-composition expansion (CCE) measures at that pressure.

    Previously this used a heaviness-weighted composition proxy that had no
    physical units and gave values inconsistent with the flash output.
    """
    return float(max(1.0 - beta, 0.0))


def condensate_bank_damage_factor(
    dropout: float,
    relperm_params: "RelPermParams | None" = None,
    *,
    damage_strength: float = 3.0,
    critical_dropout: float = 0.05,
    min_damage_factor: float = 0.20,
) -> float:
    """Condensate-bank productivity impairment factor.

    Applies an exponential penalty once the molar liquid dropout exceeds a
    critical threshold, representing the near-wellbore two-phase flow zone
    described by Fevang & Whitson (1996).

    Parameters
    ----------
    dropout          : molar liquid fraction from the well-cell flash (1 - beta)
    relperm_params   : if supplied, overrides the scalar kwargs with the
                       values stored on RelPermParams (preferred call path)
    damage_strength  : exponential rate constant (higher = sharper damage onset)
    critical_dropout : liquid fraction below which no impairment is applied
    min_damage_factor: floor — damage factor never falls below this value

    The circular kro dependence from the previous version has been removed.
    Damage is now driven purely by excess liquid dropout, which is the
    physically meaningful quantity that controls near-well condensate banking.
    """
    if relperm_params is not None:
        damage_strength   = relperm_params.condensate_damage_strength
        critical_dropout  = relperm_params.condensate_critical_dropout
        min_damage_factor = relperm_params.condensate_min_damage_factor

    dropout = float(max(dropout, 0.0))
    excess  = max(dropout - critical_dropout, 0.0)
    damage  = math.exp(-damage_strength * excess)
    return float(np.clip(damage, min_damage_factor, 1.0))


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
        well: "Well | List[Well]",
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
        # Accept either a single Well or a list of Wells.
        # self.wells is the authoritative list; self.well is kept as an alias
        # to the first well for all single-well code paths (backward-compatible).
        if isinstance(well, list):
            self.wells: List[Well] = well if well else []
        else:
            self.wells = [well]
        if not self.wells:
            raise ValueError("At least one producer well must be provided")
        self.well = self.wells[0]
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
        # Correct pore volume: π×r² × h × φ distributed by dx fractions.
        # grid.bulk_volume(i) = cell_width × area_ft2 = dx × π×r², which gives
        # π×r² × length_ft (wrong — length_ft is the radial flow distance, not
        # the pay thickness). The correct cylindrical bulk volume is π×r² × h.
        correct_total_pv = self.grid.area_ft2 * self.grid.thickness_ft * self.phi
        pv_fractions     = self.grid.dx_array / self.grid.length_ft
        self.pv          = correct_total_pv * pv_fractions
        self.Tc, self.Pc, _, self.mw_components = self.fluid.critical_arrays()

        # Pre-compute Peaceman WI for each well — result is static (geometry + well params only)
        self._well_wi: Dict[int, float] = {}
        for _w in self.wells:
            self._well_wi[id(_w)] = self._compute_peaceman_wi(_w)

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
        self._cache_state_id = None

    def _set_cache_state(self, state: SimulationState) -> None:
        """Update cache key and invalidate caches when state content changes.

        Uses a content-based key (hash of pressure + z arrays) rather than
        object id.  This means the cache survives when work_state is rebuilt
        as a new SimulationState object each subcycle — cells whose pressure
        and composition haven't changed get cache hits rather than reflashes.
        Typical saving: ~5× fewer flash calls per subcycle.
        """
        # Cheap content fingerprint: combine a few elements and the array shapes
        p = state.pressure
        z = state.z
        key = hash((p.tobytes(), z.tobytes()))
        if key != self._cache_state_id:
            self._cache_state_id = key
            self._flash_cache = {}
            self._mobility_cache = {}

    def _invalidate_cell_cache(self, i: int) -> None:
        """Invalidate caches for a single cell (after a well sink modifies it)."""
        self._flash_cache.pop(i, None)
        self._mobility_cache.pop(i, None)

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
        """Compute per-cell flowing gas saturation Sg = β × (1 - Sw).

        With the direct beta_to_flowing_saturations mapping this reduces to a
        vectorised operation: flash each cell (using the cache if warm) and
        multiply the vapour fraction by the hydrocarbon pore space.  We do NOT
        invalidate the flash cache here — the cache is keyed by state id and
        will already be valid if this state was just used in transport.
        """
        if self._cache_state_id != id(state):
            self._set_cache_state(state)
        sg_profile = np.empty(self.grid.nx)
        for i in range(self.grid.nx):
            fl   = self.cell_flash_cached(state, i)
            beta = float(fl["beta"])
            Sw   = float(np.clip(state.sw[i], 0.0, 0.95))
            Shc  = max(1.0 - Sw, 0.0)
            Sg_raw = beta * Shc
            total  = Sg_raw + (1.0 - beta) * Shc + Sw
            sg_profile[i] = Sg_raw / total if total > 1e-12 else 0.0
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
        mu_o = pvt_props.get('oil_viscosity_cp') if pvt_props.get('oil_viscosity_cp') is not None else phase_viscosity_cp("l", x, state.pressure[i], self.T, self.Tc, self.Pc, self.mw_components)
        mu_g = pvt_props.get('gas_viscosity_cp') if pvt_props.get('gas_viscosity_cp') is not None else phase_viscosity_cp("v", y, state.pressure[i], self.T, self.Tc, self.Pc, self.mw_components)
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
        upstream_vel = i if dp >= 0.0 else j
        _p_v = max(float(state.pressure[upstream_vel]), 14.7)
        try:
            _Z_v = float(self.eos.z_factor(state.z[upstream_vel], _p_v, self.T, phase="v"))
            _Z_v = max(_Z_v, 0.05)
        except Exception:
            _Z_v = 0.85
        rho_v = _p_v / max(_Z_v * R * self.T, 1e-12) * FT3_PER_STB
        q_hc = 0.00708 * rho_v * trans * lam_face * dp
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

        # Darcy flux in field units: q [lbmol/day] = 0.00708 * T [md*ft/cp] * dP [psi] * rho [lbmol/RB]
        # T does NOT include the 0.00708 constant (that lives inside peaceman_well_index).
        # rho is evaluated at upstream conditions for the molar conversion.
        _p_up = max(float(state.pressure[upstream]), 14.7)
        try:
            _Z_up = float(self.eos.z_factor(state.z[upstream], _p_up, self.T, phase="v"))
            _Z_up = max(_Z_up, 0.05)
        except Exception:
            _Z_up = 0.85
        rho_up = _p_up / max(_Z_up * R * self.T, 1e-12) * FT3_PER_STB  # lbmol/RB
        _DARCY = 0.00708

        gas_advective = np.zeros(self.nc)
        oil_advective = np.zeros(self.nc)
        if self.transport.phase_split_advection:
            qg = _DARCY * rho_up * trans * lam_g * dp
            qo = _DARCY * rho_up * trans * lam_o * dp
            gas_advective = qg * beta_up * y_up
            oil_advective = qo * (1.0 - beta_up) * x_up
        else:
            q_total = _DARCY * rho_up * trans * (lam_g + lam_o) * dp
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
            # Proper Fickian dispersion: J = -D * phi * A * grad(z)  [lbmol/day]
            # grad(z) approximated as (z_dn - z_up) / distance
            grad_z = (z_dn - z_up) / max(distance, 1e-12)
            dispersive_total = -dispersion_coeff * self.phi * self.grid.area_ft2 * grad_z
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

        upstream_w = i if dp >= 0.0 else j
        mob = self.phase_mobility_data(state, upstream_w)
        lam_w = mob["lam_w"]

        distance = self.grid.interface_distance(i, j)
        trans = self.k * self.grid.area_ft2 / max(distance, 1e-12)
        # Water: use liquid-water molar density (incompressible approximation)
        # rho_w [lbmol/RB] = MW_WATER / (WATER_DENSITY_LBFT3 * FT3_PER_STB)
        rho_w = MW_WATER / (WATER_DENSITY_LBFT3 * FT3_PER_STB)
        qw = 0.00708 * rho_w * trans * lam_w * dp
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

    def _compute_peaceman_wi(self, w: "Well") -> float:
        """Compute Peaceman WI for a given Well object. Result is static — call once and cache."""
        i   = w.cell_index
        dx  = self.grid.cell_width(i)
        dy  = self.grid.width_ft
        h   = self.grid.thickness_ft
        rw  = max(w.rw_ft, 1e-4)
        skin = w.skin
        re  = 0.28 * math.sqrt(dx * dx + dy * dy) / ((dy / dx) ** 0.5 + (dx / dy) ** 0.5)
        re  = max(re, 1.01 * rw)
        wi_geom = 0.00708 * self.k * h / max(math.log(re / rw) + skin, 1e-8)
        return w.productivity_index * wi_geom

    def peaceman_well_index(self, i: int) -> float:
        """Return cached Peaceman WI for the current self.well at cell i."""
        key = id(self.well)
        if key not in self._well_wi:
            self._well_wi[key] = self._compute_peaceman_wi(self.well)
        return self._well_wi[key]

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
        # Use the user-configured minimum BHP as the physical floor,
        # not a hard-coded 50 psia which ignores well-specific constraints.
        min_pwf = float(getattr(self.well, "min_bhp_psia", 50.0))
        pwf = float(np.clip(pwf_psia, min_pwf, pr))
        dp = max(pr - pwf, 0.0)

        # Evaluate the producing-cell flash at flowing bottomhole pressure,
        # not just at the reservoir-cell pressure. This keeps the wellstream
        # split responsive to drawdown and depletion.
        fl = self.flash.flash(state.z[i], pwf, self.T)
        beta = float(fl["beta"])
        x = np.asarray(fl["x"])
        y = np.asarray(fl["y"])
        self.last_produced_gas_composition = y.copy()

        mu_o = phase_viscosity_cp("l", x, pwf, self.T, self.Tc, self.Pc, self.mw_components)
        mu_g = phase_viscosity_cp("v", y, pwf, self.T, self.Tc, self.Pc, self.mw_components)
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

        # Correct unit conversion for the Peaceman rate equation.
        # WI from _compute_peaceman_wi is in [md·ft] (the 0.00708 constant in
        # field units).  The Darcy rate equation in field units is:
        #   q [RB/day] = WI [md·ft] × lam_t [kr/cp] × dp [psi]
        # Converting RB/day to lbmol/day requires the in-situ molar density:
        #   ρ [lbmol/RB] = P / (Z · R · T) × 5.614583
        # where R = 10.7316 psia·ft³/(lbmol·R).
        # The previous factor of 1e-2 was ~400× too small, causing the well to
        # deliver ~400× less than a correct Darcy calculation would predict.
        Z_well = self.eos.z_factor(state.z[i], pr, self.T, phase="v")
        Z_well = max(float(Z_well), 0.05)
        rho_lbmol_RB = max(pr, 14.7) / max(Z_well * R * self.T, 1e-12) * 5.614583
        darcy_conv = rho_lbmol_RB  # lbmol/RB  (replaces the old 1e-2)

        q_undamaged = wi_geom * lam_t * dp * darcy_conv

        # Evaluate dropout at the near-wellbore pressure, which is the geometric
        # mean of BHP and reservoir pressure.  This represents conditions in the
        # condensate bank region (Fevang & Whitson Region 1/2 boundary) rather
        # than at either extreme.  Only computed when there is meaningful drawdown.
        if dp > 1.0:
            p_nearwell = math.sqrt(max(pwf, 14.7) * max(pr, 14.7))
            fl_nearwell = self.flash.flash(state.z[i], p_nearwell, self.T)
            beta_nearwell = float(fl_nearwell["beta"])
        else:
            beta_nearwell = beta
        dropout = liquid_dropout_fraction(beta_nearwell)
        damage_factor = condensate_bank_damage_factor(dropout, self.relperm)
        wi_eff = wi_geom * damage_factor
        q_total = wi_eff * lam_t * dp * darcy_conv

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

    def estimate_mixture_density_lbft3(
        self,
        state: SimulationState,
        response: Dict[str, float | np.ndarray],
        pressure_psia: float | None = None,
    ) -> float:
        """In-situ mixture density at a given tubing pressure.

        Uses the real-gas law for gas density (ρ_g = P·Mg / (Z·R·T)) evaluated
        at the supplied pressure rather than surface or reservoir densities.
        Liquid densities are treated as incompressible (reasonable for condensate
        and water over typical wellbore conditions).

        Parameters
        ----------
        pressure_psia : if None, uses the well-cell reservoir pressure.
                        For hydrostatic calculations pass the average tubing
                        pressure (arithmetic mean of BHP and THP).
        """
        qg = float(response["qg_lbmol_day"])
        qo = float(response["qo_lbmol_day"])
        qw = float(response["qw_lbmol_day"])

        total = qg + qo + qw
        if total <= 1e-12:
            return 0.1

        p_eval = float(pressure_psia) if pressure_psia is not None else float(state.pressure[self.well.cell_index])
        p_eval = max(p_eval, 14.7)

        # Gas: real-gas law density at tubing conditions.
        # ρ_g = P · Mg / (Z · R · T)  [lb/ft³]
        # Mg (gas molecular weight) from mole-fraction-weighted composition.
        y = np.asarray(response.get("y", np.zeros(self.nc)), dtype=float)
        Mg = float(np.dot(y, self.mw_components)) if float(np.sum(y)) > 1e-6 else float(np.dot(np.ones(self.nc)/self.nc, self.mw_components))
        # Z-factor: try flash, fall back to PVT table, then ideal-gas approximation
        try:
            Z_gas = self.eos.z_factor(y / max(float(np.sum(y)), 1e-12), p_eval, self.T, phase="v")
            Z_gas = max(Z_gas, 0.1)
        except Exception:
            pvt_props = self.pvt_lookup(p_eval)
            Z_gas = pvt_props.get("z_factor") or max(p_eval * 0.85 / max(p_eval, 1.0), 0.1)
        rho_gas = p_eval * Mg / max(Z_gas * R * self.T, 1e-12)  # lb/ft³

        # Liquid densities: treat as incompressible (pressure-independent)
        rho_cond  = self.reporting.condensate_density_lbft3
        rho_water = WATER_DENSITY_LBFT3

        # Molar masses for converting lbmol/day to mass flow
        x = np.asarray(response.get("x", np.zeros(self.nc)), dtype=float)
        Mo = float(np.dot(x, self.mw_components)) if float(np.sum(x)) > 1e-6 else rho_cond * FT3_PER_STB / max(1.0, 1.0)
        Mw = MW_WATER

        # Volume-weighted mixture density at in-situ conditions
        vol_gas   = qg * Z_gas * R * self.T / max(p_eval, 1e-12)  # ft³/day (real-gas)
        vol_cond  = qo * Mo / max(rho_cond, 1e-12)                 # ft³/day
        vol_water = qw * Mw / max(rho_water, 1e-12)                # ft³/day
        vol_total = vol_gas + vol_cond + vol_water

        if vol_total <= 1e-12:
            return 0.1

        rho_mix = (qg * Mg + qo * Mo + qw * Mw) / vol_total
        return float(np.clip(rho_mix, 0.01, 62.4))

    def tubing_pressure_profile_from_response(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> Dict[str, float]:
        """Compute THP from BHP using a single-pass vertical lift model.

        Fixes applied vs the previous version
        --------------------------------------
        1. In-situ gas density and velocity use the real-gas law at average
           tubing pressure rather than surface volumetric rates.
        2. Friction uses Darcy-Weisbach only (mechanistic model) or the simple
           quadratic coefficient only (simple model) — no double-counting.
        3. Acceleration pressure drop is computed from the actual kinetic-energy
           change between BHP and an estimated THP, not as an arbitrary fraction
           of friction.
        4. Hydrostatic head uses in-situ mixture density at the average tubing
           pressure (average of BHP and atmospheric THP estimate) rather than
           reservoir-condition density.
        """
        pwf = float(response["pwf_psia"])
        qg_lbmol_day = float(response["qg_lbmol_day"])
        qo_lbmol_day = float(response["qo_lbmol_day"])
        qw_lbmol_day = float(response["qw_lbmol_day"])

        gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(qg_lbmol_day)
        liquid_rate_stb_day = (
            liquid_lbmol_day_to_stb_day(qo_lbmol_day, np.asarray(response["x"]), self.mw_components, self.reporting.condensate_density_lbft3)
            + water_lbmol_day_to_stb_day(qw_lbmol_day)
        )

        # ── Estimate average tubing pressure for in-situ property evaluation ──
        # First pass: rough hydrostatic with surface-condition density to get a
        # provisional THP, then average BHP and provisional THP.
        rho_mix_bh = self.estimate_mixture_density_lbft3(state, response, pressure_psia=pwf)
        hydrostatic_rough = 0.006944444444444444 * rho_mix_bh * self.well.tvd_ft
        thp_provisional = max(pwf - hydrostatic_rough, 14.7)
        p_avg_tubing = 0.5 * (pwf + thp_provisional)

        # Mixture density at average tubing pressure (physically correct for hydrostatic)
        rho_mix = self.estimate_mixture_density_lbft3(state, response, pressure_psia=p_avg_tubing)
        hydrostatic_psi = 0.006944444444444444 * rho_mix * self.well.tvd_ft

        if self.well.tubing_model == "simple":
            # Simple quadratic model — friction coefficient set by user
            friction_psi   = self.well.thp_friction_coeff * (gas_rate_mmscf_day + 0.001 * liquid_rate_stb_day) ** 2
            accel_psi      = 0.0
            velocity_ft_s  = 0.0
            reynolds       = 0.0
            friction_factor = float(self.well.thp_friction_coeff)

        else:
            # Mechanistic Darcy-Weisbach model using in-situ volumetric rates
            area_ft2   = math.pi * (max(self.well.tubing_id_in, 0.25) / 12.0) ** 2 / 4.0
            diameter_ft = max(self.well.tubing_id_in / 12.0, 1e-6)

            # In-situ gas volume at average tubing conditions (real-gas law)
            y = np.asarray(response.get("y", np.zeros(self.nc)), dtype=float)
            Mg = float(np.dot(y, self.mw_components)) if float(np.sum(y)) > 1e-6 else 20.0
            try:
                Z_avg = self.eos.z_factor(y / max(float(np.sum(y)), 1e-12), p_avg_tubing, self.T, phase="v")
                Z_avg = max(Z_avg, 0.1)
            except Exception:
                Z_avg = 0.85
            # ft³/day at average tubing pressure
            gas_vol_insitu_ft3_day  = qg_lbmol_day * Z_avg * R * self.T / max(p_avg_tubing, 1e-12)

            x = np.asarray(response.get("x", np.zeros(self.nc)), dtype=float)
            rho_cond  = self.reporting.condensate_density_lbft3
            rho_water = WATER_DENSITY_LBFT3
            Mo = float(np.dot(x, self.mw_components)) if float(np.sum(x)) > 1e-6 else 100.0
            liq_vol_insitu_ft3_day  = (qo_lbmol_day * Mo / max(rho_cond, 1e-12)
                                       + qw_lbmol_day * MW_WATER / max(rho_water, 1e-12))
            mix_vol_insitu_ft3_day  = gas_vol_insitu_ft3_day + liq_vol_insitu_ft3_day
            velocity_ft_s = mix_vol_insitu_ft3_day / max(area_ft2 * 86400.0, 1e-12)

            # Mixture viscosity at average tubing pressure
            gas_frac   = float(response.get("gas_frac", 0.0))
            oil_frac   = float(response.get("oil_frac", 0.0))
            water_frac = float(response.get("water_frac", 0.0))
            mu_mix_cp  = (
                gas_frac   * phase_viscosity_cp("v", y, p_avg_tubing, self.T, self.Tc, self.Pc, self.mw_components)
                + oil_frac * phase_viscosity_cp("l", x, p_avg_tubing, self.T, self.Tc, self.Pc, self.mw_components)
                + water_frac * WATER_VISCOSITY_CP
            )
            mu_mix_lb_ft_s = max(mu_mix_cp, 1e-6) * 0.000671968975

            reynolds = rho_mix * max(velocity_ft_s, 0.0) * diameter_ft / max(mu_mix_lb_ft_s, 1e-12)
            roughness_ft  = max(self.well.tubing_roughness_in, 0.0) / 12.0
            rel_roughness = roughness_ft / max(diameter_ft, 1e-12)

            if reynolds < 1e-8:
                friction_factor = 0.0
            elif reynolds < 2300.0:
                friction_factor = 64.0 / reynolds
            else:
                # Swamee-Jain approximation to Colebrook-White (valid Re > 5000)
                # falls back gracefully in transition zone
                arg = rel_roughness / 3.7 + 5.74 / max(reynolds ** 0.9, 1e-12)
                friction_factor = 0.25 / max(math.log10(max(arg, 1e-12)) ** 2, 1e-12)

            friction_grad_psi_ft = (friction_factor * rho_mix * max(velocity_ft_s, 0.0) ** 2
                                    / max(2.0 * 32.174 * diameter_ft * 144.0, 1e-12))
            # Friction: Darcy-Weisbach only, scaled by calibration factor.
            # The simple-model quadratic term is NOT added here (that would double-count).
            friction_psi = self.well.tubing_calibration_factor * friction_grad_psi_ft * self.well.tvd_ft

            # Acceleration pressure drop: kinetic-energy change from BHP to THP.
            # Δp_acc = ρ_mix · (v_top² - v_bottom²) / (2·g_c·144)
            # Gas expands toward surface so v_top > v_bottom; use ratio of densities.
            rho_mix_top = self.estimate_mixture_density_lbft3(state, response, pressure_psia=max(thp_provisional, 14.7))
            rho_ratio = rho_mix / max(rho_mix_top, 1e-3)   # > 1 (denser at bottom)
            v_top = velocity_ft_s * rho_ratio               # conservation of mass
            accel_psi = rho_mix * max(v_top ** 2 - velocity_ft_s ** 2, 0.0) / max(2.0 * 32.174 * 144.0, 1e-12)
            accel_psi = float(np.clip(accel_psi, 0.0, 0.05 * friction_psi))  # sanity cap at 5% of friction

        thp = pwf - hydrostatic_psi - friction_psi - accel_psi
        return {
            "estimated_thp_psia":            float(max(thp, 14.7)),
            "raw_estimated_thp_psia":        float(thp),
            "tubing_hydrostatic_psi":        float(max(hydrostatic_psi, 0.0)),
            "tubing_friction_psi":           float(max(friction_psi, 0.0)),
            "tubing_acceleration_psi":       float(max(accel_psi, 0.0)),
            "tubing_mixture_velocity_ft_s":  float(max(velocity_ft_s, 0.0)),
            "tubing_reynolds_number":        float(max(reynolds, 0.0)),
            "tubing_friction_factor":        float(max(friction_factor, 0.0)),
        }

    def tubing_head_pressure_from_response(self, state: SimulationState, response: Dict[str, float | np.ndarray]) -> float:
        return float(self.tubing_pressure_profile_from_response(state, response)["estimated_thp_psia"])

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
        # pwf_max_dp gives the highest drawdown (lowest BHP) → highest rate
        # pwf_min_dp gives zero drawdown → near-zero rate
        # Variable names now reflect which end of the rate range each represents.
        min_bhp = max(float(getattr(self.well, "min_bhp_psia", 50.0)), 14.7)
        pwf_max_dp = min_bhp
        pwf_min_dp = max(pr - 1e-6, pwf_max_dp + 1e-6)

        resp_max_dp = self.well_response_at_pwf(state, pwf_max_dp)  # highest achievable rate
        resp_min_dp = self.well_response_at_pwf(state, pwf_min_dp)  # near-zero rate

        qg_max = float(resp_max_dp["qg_lbmol_day"])
        qg_min = float(resp_min_dp["qg_lbmol_day"])

        # Target above maximum achievable → constrain to max drawdown
        if target_qg_lbmol_day >= qg_max:
            return resp_max_dp
        # Target at or below minimum → near-zero drawdown
        if target_qg_lbmol_day <= qg_min:
            return resp_min_dp

        # Sample the rate curve to detect non-monotonicity (condensate bank damage
        # and phase behaviour mean qg is not always monotone with PWF).
        n_probe = 20
        probe_pwfs = np.linspace(pwf_max_dp, pwf_min_dp, n_probe + 2)[1:-1]
        probe_pairs = []
        for p in probe_pwfs:
            r = self.well_response_at_pwf(state, float(p))
            probe_pairs.append((float(p), float(r["qg_lbmol_day"]), r))

        # Find the probe pair that brackets the target (if any)
        # and return the response with the smallest error
        best_resp = resp_max_dp
        best_err = abs(float(resp_max_dp["qg_lbmol_day"]) - target_qg_lbmol_day)

        for _, qg_p, r_p in probe_pairs:
            err = abs(qg_p - target_qg_lbmol_day)
            if err < best_err:
                best_err = err
                best_resp = r_p

        # If we're already within tolerance, return immediately
        if gas_lbmol_day_to_mmscf_day(best_err) < tol_mmscf_day:
            return best_resp

        # Bisect around the best probe point for fine convergence
        # Only bisect if rate is locally monotone around that point
        best_idx = min(range(len(probe_pairs)), key=lambda j: abs(probe_pairs[j][1] - target_qg_lbmol_day))
        lo_idx = max(best_idx - 1, 0)
        hi_idx = min(best_idx + 1, len(probe_pairs) - 1)
        p_a, qg_a = probe_pairs[lo_idx][0], probe_pairs[lo_idx][1]
        p_b, qg_b = probe_pairs[hi_idx][0], probe_pairs[hi_idx][1]

        # Ensure the target is bracketed and the function is monotone in this interval
        if (qg_a - target_qg_lbmol_day) * (qg_b - target_qg_lbmol_day) < 0:
            for _ in range(max_iter):
                p_mid = 0.5 * (p_a + p_b)
                r_mid = self.well_response_at_pwf(state, p_mid)
                qg_mid = float(r_mid["qg_lbmol_day"])
                if gas_lbmol_day_to_mmscf_day(abs(qg_mid - target_qg_lbmol_day)) < tol_mmscf_day:
                    return r_mid
                if (qg_a - target_qg_lbmol_day) * (qg_mid - target_qg_lbmol_day) < 0:
                    p_b, qg_b = p_mid, qg_mid
                else:
                    p_a, qg_a = p_mid, qg_mid
                best_resp = r_mid

        return best_resp

    def solve_thp_control(self, state: SimulationState, tol_psia: float = 1.0, max_iter: int = 60) -> Dict[str, float | np.ndarray]:
        target_thp = float(self.well.thp_psia)
        i = self.well.cell_index
        pr = float(state.pressure[i])

        min_bhp = max(float(getattr(self.well, "min_bhp_psia", 50.0)), 14.7)
        pwf_max_dp = min_bhp
        pwf_min_dp = max(pr - 1e-6, pwf_max_dp + 1e-6)

        resp_max_dp = self.well_response_at_pwf(state, pwf_max_dp)
        resp_min_dp = self.well_response_at_pwf(state, pwf_min_dp)

        profile_max_dp = self.tubing_pressure_profile_from_response(state, resp_max_dp)
        profile_min_dp = self.tubing_pressure_profile_from_response(state, resp_min_dp)
        thp_max_dp = float(profile_max_dp["estimated_thp_psia"])  # THP at max drawdown
        thp_min_dp = float(profile_min_dp["estimated_thp_psia"])  # THP at min drawdown

        # THP is normally monotonically increasing with PWF (less friction at low rates).
        # If THP at max drawdown is already above target, constrain to that end.
        if target_thp <= thp_max_dp:
            resp_max_dp.update(profile_max_dp)
            return resp_max_dp
        if target_thp >= thp_min_dp:
            resp_min_dp.update(profile_min_dp)
            return resp_min_dp

        # Detect non-monotonic THP (liquid loading regime): sample a few interior
        # points and fall back to the response giving the closest THP if the
        # relationship is not monotone.
        n_probe = 6
        probe_pwfs = np.linspace(pwf_max_dp, pwf_min_dp, n_probe + 2)[1:-1]
        probe_thps = []
        for p in probe_pwfs:
            r = self.well_response_at_pwf(state, float(p))
            pf = self.tubing_pressure_profile_from_response(state, r)
            probe_thps.append((float(p), float(pf["estimated_thp_psia"]), r, pf))

        # Check monotonicity
        thp_vals = [t for _, t, _, _ in probe_thps]
        is_monotone = all(thp_vals[j] >= thp_vals[j-1] for j in range(1, len(thp_vals)))

        if not is_monotone:
            # Non-monotonic: pick the probe point whose THP is closest to target
            best_probe = min(probe_thps, key=lambda item: abs(item[1] - target_thp))
            best_r, best_pf = best_probe[2], best_probe[3]
            best_r.update(best_pf)
            return best_r

        # Monotone: bisect in PWF space
        pwf_lo = pwf_max_dp
        pwf_hi = pwf_min_dp
        best = resp_min_dp
        best.update(profile_min_dp)
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
                pwf_lo = pwf_mid   # increase drawdown to raise THP
            else:
                pwf_hi = pwf_mid   # reduce drawdown to lower THP

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
            response["control_mode"] = "gas_rate"
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
        """Compute surface gas and condensate rates from the well response.

        Condensate rate priority
        ------------------------
        1. PVT table CGR (vaporized_cgr or reservoir_cgr column).
           When a CGR column is present in the uploaded PVT table it is applied
           directly:  condensate_rate = CGR_table(P_res) × gas_rate_MMscfd.
           This takes priority because the PVT table comes from a tuned study.

        2. Wellstream liquid rate (default when no PVT table).
           The wellstream from the well model already carries qo (lbmol/day of
           liquid-phase hydrocarbons) evaluated at BHP by the PR-EOS.  Converting
           qo to STB/day using the liquid composition x and the user API-gravity
           density gives a physically consistent condensate rate that is NOT
           subject to the EOS extrapolation error at cold separator conditions.
           condensate_rate = liquid_lbmol_day_to_stb_day(qo, x, mw, rho_API)

           The EOS separator flash is still run (to get the sales-gas split and
           stage-by-stage gas volumes) but its stock-tank liquid volume is not
           used as the primary condensate number.

        Why not the EOS separator flash for condensate?
        ------------------------------------------------
        The PR-EOS is tuned to reservoir temperature (~220°F here).  At separator
        conditions (74°F, 300 psia) it over-volatilises the heavier C5-C7+
        pseudocomponents, predicting too little liquid dropout.  The wellstream
        qo route avoids this extrapolation because the EOS is evaluated at the
        more reliable BHP conditions, and the liquid composition x at BHP is
        close to the stock-tank composition for a lean gas condensate.

        Reported diagnostic fields
        --------------------------
        condensate_reporting_basis      : 'pvt_table_cgr' | 'wellstream_qo'
        wellstream_cgr_stb_mmscf        : CGR from wellstream qo route
        eos_separator_cgr_stb_mmscf     : CGR from EOS separator train (diagnostic)
        pvt_table_cgr_stb_mmscf        : CGR from PVT table (nan if absent)
        effective_cgr_stb_mmscf        : CGR actually used for reporting
        """
        sep = self.separator_flash(response)
        q_sep_gas_lbmol_day = float(sep["total_separator_gas_lbmol_day"])
        q_sep_liq_lbmol_day = float(sep["stock_tank_liquid_lbmol_day"])
        q_w_lbmol_day       = float(response.get("qw_lbmol_day", 0.0))
        x_sep               = np.asarray(sep["x_sep"], dtype=float)

        # Reported gas rate
        qg_report_lbmol_day = max(float(response.get("qg_lbmol_day", q_sep_gas_lbmol_day)), 0.0)
        qo_report_lbmol_day = max(float(response.get("qo_lbmol_day", q_sep_liq_lbmol_day)), 0.0)
        x_report = np.asarray(response.get("x", x_sep), dtype=float)
        if x_report.ndim != 1 or x_report.size == 0 or float(np.sum(x_report)) <= 0.0:
            x_report = x_sep.copy()
        x_report = x_report / np.sum(x_report)
        gas_rate_mmscf_day = gas_lbmol_day_to_mmscf_day(qg_report_lbmol_day)

        # ── Route A: EOS separator train → STB/day ──────────────────────────
        # The wellstream is flashed at separator conditions. This is the
        # primary route for single-phase gas (beta=1 at reservoir), where
        # qo=0 but the separator still produces some condensate.
        eos_sep_stb_day = liquid_lbmol_day_to_stb_day(
            q_sep_liq_lbmol_day,
            x_sep,
            self.mw_components,
            self.reporting.condensate_density_lbft3,
        )
        eos_sep_cgr = eos_sep_stb_day / max(gas_rate_mmscf_day, 1e-12)

        # ── Route B: wellstream qo → STB/day ────────────────────────────────
        # When the reservoir is genuinely two-phase (below dew point), qo > 0
        # and represents the liquid-phase molar flow at BHP. Converting it to
        # STB/day using the liquid composition x at BHP gives a condensate
        # estimate that avoids the EOS extrapolation error at cold separator
        # conditions. When the reservoir is single-phase gas, qo=0 and this
        # route contributes nothing — we fall back entirely to the separator.
        wellstream_stb_day = liquid_lbmol_day_to_stb_day(
            qo_report_lbmol_day,
            x_report,
            self.mw_components,
            self.reporting.condensate_density_lbft3,
        )
        wellstream_cgr = wellstream_stb_day / max(gas_rate_mmscf_day, 1e-12)

        # Take the larger of the two EOS-based estimates.
        # - Above dew point: qo=0, separator governs.
        # - Below dew point: wellstream qo is large, typically exceeds
        #   the separator train which under-predicts liquid at low temperature.
        if wellstream_stb_day >= eos_sep_stb_day:
            default_stb_day = wellstream_stb_day
            default_cgr     = wellstream_cgr
            default_basis   = "wellstream_qo"
        else:
            default_stb_day = eos_sep_stb_day
            default_cgr     = eos_sep_cgr
            default_basis   = "eos_separator_train"

        # ── Route C: PVT table CGR (highest priority when available) ────────
        pressure_for_cgr = float(response.get("pr_psia", response.get("pwf_psia", self.separator.pressure_psia)))
        pvt_props        = self.pvt_lookup(pressure_for_cgr)
        cgr_table        = pvt_props.get("vaporized_cgr_stb_mmscf") or pvt_props.get("reservoir_cgr_stb_mmscf")
        pvt_cgr          = float(cgr_table) if cgr_table is not None and np.isfinite(float(cgr_table)) and float(cgr_table) > 0.0 else float("nan")
        pvt_cgr_valid    = np.isfinite(pvt_cgr)

        # ── Select condensate rate ────────────────────────────────────────────
        if gas_rate_mmscf_day <= 1e-12:
            condensate_rate_stb_day    = 0.0
            effective_cgr_stb_mmscf   = 0.0
            condensate_reporting_basis = "zero_gas"
        elif pvt_cgr_valid:
            effective_cgr_stb_mmscf   = pvt_cgr
            condensate_rate_stb_day   = pvt_cgr * gas_rate_mmscf_day
            condensate_reporting_basis = "pvt_table_cgr"
        else:
            effective_cgr_stb_mmscf   = default_cgr
            condensate_rate_stb_day   = default_stb_day
            condensate_reporting_basis = default_basis

        # ── Surface fractions ─────────────────────────────────────────────────
        total_surface      = max(qg_report_lbmol_day + qo_report_lbmol_day + q_w_lbmol_day, 1e-12)
        surface_gas_frac   = qg_report_lbmol_day / total_surface
        surface_oil_frac   = qo_report_lbmol_day / total_surface
        surface_water_frac = q_w_lbmol_day        / total_surface

        stages = sep.get("stages", [])
        stage_gas_rates = {
            s.get("label", f"stage{i+1}"): gas_lbmol_day_to_mmscf_day(float(s.get("q_gas_lbmol_day", 0.0)))
            for i, s in enumerate(stages)
        }

        return {
            "gas_rate_mmscf_day":                      float(gas_rate_mmscf_day),
            "condensate_rate_stb_day":                 float(condensate_rate_stb_day),
            "separator_vapor_fraction":                float(sep["beta_sep"]),
            "surface_gas_fraction":                    float(surface_gas_frac),
            "surface_oil_fraction":                    float(surface_oil_frac),
            "surface_water_fraction":                  float(surface_water_frac),
            # Diagnostics
            "effective_cgr_stb_mmscf":                 float(effective_cgr_stb_mmscf),
            "wellstream_cgr_stb_mmscf":                float(wellstream_cgr),
            "eos_separator_cgr_stb_mmscf":             float(eos_sep_cgr),
            "pvt_table_cgr_stb_mmscf":                 float(pvt_cgr) if pvt_cgr_valid else float("nan"),
            "condensate_reporting_basis":              condensate_reporting_basis,
            # Per-stage gas rates
            "raw_separator_gas_rate_mmscf_day":        float(gas_lbmol_day_to_mmscf_day(q_sep_gas_lbmol_day)),
            "total_separator_gas_rate_mmscf_day":      float(gas_lbmol_day_to_mmscf_day(q_sep_gas_lbmol_day)),
            "stock_tank_condensate_rate_stb_day":      float(eos_sep_stb_day),
            "separator_stage1_gas_rate_mmscf_day":     float(stage_gas_rates.get("stage1", 0.0)),
            "separator_stage2_gas_rate_mmscf_day":     float(stage_gas_rates.get("stage2", 0.0)),
            "separator_stage3_gas_rate_mmscf_day":     float(stage_gas_rates.get("stage3", 0.0)),
            "separator_stock_tank_gas_rate_mmscf_day": float(stage_gas_rates.get("stock_tank", 0.0)),
            "separator_stage_count":                   float(sep.get("stage_count", 1)),
        }

    def reported_well_response(self, state: SimulationState) -> Dict[str, float | np.ndarray]:
        response = self.solve_well_control(state)
        response = dict(response)

        if self.well.control_mode == "gas_rate":
            response["reported_gas_rate_mmscf_day"] = float(self.well.target_gas_rate_mmscf_day)
        else:
            response["reported_gas_rate_mmscf_day"] = gas_lbmol_day_to_mmscf_day(float(response["qg_lbmol_day"]))

        return response

    def well_sink(self, state: SimulationState) -> WellSinkResult:
        response = self.solve_well_control(state)
        return WellSinkResult(
            hc_sink=np.asarray(response["hc_sink"]),
            qw_lbmol_day=float(response["qw_lbmol_day"]),
            q_total_lbmol_day=float(response["q_total_lbmol_day"]),
            gas_frac=float(response["gas_frac"]),
            oil_frac=float(response["oil_frac"]),
            water_frac=float(response["water_frac"]),
            dropout=float(response["dropout"]),
            krg=float(response["krg"]),
            kro=float(response["kro"]),
            krw=float(response["krw"]),
            damage_factor=float(response["damage_factor"]),
            wi_eff=float(response["wi_eff"]),
            q_undamaged=float(response["q_undamaged"]),
            ploss=float(response["ploss"]),
        )

    def well_sinks_all(self, state: SimulationState) -> Tuple[np.ndarray, float, Dict[str, float], List[Dict]]:
        """Aggregate sink contributions from every producer in self.wells.

        Returns
        -------
        agg_hc_sink  : ndarray (nx, nc) — total HC moles removed per cell per day
        agg_qw       : float            — total water lbmol/day across all wells
        agg_scalars  : dict             — production-weighted aggregate diagnostics
        per_well     : list of dicts    — per-well breakdown for reporting
        """
        nx, nc = self.grid.nx, self.nc
        agg_hc_sink = np.zeros((nx, nc), dtype=float)
        agg_qw = agg_q_total = agg_qg = agg_qo = 0.0
        agg_dropout = agg_krg = agg_kro = agg_krw = 0.0
        agg_damage = agg_wi_eff = agg_q_undamaged = agg_ploss = 0.0
        per_well: List[Dict] = []
        n_wells = len(self.wells)

        # Temporarily swap self.well to each producer so that solve_well_control
        # (which reads self.well) operates correctly for each one in turn.
        orig_well = self.well
        for w in self.wells:
            self.well = w
            self._mobility_cache = {}   # invalidate per-well so each well gets its own cell's data
            ws = self.well_sink(state)
            wi = w.cell_index
            agg_hc_sink[wi] += np.asarray(ws.hc_sink, dtype=float)
            agg_qw          += ws.qw_lbmol_day
            agg_q_total     += ws.q_total_lbmol_day
            agg_qg          += ws.q_total_lbmol_day * ws.gas_frac
            agg_qo          += ws.q_total_lbmol_day * ws.oil_frac
            agg_dropout     += ws.dropout
            agg_krg         += ws.krg
            agg_kro         += ws.kro
            agg_krw         += ws.krw
            agg_damage      += ws.damage_factor
            agg_wi_eff      += ws.wi_eff
            agg_q_undamaged += ws.q_undamaged
            agg_ploss       += ws.ploss
            per_well.append({
                "cell_index":          wi,
                "name":                getattr(w, "name", f"Well {len(per_well)+1}"),
                "q_total_lbmol_day":   ws.q_total_lbmol_day,
                "qg_lbmol_day":        ws.q_total_lbmol_day * ws.gas_frac,
                "qo_lbmol_day":        ws.q_total_lbmol_day * ws.oil_frac,
                "qw_lbmol_day":        ws.qw_lbmol_day,
                "gas_frac":            ws.gas_frac,
                "oil_frac":            ws.oil_frac,
                "water_frac":          ws.water_frac,
                "dropout":             ws.dropout,
                "krg":                 ws.krg,
                "kro":                 ws.kro,
                "krw":                 ws.krw,
                "damage_factor":       ws.damage_factor,
                "wi_eff":              ws.wi_eff,
                "q_undamaged":         ws.q_undamaged,
                "ploss":               ws.ploss,
                "pressure_psia":       float(state.pressure[wi]),
            })

        # Restore primary well alias
        self.well = orig_well
        self._mobility_cache = {}

        denom_wells = max(n_wells, 1)
        agg_scalars = {
            "q_total_lbmol_day": agg_q_total,
            "qw_lbmol_day":      agg_qw,
            "gas_frac":          agg_qg / max(agg_q_total, 1e-12),
            "oil_frac":          agg_qo / max(agg_q_total, 1e-12),
            "water_frac":        agg_qw / max(agg_q_total + agg_qw, 1e-12),
            "dropout":           agg_dropout  / denom_wells,
            "krg":               agg_krg      / denom_wells,
            "kro":               agg_kro      / denom_wells,
            "krw":               agg_krw      / denom_wells,
            "damage_factor":     agg_damage   / denom_wells,
            "wi_eff":            agg_wi_eff   / denom_wells,
            "q_undamaged":       agg_q_undamaged,
            "ploss":             agg_ploss    / denom_wells,
        }
        return agg_hc_sink, agg_qw, agg_scalars, per_well

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

        # ── Per-cell molar density and real-gas compressibility ───────────────
        # rho_i = P_i / (Z_i * R * T) * 5.614583  [lbmol/RB]
        # cg_i  = 1 / P_i                           [1/psi, ideal-gas approx]
        # ct_i  = cg_i + cr                         [1/psi, total compressibility]
        #
        # The ideal-gas approximation to cg is exact for Z=1 and accurate to
        # within ~15% for typical gas condensate pressures. The full EOS form
        # cg = 1/P - (1/Z)*(dZ/dP) requires numerical dZ/dP which is expensive
        # for every cell every timestep.
        cr = getattr(self.rock, "compressibility_1_psi", 3.0e-6)
        rho_cell = np.empty(n)
        acc_scale = np.empty(n)
        # Use self.pv — the corrected per-cell pore volume (π×r²×h×φ distributed
        # by dx fractions).  Do NOT recompute from grid.area_ft2 × dx_array here,
        # because area_ft2 = π×r² and dx_array sums to length_ft (the radial flow
        # distance), making that product π×r²×length_ft — the wrong bulk volume.
        pv_cell = self.pv

        for i in range(n):
            Pi = max(float(p_old[i]), 14.7)
            try:
                Zi = float(self.eos.z_factor(state.z[i], Pi, self.T, phase="v"))
                Zi = max(Zi, 0.05)
            except Exception:
                Zi = max(Pi * 0.85 / max(Pi, 1.0), 0.05)
            # rho in lbmol/ft³ (NOT lbmol/RB) — consistent with pv_cell in ft³.
            # Using lbmol/RB here would make acc_scale 5.6× too large (1 RB = 5.6146 ft³),
            # causing pressure to deplete 5× too slowly.
            rho_cell[i] = Pi / max(Zi * R * self.T, 1e-12)   # lbmol/ft³

            # Real gas compressibility: cg = 1/P - (1/Z)(dZ/dP)
            # Use centred finite difference for dZ/dP from the EOS.
            # Step size: 1% of P, clamped to [5, 200] psia for numerical stability.
            dP_num = float(np.clip(Pi * 0.01, 5.0, 200.0))
            try:
                Zi_hi = float(self.eos.z_factor(state.z[i], Pi + dP_num, self.T, phase="v"))
                Zi_lo = float(self.eos.z_factor(state.z[i], max(Pi - dP_num, 14.7), self.T, phase="v"))
                dZdP  = (Zi_hi - Zi_lo) / (2.0 * dP_num)
                cg_i  = 1.0 / Pi - dZdP / max(Zi, 1e-6)
                cg_i  = max(cg_i, 5e-7)   # floor: always positive
            except Exception:
                cg_i = 1.0 / Pi            # fallback to ideal gas
            ct_i = cg_i + cr
            acc_scale[i] = rho_cell[i] * ct_i * pv_cell[i] / max(dt_days, 1e-12)

        # ── Build the tridiagonal system ──────────────────────────────────────
        # Darcy's law in field units:
        #   q [lbmol/day] = 0.00708 * T [md*ft/cp] * dP [psi] * rho [lbmol/RB]
        # The 0.00708 constant already appears inside peaceman_well_index via
        # _compute_peaceman_wi (used for the well coupling).  For inter-cell
        # transmissibility the factor must be applied explicitly.
        DARCY = 0.00708  # field-unit conversion for inter-cell Darcy flow
        ab = np.zeros((3, n), dtype=float)
        b  = np.zeros(n, dtype=float)

        # Accumulation terms
        ab[1, :] += acc_scale
        b        += acc_scale * p_old

        # Inter-cell transmissibilities
        for i in range(n - 1):
            rho_face = 0.5 * (rho_cell[i] + rho_cell[i + 1])
            T_raw = self.transmissibility(state, i, i + 1)  # md*ft/cp (no 0.00708)
            Tij   = DARCY * rho_face * T_raw                # lbmol/day/psi
            ab[1, i    ] += Tij
            ab[1, i + 1] += Tij
            ab[0, i + 1] -= Tij   # superdiagonal
            ab[2, i    ] -= Tij   # subdiagonal

        # Boundary conditions (constant-pressure sides)
        for side in ("left", "right"):
            p_ext = self.boundary_pressure_target(side)
            if p_ext is not None:
                bi  = self.boundary_cell_index(side)
                Tb  = self.boundary_transmissibility(side)
                Tb_scaled = DARCY * rho_cell[bi] * Tb  # lbmol/day/psi
                ab[1, bi] += Tb_scaled
                b[bi]     += Tb_scaled * p_ext

        # Aquifer coupling — J_aq is already in lbmol/day/psi
        if self.aquifer.enabled:
            ai  = self.boundary_cell_index(self.aquifer.side)
            p_aq = self.aquifer_current_pressure_psia()
            Jaq  = self.aquifer.productivity_index_lbmol_day_psi
            ab[1, ai] += Jaq
            b[ai]     += Jaq * p_aq

        # Producer well coupling
        # WI from _compute_peaceman_wi already includes the 0.00708 factor,
        # so J_w = WI * lam_t * rho  [lbmol/day/psi].
        # The 0.00708 constant converts md·ft/cp·psi → RB/day, so rho must be
        # in lbmol/RB here (not lbmol/ft³ as used in the accumulation term).
        orig_well = self.well
        for _w in self.wells:
            self.well = _w
            self._mobility_cache = {}
            _wi  = _w.cell_index
            _mob = self.phase_mobility_data(state, _wi)
            _rho_RB = rho_cell[_wi] * FT3_PER_STB   # lbmol/ft³ → lbmol/RB
            _Jw  = self.peaceman_well_index(_wi) * _mob["lam_t"] * _rho_RB
            _response = self.solve_well_control(state)
            _pwf = float(_response["pwf_psia"])
            ab[1, _wi] += _Jw
            b[_wi]     += _Jw * _pwf
        self.well = orig_well
        self._mobility_cache = {}

        # Injector well coupling (same: WI contains 0.00708, rho must be lbmol/RB)
        injector     = self.active_injector()
        injector_perf = self.injection_performance(state)
        actual_inj_rate   = float(injector_perf.get("actual_rate_lbmol_day", 0.0))
        effective_inj_bhp = float(injector_perf.get("effective_bhp_psia", 0.0))
        if injector is not None and self.scenario.name != "natural_depletion" and actual_inj_rate > 0.0:
            ii      = injector.cell_index
            inj_mob = self.phase_mobility_data(state, ii)
            _rho_RB_inj = rho_cell[ii] * FT3_PER_STB
            Jin     = self.peaceman_well_index(ii) * inj_mob["lam_t"] * _rho_RB_inj
            rate_implied_bhp = p_old[ii] + actual_inj_rate / max(Jin, 1e-12)
            pinj_equiv = float(np.clip(rate_implied_bhp, p_old[ii],
                                       max(effective_inj_bhp, p_old[ii])))
            ab[1, ii] += Jin
            b[ii]     += Jin * pinj_equiv

        # Solve the banded tridiagonal system
        try:
            p_solved = solve_banded((1, 1), ab, b)
        except Exception:
            A_dense = np.diag(ab[1]) + np.diag(ab[0, 1:], 1) + np.diag(ab[2, :-1], -1)
            try:
                p_solved = np.linalg.solve(A_dense, b)
            except np.linalg.LinAlgError:
                p_solved = p_old.copy()

        # The pressure system is now dimensionally consistent (all terms in
        # lbmol/day/psi) so the linear solve is well-conditioned.  The previous
        # 0.5 under-relaxation was needed to tame the poorly-scaled empirical
        # system; with correct units it is no longer required and wastes a factor
        # of 2 in pressure responsiveness per timestep.
        p_new = np.asarray(p_solved, dtype=float)
        min_bhp_floor = min((w.min_bhp_psia for w in self.wells), default=14.7)
        return np.maximum(p_new, max(min_bhp_floor * 0.9, 14.7))

    def transport_update(self, state: SimulationState, dt_days: float) -> TransportResult:
        """
        Transport is sub-cycled so the producer-cell composition and phase state
        are refreshed within a large reporting timestep.

        Subcycle count is chosen adaptively: fluxes are computed once at the
        start of the step, used to determine the CFL-safe n_sub, then reused
        directly as the first subcycle's flux.  This avoids the double
        computation that a separate probe loop would require.

        CFL target: 0.5 (stable for first-order upwind explicit advection).
        Minimum: 1 subcycle.  Hard cap: 20 to prevent runaway at very high flux.
        """
        # ── Compute all inter-cell fluxes once ───────────────────────────────
        # These serve two purposes: (1) determine n_sub from the CFL, and
        # (2) provide the actual fluxes for the first transport subcycle.
        self._set_cache_state(state)
        nx = self.grid.nx
        cached_hc_fluxes = []  # List[np.ndarray], length nx-1
        cached_w_fluxes  = []  # List[float],      length nx-1
        max_cfl_per_day  = 0.0

        for i in range(nx - 1):
            fluxes = self.component_flux_breakdown_between(state, i, i + 1)
            hc_flux = np.asarray(fluxes["total"], dtype=float)
            w_flux  = self.water_flux_between(state, i, i + 1)
            cached_hc_fluxes.append(hc_flux)
            cached_w_fluxes.append(w_flux)
            hc_flux_mag = float(np.sum(np.abs(hc_flux)))
            if hc_flux_mag > 0.0:
                # Use the upstream cell pv (the donor cell for upwind transport).
                # Using pv_min of the two adjacent cells causes the tiny near-well
                # LGR cells to dominate n_sub even when their flux is modest,
                # leading to n_sub=20 and ~20× cost blowup. The upstream cell is
                # the physically correct donor volume for the CFL condition.
                upstream = i if hc_flux[hc_flux != 0][0] > 0 else i + 1
                pv_upstream = max(self.pv[upstream], 1e-12)
                max_cfl_per_day = max(max_cfl_per_day, hc_flux_mag / pv_upstream)

        CFL_TARGET = 0.5
        if max_cfl_per_day > 0.0:
            # Hard cap at 4: beyond 4 subcycles the marginal accuracy gain is
            # negligible for the smooth pressure-driven flows in this simulator,
            # and the cost scales linearly with n_sub.
            n_sub = max(1, min(4, int(math.ceil(max_cfl_per_day * dt_days / CFL_TARGET))))
        else:
            n_sub = 1
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
        # Initialize explicitly — never rely on locals() to check assignment
        aquifer_rate_lbmol_day: float = 0.0
        # Accumulate the exact moles added by the injector across all subcycles
        hc_injected_lbmol_total: float = 0.0

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
                inj_added = float(np.sum(inj_source)) * dt_sub  # exact lbmol added this subcycle
                ncomp_new[inj_cell] += dt_sub * inj_source
                hc_injected_lbmol_total += inj_added

            aquifer_cell, aquifer_rate_lbmol_day, aquifer_pressure_psia = self.aquifer_water_influx_lbmol_day(work_state)
            if aquifer_cell is not None and abs(aquifer_rate_lbmol_day) > 0.0:
                nw_new[aquifer_cell] += dt_sub * aquifer_rate_lbmol_day
                self.cum_aquifer_water_lbmol += max(aquifer_rate_lbmol_day, 0.0) * dt_sub

            agg_hc_sink, agg_qw, agg_scalars, _per_well = self.well_sinks_all(work_state)
            q_total     = agg_scalars["q_total_lbmol_day"]
            gas_frac    = agg_scalars["gas_frac"]
            oil_frac    = agg_scalars["oil_frac"]
            water_frac  = agg_scalars["water_frac"]
            dropout     = agg_scalars["dropout"]
            krg         = agg_scalars["krg"]
            kro         = agg_scalars["kro"]
            krw         = agg_scalars["krw"]
            damage_factor = agg_scalars["damage_factor"]
            wi_eff      = agg_scalars["wi_eff"]
            q_undamaged = agg_scalars["q_undamaged"]
            ploss       = agg_scalars["ploss"]

            # Apply all well sinks: agg_hc_sink is (nx, nc) already cell-indexed
            ncomp_new -= dt_sub * agg_hc_sink
            # Water sink per well cell
            for _pw in _per_well:
                nw_new[_pw["cell_index"]] -= dt_sub * float(_pw["qw_lbmol_day"])

            # Clip to physical floor.  Any cell driven negative (well drains faster
            # than neighbours can supply) gets raised to 1e-12.  This creates
            # spurious mass that would break closure.  Track it and add it to the
            # injected total so record_mass_balance sees the correct net balance:
            #   initial + (real_inj + clip_added) - produced - final = 0
            ncomp_pre_clip = ncomp_new.copy()
            ncomp_new = np.clip(ncomp_new, 1e-12, None)
            clip_mass_added = float(np.sum(ncomp_new - ncomp_pre_clip))  # >= 0
            hc_injected_lbmol_total += clip_mass_added
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

        return TransportResult(
            nt=work_state.nt,
            z=work_state.z,
            nw=work_state.nw,
            sw=work_state.sw,
            q_total=q_total,
            gas_frac=gas_frac,
            oil_frac=oil_frac,
            water_frac=water_frac,
            dropout=dropout,
            krg=krg,
            kro=kro,
            krw=krw,
            damage_factor=damage_factor,
            wi_eff=wi_eff,
            q_undamaged=q_undamaged,
            ploss=ploss,
            injector_actual_rate_lbmol_day=injector_actual_rate_lbmol_day,
            transport_advective_total=float(transport_advective_total),
            transport_dispersive_total=float(transport_dispersive_total),
            transport_total_flux=float(transport_total_flux),
            aquifer_rate_lbmol_day=float(aquifer_rate_lbmol_day),
            hc_injected_lbmol=float(hc_injected_lbmol_total),
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

        tr = self.transport_update(state_mid, dt_days)

        # Average pressure correction: after transport updates the mole inventory,
        # compute the spatially-averaged pressure consistent with total remaining
        # moles via the real gas law, then shift all cell pressures by the same
        # scalar offset to preserve the spatial gradient from p_pred.
        #
        # The per-cell pressure_update solve gives the correct Darcy gradient but
        # the wrong average level when well-cell BHP coupling dominates accumulation.
        # Per-cell correction fails because inter-cell fluxes redistribute moles
        # between cells, creating artificial gradients opposite to reality.
        try:
            z_avg    = np.average(tr.z, axis=0, weights=tr.nt)
            z_avg   /= max(float(z_avg.sum()), 1e-12)
            nt_total  = float(tr.nt.sum())
            pv_hc_tot = float(self.pv.sum()) * max(1.0 - float(np.mean(tr.sw)), 0.0)
            if pv_hc_tot > 1e-6 and nt_total > 0.0:
                pZ_avg    = nt_total * R * self.T / pv_hc_tot
                p_avg_old = float(np.mean(p_pred))
                p_avg_new = p_avg_old
                for _ in range(15):
                    Z_avg = float(self.eos.z_factor(
                        z_avg, max(p_avg_new, 14.7), self.T, phase="v"))
                    Z_avg = max(Z_avg, 0.05)
                    p_iter = pZ_avg * Z_avg
                    if abs(p_iter - p_avg_new) < 0.05:
                        break
                    p_avg_new = 0.5 * (p_avg_new + p_iter)
                dp_corr   = p_avg_new - p_avg_old
                # Limit correction: no more than one timestep's worth of
                # pressure drop (500 psia) and never below abandonment pressure.
                min_p = max(14.7, min((w.min_bhp_psia for w in self.wells), default=14.7))
                dp_corr = float(np.clip(dp_corr, -500.0, 500.0))
                p_corrected = np.clip(p_pred + dp_corr, min_p * 0.5, 50000.0)
            else:
                p_corrected = p_pred.copy()
        except Exception:
            p_corrected = p_pred.copy()


        # Single pressure solve only (predictor without corrector).
        # The corrector step was halving simulation speed with negligible
        # accuracy improvement given the explicit transport scheme.
        new_state = SimulationState(
            pressure=p_corrected,
            z=tr.z,
            nt=tr.nt,
            nw=tr.nw,
            sw=tr.sw,
            sg_max=state_mid.sg_max.copy(),
        )
        summary = {
            "well_rate_total": tr.q_total,
            "well_gas_fraction": tr.gas_frac,
            "well_oil_fraction": tr.oil_frac,
            "well_water_fraction": tr.water_frac,
            "well_dropout_indicator": tr.dropout,
            "well_krg": tr.krg,
            "well_kro": tr.kro,
            "well_krw": tr.krw,
            "well_damage_factor": tr.damage_factor,
            "well_effective_wi": tr.wi_eff,
            "well_rate_undamaged": tr.q_undamaged,
            "productivity_loss_fraction": tr.ploss,
            "avg_pressure": float(np.mean(p_corrected)),
            "min_pressure": float(np.min(p_corrected)),
            "avg_sw": float(np.mean(tr.sw)),
            "injector_actual_rate_lbmol_day": float(tr.injector_actual_rate_lbmol_day),
            "transport_advective_flux_lbmol_day": float(tr.transport_advective_total),
            "transport_dispersive_flux_lbmol_day": float(tr.transport_dispersive_total),
            "transport_total_flux_lbmol_day": float(tr.transport_total_flux),
            "aquifer_rate_lbmol_day": float(tr.aquifer_rate_lbmol_day),
            "aquifer_pressure_psia": float(self.aquifer_current_pressure_psia()),
            "left_boundary_pressure_psia": float(self.boundary.left_pressure_psia if self.boundary.left_mode == "constant_pressure" else p_corrected[0]),
            "right_boundary_pressure_psia": float(self.boundary.right_pressure_psia if self.boundary.right_mode == "constant_pressure" else p_corrected[-1]),
            "hc_injected_lbmol": float(tr.hc_injected_lbmol),
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
                    # Ramp the accepted dt back toward the target on consecutive
                    # successes (growth factor 1.2, capped at the requested dt).
                    accepted_dt = min(trial_dt * 1.2, dt_days)
                    return new_state, summary, accepted_dt, retries
                last_msg = msg
            except Exception as exc:
                last_msg = str(exc)

            trial_dt *= 0.5
            retries += 1
            if trial_dt < min_dt_days:
                raise RuntimeError(f"Adaptive timestep failed. Last reason: {last_msg}")

        raise RuntimeError(f"Adaptive timestep failed after maximum retries. Last reason: {last_msg}")

    def record_mass_balance(
        self,
        history: SimulationHistory,
        state_before: SimulationState,
        state_after: SimulationState,
        qw_produced_lbmol: float,
        hc_injected_lbmol: float,
        dt_days: float,
    ) -> None:
        """Record hydrocarbon and water mass balance errors.

        Both HC production and injection are derived from exact quantities:

        - hc_injected_lbmol: accumulated directly from inj_source × dt_sub
          inside transport_update across all subcycles — exact, not rate×dt.
        - hc_produced_this_step: nt_before - nt_after + hc_injected — the net
          nt change is (injected - produced), so production = injected - net_gain.
          This is guaranteed consistent with the physics because both quantities
          come from the same ncomp_new arrays.

        For the final zero-dt record (state_before == state_after, dt=0), both
        hc_injected_lbmol=0 and net_change=0, so hc_produced_this_step=0. Correct.
        """
        hc_before = float(np.sum(state_before.nt))
        hc_after  = float(np.sum(state_after.nt))
        net_gain  = hc_after - hc_before          # +ve when injection > production
        # produced = injected - net_gain  (exact, derived from same ncomp arrays)
        hc_produced_this_step = max(hc_injected_lbmol - net_gain, 0.0)

        self.cum_hc_produced_lbmol    += hc_produced_this_step
        self.cum_water_produced_lbmol += qw_produced_lbmol * dt_days
        self.cum_hc_injected_lbmol    += hc_injected_lbmol

        hc_err = (self.initial_hc_lbmol
                  + self.cum_hc_injected_lbmol
                  - self.cum_hc_produced_lbmol
                  - hc_after)
        w_current = float(np.sum(state_after.nw))
        # Water error: computed directly as (initial - current - net_produced).
        # Net water produced = initial_nw - current_nw - aquifer_influx_added.
        # This is exact regardless of any rate-averaging discrepancy.
        net_water_produced = (self.initial_water_lbmol - w_current
                              - self.cum_aquifer_water_lbmol)
        w_err = net_water_produced - self.cum_water_produced_lbmol

        history.hc_mass_balance_error_lbmol.append(float(hc_err))
        history.water_mass_balance_error_lbmol.append(float(w_err))

    def _record_history_step(
        self,
        history: SimulationHistory,
        state: SimulationState,
        t: float,
        dew_point: float,
        response: Dict[str, float | np.ndarray],
        injector: "InjectorWell | None",
        injector_perf: Dict[str, float | bool | np.ndarray | None],
        injector_rate_lbmol_day: float,
        q_total: float,
        gas_frac: float,
        oil_frac: float,
        water_frac: float,
        qw_lbmol_day: float,
        dropout: float,
        krg: float,
        kro: float,
        krw: float,
        damage_factor: float,
        wi_eff: float,
        q_undamaged: float,
        ploss: float,
        sep_rates: Dict[str, float],
        gas_rate_mmscf_day: float,
        condensate_rate_stb_day: float,
        water_rate_stb_day: float,
        cum_gas_scf: float,
        cum_condensate_stb: float,
        cum_water_stb: float,
        summary: Dict[str, float] | None,
        accepted_dt: float,
        retries: int,
        state_before: "SimulationState",
        dt_days: float,
        hc_injected_lbmol: float = 0.0,
    ) -> None:
        """Append one row to SimulationHistory. Called from both the final-step
        and the normal-step paths inside run(), eliminating ~100 lines of
        duplicated code."""
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

        history.well_control_mode.append(str(response["control_mode"]))
        history.well_flowing_pwf_psia.append(float(response["pwf_psia"]))
        history.well_estimated_thp_psia.append(float(response["estimated_thp_psia"]))
        history.well_tubing_hydrostatic_psi.append(float(response.get("tubing_hydrostatic_psi", 0.0)))
        history.well_tubing_friction_psi.append(float(response.get("tubing_friction_psi", 0.0)))
        history.well_tubing_acceleration_psi.append(float(response.get("tubing_acceleration_psi", 0.0)))
        history.well_tubing_mixture_velocity_ft_s.append(float(response.get("tubing_mixture_velocity_ft_s", 0.0)))
        history.well_tubing_reynolds_number.append(float(response.get("tubing_reynolds_number", 0.0)))
        history.well_tubing_friction_factor.append(float(response.get("tubing_friction_factor", 0.0)))
        history.well_target_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
        if self.well.control_mode == "gas_rate":
            history.controlled_gas_rate_mmscf_day.append(float(self.well.target_gas_rate_mmscf_day))
        else:
            history.controlled_gas_rate_mmscf_day.append(float(response["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE)
        history.reported_gas_rate_mmscf_day.append(
            float(response.get("reported_gas_rate_mmscf_day", float(response["qg_lbmol_day"]) * SCF_PER_LBMOL * MM_PER_ONE))
        )

        adv_flux = float(summary.get("transport_advective_flux_lbmol_day", 0.0)) if summary else 0.0
        disp_flux = float(summary.get("transport_dispersive_flux_lbmol_day", 0.0)) if summary else 0.0
        tot_flux = float(summary.get("transport_total_flux_lbmol_day", 0.0)) if summary else 0.0
        aq_rate = float(summary.get("aquifer_rate_lbmol_day", 0.0)) if summary else 0.0
        aq_pres = float(summary.get("aquifer_pressure_psia", self.aquifer_current_pressure_psia())) if summary else float(self.aquifer_current_pressure_psia())
        lb_pres = float(summary.get("left_boundary_pressure_psia", state.pressure[0])) if summary else float(self.boundary.left_pressure_psia if self.boundary.left_mode == "constant_pressure" else state.pressure[0])
        rb_pres = float(summary.get("right_boundary_pressure_psia", state.pressure[-1])) if summary else float(self.boundary.right_pressure_psia if self.boundary.right_mode == "constant_pressure" else state.pressure[-1])

        history.transport_advective_flux_lbmol_day.append(adv_flux)
        history.transport_dispersive_flux_lbmol_day.append(disp_flux)
        history.transport_total_flux_lbmol_day.append(tot_flux)
        history.aquifer_rate_lbmol_day.append(aq_rate)
        history.aquifer_cumulative_mlbmol.append(float(self.cum_aquifer_water_lbmol / 1.0e6))
        history.aquifer_pressure_psia.append(aq_pres)
        history.left_boundary_pressure_psia.append(lb_pres)
        history.right_boundary_pressure_psia.append(rb_pres)

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
            history.injector_rate_target_lbmol_day.append(float(injector_perf.get("target_rate_lbmol_day", 0.0)))
            history.injector_rate_achievable_lbmol_day.append(float(injector_perf.get("achievable_rate_lbmol_day", 0.0)))
            history.injector_cell_pressure_psia.append(float(injector_perf.get("cell_pressure_psia", state.pressure[injector.cell_index])))
            history.injector_pressure_delta_psia.append(float(injector_perf.get("delta_p_psia", 0.0)))
            history.injector_effective_bhp_psia.append(float(injector_perf.get("effective_bhp_psia", 0.0)))
            history.injector_active_flag.append(1.0 if injector_perf.get("active", False) else 0.0)

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
        history.well_hysteresis_trap_fraction.append(float(response.get("hysteresis_trap_fraction", 0.0)))
        history.well_hysteresis_imbibition_flag.append(float(response.get("hysteresis_imbibition_flag", 0.0)))
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

        self.record_mass_balance(
            history, state_before, state, qw_lbmol_day, hc_injected_lbmol, dt_days
        )
        history.accepted_dt_days.append(float(accepted_dt))
        history.timestep_retries.append(int(retries))

    def run(self, state0: SimulationState, t_end_days: float, dt_days: float) -> Tuple[SimulationState, SimulationHistory]:
        if dt_days <= 0.0:
            raise ValueError("Time step must be positive")

        history = SimulationHistory()
        state = state0
        t = 0.0
        self._initialize_accounting(state0)

        cum_gas_scf = 0.0
        cum_condensate_stb = 0.0
        cum_water_stb = 0.0

        # Dew point is a fluid property — fixed at the value from the PVT table
        # (interpolated at initial pressure) or the user-configured value.
        # It must NOT follow reservoir pressure: the visible drop in the first
        # year was caused by the previous dynamic logic resetting dew_point to
        # current_pressure when the flash returned two_phase below the dew point.
        p_init_for_dp = float(np.mean(state0.pressure))
        if self.pvt_table is not None and self.pvt_table.has("dew_point_psia"):
            dp_from_table = self.pvt_table.interp(p_init_for_dp, "dew_point_psia", None)
            dew_point = float(dp_from_table) if dp_from_table is not None and np.isfinite(dp_from_table) else float(self.pvt.dew_point_psia)
        else:
            dew_point = float(self.pvt.dew_point_psia)

        # Record the initial state at t=0 so history[0] reflects true initial conditions
        # (p_init_psia, no production) rather than the state after the first timestep.
        self._set_cache_state(state0)
        response_init  = self.reported_well_response(state0)
        sep_rates_init = self.separator_rates(response_init)
        injector_init  = self.active_injector()
        inj_perf_init  = self.injection_performance(state0)
        self._record_history_step(
            history=history, state=state0, t=0.0, dew_point=dew_point,
            response=response_init,
            injector=injector_init, injector_perf=inj_perf_init,
            injector_rate_lbmol_day=float(inj_perf_init.get("actual_rate_lbmol_day", 0.0)),
            q_total=float(response_init["q_total_lbmol_day"]),
            gas_frac=float(response_init["gas_frac"]),
            oil_frac=float(response_init["oil_frac"]),
            water_frac=float(response_init["water_frac"]),
            qw_lbmol_day=float(response_init.get("qw_lbmol_day", 0.0)),
            dropout=float(response_init["dropout"]),
            krg=float(response_init["krg"]),
            kro=float(response_init["kro"]),
            krw=float(response_init["krw"]),
            damage_factor=float(response_init["damage_factor"]),
            wi_eff=float(response_init["wi_eff"]),
            q_undamaged=float(response_init["q_undamaged"]),
            ploss=float(response_init["ploss"]),
            sep_rates=sep_rates_init,
            gas_rate_mmscf_day=float(sep_rates_init["gas_rate_mmscf_day"]),
            condensate_rate_stb_day=float(sep_rates_init["condensate_rate_stb_day"]),
            water_rate_stb_day=0.0,
            cum_gas_scf=0.0,
            cum_condensate_stb=0.0,
            cum_water_stb=0.0,
            summary=None,
            accepted_dt=0.0,
            retries=0,
            state_before=state0,
            dt_days=0.0,
        )

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
                # Final timestep: record terminal state and break
                sep_rates = self.separator_rates(response0)
                qw_lbmol_day_0 = float(response0.get("qw_lbmol_day", q_total_0 * water_frac_0))
                gas_rate_mmscf_day = float(sep_rates["gas_rate_mmscf_day"])
                condensate_rate_stb_day = float(sep_rates["condensate_rate_stb_day"])
                water_rate_stb_day = water_lbmol_day_to_stb_day(qw_lbmol_day_0)

                self._record_history_step(
                    history=history, state=state, t=t, dew_point=dew_point,
                    response=response0,
                    injector=injector, injector_perf=injector_perf_0,
                    injector_rate_lbmol_day=injector_rate_lbmol_day,
                    q_total=q_total_0, gas_frac=gas_frac_0, oil_frac=oil_frac_0,
                    water_frac=water_frac_0, qw_lbmol_day=qw_lbmol_day_0, dropout=dropout_0,
                    krg=krg_0, kro=kro_0, krw=krw_0,
                    damage_factor=damage_factor_0, wi_eff=wi_eff_0,
                    q_undamaged=q_undamaged_0, ploss=ploss_0,
                    sep_rates=sep_rates,
                    gas_rate_mmscf_day=gas_rate_mmscf_day,
                    condensate_rate_stb_day=condensate_rate_stb_day,
                    water_rate_stb_day=water_rate_stb_day,
                    cum_gas_scf=cum_gas_scf, cum_condensate_stb=cum_condensate_stb,
                    cum_water_stb=cum_water_stb,
                    summary=None, accepted_dt=0.0, retries=0,
                    state_before=state, dt_days=0.0, hc_injected_lbmol=0.0,
                )
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

            # Trapezoid average for reporting quantities over the accepted timestep
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
            # Use explicit qw from responses — more accurate than q_total * water_frac
            qw_lbmol_day = 0.5 * (float(response0.get("qw_lbmol_day", q_total_0 * water_frac_0))
                                  + float(response1.get("qw_lbmol_day", q_total_1 * water_frac_1)))

            response_avg = dict(response1)
            response_avg["qg_lbmol_day"] = qg_lbmol_day
            response_avg["qo_lbmol_day"] = qo_lbmol_day
            x_avg = 0.5 * (np.asarray(response0["x"], dtype=float) + np.asarray(response1["x"], dtype=float))
            y_avg = 0.5 * (np.asarray(response0["y"], dtype=float) + np.asarray(response1["y"], dtype=float))
            response_avg["x"] = x_avg / np.sum(x_avg)
            response_avg["y"] = y_avg / np.sum(y_avg)

            sep_rates = self.separator_rates(response_avg)
            gas_rate_mmscf_day = float(sep_rates["gas_rate_mmscf_day"])
            condensate_rate_stb_day = float(sep_rates["condensate_rate_stb_day"])
            water_rate_stb_day = water_lbmol_day_to_stb_day(qw_lbmol_day)

            cum_gas_scf += gas_rate_mmscf_day * 1.0e6 * accepted_dt
            cum_condensate_stb += condensate_rate_stb_day * accepted_dt
            cum_water_stb += water_rate_stb_day * accepted_dt

            self._record_history_step(
                history=history, state=state_new, t=t_new, dew_point=dew_point,
                response=response1,
                injector=injector, injector_perf=injector_perf_1,
                injector_rate_lbmol_day=actual_inj_rate_lbmol_day,
                q_total=q_total, gas_frac=gas_frac, oil_frac=oil_frac,
                water_frac=water_frac, qw_lbmol_day=qw_lbmol_day, dropout=dropout,
                krg=krg, kro=kro, krw=krw,
                damage_factor=damage_factor, wi_eff=wi_eff,
                q_undamaged=q_undamaged, ploss=ploss,
                sep_rates=sep_rates,
                gas_rate_mmscf_day=gas_rate_mmscf_day,
                condensate_rate_stb_day=condensate_rate_stb_day,
                water_rate_stb_day=water_rate_stb_day,
                cum_gas_scf=cum_gas_scf, cum_condensate_stb=cum_condensate_stb,
                cum_water_stb=cum_water_stb,
                summary=summary, accepted_dt=accepted_dt, retries=retries,
                state_before=state, dt_days=accepted_dt,
                hc_injected_lbmol=float(summary.get("hc_injected_lbmol", 0.0)) if summary else 0.0,
            )

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
            dropout_arr[i] = liquid_dropout_fraction(beta)
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

    # Published binary interaction parameters (kij) for PR-EOS from:
    # Katz & Firoozabadi (1978), Whitson & Brulé "Phase Behavior of Petroleum
    # Reservoir Fluids" (2000), and Reid, Prausnitz & Poling (1987).
    # Component order: 0=H2S, 1=CO2, 2=N2, 3=C1, 4=C2, 5=C3,
    #                  6=iC4, 7=nC4, 8=iC5, 9=nC5, 10=C6, 11=C7+
    #
    # Key non-zero pairs that significantly affect phase behaviour:
    #   H2S–hydrocarbons : moderately positive (H2S is polar)
    #   CO2–C1           : 0.103 (large — CO2 is non-polar but very different from CH4)
    #   CO2–hydrocarbons : 0.12–0.15 declining for heavier components
    #   N2–C1            : 0.025 (small — both non-polar, similar size)
    #   N2–hydrocarbons  : 0.08–0.16 increasing for heavier components
    #   C1–heavier HC    : small, 0.01–0.02, essentially zero for C1–C2

    # fmt: off
    kij_pairs = {
        # H2S (0) interactions
        (0, 1): 0.089,   # H2S–CO2
        (0, 2): 0.178,   # H2S–N2
        (0, 3): 0.080,   # H2S–C1
        (0, 4): 0.070,   # H2S–C2
        (0, 5): 0.070,   # H2S–C3
        (0, 6): 0.060,   # H2S–iC4
        (0, 7): 0.060,   # H2S–nC4
        (0, 8): 0.060,   # H2S–iC5
        (0, 9): 0.060,   # H2S–nC5
        (0,10): 0.050,   # H2S–C6
        (0,11): 0.050,   # H2S–C7+
        # CO2 (1) interactions
        (1, 2): 0.000,   # CO2–N2  (negligible)
        (1, 3): 0.103,   # CO2–C1  (key pair for gas injection)
        (1, 4): 0.130,   # CO2–C2
        (1, 5): 0.135,   # CO2–C3
        (1, 6): 0.130,   # CO2–iC4
        (1, 7): 0.130,   # CO2–nC4
        (1, 8): 0.125,   # CO2–iC5
        (1, 9): 0.125,   # CO2–nC5
        (1,10): 0.120,   # CO2–C6
        (1,11): 0.115,   # CO2–C7+
        # N2 (2) interactions
        (2, 3): 0.025,   # N2–C1
        (2, 4): 0.010,   # N2–C2  (N2 less miscible with heavier HC — can be zero)
        (2, 5): 0.090,   # N2–C3
        (2, 6): 0.095,   # N2–iC4
        (2, 7): 0.095,   # N2–nC4
        (2, 8): 0.100,   # N2–iC5
        (2, 9): 0.100,   # N2–nC5
        (2,10): 0.110,   # N2–C6
        (2,11): 0.110,   # N2–C7+
        # C1 (3) interactions with heavier HC — small but non-trivial for C7+
        (3, 4): 0.000,   # C1–C2
        (3, 5): 0.000,   # C1–C3
        (3, 6): 0.000,   # C1–iC4
        (3, 7): 0.000,   # C1–nC4
        (3, 8): 0.000,   # C1–iC5
        (3, 9): 0.000,   # C1–nC5
        (3,10): 0.010,   # C1–C6
        (3,11): 0.020,   # C1–C7+  (most important C1–HC pair for condensate)
        # C2 and heavier — all effectively zero
    }
    # fmt: on

    for (i, j), val in kij_pairs.items():
        kij[i, j] = val
        kij[j, i] = val  # symmetric

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


def compute_volumetrics(
    sim: "CompositionalSimulator1D",
    state0: "SimulationState",
    sw_init: float,
    pvt_table: "PVTTable | None" = None,
) -> Dict[str, float]:
    """
    Compute static volumetric in-place quantities from initial conditions.

    Accepts the simulator directly so callers don't need to pass its
    constituent parts (grid, rock, fluid, eos, temperature_R, pv) separately.

    Uses the standard industry formula:

        G = 43,560 × A(acres) × h(ft) × φ × (1 − Swi) / Bgi

    which is equivalent to G = HCPV / Bgi when working in ft³ throughout,
    since GRV already absorbs the 43,560 ft²/acre conversion via grid geometry.
    """
    grid        = sim.grid
    eos         = sim.eos
    fluid       = sim.fluid
    temperature_R = sim.T
    pv_cells    = sim.pv

    # -----------------------------------------------------------------
    # GRV — Gross Rock Volume
    # GRV = cylindrical volume = π r² × h (net pay)
    grv_ft3 = float(grid.area_ft2 * grid.thickness_ft)

    # Drainage area = π r² (already stored as area_ft2)
    drainage_area_ft2   = float(grid.area_ft2)
    drainage_area_acres = drainage_area_ft2 / 43560.0

    # -----------------------------------------------------------------
    # PV — Pore Volume  (exact sum of per-cell simulator PV)
    # -----------------------------------------------------------------
    pv_ft3 = float(np.sum(pv_cells))

    # Hydrocarbon and water pore volumes using scalar sw_init
    hcpv_ft3 = pv_ft3 * (1.0 - sw_init)
    wpv_ft3  = pv_ft3 * sw_init

    # -----------------------------------------------------------------
    # Bgi — Initial gas formation volume factor  [res ft³ / scf]
    #
    # Route 1: PVT table gas_fvf_ft3_scf column (preferred)
    # Route 2: PR-EOS Z-factor
    #   Bgi = (P_std / T_std) × (Zi × T / Pi)
    #       = Z_init × R × T / (Pi × SCF_PER_LBMOL)
    # -----------------------------------------------------------------
    p_init = float(np.mean(state0.pressure))
    z_avg  = np.mean(state0.z, axis=0)
    z_avg  = z_avg / np.sum(z_avg)

    bg_init: float | None = None
    zi: float | None = None

    if pvt_table is not None and pvt_table.has("gas_fvf_ft3_scf"):
        bg_pvt = pvt_table.interp(p_init, "gas_fvf_ft3_scf", None)
        if bg_pvt is not None and float(bg_pvt) > 0.0:
            bg_init = float(bg_pvt)

    if bg_init is None or bg_init <= 0.0:
        # Fall back to EOS Z-factor
        if pvt_table is not None and pvt_table.has("z_factor"):
            zi_pvt = pvt_table.interp(p_init, "z_factor", None)
            if zi_pvt is not None and float(zi_pvt) > 0.0:
                zi = float(zi_pvt)
        if zi is None:
            try:
                zi = float(eos.z_factor(z_avg, p_init, temperature_R, phase="v"))
            except Exception:
                zi = 1.0
        # Bgi = Z * R * T / (P * SCF_PER_LBMOL)  [res ft³/scf]
        bg_init = zi * R * temperature_R / (p_init * SCF_PER_LBMOL)

    # -----------------------------------------------------------------
    # GIIP  [scf → Bscf]   G = HCPV / Bgi
    # -----------------------------------------------------------------
    giip_scf  = hcpv_ft3 / max(bg_init, 1e-12)
    giip_bscf = giip_scf / 1.0e9
    giip_lbmol = giip_scf / SCF_PER_LBMOL

    # -----------------------------------------------------------------
    # CIIP  [STB → MSTB]
    # CGR lookup: vaporized_cgr first, then reservoir_cgr, then EOS flash
    # -----------------------------------------------------------------
    cgr_stb_mmscf: float | None = None

    if pvt_table is not None:
        if pvt_table.has("vaporized_cgr_stb_mmscf"):
            cgr_pvt = pvt_table.interp(p_init, "vaporized_cgr_stb_mmscf", None)
            if cgr_pvt is None and pvt_table.has("reservoir_cgr_stb_mmscf"):
                cgr_pvt = pvt_table.interp(p_init, "reservoir_cgr_stb_mmscf", None)
        elif pvt_table.has("reservoir_cgr_stb_mmscf"):
            cgr_pvt = pvt_table.interp(p_init, "reservoir_cgr_stb_mmscf", None)
        else:
            cgr_pvt = None
        if cgr_pvt is not None and np.isfinite(cgr_pvt) and float(cgr_pvt) > 0.0:
            cgr_stb_mmscf = float(cgr_pvt)

    if cgr_stb_mmscf is None:
        # EOS flash fallback: estimate CGR from liquid fraction of flash
        try:
            flash_calc = FlashCalculator(eos)
            fl    = flash_calc.flash(z_avg, p_init, temperature_R)
            beta  = float(fl["beta"])
            x     = np.asarray(fl["x"], dtype=float)
            x     = x / max(np.sum(x), 1e-12)
            _, _, _, mw_arr = fluid.critical_arrays()
            mw_liq       = float(np.dot(x, mw_arr))
            liq_lbmol    = giip_lbmol * max(1.0 - beta, 0.0)
            liq_stb      = liq_lbmol * mw_liq / CONDENSATE_DENSITY_LBFT3 / FT3_PER_STB
            gas_mmscf    = giip_bscf * 1000.0
            if gas_mmscf > 1e-6 and liq_stb > 0.0:
                cgr_stb_mmscf = liq_stb / gas_mmscf
        except Exception:
            pass  # cgr_stb_mmscf stays None → CIIP = NaN

    if cgr_stb_mmscf is not None and cgr_stb_mmscf > 0.0:
        ciip_stb  = (giip_scf / 1.0e6) * cgr_stb_mmscf   # GIIP(MMscf) × CGR(STB/MMscf)
        ciip_mstb = ciip_stb / 1.0e3
    else:
        ciip_mstb = float("nan")

    # -----------------------------------------------------------------
    # WIIP  [ft³ → STB → MSTB]   (no Bw correction — Bw ≈ 1.0)
    # -----------------------------------------------------------------
    wiip_stb  = wpv_ft3 / FT3_PER_STB
    wiip_mstb = wiip_stb / 1.0e3
    wiip_lbmol = float(np.sum(state0.nw))

    # -----------------------------------------------------------------
    # Unit conversions for return dict
    # -----------------------------------------------------------------
    grv_mmbbl = grv_ft3 / (FT3_PER_STB * 1.0e6)
    pv_mmbbl  = pv_ft3  / (FT3_PER_STB * 1.0e6)

    return {
        "drainage_area_acres":     drainage_area_acres,
        "net_pay_thickness_ft":    float(grid.thickness_ft),
        "gross_rock_volume_ft3":   grv_ft3,
        "gross_rock_volume_mmbbl": grv_mmbbl,
        "pore_volume_ft3":         pv_ft3,
        "pore_volume_mmbbl":       pv_mmbbl,
        "hc_pore_volume_ft3":      hcpv_ft3,
        "initial_swi":             float(sw_init),
        "bg_init_res_ft3_per_scf": bg_init,
        "bgi_res_ft3_per_scf":     bg_init,   # alias kept for backward compat
        "zi":                      zi if zi is not None else bg_init * p_init * SCF_PER_LBMOL / (R * temperature_R),
        "giip_bscf":               giip_bscf,
        "giip_lbmol":              giip_lbmol,
        "ciip_mstb":               ciip_mstb,
        "cgr_stb_mmscf":           cgr_stb_mmscf if cgr_stb_mmscf is not None else float("nan"),
        "wiip_mstb":               wiip_mstb,
        "wiip_lbmol":              wiip_lbmol,
    }


def generate_chapter4_summary(history: SimulationHistory, volumetrics: Dict[str, float] | None = None) -> List[List[object]]:
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

    # Volumetric rows — static values prepended to the summary table.
    # Layout mirrors fix85: Drainage Area, Net Pay, GRV, PV, GIIP, CIIP, WIIP.
    # GRV and PV are reported in MMft³ (consistent with fix85) rather than MMbbl.
    vol_rows: List[List[object]] = []
    if volumetrics:
        _NA = 0.0   # time column — zero for static rows (fix85 convention)
        drainage_acres = volumetrics.get("drainage_area_acres", 0.0)
        net_pay        = volumetrics.get("net_pay_thickness_ft", 0.0)
        grv_mmft3      = volumetrics.get("gross_rock_volume_ft3", 0.0) / 1.0e6
        pv_mmft3       = volumetrics.get("pore_volume_ft3", 0.0) / 1.0e6
        giip           = volumetrics.get("giip_bscf", 0.0)
        ciip           = volumetrics.get("ciip_mstb", float("nan"))
        wiip           = volumetrics.get("wiip_mstb", 0.0)

        vol_rows = [
            ["Drainage Area (acres)",     drainage_acres, drainage_acres, _NA, drainage_acres],
            ["Net Pay Thickness (ft)",    net_pay,        net_pay,        _NA, net_pay],
            ["Gross Rock Volume (MMft³)", grv_mmft3,      grv_mmft3,      _NA, grv_mmft3],
            ["Pore Volume (MMft³)",       pv_mmft3,       pv_mmft3,       _NA, pv_mmft3],
            ["GIIP (Bscf)",               giip,           giip,           _NA, giip],
            ["CIIP (MSTB)",               ciip,           ciip,           _NA, ciip],
            ["WIIP (MSTB)",               wiip,           wiip,           _NA, wiip],
        ]

    return vol_rows + [
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
    wells_config: "List[dict] | None" = None,
    refined_cells: int = 10,
    min_dx_ft: float = 50.0,
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
    dt_days: float = 30.0,
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
    condensate_damage_strength: float = 3.0,
    condensate_critical_dropout: float = 0.05,
    condensate_min_damage_factor: float = 0.20,
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
    condensate_api_gravity: float = 50.0,
    pvt_table: PVTTable | None = None,
) -> Dict[str, object]:
    fluid = build_example_fluid()
    eos = PengRobinsonEOS(fluid)
    flash = FlashCalculator(eos)

    if z_init is None:
        # Default composition from a representative gas condensate
        # (H2S, CO2, N2, C1, C2, C3, iC4, nC4, iC5, nC5, C6, C7+)
        z_init = np.array([0.0000, 0.0066, 0.0774, 0.6367, 0.0723, 0.0560,
                           0.0204, 0.0225, 0.0127, 0.0094, 0.0135, 0.0725], dtype=float)
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
    relperm = RelPermParams(Sgc=Sgc, Swc=Swc, Sorw=Sorw, Sorg=Sorg, krg0=krg0, kro0=kro0, krw0=krw0, ng=ng, no=no, nw=nw,
                            condensate_damage_strength=condensate_damage_strength,
                            condensate_critical_dropout=condensate_critical_dropout,
                            condensate_min_damage_factor=condensate_min_damage_factor)
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
    def _build_well(cfg: dict, idx: int) -> Well:
        """Build a Well from a config dict, falling back to the scalar params
        supplied to run_example for any key that is missing from cfg."""
        pos  = float(np.clip(float(cfg.get("position_fraction", 1.0)), 0.0, 1.0))
        cell = int(round(pos * (nx - 1)))
        cell = max(0, min(cell, nx - 1))
        w = Well(
            cell_index=cell,
            control_mode=str(cfg.get("control_mode", well_control_mode)),
            bhp_psia=float(cfg.get("bhp_psia", bhp_psia)),
            drawdown_psia=float(cfg.get("drawdown_psia", drawdown_psia)),
            min_bhp_psia=float(cfg.get("min_bhp_psia", min_bhp_psia)),
            target_gas_rate_mmscf_day=float(cfg.get("target_gas_rate_mmscf_day", target_gas_rate_mmscf_day)),
            thp_psia=float(cfg.get("thp_psia", thp_psia)),
            tvd_ft=float(cfg.get("tvd_ft", tvd_ft)),
            tubing_id_in=float(cfg.get("tubing_id_in", tubing_id_in)),
            wellhead_temperature_R=float(cfg.get("wellhead_temperature_R", wellhead_temperature_R)),
            thp_friction_coeff=float(cfg.get("thp_friction_coeff", thp_friction_coeff)),
            tubing_roughness_in=float(cfg.get("tubing_roughness_in", tubing_roughness_in)),
            tubing_calibration_factor=float(cfg.get("tubing_calibration_factor", tubing_calibration_factor)),
            tubing_model=str(cfg.get("tubing_model", tubing_model)),
            productivity_index=float(cfg.get("productivity_index", productivity_index)),
            rw_ft=float(cfg.get("rw_ft", rw_ft)),
            skin=float(cfg.get("skin", skin)),
            name=str(cfg.get("name", f"Well {idx}")),
        )
        return w

    if wells_config is not None and len(wells_config) > 0:
        wells_list = [_build_well(cfg, i + 1) for i, cfg in enumerate(wells_config)]
    else:
        # Backward-compatible single-well path using scalar parameters
        wells_list = [_build_well({"position_fraction": 1.0, "name": "Well 1"}, 1)]

    well = wells_list[0]  # primary well alias for reporting

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
        condensate_api_gravity=condensate_api_gravity,
    )

    sim = CompositionalSimulator1D(
        grid=grid,
        rock=rock,
        fluid=fluid,
        eos=eos,
        flash=flash,
        well=wells_list,
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

    # sim.pv is now computed correctly in __init__ using π×r²×h×φ distributed
    # by dx fractions — no override needed here.

    state0 = sim.initialize_state(p_init_psia=p_init_psia, z_init=z_init, sw_init=sw_init)

    # Compute static volumetrics from initial state before the run
    volumetrics = compute_volumetrics(
        sim=sim,
        state0=state0,
        sw_init=sw_init,
        pvt_table=pvt_table,
    )

    final_state, history = sim.run(state0, t_end_days=t_end_days, dt_days=dt_days)
    final_diag = sim.spatial_diagnostics(final_state)

    summary_rows = generate_chapter4_summary(history, volumetrics=volumetrics)
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Initial Value", "Peak / Critical Value", "Time of Peak / Critical (days)", "Final Value"])
    history_df = history_dataframe(history, start_date_text=start_date_text)
    spatial_df = spatial_dataframe(final_diag)

    # Per-well breakdown at the final state
    try:
        _agg_hc, _agg_qw, _agg_sc, _per_well = sim.well_sinks_all(final_state)
        per_well_rows = []
        for _pw in _per_well:
            per_well_rows.append({
                "Well":                     _pw.get("name", f'Well {_pw["cell_index"]}'),
                "Cell index":               int(_pw["cell_index"]),
                "Position (fraction)":      round(_pw["cell_index"] / max(sim.grid.nx - 1, 1), 3),
                "Pressure (psia)":          round(float(_pw["pressure_psia"]), 2),
                "Gas rate (MMscf/day)":     round(gas_lbmol_day_to_mmscf_day(float(_pw["qg_lbmol_day"])), 4),
                "Condensate rate (STB/day)": round(liquid_lbmol_day_to_stb_day(
                    float(_pw["qo_lbmol_day"]),
                    final_state.z[_pw["cell_index"]],
                    sim.mw_components,
                    sim.reporting.condensate_density_lbft3,
                ), 2),
                "Water rate (STB/day)":     round(water_lbmol_day_to_stb_day(float(_pw["qw_lbmol_day"])), 2),
                "Damage factor":            round(float(_pw["damage_factor"]), 4),
                "Productivity loss":        round(float(_pw["ploss"]), 4),
                "krg":                      round(float(_pw["krg"]), 4),
                "kro":                      round(float(_pw["kro"]), 4),
            })
        per_well_df = pd.DataFrame(per_well_rows)
    except Exception:
        per_well_df = None

    return {
        "fluid": fluid,
        "grid": grid,
        "well": well,
        "wells": wells_list,
        "scenario": scenario,
        "sim": sim,
        "final_state": final_state,
        "history": history,
        "history_df": history_df,
        "final_diag": final_diag,
        "spatial_df": spatial_df,
        "summary_df": summary_df,
        "volumetrics": volumetrics,
        "per_well_df": per_well_df,
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


def _timed_run_example(
    kwargs: Dict[str, object],
    label: str,
    current: int,
    total: int,
    placeholder: "st.delta_generator.DeltaGenerator",
) -> Tuple[Dict[str, object], float]:
    """
    Run run_example(**kwargs) in a background thread while updating *placeholder*
    every 0.25 s with a live MM:SS elapsed timer.

    Returns
    -------
    result       : the result dict from run_example
    elapsed_total: wall-clock seconds the run took

    Raises on failure (the placeholder is left showing the error context).
    """
    import threading
    import time as _time

    _result: Dict[str, object] = {}
    _error: list = []

    def _worker() -> None:
        try:
            _result.update(run_example(**kwargs))
        except Exception as exc:  # noqa: BLE001
            _error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    t0 = _time.monotonic()

    while thread.is_alive():
        elapsed = _time.monotonic() - t0
        mins, secs = divmod(int(elapsed), 60)
        batch_info = f" ({current}/{total})" if total > 1 else ""
        placeholder.info(
            f"⏱ **{label}**{batch_info} — **{mins:02d}:{secs:02d}** elapsed"
        )
        _time.sleep(0.25)

    thread.join()
    elapsed_total = _time.monotonic() - t0

    if _error:
        raise _error[0]

    return _result, elapsed_total


def display_streamlit_app() -> None:
    st.set_page_config(page_title="Compositional Reservoir Simulator", layout="wide")
    st.title("Compositional Reservoir Simulator")

    # ── Session state migration ───────────────────────────────────────────────
    # Fix stale dt_days below 1 day (old default was 0.05).
    _stale_dt = st.session_state.get("dt_days", 30.0)
    if isinstance(_stale_dt, (int, float)) and float(_stale_dt) < 1.0:
        st.session_state["dt_days"] = 30.0
    # Fix stale min_dx_ft below 20 ft (old default was 12 ft, which causes
    # very small near-well cells that drive n_sub to 20 and blow up runtime).
    _stale_dx = st.session_state.get("min_dx_ft", 50.0)
    if isinstance(_stale_dx, (int, float)) and float(_stale_dx) < 20.0:
        st.session_state["min_dx_ft"] = 50.0

    # Reset stale well values that carried over from old session defaults.
    # Well keys use the pattern w{i}_<field> (e.g. w0_qg, w0_tvd).
    for _wi in range(4):
        _wk = f"w{_wi}"
        _stale_resets = [
            (f"{_wk}_qg",  5.0,    0.0),    # old default gas rate
            (f"{_wk}_bhp", 3000.0, 100.0),  # old default BHP
            (f"{_wk}_thp", 500.0,  14.7),   # old default THP
            (f"{_wk}_tvd", 8000.0, 100.0),  # old default TVD
            (f"{_wk}_wht", 60.0,   -60.0),  # old default wellhead temp
        ]
        for _skey, _old_val, _new_val in _stale_resets:
            if _skey in st.session_state and float(st.session_state[_skey]) == _old_val:
                st.session_state[_skey] = _new_val
    # ─────────────────────────────────────────────────────────────────────────

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
        "start_date_text": "",
        "end_date_text": "",
        "dt_days": 30.0,
        "nx": 10,
        "length_ft": 500.0,
        "radius_ft": 1.0,
        "thickness_ft": 5.0,
        "refined_cells": 10,
        "min_dx_ft": 50.0,
        "growth": 1.45,
        "temperature_F": -60.0,
        "dew_point_psia": 100.0,
        "separator_stages": 4,
        "separator_pressure_psia": 14.7,
        "separator_temperature_F": -10.0,
        "separator_second_stage_pressure_psia": 14.7,
        "separator_second_stage_temperature_F": -10.0,
        "separator_third_stage_pressure_psia": 14.7,
        "separator_third_stage_temperature_F": -10.0,
        "stock_tank_pressure_psia": 14.7,
        "stock_tank_temperature_F": -10.0,
        "porosity": 0.01,
        "permeability_md": 0.01,
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
        "condensate_api_gravity": 1.0,
        "p_init_psia": 100.0,
        "sw_init": 0.0,
        "well_control_label": "Bottom-hole pressure (BHP)",
        "bhp_psia": 100.0,
        "drawdown_psia": 200.0,
        "min_bhp_psia": 50.0,
        "target_gas_rate_mmscf_day": 0.0,
        "thp_psia": 14.7,
        "tvd_ft": 100.0,
        "tubing_id_in": 2.441,
        "wellhead_temperature_F": -60.0,
        "tubing_roughness_in": 0.0006,
        "tubing_calibration_factor": 1.0,
        "tubing_model": "Mechanistic proxy",
        "thp_friction_coeff": 0.02,
        "productivity_index": 1.0,
        "rw_ft": 0.35,
        "skin": 0.0,
        "injection_rate_value": 0.0,
        "injection_rate_unit": "MMscf/day",
        "injection_control_label": "Enhanced pressure-controlled",
        "injection_pressure_psia": 14.7,
        "max_injection_bhp_psia": 9000.0,
        "injectivity_index_mmscf_day_psi": 0.0,
        "injection_start_date_text": "",
        "injection_end_date_text": "",
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
        "hyst_cgr_label": "Rich  (50–100 STB/MMscf)",
        "hysteresis_gas_trapping_strength": 0.35,
        "hysteresis_imbibition_krg_reduction": 0.50,
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
    start_date_text = _ss("start_date_text", "")
    end_date_text = _ss("end_date_text", "")
    try:
        t_end_days = days_between_dates(start_date_text, end_date_text)
    except Exception:
        t_end_days = None
    dt_days = float(_ss("dt_days", 30.0))

    nx = int(_ss("nx", 40))
    length_ft = float(_ss("length_ft", 4000.0))
    radius_ft = float(_ss("radius_ft", 100.0))
    thickness_ft = float(_ss("thickness_ft", 50.0))
    equivalent_width_ft = 2.0 * radius_ft
    area_ft2 = math.pi * radius_ft ** 2   # cylindrical drainage area: π × r²
    refined_cells = int(_ss("refined_cells", min(10, max(nx - 1, 2))))
    min_dx_ft = float(_ss("min_dx_ft", 50.0))
    growth = float(_ss("growth", 1.45))

    default_initial = np.array([0.0000, 0.0066, 0.0774, 0.6367, 0.0723, 0.0560, 0.0204, 0.0225, 0.0127, 0.0094, 0.0135, 0.0725], dtype=float)
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
    dew_point_psia = float(_ss("dew_point_psia", 100.0))
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
    permeability_md = float(_ss("permeability_md", 0.01))
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
    tubing_model = _ss("tubing_model", "Mechanistic proxy")
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
    injectivity_index_mmscf_day_psi = float(_ss("injectivity_index_mmscf_day_psi", 0.0))
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
        start_date_text = st.text_input("Start date (dd/mm/yyyy)", value=st.session_state.get("start_date_text", ""), key="start_date_text")
        end_date_text = st.text_input("End date (dd/mm/yyyy)", value=st.session_state.get("end_date_text", ""), key="end_date_text")
        try:
            t_end_days = days_between_dates(start_date_text, end_date_text)
            st.caption(f"Simulation length = {t_end_days:.0f} days")
        except ValueError as e:
            t_end_days = None
            st.error(str(e))
            input_validation_errors.append(str(e))
        dt_days = st.number_input("Requested time step (days)", min_value=0.005, max_value=1000.0, value=float(st.session_state.get("dt_days", 30.0)), step=1.0, key="dt_days")
        st.caption("Adaptive timestep retry is enabled.")

    with tab_reservoir_grid:
        st.subheader("Reservoir / Grid")
        nx = st.number_input("Number of grid cells", min_value=10, max_value=200, value=int(st.session_state.get("nx", 10)), step=1, key="nx")
        length_ft = st.number_input("Reservoir length (ft)", min_value=500.0, max_value=20000.0, value=float(st.session_state.get("length_ft", 500.0)), step=100.0, key="length_ft")
        radius_ft = st.number_input("Equivalent drainage radius (ft)", min_value=1.0, max_value=100000.0, value=float(st.session_state.get("radius_ft", 1.0)), step=10.0, key="radius_ft")
        thickness_ft = st.number_input("Reservoir thickness (ft)", min_value=5.0, max_value=500.0, value=float(st.session_state.get("thickness_ft", 5.0)), step=0.1, format="%.1f", key="thickness_ft")
        equivalent_width_ft = 2.0 * radius_ft
        area_ft2 = math.pi * radius_ft ** 2   # cylindrical drainage area: π × r²
        _rc_max = max(int(nx) - 1, 2)
        _rc_val = max(2, min(int(st.session_state.get("refined_cells", min(10, int(nx)-1))), _rc_max))
        refined_cells = st.number_input("Refined cells near well", min_value=2, max_value=_rc_max, value=_rc_val, step=1, key="refined_cells")
        min_dx_ft = st.number_input("Minimum near-well cell size (ft)", min_value=1.0, max_value=500.0, value=float(st.session_state.get("min_dx_ft", 50.0)), step=5.0, key="min_dx_ft")
        growth = st.number_input("LGR growth factor", min_value=1.05, max_value=3.0, value=float(st.session_state.get("growth", 1.45)), step=0.05, key="growth")
        st.caption(f"Equivalent model width = {equivalent_width_ft:,.2f} ft")
        st.caption(f"Drainage cross-section (π × r²) = {area_ft2:,.0f} ft²  ({area_ft2/43560:.1f} acres)")
        st.info(
            "Grid: 1D nonuniform Cartesian with near-well refinement. "
            "The drainage radius defines the cylindrical pore volume: area = π × r². "
            "Producer is in the last cell; injector, when active, is in the first cell."
        )

    with tab_components_composition:
        st.subheader("Components / Composition")
        fluid_components = build_example_fluid().components
        component_names = [c.name for c in fluid_components]

        st.subheader("Initial Hydrocarbon Composition")
        default_initial = np.array([0.0000, 0.0066, 0.0774, 0.6367, 0.0723, 0.0560, 0.0204, 0.0225, 0.0127, 0.0094, 0.0135, 0.0725], dtype=float)
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
        temperature_F = st.number_input("Reservoir temperature (°F)", min_value=-60.0, max_value=740.0, value=float(st.session_state.get("temperature_F", -60.0)), step=10.0, key="temperature_F")
        temperature_R = fahrenheit_to_rankine(temperature_F)
        dew_point_psia = st.number_input("Dew-point pressure (psia)", min_value=100.0, max_value=20000.0, value=float(st.session_state.get("dew_point_psia", 100.0)), step=100.0, key="dew_point_psia")
        separator_stages = int(st.selectbox("Separator train", options=[1, 2, 3, 4], index=3, format_func=lambda x: {1: "1-stage separator", 2: "2-stage separator", 3: "3-stage separator", 4: "Stage 1 + Stage 2 + Stage 3 + Stock Tank"}[x], key="separator_stages"))
        separator_pressure_psia = st.number_input("Stage 1 separator pressure (psia)", min_value=14.7, max_value=5000.0, value=float(st.session_state.get("separator_pressure_psia", 14.7)), step=25.0, key="separator_pressure_psia")
        separator_temperature_F = st.number_input("Stage 1 separator temperature (°F)", min_value=-10.0, max_value=340.0, value=float(st.session_state.get("separator_temperature_F", -10.0)), step=5.0, key="separator_temperature_F")
        separator_temperature_R = fahrenheit_to_rankine(separator_temperature_F)
        separator_second_stage_pressure_psia = st.number_input("Stage 2 separator pressure (psia)", min_value=14.7, max_value=5000.0, value=float(st.session_state.get("separator_second_stage_pressure_psia", 14.7)), step=10.0, disabled=separator_stages < 2, key="separator_second_stage_pressure_psia")
        separator_second_stage_temperature_F = st.number_input("Stage 2 separator temperature (°F)", min_value=-10.0, max_value=340.0, value=float(st.session_state.get("separator_second_stage_temperature_F", -10.0)), step=5.0, disabled=separator_stages < 2, key="separator_second_stage_temperature_F")
        separator_second_stage_temperature_R = fahrenheit_to_rankine(separator_second_stage_temperature_F)
        separator_third_stage_pressure_psia = st.number_input("Stage 3 separator pressure (psia)", min_value=14.7, max_value=5000.0, value=float(st.session_state.get("separator_third_stage_pressure_psia", 14.7)), step=5.0, disabled=separator_stages < 3, key="separator_third_stage_pressure_psia")
        separator_third_stage_temperature_F = st.number_input("Stage 3 separator temperature (°F)", min_value=-10.0, max_value=340.0, value=float(st.session_state.get("separator_third_stage_temperature_F", -10.0)), step=5.0, disabled=separator_stages < 3, key="separator_third_stage_temperature_F")
        separator_third_stage_temperature_R = fahrenheit_to_rankine(separator_third_stage_temperature_F)
        stock_tank_pressure_psia = st.number_input("Stock tank pressure (psia)", min_value=14.7, max_value=100.0, value=float(st.session_state.get("stock_tank_pressure_psia", 14.7)), step=1.0, disabled=separator_stages < 4, key="stock_tank_pressure_psia")
        stock_tank_temperature_F = st.number_input("Stock tank temperature (°F)", min_value=-10.0, max_value=340.0, value=float(st.session_state.get("stock_tank_temperature_F", -10.0)), step=5.0, disabled=separator_stages < 4, key="stock_tank_temperature_F")
        stock_tank_temperature_R = fahrenheit_to_rankine(stock_tank_temperature_F)
        condensate_api_gravity = st.number_input("Condensate API gravity", min_value=1.0, max_value=100.0, value=float(st.session_state.get("condensate_api_gravity", 1.0)), step=1.0, format="%.1f", key="condensate_api_gravity")

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
        porosity = st.number_input("Porosity", min_value=0.01, max_value=0.50, value=float(st.session_state.get("porosity", 0.01)), step=0.01, format="%.2f", key="porosity")
        permeability_md = st.number_input("Permeability (mD)", min_value=0.01, max_value=10000.0, value=float(st.session_state.get("permeability_md", 0.01)), step=1.0, key="permeability_md")
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
        p_init_psia = st.number_input("Initial reservoir pressure (psia)", min_value=100.0, max_value=20000.0, value=float(st.session_state.get("p_init_psia", 100.0)), step=100.0, key="p_init_psia")
        sw_init = st.number_input("Initial water saturation, Swi", min_value=0.0, max_value=0.95, value=float(st.session_state.get("sw_init", 0.0)), step=0.01, format="%.2f", key="sw_init")
        st.caption("Initial hydrocarbon composition is taken from the Components / Composition section.")

    wells_config = None  # populated inside tab_producer_well; used in run_kwargs
    with tab_producer_well:
        st.subheader("Producer Wells")
        n_producers = int(st.number_input(
            "Number of producer wells", min_value=1, max_value=5,
            value=int(st.session_state.get("n_producers", 1)), step=1, key="n_producers",
        ))
        st.caption("Each well is placed at a fractional position along the 1D flow path "
                   "(0 = injector end, 1 = outer boundary end).")

        _control_map = {
            "Fixed BHP":        "bhp",
            "Fixed Drawdown":   "drawdown",
            "Fixed Gas Rate":   "gas_rate",
            "Fixed THP":        "thp",
        }

        wells_config = []
        for _wi in range(n_producers):
            _wk = f"w{_wi}"
            _default_pos = round((_wi + 1) / n_producers, 2)
            with st.expander(f"Well {_wi + 1}", expanded=(_wi == 0)):
                _name  = st.text_input("Well name", value=st.session_state.get(f"{_wk}_name", f"Well {_wi + 1}"), key=f"{_wk}_name")
                _pos   = float(st.slider("Position along flow path", 0.0, 1.0, float(st.session_state.get(f"{_wk}_pos", _default_pos)), step=0.05, key=f"{_wk}_pos", help="0 = injector/left end  ·  1 = outer boundary/right end"))
                _ctrl_label = st.selectbox("Control mode", list(_control_map.keys()), index=0, key=f"{_wk}_ctrl")
                _ctrl  = _control_map[_ctrl_label]
                _bhp   = float(st.number_input("BHP target (psia)",          min_value=100.0,  max_value=20000.0, value=float(st.session_state.get(f"{_wk}_bhp",  100.0)), step=100.0,  disabled=(_ctrl != "bhp"),      key=f"{_wk}_bhp"))
                _dd    = float(st.number_input("Drawdown target (psi)",       min_value=1.0,    max_value=10000.0, value=float(st.session_state.get(f"{_wk}_dd",    1.0)),  step=10.0,   disabled=(_ctrl != "drawdown"), key=f"{_wk}_dd"))
                _qg    = float(st.number_input("Target gas rate (MMscf/day)", min_value=0.0,    max_value=5000.0,  value=float(st.session_state.get(f"{_wk}_qg",    0.0)),    step=0.5,    disabled=(_ctrl != "gas_rate"), key=f"{_wk}_qg"))
                _thp   = float(st.number_input("THP target (psia)",           min_value=14.7,   max_value=10000.0, value=float(st.session_state.get(f"{_wk}_thp",   14.7)),  step=25.0,   disabled=(_ctrl != "thp"),      key=f"{_wk}_thp"))
                _min_bhp = float(st.number_input(
                    "Minimum BHP (psia)", min_value=50.0, max_value=5000.0,
                    value=float(st.session_state.get(f"{_wk}_min_bhp", 500.0)),
                    step=50.0, disabled=(_ctrl not in {"gas_rate", "drawdown"}),
                    key=f"{_wk}_min_bhp",
                    help="Floor BHP for gas-rate and drawdown control. Well enters decline when BHP hits this value.",
                ))
                _tvd   = float(st.number_input("TVD (ft)",              min_value=100.0,  max_value=30000.0, value=float(st.session_state.get(f"{_wk}_tvd",   100.0)),  step=100.0, key=f"{_wk}_tvd"))
                _tid   = float(st.number_input("Tubing ID (in)",        min_value=1.0,    max_value=6.0,     value=float(st.session_state.get(f"{_wk}_tid",  1.0)),   step=0.1,   key=f"{_wk}_tid"))
                _skin  = float(st.number_input("Skin factor",           min_value=-10.0,  max_value=50.0,    value=float(st.session_state.get(f"{_wk}_skin", 0.0)),     step=0.5,   key=f"{_wk}_skin"))
                _rw    = float(st.number_input("Wellbore radius (ft)",  min_value=0.05,   max_value=5.0,     value=float(st.session_state.get(f"{_wk}_rw",   0.05)),    step=0.05,  key=f"{_wk}_rw"))
                _pi    = float(st.number_input("PI multiplier",         min_value=0.01,   max_value=100.0,   value=float(st.session_state.get(f"{_wk}_pi",   0.01)),     step=0.1,   key=f"{_wk}_pi"))
                _wht_F = float(st.number_input("Wellhead temp (°F)",    min_value=-60.0,  max_value=340.0,   value=float(st.session_state.get(f"{_wk}_wht",  -60.0)),    step=5.0,   key=f"{_wk}_wht"))
                wells_config.append({
                    "name":                       _name,
                    "position_fraction":          _pos,
                    "control_mode":               _ctrl,
                    "bhp_psia":                   _bhp,
                    "drawdown_psia":              _dd,
                    "target_gas_rate_mmscf_day":  _qg,
                    "thp_psia":                   _thp,
                    "min_bhp_psia":               _min_bhp,
                    "tvd_ft":                     _tvd,
                    "tubing_id_in":               _tid,
                    "skin":                       _skin,
                    "rw_ft":                      _rw,
                    "productivity_index":         _pi,
                    "wellhead_temperature_R":     fahrenheit_to_rankine(_wht_F),
                    "tubing_model":               "mechanistic",
                    "tubing_roughness_in":        0.0006,
                    "tubing_calibration_factor":  1.0,
                    "thp_friction_coeff":         0.02,
                })

        # Defensive guard
        if not wells_config:
            wells_config = [{"name": "Well 1", "position_fraction": 1.0,
                             "control_mode": "bhp", "bhp_psia": 3000.0,
                             "drawdown_psia": 200.0, "target_gas_rate_mmscf_day": 5.0,
                             "thp_psia": 500.0, "min_bhp_psia": 500.0,
                             "tvd_ft": 8000.0, "tubing_id_in": 2.441,
                             "skin": 0.0, "rw_ft": 0.35, "productivity_index": 1.0,
                             "wellhead_temperature_R": fahrenheit_to_rankine(60.0),
                             "tubing_model": "mechanistic", "tubing_roughness_in": 0.0006,
                             "tubing_calibration_factor": 1.0, "thp_friction_coeff": 0.02}]

        # Expose primary well scalars so existing run_kwargs keys still work
        well_control_mode           = wells_config[0]["control_mode"]
        bhp_psia                    = wells_config[0]["bhp_psia"]
        drawdown_psia               = wells_config[0]["drawdown_psia"]
        target_gas_rate_mmscf_day   = wells_config[0]["target_gas_rate_mmscf_day"]
        thp_psia                    = wells_config[0]["thp_psia"]
        min_bhp_psia                = wells_config[0]["min_bhp_psia"]
        tvd_ft                      = wells_config[0]["tvd_ft"]
        tubing_id_in                = wells_config[0]["tubing_id_in"]
        wellhead_temperature_R      = wells_config[0]["wellhead_temperature_R"]
        tubing_model                = wells_config[0]["tubing_model"]
        tubing_roughness_in         = wells_config[0]["tubing_roughness_in"]
        tubing_calibration_factor   = wells_config[0]["tubing_calibration_factor"]
        thp_friction_coeff          = wells_config[0]["thp_friction_coeff"]
        productivity_index          = wells_config[0]["productivity_index"]
        rw_ft                       = wells_config[0]["rw_ft"]
        skin                        = wells_config[0]["skin"]

    with tab_injector_well:
        st.subheader("Injector Well")
        injection_rate_value = st.number_input("Gas injection rate", min_value=0.0, max_value=1.0e6, value=float(st.session_state.get("injection_rate_value", 0.0)), step=0.5, disabled=(scenario_name == "natural_depletion"), key="injection_rate_value")
        injection_rate_unit = st.selectbox("Gas injection rate unit", ["Mscf/day", "MMscf/day"], index=1, key="injection_rate_unit")
        injection_control_label = st.selectbox("Injection mode", ["Simple schedule-driven", "Enhanced pressure-controlled"], index=1, disabled=(scenario_name == "natural_depletion"), key="injection_control_label")
        injection_control_mode = "enhanced" if injection_control_label == "Enhanced pressure-controlled" else "simple"
        injection_pressure_psia = st.number_input("Injection pressure (psia)", min_value=14.7, max_value=20000.0, value=float(st.session_state.get("injection_pressure_psia", 14.7)), step=100.0, disabled=(scenario_name == "natural_depletion"), key="injection_pressure_psia")
        max_injection_bhp_psia = st.number_input("Maximum bottom-hole injection pressure (psia)", min_value=14.7, max_value=25000.0, value=float(st.session_state.get("max_injection_bhp_psia", 14.7)), step=100.0, disabled=(scenario_name == "natural_depletion" or injection_control_mode != "enhanced"), key="max_injection_bhp_psia")
        injectivity_index_mmscf_day_psi = st.number_input("Injectivity index (MMscf/day/psi)", min_value=0.0, max_value=1.0e4, value=float(st.session_state.get("injectivity_index_mmscf_day_psi", 0.0)), step=0.01, format="%.4f", disabled=(scenario_name == "natural_depletion" or injection_control_mode != "enhanced"), key="injectivity_index_mmscf_day_psi")
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
        left_boundary_pressure_psia = st.number_input("Left boundary pressure (psia)", min_value=14.7, max_value=20000.0, value=float(st.session_state.get("left_boundary_pressure_psia", 14.7)), step=100.0, disabled=(left_boundary_mode != "constant_pressure"), key="left_boundary_pressure_psia")
        left_boundary_transmissibility_multiplier = st.number_input("Left boundary transmissibility multiplier", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="left_boundary_transmissibility_multiplier")
        right_boundary_label = st.selectbox("Right boundary", ["Closed", "Constant pressure"], index=0, key="right_boundary_label")
        right_boundary_mode = "closed" if right_boundary_label == "Closed" else "constant_pressure"
        right_boundary_pressure_psia = st.number_input("Right boundary pressure (psia)", min_value=14.7, max_value=20000.0, value=float(st.session_state.get("right_boundary_pressure_psia", 14.7)), step=100.0, disabled=(right_boundary_mode != "constant_pressure"), key="right_boundary_pressure_psia")
        right_boundary_transmissibility_multiplier = st.number_input("Right boundary transmissibility multiplier", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="right_boundary_transmissibility_multiplier")
        aquifer_enabled = st.checkbox("Enable analytic aquifer", value=False, key="aquifer_enabled")
        aquifer_side = st.selectbox("Aquifer side", ["left", "right"], index=0, disabled=(not aquifer_enabled), key="aquifer_side")
        aquifer_initial_pressure_psia = st.number_input("Aquifer initial pressure (psia)", min_value=14.7, max_value=20000.0, value=float(st.session_state.get("aquifer_initial_pressure_psia", 14.7)), step=100.0, disabled=(not aquifer_enabled), key="aquifer_initial_pressure_psia")
        _aq_pi_default_stb_day_psi = 500.0 / water_lbmol_per_stb()
        _aq_cap_default_mstb_psi = 5.0e5 / water_lbmol_per_mstb()
        aquifer_productivity_index_stb_day_psi = st.number_input("Aquifer productivity index (STB/day/psi)", min_value=0.0, max_value=1.0e7, value=float(st.session_state.get("aquifer_productivity_index_stb_day_psi", 0.0)), step=1.0, format="%.2f", disabled=(not aquifer_enabled), key="aquifer_productivity_index_stb_day_psi")
        aquifer_total_capacity_mstb_per_psi = st.number_input("Aquifer total capacity (MSTB/psi)", min_value=0.001, max_value=1.0e7, value=float(st.session_state.get("aquifer_total_capacity_mstb_per_psi", 0.001)), step=1.0, format="%.2f", disabled=(not aquifer_enabled), key="aquifer_total_capacity_mstb_per_psi")
        aquifer_water_influx_fraction = st.number_input("Aquifer influx fraction", min_value=0.0, max_value=1.0, value=float(st.session_state.get("aquifer_water_influx_fraction", 0.0)), step=0.05, format="%.2f", disabled=(not aquifer_enabled), key="aquifer_water_influx_fraction")
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
        hysteresis_reversal_tolerance = st.number_input("Reversal tolerance in Sg", min_value=0.0, max_value=0.30, value=0.01, step=0.005, format="%.3f", disabled=not hysteresis_enabled, key="hysteresis_reversal_tolerance")

        # ── Literature-based hysteresis calibration helper ─────────────────
        # Parameter recommendations derived from:
        #   Land (1968) — trapping coefficient framework
        #   Skauge & Larsen (1994) — gas condensate core measurements
        #   Blom & Hagoort (1998) — krg reduction in condensate systems
        #   Al-Anazi et al. (2005) — rich condensate specific recommendations
        _CGR_PRESETS = {
            "Lean  (<20 STB/MMscf)":       {"trapping": 0.55, "krg_red": 0.65, "kro_red": 0.10,
                                             "land_c_lo": 3.0, "land_c_hi": 4.0,
                                             "ref": "Land (1968); Carlson (1981)"},
            "Moderate  (20–50 STB/MMscf)": {"trapping": 0.45, "krg_red": 0.55, "kro_red": 0.12,
                                             "land_c_lo": 2.5, "land_c_hi": 3.5,
                                             "ref": "Skauge & Larsen (1994)"},
            "Rich  (50–100 STB/MMscf)":    {"trapping": 0.35, "krg_red": 0.50, "kro_red": 0.15,
                                             "land_c_lo": 1.5, "land_c_hi": 2.5,
                                             "ref": "Al-Anazi et al. (2005); Blom & Hagoort (1998)"},
            "Very rich  (>100 STB/MMscf)": {"trapping": 0.25, "krg_red": 0.40, "kro_red": 0.18,
                                             "land_c_lo": 1.0, "land_c_hi": 2.0,
                                             "ref": "Blom & Hagoort (1998); Al-Anazi et al. (2005)"},
            "Manual (no preset)":           None,
        }
        _cgr_labels = list(_CGR_PRESETS.keys())
        _cgr_default = _ss("hyst_cgr_label", "Rich  (50–100 STB/MMscf)")
        if _cgr_default not in _cgr_labels:
            _cgr_default = "Rich  (50–100 STB/MMscf)"
        _cgr_label = st.selectbox(
            "Fluid CGR range (for literature-based calibration)",
            _cgr_labels,
            index=_cgr_labels.index(_cgr_default),
            key="hyst_cgr_label",
            disabled=not hysteresis_enabled,
            help="Selects published parameter recommendations for your CGR range. "
                 "Values are pre-filled below but remain fully editable.",
        )
        _preset = _CGR_PRESETS[_cgr_label]
        if _preset is not None:
            st.caption(
                f"Recommended: trapping strength {_preset['trapping']:.2f}, "
                f"krg reduction {_preset['krg_red']:.2f}, "
                f"kro reduction {_preset['kro_red']:.2f} "
                f"(Land C ≈ {_preset['land_c_lo']:.1f}–{_preset['land_c_hi']:.1f}). "
                f"Source: {_preset['ref']}."
            )
            _trap_default = _preset["trapping"]
            _krg_default  = _preset["krg_red"]
            _kro_default  = _preset["kro_red"]
        else:
            _trap_default = float(_ss("hysteresis_gas_trapping_strength", 0.60))
            _krg_default  = float(_ss("hysteresis_imbibition_krg_reduction", 0.75))
            _kro_default  = float(_ss("hysteresis_imbibition_kro_reduction", 0.15))

        hysteresis_gas_trapping_strength = st.number_input(
            "Gas trapping strength  [higher = more trapping]",
            min_value=0.0, max_value=1.0,
            value=float(_trap_default),
            step=0.05, format="%.2f",
            disabled=not hysteresis_enabled,
            key="hysteresis_gas_trapping_strength",
            help="Fraction of gas saturation that becomes trapped during imbibition. "
                 "Lower values = less trapping. Al-Anazi et al. (2005): 0.25–0.45 for rich condensates.",
        )
        hysteresis_imbibition_krg_reduction = st.number_input(
            "Imbibition krg reduction multiplier",
            min_value=0.0, max_value=1.5,
            value=float(_krg_default),
            step=0.05, format="%.2f",
            disabled=not hysteresis_enabled,
            key="hysteresis_imbibition_krg_reduction",
            help="Multiplier applied to krg during imbibition (condensate re-evaporation). "
                 "Blom & Hagoort (1998): 0.40–0.60 for rich condensates (CGR > 50 STB/MMscf).",
        )
        hysteresis_imbibition_kro_reduction = st.number_input(
            "Imbibition kro reduction multiplier",
            min_value=0.0, max_value=1.0,
            value=float(_kro_default),
            step=0.05, format="%.2f",
            disabled=not hysteresis_enabled,
            key="hysteresis_imbibition_kro_reduction",
            help="Multiplier applied to kro during imbibition. Less sensitive for gas condensate systems. "
                 "Typical range 0.10–0.20.",
        )
        # ── End calibration helper ─────────────────────────────────────────
        transport_enabled = st.checkbox("Enable dispersive transport", value=True, key="transport_enabled")
        transport_phase_split_advection = st.checkbox("Use phase-split hydrocarbon advection", value=True, key="transport_phase_split_advection")
        transport_dispersivity_ft = st.number_input("Longitudinal dispersivity (ft)", min_value=0.0, max_value=5000.0, value=15.0, step=1.0, key="transport_dispersivity_ft")
        transport_molecular_diffusion_ft2_day = st.number_input("Molecular diffusion (ft²/day)", min_value=0.0, max_value=100.0, value=0.15, step=0.05, format="%.3f", key="transport_molecular_diffusion_ft2_day")
        transport_max_dispersive_fraction = st.number_input("Max dispersive/advective flux ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.05, format="%.2f", key="transport_max_dispersive_fraction")

        st.subheader("Condensate Bank Impairment")
        st.caption(
            "Impairment is evaluated at the near-wellbore pressure (geometric mean of BHP and "
            "reservoir pressure) using molar liquid dropout as the damage driver. "
            "Damage factor = exp(−strength × max(dropout − critical, 0)), floored at the minimum value."
        )
        condensate_damage_strength = st.number_input(
            "Damage strength", min_value=0.0, max_value=20.0, value=3.0, step=0.5, format="%.1f",
            key="condensate_damage_strength",
            help="Exponential decay rate. Higher values give sharper onset of impairment once dropout exceeds the critical threshold.",
        )
        condensate_critical_dropout = st.number_input(
            "Critical liquid dropout (molar fraction)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f",
            key="condensate_critical_dropout",
            help="Molar liquid fraction below which no productivity impairment is applied.",
        )
        condensate_min_damage_factor = st.number_input(
            "Minimum damage factor", min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f",
            key="condensate_min_damage_factor",
            help="Floor on the damage factor — the well never loses more than (1 − this value) of its undamaged productivity index.",
        )


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
        st.caption("Results, plots, tables, and downloads appear here after you click **Run Simulation**.")

    st.sidebar.divider()
    run_clicked = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

    if not run_clicked:
        return

    if not relperm_input_valid:
        input_validation_errors.append("Invalid relative permeability inputs. Check saturation constraints.")

    # Check that all required inputs have been supplied by the user
    _sentinel_checks = [
        # Dates
        (not start_date_text.strip(),  "Case Setup → Start date must be entered (dd/mm/yyyy)"),
        (not end_date_text.strip(),    "Case Setup → End date must be entered (dd/mm/yyyy)"),
        # Reservoir geometry
        (float(radius_ft)       <= 1.0,   "Reservoir / Grid → Drainage radius must be entered (currently at minimum 1 ft)"),
        (float(thickness_ft)    <= 5.0,   "Reservoir / Grid → Reservoir thickness must be entered (currently at minimum 5 ft)"),
        (float(length_ft)       <= 500.0, "Reservoir / Grid → Reservoir length must be entered (currently at minimum 500 ft)"),
        # Rock
        (float(porosity)        <= 0.01,  "Rock & Rock-Fluid → Porosity must be entered (currently at minimum 0.01)"),
        (float(permeability_md) <= 0.01,  "Rock & Rock-Fluid → Permeability must be entered (currently at minimum 0.01 mD)"),
        # Components & PVT
        (float(temperature_F)   <= -60.0, "Components & PVT → Reservoir temperature must be entered"),
        (float(dew_point_psia)  <= 100.0, "Components & PVT → Dew-point pressure must be entered"),
        (float(separator_pressure_psia) <= 14.7, "Components & PVT → Stage 1 separator pressure must be entered"),
        (float(separator_temperature_F) <= -10.0, "Components & PVT → Stage 1 separator temperature must be entered"),
        (float(condensate_api_gravity)  <= 1.0,   "Components & PVT → Condensate API gravity must be entered"),
        # Initial conditions
        (float(p_init_psia)     <= 100.0, "Initial Conditions → Initial reservoir pressure must be entered"),
        # Producer well — check each defined producer using the already-built wells_config
        *[
            (w["control_mode"] == "bhp" and float(w["bhp_psia"]) <= 100.0,
             f"Producer Well {_wi+1} → BHP target must be entered (currently at minimum 100 psia)")
            for _wi, w in enumerate(wells_config)
        ],
        *[
            (w["control_mode"] == "gas_rate" and float(w["target_gas_rate_mmscf_day"]) <= 0.0,
             f"Producer Well {_wi+1} → Gas rate target must be entered")
            for _wi, w in enumerate(wells_config)
        ],
        *[
            (float(w["tvd_ft"]) <= 100.0,
             f"Producer Well {_wi+1} → TVD must be entered (currently at minimum 100 ft)")
            for _wi, w in enumerate(wells_config)
        ],
    ]
    for condition, msg in _sentinel_checks:
        if condition:
            input_validation_errors.append(msg)

    if input_validation_errors:
        st.error("Please supply all required inputs before running the simulation.")
        for _err in input_validation_errors:
            st.warning(_err)
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
        condensate_damage_strength=float(condensate_damage_strength),
        condensate_critical_dropout=float(condensate_critical_dropout),
        condensate_min_damage_factor=float(condensate_min_damage_factor),
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
        condensate_api_gravity=float(condensate_api_gravity),
        pvt_table=pvt_table,
        wells_config=wells_config if wells_config else None,
    )

    # Show the live timer inside the Results tab so it doesn't bleed across tabs
    with tab_results:
        timer_placeholder = st.empty()

    try:
        result, _elapsed_total = _timed_run_example(
            run_kwargs, "Compositional simulation", 1, 1, timer_placeholder
        )
    except Exception as e:
        with tab_results:
            timer_placeholder.empty()
            st.error(f"Simulation failed: {e}")
        return

    # Clear the live timer — the success banner below replaces it
    timer_placeholder.empty()

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
            comp_timer = st.empty()
            for idx, label in enumerate(selected_comparison_labels):
                compare_kwargs = dict(run_kwargs)
                compare_kwargs["scenario_name"] = comparison_map[label]
                if comparison_map[label] == "natural_depletion":
                    compare_kwargs["injection_rate_value"] = 0.0
                comp_timer.info(f"⏱ Scenario comparison — running **{label}** ({idx+1}/{len(selected_comparison_labels)})…")
                try:
                    comp_result, _ = _timed_run_example(compare_kwargs, label, idx + 1, len(selected_comparison_labels), comp_timer)
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
            comp_timer.empty()
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
        sens_timer = st.empty()
        n_sens_cases = len(pct_cases)
        for sens_idx, (case_label, pct_change) in enumerate(pct_cases):
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
                sens_result, _ = _timed_run_example(case_kwargs, f"{case_label} ({pct_change:+.0f}%)", sens_idx + 1, n_sens_cases, sens_timer)
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
        sens_timer.empty()
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
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time as _time

            rng = np.random.default_rng(int(monte_carlo_seed))
            n_mc_runs = int(monte_carlo_runs)

            # Build all case kwargs up-front (deterministic sampling order)
            mc_cases = []
            for run_idx in range(n_mc_runs):
                case_kwargs = dict(run_kwargs)
                sampled_inputs = {}
                for label in selected_mc:
                    param_key, param_title = mc_param_map[label]
                    base_value = float(run_kwargs.get(param_key, 0.0))
                    sampled_value = _sample_parameter_value(
                        param_key, base_value,
                        float(monte_carlo_low_pct), float(monte_carlo_high_pct),
                        rng, str(case_kwargs.get("scenario_name", "")),
                    )
                    case_kwargs[param_key] = sampled_value
                    sampled_inputs[param_title] = sampled_value
                mc_cases.append((run_idx, case_kwargs, sampled_inputs))

            mc_rows = []
            mc_errors = []
            mc_timer = st.empty()
            completed = [0]

            def _run_mc_case(args):
                run_idx, ckwargs, sinputs = args
                result = run_example(**ckwargs)
                return run_idx, result["history"], sinputs

            # Use up to 4 worker threads — each run is CPU-bound but GIL is released
            # during NumPy operations, giving meaningful parallelism.
            n_workers = min(4, n_mc_runs)
            t0_mc = _time.monotonic()
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_mc_case, args): args[0] for args in mc_cases}
                for fut in as_completed(futures):
                    elapsed = _time.monotonic() - t0_mc
                    completed[0] += 1
                    mins, secs = divmod(int(elapsed), 60)
                    mc_timer.info(
                        f"⏱ Monte Carlo — **{completed[0]}/{n_mc_runs}** runs done "
                        f"(**{mins:02d}:{secs:02d}** elapsed)"
                    )
                    try:
                        run_idx, mc_history, sampled_inputs = fut.result()
                        row = {
                            "Run": run_idx + 1,
                            "Final Avg Pressure (psia)": float(mc_history.avg_pressure_psia[-1]),
                            "Final Well Pressure (psia)": float(mc_history.well_pressure_psia[-1]),
                            "Final Gas Rate (MMscf/day)": float(mc_history.gas_rate_mmscf_day[-1]),
                            "Final Condensate Rate (STB/day)": float(mc_history.condensate_rate_stb_day[-1]),
                            "Cumulative Gas (Bscf)": float(mc_history.cum_gas_bscf[-1]),
                            "Cumulative Condensate (MSTB)": float(mc_history.cum_condensate_mstb[-1]),
                            "Cumulative Water (MSTB)": float(mc_history.cum_water_mstb[-1]),
                            "Final Well P - Dew Point (psia)": float(mc_history.well_pressure_minus_dewpoint_psia[-1]),
                            "Final Productivity Loss Fraction": float(mc_history.productivity_loss_fraction[-1]),
                        }
                        row.update(sampled_inputs)
                        mc_rows.append(row)
                    except Exception as e:
                        mc_errors.append(f"Run {futures[fut] + 1}: {e}")

            mc_timer.success(
                f"✅ Monte Carlo complete — {len(mc_rows)}/{n_mc_runs} successful "
                f"in **{_time.monotonic() - t0_mc:.1f} s**"
            )
            if mc_rows:
                mc_df = pd.DataFrame(mc_rows).sort_values("Run").reset_index(drop=True)
                summary_rows = []
                for metric in [
                    "Final Avg Pressure (psia)",
                    "Final Well Pressure (psia)",
                    "Final Gas Rate (MMscf/day)",
                    "Final Condensate Rate (STB/day)",
                    "Cumulative Gas (Bscf)",
                    "Cumulative Condensate (MSTB)",
                    "Final Well P - Dew Point (psia)",
                    "Final Productivity Loss Fraction",
                ]:
                    vals = pd.to_numeric(mc_df[metric], errors="coerce").to_numpy(dtype=float)
                    p10, p50, p90 = _percentile_summary(vals)
                    summary_rows.append({
                        "Metric": metric,
                        "Mean": float(np.nanmean(vals)),
                        "Std Dev": float(np.nanstd(vals)),
                        "P10": p10, "P50": p50, "P90": p90,
                        "Min": float(np.nanmin(vals)),
                        "Max": float(np.nanmax(vals)),
                    })
                monte_carlo_bundle = {
                    "runs_df": mc_df,
                    "summary_df": pd.DataFrame(summary_rows),
                    "selected_labels": selected_mc,
                    "errors": mc_errors,
                    "successful_runs": len(mc_df),
                    "requested_runs": n_mc_runs,
                }
            elif mc_errors:
                st.warning("Monte Carlo analysis could not be computed. " + "; ".join(mc_errors[:5]))
        else:
            st.info("Select at least one uncertain parameter to run Monte Carlo analysis.")

    with tab_results:
        st.success(f"Simulation complete in {_elapsed_total:.0f} seconds.")

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
            _wpdp = history.well_pressure_minus_dewpoint_psia[-1]
            c3.metric(
                "Well P − Dew Point (psia)",
                f"{_wpdp:.2f}",
                help=(
                    "Flowing well pressure minus dew-point pressure. "
                    "Negative values mean the well is producing below the dew point — "
                    "condensate is dropping out in the reservoir near the wellbore. "
                    f"Here: Pwf={history.well_flowing_pwf_psia[-1]:.0f} psia, "
                    f"dew point={history.dew_point_psia[-1]:.0f} psia."
                ),
            )
    
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

            # ── Recovery factors ─────────────────────────────────────────────
            volumetrics = result.get("volumetrics") or {}
            giip_bscf = volumetrics.get("giip_bscf", None)
            ciip_mstb = volumetrics.get("ciip_mstb", None)
            cum_gas_bscf  = float(history.cum_gas_bscf[-1])
            cum_cond_mstb = float(history.cum_condensate_mstb[-1])

            st.subheader("Recovery Factors")
            rf1, rf2 = st.columns(2)
            if giip_bscf and np.isfinite(giip_bscf) and giip_bscf > 0:
                rf_gas = cum_gas_bscf / giip_bscf * 100.0
                rf1.metric(
                    "Gas Recovery Factor",
                    f"{rf_gas:.2f}%",
                    help=f"{cum_gas_bscf:.4f} Bscf produced / {giip_bscf:.3f} Bscf GIIP",
                )
            else:
                rf1.metric("Gas Recovery Factor", "N/A", help="GIIP not available")
            if ciip_mstb and np.isfinite(ciip_mstb) and ciip_mstb > 0:
                rf_cond = cum_cond_mstb / ciip_mstb * 100.0
                rf2.metric(
                    "Condensate Recovery Factor",
                    f"{rf_cond:.2f}%",
                    help=f"{cum_cond_mstb:.2f} MSTB produced / {ciip_mstb:.2f} MSTB CIIP",
                )
            else:
                rf2.metric("Condensate Recovery Factor", "N/A", help="CIIP not available")

            # ── Grid diagram ─────────────────────────────────────────────────
            st.subheader("Grid Diagram")
            _g   = result["grid"]
            _nx  = _g.nx
            _dx  = _g.dx_array
            _x_edges = np.concatenate([[0.0], np.cumsum(_dx)])
            _x_centers = 0.5 * (_x_edges[:-1] + _x_edges[1:])
            _L   = float(_g.length_ft)

            # Identify producer and injector cells
            _wells_list = result.get("wells", [result["well"]])
            _scenario   = result.get("scenario")
            _injector   = getattr(_scenario, "injector", None) if _scenario else None

            # Build SVG
            W, H = 700, 160
            cell_h = 60
            y_top  = 50
            y_bot  = y_top + cell_h

            def _cx(cell_i):
                """Cell centre x in SVG coords."""
                return int(W * _x_centers[cell_i] / max(_L, 1e-6))

            def _cell_x0(cell_i):
                return int(W * _x_edges[cell_i] / max(_L, 1e-6))

            def _cell_x1(cell_i):
                return int(W * _x_edges[cell_i + 1] / max(_L, 1e-6))

            # Colour for each cell: default, producer, injector
            producer_cells  = {w.cell_index for w in _wells_list}
            injector_cells  = {_injector.cell_index} if _injector else set()

            rects = []
            for ci in range(_nx):
                x0, x1 = _cell_x0(ci), _cell_x1(ci)
                if ci in injector_cells:
                    fill, stroke = "#b5d4f4", "#185fa5"
                elif ci in producer_cells:
                    fill, stroke = "#f5c4b3", "#993c1d"
                else:
                    fill, stroke = "#f1efe8", "#5f5e5a"
                rects.append(
                    f'<rect x="{x0}" y="{y_top}" width="{x1-x0}" height="{cell_h}" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" rx="2"/>'
                )

            # Well symbols and labels
            well_marks = []
            for w in _wells_list:
                ci = w.cell_index
                cx = _cx(ci)
                name = getattr(w, "name", "Producer")
                well_marks.append(
                    f'<line x1="{cx}" y1="{y_bot}" x2="{cx}" y2="{y_bot+28}" '
                    f'stroke="#993c1d" stroke-width="2.5"/>'
                )
                well_marks.append(
                    f'<circle cx="{cx}" cy="{y_bot+34}" r="8" fill="#993c1d"/>'
                )
                well_marks.append(
                    f'<text x="{cx}" y="{y_top-10}" text-anchor="middle" '
                    f'font-size="11" fill="#993c1d" font-family="sans-serif">{name}</text>'
                )
            if _injector:
                ci  = _injector.cell_index
                cx  = _cx(ci)
                well_marks.append(
                    f'<line x1="{cx}" y1="{y_bot}" x2="{cx}" y2="{y_bot+28}" '
                    f'stroke="#185fa5" stroke-width="2.5"/>'
                )
                well_marks.append(
                    f'<polygon points="{cx},{y_bot+26} {cx-8},{y_bot+42} {cx+8},{y_bot+42}" '
                    f'fill="#185fa5"/>'
                )
                well_marks.append(
                    f'<text x="{cx}" y="{y_top-10}" text-anchor="middle" '
                    f'font-size="11" fill="#185fa5" font-family="sans-serif">Injector</text>'
                )

            # Axis ticks: 0, L/2, L
            axis = []
            for frac, label in [(0.0, "0 ft"), (0.5, f"{_L/2:,.0f} ft"), (1.0, f"{_L:,.0f} ft")]:
                ax = int(W * frac)
                axis.append(f'<line x1="{ax}" y1="{y_bot}" x2="{ax}" y2="{y_bot+6}" stroke="#888" stroke-width="1"/>')
                axis.append(
                    f'<text x="{ax}" y="{y_bot+18}" text-anchor="middle" '
                    f'font-size="10" fill="#888" font-family="sans-serif">{label}</text>'
                )

            # Legend
            legend = (
                f'<rect x="10" y="{H-28}" width="12" height="12" fill="#f5c4b3" stroke="#993c1d" stroke-width="1" rx="2"/>'
                f'<text x="26" y="{H-18}" font-size="10" fill="#5f5e5a" font-family="sans-serif">Producer</text>'
            )
            if _injector:
                legend += (
                    f'<rect x="90" y="{H-28}" width="12" height="12" fill="#b5d4f4" stroke="#185fa5" stroke-width="1" rx="2"/>'
                    f'<text x="106" y="{H-18}" font-size="10" fill="#5f5e5a" font-family="sans-serif">Injector</text>'
                )
            legend += (
                f'<rect x="{(160 if _injector else 90)}" y="{H-28}" width="12" height="12" fill="#f1efe8" stroke="#5f5e5a" stroke-width="1" rx="2"/>'
                f'<text x="{(176 if _injector else 106)}" y="{H-18}" font-size="10" fill="#5f5e5a" font-family="sans-serif">Reservoir cell</text>'
            )

            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%">'
                f'<title>1D reservoir grid with {_nx} cells</title>'
                f'<desc>Grid showing {len(_wells_list)} producer(s)'
                + (f' and 1 injector' if _injector else '') + '</desc>'
                + "".join(rects)
                + "".join(well_marks)
                + "".join(axis)
                + legend
                + '</svg>'
            )
            st.components.v1.html(svg, height=H + 20)
    
            if result.get("per_well_df") is not None and len(result["per_well_df"]) > 1:
                st.subheader("Per-Well Breakdown (final timestep)")
                st.dataframe(result["per_well_df"], use_container_width=True)
    
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
            d3.write(f"Drainage cross-section area π×r² (ft²): {area_ft2:,.0f}  ({area_ft2/43560:.1f} acres)")
            d4.write(f"Near-well minimum dx (ft): {grid.cell_width(well.cell_index):.3f}")
    
            d5 = st.columns(1)[0]
            d5.write(f"Peaceman WI geom: {sim.peaceman_well_index(well.cell_index):.5f}")
    
        with result_tabs[1]:
            figs = [
                make_line_figure(time_axis, [(annual_history_df["avg_pressure_psia"].to_numpy(dtype=float), "Average Reservoir Pressure"), (annual_history_df["well_flowing_pwf_psia"].to_numpy(dtype=float), "Flowing BHP (Pwf)"), (annual_history_df["dew_point_psia"].to_numpy(dtype=float), "Dew Point")], time_axis_label, "Pressure (psia)", "Pressure and Dew-Point History"),
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
