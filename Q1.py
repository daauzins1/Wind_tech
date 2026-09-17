import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DTU 10 MW WIND TURBINE - ASSIGNMENT #1 - QUESTION 1
# BEM: Cp(lambda, theta_p) and CT(lambda, theta_p)
# Momentum equations (1) and (2) from the assignment
# ============================================================

# -------------------------
# 1. Turbine constants
# -------------------------
R = 89.17              # rotor radius [m]
B = 3                  # number of blades
rho = 1.225            # air density [kg/m^3]
V0 = 10.0              # reference wind speed [m/s]
A = np.pi * R**2       # rotor swept area [m^2]

# -------------------------
# 2. Blade geometry
# r [m], chord [m], twist beta [deg], t/c [%]
# -------------------------
blade = np.array([
    [2.80, 5.38, 14.50, 100.00],
    [11.00, 5.45, 14.43, 86.05],
    [16.87, 5.87, 12.55, 61.10],
    [22.96, 6.18, 8.89, 43.04],
    [32.31, 6.02, 6.38, 32.42],
    [41.57, 5.42, 4.67, 27.81],
    [50.41, 4.70, 2.89, 25.32],
    [58.53, 4.00, 1.21, 24.26],
    [65.75, 3.40, -0.13, 24.10],
    [71.97, 2.91, -1.11, 24.10],
    [77.19, 2.54, -1.86, 24.10],
    [78.71, 2.43, -2.08, 24.10],
    [80.14, 2.33, -2.28, 24.10],
    [82.71, 2.13, -2.64, 24.10],
    [84.93, 1.90, -2.95, 24.10],
    [86.83, 1.63, -3.18, 24.10],
    [88.45, 1.18, -3.36, 24.10],
    [89.17, 0.60, -3.43, 24.10],
], dtype=float)

# -----------------------------------------------------------------
# 3. Airfoil data
#    The code accepts either the filenames shown in the assignment
#    or your local uploaded names with "(6)" added by ChatGPT.
# -----------------------------------------------------------------

def find_file(base_dir: Path, normal_name: str) -> Path:
    p = base_dir / normal_name
    if p.exists():
        return p

    stem = p.stem
    suffix = p.suffix
    p2 = base_dir / f"{stem}(6){suffix}"
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"Cannot find {normal_name} or {p2.name} in:\n{base_dir}"
    )


def load_airfoils(base_dir: Path):
    # t/c [%] : filename
    files = {
        100.0: "cylinder.txt",
        60.0: "FFA-W3-600.txt",
        48.0: "FFA-W3-480.txt",
        36.0: "FFA-W3-360.txt",
        30.1: "FFA-W3-301.txt",
        24.1: "FFA-W3-241.txt",
    }

    data = {}

    for tc, name in files.items():
        file_path = find_file(base_dir, name)
        raw = np.loadtxt(file_path, usecols=(0, 1, 2), dtype=float)
        raw = raw[np.argsort(raw[:, 0])]

        # Remove duplicate alpha rows if any
        alpha, unique_idx = np.unique(raw[:, 0], return_index=True)
        cl = raw[unique_idx, 1]
        cd = raw[unique_idx, 2]

        data[tc] = (alpha, cl, cd)

    return data


# -----------------------------------------------------------------
# 4. Interpolate Cl and Cd in:
#       (a) angle of attack
#       (b) thickness-to-chord ratio
# -----------------------------------------------------------------

def interpolate_cl_cd(tc, alpha_deg, airfoil_data):
    thicknesses = np.array(sorted(airfoil_data.keys()), dtype=float)

    tc = np.asarray(tc, dtype=float)
    alpha_deg = np.asarray(alpha_deg, dtype=float)

    # Interpolate alpha for every available airfoil.
    cl_all = []
    cd_all = []

    for t in thicknesses:
        alpha_tab, cl_tab, cd_tab = airfoil_data[t]
        cl_all.append(np.interp(alpha_deg, alpha_tab, cl_tab))
        cd_all.append(np.interp(alpha_deg, alpha_tab, cd_tab))

    cl_all = np.asarray(cl_all)
    cd_all = np.asarray(cd_all)

    # Find thickness bracket.
    idx = np.searchsorted(thicknesses, tc, side="right")
    idx = np.clip(idx, 1, len(thicknesses) - 1)

    i0 = idx - 1
    i1 = idx
    row = np.arange(alpha_deg.size)

    t0 = thicknesses[i0]
    t1 = thicknesses[i1]

    w = np.divide(
        tc - t0,
        t1 - t0,
        out=np.zeros_like(tc, dtype=float),
        where=(t1 != t0),
    )

    cl0 = cl_all[i0, row]
    cl1 = cl_all[i1, row]
    cd0 = cd_all[i0, row]
    cd1 = cd_all[i1, row]

    cl = cl0 + w * (cl1 - cl0)
    cd = cd0 + w * (cd1 - cd0)

    return cl, cd


# -----------------------------------------------------------------
# 5. Momentum Equation (1)
#
# Assignment:
# CT/F = 4 a (1-a),                    a <= 0.33
# CT/F = 4 a [1 - 1/4 (5 - 3a) a],      a > 0.33
#
# For the high-induction branch we solve the cubic monotonically
# by bisection instead of using a fragile closed-form expression.
# -----------------------------------------------------------------

def axial_induction_eq1(ct_over_f):
    x = np.asarray(ct_over_f, dtype=float)
    out = np.zeros_like(x)

    finite = np.isfinite(x)
    positive = finite & (x > 0.0)

    # Low-induction branch
    low = positive & (x <= 8.0 / 9.0)
    out[low] = 0.5 * (
        1.0 - np.sqrt(np.maximum(0.0, 1.0 - x[low]))
    )

    # High-induction branch:
    # x = 4a - 5a^2 + 3a^3, 1/3 < a < 1
    high = positive & (x > 8.0 / 9.0)

    if np.any(high):
        target = x[high]
        lo = np.full(target.shape, 1.0 / 3.0)
        hi = np.ones_like(target)

        # At a = 1, the cubic gives x = 2.
        too_large = target >= 2.0
        out_high = np.empty_like(target)
        out_high[too_large] = 0.999999

        solve = ~too_large
        if np.any(solve):
            lo_s = lo[solve]
            hi_s = hi[solve]
            target_s = target[solve]

            for _ in range(50):
                mid = 0.5 * (lo_s + hi_s)
                fmid = 4.0 * mid - 5.0 * mid**2 + 3.0 * mid**3
                move_right = fmid < target_s
                lo_s[move_right] = mid[move_right]
                hi_s[~move_right] = mid[~move_right]

            out_high[solve] = 0.5 * (lo_s + hi_s)

        out[high] = out_high

    return np.clip(out, 0.0, 0.999999)


# -----------------------------------------------------------------
# 6. Momentum Equation (2) from the assignment
#
# a = 0.246 x + 0.0586 x^2 + 0.0883 x^3,
# where x = CT/F
# -----------------------------------------------------------------

def axial_induction_eq2(ct_over_f):
    x = np.asarray(ct_over_f, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.maximum(x, 0.0)

    a = (
        0.246 * x
        + 0.0586 * x**2
        + 0.0883 * x**3
    )

    return np.clip(a, 0.0, 0.95)


# -----------------------------------------------------------------
# 7. One BEM solution for one (lambda, pitch)
# -----------------------------------------------------------------

def bem(lam, pitch_deg, equation, airfoil_data, n_radial=80):
    # Use a smooth radial grid based on the supplied blade stations.
    r = np.linspace(blade[0, 0], R, n_radial)

    chord = np.interp(r, blade[:, 0], blade[:, 1])
    beta = np.interp(r, blade[:, 0], blade[:, 2])
    tc = np.interp(r, blade[:, 0], blade[:, 3])

    omega = lam * V0 / R

    # Initial guesses
    a = np.full_like(r, 0.2)
    a_prime = np.full_like(r, 0.01)

    relaxation = 0.30

    for _ in range(100):
        # Flow angle
        phi = np.arctan2(
            V0 * (1.0 - a),
            omega * r * (1.0 + a_prime),
        )

        sin_phi = np.maximum(np.abs(np.sin(phi)), 1.0e-8)
        cos_phi = np.cos(phi)

        # Prandtl tip loss, as requested in the assignment.
        exponent = -(B / 2.0) * (R - r) / (r * sin_phi)
        F = (2.0 / np.pi) * np.arccos(
            np.clip(np.exp(exponent), 0.0, 1.0)
        )

        # The tip element itself has zero aerodynamic load.
        F[-1] = 1.0
        F = np.clip(F, 1.0e-4, 1.0)

        # Angle of attack
        alpha = np.degrees(phi) - beta - pitch_deg

        # Airfoil coefficients
        cl, cd = interpolate_cl_cd(tc, alpha, airfoil_data)

        # Normal and tangential force coefficients
        cn = cl * cos_phi + cd * sin_phi
        ct = cl * sin_phi - cd * cos_phi

        # Local solidity
        sigma = B * chord / (2.0 * np.pi * r)

        # Blade-element local thrust coefficient CT/F
        ct_over_f = (
            sigma * cn * (1.0 - a)**2
            / np.maximum(sin_phi**2, 1.0e-12)
        )

        if equation == 1:
            a_new = axial_induction_eq1(ct_over_f)
        elif equation == 2:
            a_new = axial_induction_eq2(ct_over_f)
        else:
            raise ValueError("equation must be 1 or 2")

        # Tangential induction from BEM momentum relation
        denominator = 4.0 * F * sin_phi * cos_phi - sigma * ct

        a_prime_new = np.divide(
            sigma * ct,
            denominator,
            out=a_prime.copy(),
            where=np.abs(denominator) > 1.0e-10,
        )

        # Numerical protection in difficult local regions.
        a_prime_new = np.clip(a_prime_new, -0.5, 0.5)

        error = max(
            np.max(np.abs(a_new - a)),
            np.max(np.abs(a_prime_new - a_prime)),
        )

        a = relaxation * a_new + (1.0 - relaxation) * a
        a_prime = relaxation * a_prime_new + (1.0 - relaxation) * a_prime

        if error < 1.0e-7:
            break

    # Final loads
    phi = np.arctan2(
        V0 * (1.0 - a),
        omega * r * (1.0 + a_prime),
    )

    sin_phi = np.maximum(np.abs(np.sin(phi)), 1.0e-8)
    cos_phi = np.cos(phi)

    alpha = np.degrees(phi) - beta - pitch_deg
    cl, cd = interpolate_cl_cd(tc, alpha, airfoil_data)

    cn = cl * cos_phi + cd * sin_phi
    ct = cl * sin_phi - cd * cos_phi

    relative_speed = np.sqrt(
        (V0 * (1.0 - a))**2
        + (omega * r * (1.0 + a_prime))**2
    )

    # Distributed thrust and torque loads
    dT_dr = (
        0.5 * rho * relative_speed**2 * B * chord * cn
    )

    dQ_dr = (
        0.5 * rho * relative_speed**2 * B * chord * ct * r
    )

    # Assignment requirement: tip element at r = R has zero load.
    dT_dr[-1] = 0.0
    dQ_dr[-1] = 0.0

    # Integrate along the blade.
    if hasattr(np, "trapezoid"):
        T = np.trapezoid(dT_dr, r)
        Q = np.trapezoid(dQ_dr, r)
    else:
        T = np.trapz(dT_dr, r)
        Q = np.trapz(dQ_dr, r)

    power = Q * omega

    Cp = power / (0.5 * rho * A * V0**3)
    CT = T / (0.5 * rho * A * V0**2)

    return float(Cp), float(CT)


# -----------------------------------------------------------------
# 8. Grid evaluation
#
# The assignment says the optimum is between:
#   lambda = 5 ... 10
#   theta_p = -4 ... 3 deg
# -----------------------------------------------------------------

def make_grid(equation, airfoil_data, lambda_values, pitch_values):
    cp_grid = np.empty((len(pitch_values), len(lambda_values)))
    ct_grid = np.empty_like(cp_grid)

    for j, pitch in enumerate(pitch_values):
        for i, lam in enumerate(lambda_values):
            cp, ct = bem(
                lam,
                pitch,
                equation,
                airfoil_data,
            )
            cp_grid[j, i] = cp
            ct_grid[j, i] = ct

    # Stop a bad numerical result from contaminating the contour plot.
    cp_grid = np.where(np.isfinite(cp_grid), cp_grid, np.nan)
    ct_grid = np.where(np.isfinite(ct_grid), ct_grid, np.nan)

    return cp_grid, ct_grid


# -----------------------------------------------------------------
# 9. Find the maximum, then refine locally
# -----------------------------------------------------------------

def locate_best(lambda_values, pitch_values, cp_grid):
    if not np.isfinite(cp_grid).any():
        raise RuntimeError("No finite Cp values were obtained.")

    idx = np.nanargmax(cp_grid)
    j, i = np.unravel_index(idx, cp_grid.shape)

    return (
        lambda_values[i],
        pitch_values[j],
        cp_grid[j, i],
    )


def refine_solution(equation, airfoil_data, lam0, pitch0):
    lambda_refined = np.arange(
        max(5.0, lam0 - 0.30),
        min(10.0, lam0 + 0.30) + 1.0e-12,
        0.05,
    )

    pitch_refined = np.arange(
        max(-4.0, pitch0 - 0.30),
        min(3.0, pitch0 + 0.30) + 1.0e-12,
        0.05,
    )

    cp_grid, ct_grid = make_grid(
        equation,
        airfoil_data,
        lambda_refined,
        pitch_refined,
    )

    lam_best, pitch_best, cp_best = locate_best(
        lambda_refined,
        pitch_refined,
        cp_grid,
    )

    j, i = np.unravel_index(
        np.nanargmax(cp_grid),
        cp_grid.shape,
    )

    return (
        lam_best,
        pitch_best,
        cp_best,
        ct_grid[j, i],
    )


# -----------------------------------------------------------------
# 10. Main program
# -----------------------------------------------------------------

def main():
    base_dir = Path(__file__).resolve().parent
    airfoil_data = load_airfoils(base_dir)

    # Coarse contour grid.
    lambda_values = np.arange(5.0, 10.0 + 1.0e-12, 0.20)
    pitch_values = np.arange(-4.0, 3.0 + 1.0e-12, 0.20)

    print("\nDTU 10 MW - Assignment #1 - Question 1")
    print("========================================")
    print(f"Using data folder: {base_dir}")
    print("Airfoil files loaded successfully.")

    results = {}

    for equation in (1, 2):
        cp_grid, ct_grid = make_grid(
            equation,
            airfoil_data,
            lambda_values,
            pitch_values,
        )

        lam0, pitch0, cp0 = locate_best(
            lambda_values,
            pitch_values,
            cp_grid,
        )

        lam_best, pitch_best, cp_best, ct_best = refine_solution(
            equation,
            airfoil_data,
            lam0,
            pitch0,
        )

        results[equation] = {
            "cp_grid": cp_grid,
            "ct_grid": ct_grid,
            "lambda_max": lam_best,
            "pitch_max": pitch_best,
            "cp_max": cp_best,
            "ct_at_cpmax": ct_best,
        }

        print(f"\nMomentum Equation {equation}")
        print(f"Cp,max       = {cp_best:.6f}")
        print(f"lambda_max   = {lam_best:.2f}")
        print(f"pitch_max    = {pitch_best:.2f} deg")
        print(f"CT(Cp,max)   = {ct_best:.6f}")

    # ------------------------------------------------------------
    # 11. Contour plots requested by Question 1
    # ------------------------------------------------------------
    Lambda, Pitch = np.meshgrid(lambda_values, pitch_values)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    plots = [
        (axes[0, 0], results[1]["cp_grid"], "Cp", "Cp(lambda, theta_p) - Momentum Equation 1"),
        (axes[0, 1], results[1]["ct_grid"], "CT", "CT(lambda, theta_p) - Momentum Equation 1"),
        (axes[1, 0], results[2]["cp_grid"], "Cp", "Cp(lambda, theta_p) - Momentum Equation 2"),
        (axes[1, 1], results[2]["ct_grid"], "CT", "CT(lambda, theta_p) - Momentum Equation 2"),
    ]

    for ax, grid, cbar_label, title in plots:
        levels = 30
        contour = ax.contourf(
            Lambda,
            Pitch,
            grid,
            levels=levels,
        )
        ax.set_xlabel("Tip Speed Ratio lambda")
        ax.set_ylabel("Pitch Angle theta_p [deg]")
        ax.set_title(title)
        fig.colorbar(contour, ax=ax, label=cbar_label)

    # Mark optimum points on Cp plots.
    for ax, eq in ((axes[0, 0], 1), (axes[1, 0], 2)):
        ax.plot(
            results[eq]["lambda_max"],
            results[eq]["pitch_max"],
            "ro",
            markersize=6,
        )

    fig.suptitle("DTU 10 MW BEM - Assignment #1 Question 1", fontsize=15)

    plt.show()


if __name__ == "__main__":
    main()
