import math
import numpy as np
import matplotlib.pyplot as plt

R = 89.17
rho = 1.225
P_rated = 10.64e6
V_cut_in = 4.0
V_cut_out = 25.0

Cp_eq1 = 0.494362
lambda_eq1 = 7.60

Cp_eq2 = 0.500987
lambda_eq2 = 7.65

A = math.pi * R**2


def calculate_rated_wind_speed(cp_max):
    return (P_rated / (0.5 * rho * A * cp_max))**(1.0 / 3.0)


def calculate_omega(lam, V0):
    return lam * V0 / R


V_rated_eq1 = calculate_rated_wind_speed(Cp_eq1)
V_rated_eq2 = calculate_rated_wind_speed(Cp_eq2)

omega_max_eq1 = calculate_omega(lambda_eq1, V_rated_eq1)
omega_max_eq2 = calculate_omega(lambda_eq2, V_rated_eq2)

rpm_eq1 = omega_max_eq1 * 60.0 / (2.0 * math.pi)
rpm_eq2 = omega_max_eq2 * 60.0 / (2.0 * math.pi)

if not (V_cut_in < V_rated_eq1 < V_cut_out):
    raise ValueError("Equation 1: rated wind speed is outside the valid wind-speed range.")

if not (V_cut_in < V_rated_eq2 < V_cut_out):
    raise ValueError("Equation 2: rated wind speed is outside the valid wind-speed range.")


print("DTU 10 MW BEM - Assignment #1 Question 2")
print()

print("Momentum Equation 1")
print(f"Cp_max = {Cp_eq1:.6f}")
print(f"lambda_opt = {lambda_eq1:.2f}")
print(f"V_rated = {V_rated_eq1:.4f} m/s")
print(f"omega_max = {omega_max_eq1:.4f} rad/s")
print(f"omega_max = {rpm_eq1:.4f} rpm")

print()

print("Momentum Equation 2")
print(f"Cp_max = {Cp_eq2:.6f}")
print(f"lambda_opt = {lambda_eq2:.2f}")
print(f"V_rated = {V_rated_eq2:.4f} m/s")
print(f"omega_max = {omega_max_eq2:.4f} rad/s")
print(f"omega_max = {rpm_eq2:.4f} rpm")


V1 = np.linspace(V_cut_in, V_rated_eq1, 300)
V2 = np.linspace(V_cut_in, V_rated_eq2, 300)

omega1 = calculate_omega(lambda_eq1, V1)
omega2 = calculate_omega(lambda_eq2, V2)


plt.figure(figsize=(9, 6))

plt.plot(
    V1,
    omega1,
    linewidth=2,
    label="Momentum Equation 1"
)

plt.plot(
    V2,
    omega2,
    linewidth=2,
    label="Momentum Equation 2"
)

plt.scatter(
    V_rated_eq1,
    omega_max_eq1,
    s=60
)

plt.scatter(
    V_rated_eq2,
    omega_max_eq2,
    s=60
)

plt.axvline(
    V_rated_eq1,
    linestyle="--",
    linewidth=1
)

plt.axvline(
    V_rated_eq2,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Wind Speed $V_0$ [m/s]")
plt.ylabel("Rotational Speed $\\omega$ [rad/s]")
plt.title("DTU 10 MW Wind Turbine - Assignment #1 Question 2")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()