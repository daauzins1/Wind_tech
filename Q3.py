
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# READ DATA
# ============================================================

# Use whitespace for ALL files
cylinder = pd.read_csv('cylinder.txt', sep=r'\s+', header=None)
blade600 = pd.read_csv('FFA-W3-600.txt', sep=r'\s+', header=None)
blade480 = pd.read_csv('FFA-W3-480.txt', sep=r'\s+', header=None)
blade360 = pd.read_csv('FFA-W3-360.txt', sep=r'\s+', header=None)
blade301 = pd.read_csv('FFA-W3-301.txt', sep=r'\s+', header=None)
blade241 = pd.read_csv('FFA-W3-241.txt', sep=r'\s+', header=None)
blade_data = pd.read_csv('bladedat.txt', sep=r'\s+', header=None)
# ============================================================
# BLADE GEOMETRY
# ============================================================

radius = blade_data[0].to_numpy(dtype=float)
chord = blade_data[1].to_numpy(dtype=float)
twist = blade_data[2].to_numpy(dtype=float)
thickness = blade_data[3].to_numpy(dtype=float)


# ============================================================
# AIRFOIL DATA
# ============================================================

aoa_100 = cylinder[0].to_numpy(dtype=float)
cl_100_data = cylinder[1].to_numpy(dtype=float)
cd_100_data = cylinder[2].to_numpy(dtype=float)

aoa_60 = blade600[0].to_numpy(dtype=float)
cl_60_data = blade600[1].to_numpy(dtype=float)
cd_60_data = blade600[2].to_numpy(dtype=float)

aoa_48 = blade480[0].to_numpy(dtype=float)
cl_48_data = blade480[1].to_numpy(dtype=float)
cd_48_data = blade480[2].to_numpy(dtype=float)

aoa_36 = blade360[0].to_numpy(dtype=float)
cl_36_data = blade360[1].to_numpy(dtype=float)
cd_36_data = blade360[2].to_numpy(dtype=float)

aoa_30 = blade301[0].to_numpy(dtype=float)
cl_30_data = blade301[1].to_numpy(dtype=float)
cd_30_data = blade301[2].to_numpy(dtype=float)

aoa_24 = blade241[0].to_numpy(dtype=float)
cl_24_data = blade241[1].to_numpy(dtype=float)
cd_24_data = blade241[2].to_numpy(dtype=float)


# ============================================================
# SORT BLADE DATA
# ============================================================

# np.interp requires the x-data to be increasing.
sort_radius = np.argsort(radius)

radius = radius[sort_radius]
chord = chord[sort_radius]
twist = twist[sort_radius]
thickness = thickness[sort_radius]


# ============================================================
# AIRFOIL RADIAL LOCATIONS
# ============================================================

R_100 = 2.8

R_60 = np.interp(60, thickness, radius)
R_48 = np.interp(48, thickness, radius)
R_36 = np.interp(36, thickness, radius)
R_30 = np.interp(30.1, thickness, radius)
R_24 = np.interp(24.1, thickness, radius)


# Make sure airfoil radius locations are increasing
R_airfoil = np.array([R_100, R_60, R_48, R_36, R_30, R_24])
sort_airfoil = np.argsort(R_airfoil)
R_airfoil = R_airfoil[sort_airfoil]


def loop_through_blade(wind,omega,pitch_offset):
    r = np.linspace(0.1, 0.98, r_res)
    power_list = []
    for r_local in r:
    
            radius_local = r_local * R
    
    
            # ------------------------------------------------
            # BLADE GEOMETRY
            # ------------------------------------------------
    
            chord_local = np.interp(radius_local, radius, chord)
            twist_local = np.interp(radius_local, radius, twist)
    
    
            # ------------------------------------------------
            # INITIAL INDUCTION FACTORS
            # ------------------------------------------------
    
            a = 0.0
            a_prime = 0.0
    
    
            # ------------------------------------------------
            # BEM ITERATION
            # ------------------------------------------------
    
            for iteration in range(max_it):
    
                # --------------------------------------------
                # FLOW ANGLE
                # --------------------------------------------
    
                numerator = (1 - a) * Wind_speed
                denominator = (1 + a_prime) * omega * radius_local
                phi = np.arctan2(numerator, denominator)
    
    
                # --------------------------------------------
                # ANGLE OF ATTACK
                # --------------------------------------------
    
                aoa = np.rad2deg(phi) - pitch - twist_local - pitch_offset
                aoa = (aoa + 180) % 360 - 180   # wrap to [-180, 180]
    
    
                # --------------------------------------------
                # Cl AND Cd
                # --------------------------------------------
    
                cl_100 = np.interp(aoa, aoa_100, cl_100_data)
                cd_100 = np.interp(aoa, aoa_100, cd_100_data)
    
                cl_60 = np.interp(aoa, aoa_60, cl_60_data)
                cd_60 = np.interp(aoa, aoa_60, cd_60_data)
    
                cl_48 = np.interp(aoa, aoa_48, cl_48_data)
                cd_48 = np.interp(aoa, aoa_48, cd_48_data)
    
                cl_36 = np.interp(aoa, aoa_36, cl_36_data)
                cd_36 = np.interp(aoa, aoa_36, cd_36_data)
    
                cl_30 = np.interp(aoa, aoa_30, cl_30_data)
                cd_30 = np.interp(aoa, aoa_30, cd_30_data)
    
                cl_24 = np.interp(aoa, aoa_24, cl_24_data)
                cd_24 = np.interp(aoa, aoa_24, cd_24_data)
    
    
                # --------------------------------------------
                # INTERPOLATE BETWEEN AIRFOILS
                # --------------------------------------------
    
                cl_values = np.array([cl_100, cl_60, cl_48, cl_36, cl_30, cl_24])
                cd_values = np.array([cd_100, cd_60, cd_48, cd_36, cd_30, cd_24])
    
                # Sort the values in the same way as R_airfoil
                cl_values = cl_values[sort_airfoil]
                cd_values = cd_values[sort_airfoil]
    
                Cl = np.interp(radius_local, R_airfoil, cl_values)
                Cd = np.interp(radius_local, R_airfoil, cd_values)
    
    
                # --------------------------------------------
                # AERODYNAMIC COEFFICIENTS
                # --------------------------------------------
    
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)
    
                Cn = Cl * cos_phi + Cd * sin_phi
                Ct = Cl * sin_phi - Cd * cos_phi
    
    
                # --------------------------------------------
                # LOCAL SOLIDITY
                # --------------------------------------------
    
                sigma = B * chord_local / (2 * np.pi * radius_local)
    
    
                # --------------------------------------------
                # PRANDTL TIP LOSS
                # --------------------------------------------
    
                sin_phi_abs = max(abs(sin_phi), 1e-8)
    
                exponent = -(B / 2) * (R - radius_local) / (radius_local * sin_phi_abs)
    
                F = (2 / np.pi) * np.arccos(np.clip(np.exp(exponent), 0, 1))
    
                # Prevent division by zero
                F = max(F, 1e-4)
    
    
                # --------------------------------------------
                # THRUST COEFFICIENT
                # --------------------------------------------
    
                CT = sigma * Cn * (1 - a)**2 / max(sin_phi**2, 1e-8)
    
    
                # --------------------------------------------
                # AXIAL INDUCTION
                # --------------------------------------------
    
                if CT <= 8 / 9:
    
                    inside = max(1 - CT, 0)
    
                    a_star = 0.5 * (1 - np.sqrt(inside))
    
                else:
    
                    # Bisection for:
                    #
                    # CT = 4a - 5a² + 3a³
    
                    a_low = 1 / 3
                    a_high = 1.0
    
                    for k in range(50):
    
                        a_middle = (a_low + a_high) / 2
    
                        CT_middle = 4 * a_middle - 5 * a_middle**2 + 3 * a_middle**3
    
                        if CT_middle < CT:
                            a_low = a_middle
                        else:
                            a_high = a_middle
    
                    a_star = (a_low + a_high) / 2
    
    
                # --------------------------------------------
                # TANGENTIAL INDUCTION
                # --------------------------------------------
    
                a_prime_star = sigma * Ct * (1 + a_prime) / (
                    4 * F * sin_phi * cos_phi
                    if abs(sin_phi * cos_phi) > 1e-8
                    else 1e-8
                )
    
    
                # --------------------------------------------
                # RELAXATION
                # --------------------------------------------
    
                a_next = relaxation * a_star + (1 - relaxation) * a
    
                a_prime_next = relaxation * a_prime_star + (1 - relaxation) * a_prime
    
    
                # --------------------------------------------
                # LIMIT VALUES
                # --------------------------------------------
    
                a_next = np.clip(a_next, -0.5, 0.95)
                a_prime_next = np.clip(a_prime_next, -0.95, 2.0)
    
    
                # --------------------------------------------
                # CONVERGENCE CHECK
                # --------------------------------------------
    
                if (
                    abs(a_next - a) < tolerance
                    and
                    abs(a_prime_next - a_prime) < tolerance
                ):
    
                    a = a_next
                    a_prime = a_prime_next
    
                    break
    
                a = a_next
                a_prime = a_prime_next
    
    
            # =================================================
            # CALCULATE FINAL AERODYNAMICS
            # =================================================
    
            Vrel = np.sqrt(
                ((1 - a) * Wind_speed)**2
                + ((1 + a_prime) * omega * radius_local)**2
            )
    
    
            # Recalculate tangential coefficient
            phi = np.arctan2(
                (1 - a) * Wind_speed,
                (1 + a_prime) * omega * radius_local
            )
    
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
    
            aoa = np.rad2deg(phi) - pitch - twist_local - pitch_offset
            aoa = (aoa + 180) % 360 - 180   # wrap to [-180, 180]
    
    
            # Get Cl and Cd again
            cl_100 = np.interp(aoa, aoa_100, cl_100_data)
            cd_100 = np.interp(aoa, aoa_100, cd_100_data)
    
            cl_60 = np.interp(aoa, aoa_60, cl_60_data)
            cd_60 = np.interp(aoa, aoa_60, cd_60_data)
    
            cl_48 = np.interp(aoa, aoa_48, cl_48_data)
            cd_48 = np.interp(aoa, aoa_48, cd_48_data)
    
            cl_36 = np.interp(aoa, aoa_36, cl_36_data)
            cd_36 = np.interp(aoa, aoa_36, cd_36_data)
    
            cl_30 = np.interp(aoa, aoa_30, cl_30_data)
            cd_30 = np.interp(aoa, aoa_30, cd_30_data)
    
            cl_24 = np.interp(aoa, aoa_24, cl_24_data)
            cd_24 = np.interp(aoa, aoa_24, cd_24_data)
    
    
            Cl = np.interp(
                radius_local,
                R_airfoil,
                np.array([cl_100, cl_60, cl_48, cl_36, cl_30, cl_24])[sort_airfoil]
            )
    
            Cd = np.interp(
                radius_local,
                R_airfoil,
                np.array([cd_100, cd_60, cd_48, cd_36, cd_30, cd_24])[sort_airfoil]
            )
    
    
            Ct = Cl * sin_phi - Cd * cos_phi
    
    
            # =================================================
            # TANGENTIAL FORCE
            # =================================================
    
            pt = 0.5 * rho * Vrel**2 * chord_local * Ct
    
    
            # =================================================
            # TORQUE
            # =================================================
    
            torque = B * radius_local * pt
    
    
            # =================================================
            # POWER
            # =================================================
    
            power = omega * torque
    
            power_list.append(power)
    
    power_total = np.trapezoid(power_list, r * R)
    return power_total



R = 89.17
rho = 1.225
P_rated = 10.64e6
V_cut_in = 4.0
V_cut_out = 25.0

Cp_eq1 = 0.494362
lambda_eq1 = 7.60

Cp_eq2 = 0.500987
lambda_eq2 = 7.65

A = np.pi * R**2


def calculate_rated_wind_speed(cp_max):
    return (P_rated / (0.5 * rho * A * cp_max))**(1.0 / 3.0)


def calculate_omega(lam, V0):
    return lam * V0 / R


V_rated = calculate_rated_wind_speed(Cp_eq1)        #11.2047202
omega_max = calculate_omega(lambda_eq1, V_rated)    #0.9549834423130436

print("-------------------")
print("Rated wind speed: "+ str(V_rated))
print('Rated omega: '+ str(omega_max))
print("-------------------")




# ============================================================
# TURBINE CONSTANTS
# ============================================================

R = 89.17
B = 3
rho = 1.225
res_wind = 50

Vinf = np.linspace(V_cut_in,V_cut_out,res_wind)


# ============================================================
# ITERATION SETTINGS
# ============================================================

relaxation = 0.1

# 50000 is WAY too high
max_it = 500

tolerance = 1e-6


# ============================================================
# CALCULATION SETTINGS
# ============================================================

TSR_res = 1
pitch_res = 30
r_res = 20

TSR = 8.06
pitch = 2.344



# ============================================================
# BEM CALCULATION
# ============================================================
power_per_wind_speed = []
pitch_per_wind_speed = []
for j, Wind_speed in enumerate(Vinf):

    print("Started working Boss..."+str(Wind_speed))

    omega = TSR * Wind_speed / R
    power_list = []
    pitch_control = 0
    for i in range(500000):
        power_total = loop_through_blade(Wind_speed,omega, pitch_offset=pitch_control)
        print(f"pitch={pitch_control:.1f}  power={power_total:.3e}")
        if power_total > P_rated:
            pitch_control += 0.01
        else:
            print('Pitch value is : '+str(pitch_control))
            break

    power_per_wind_speed.append(power_total)
    pitch_per_wind_speed.append(pitch_control)
print("Done :()")

plt.plot(Vinf,power_per_wind_speed)
plt.title('Power VS Wind Speed Graph', fontsize=14)
plt.xlabel('Wind speed [m/s]', fontsize=12)
plt.ylabel('Power [Watts]', fontsize=12)
plt.legend()
plt.show()

plt.plot(Vinf,pitch_per_wind_speed)
plt.title('Pitch VS Wind Speed Graph', fontsize=14)
plt.xlabel('Wind speed', fontsize=12)
plt.ylabel('Pitch angle [deg]', fontsize=12)
plt.show()

