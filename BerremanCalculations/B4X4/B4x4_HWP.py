import Berreman4x4 as b4
import numpy as np

def jones_to_mueller(J):
    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]]) / np.sqrt(2)
    return np.real(U @ np.kron(J, np.conj(J)) @ U.conj().T)

# 1. Define Birefringent Material
n_o = 1.5
n_e = 1.6
hwp_material = b4.UniaxialNonDispersiveMaterial(no=n_o, ne=n_e)

# 2. Geometry: HWP at 45 degrees
angle_deg = 45
angle_rad = np.deg2rad(angle_deg + 90)
rotated_tensor = hwp_material.rotated(b4.rotation_Euler((angle_rad, np.pi/2, 0)))

# 3. Exact thickness for Half Wave Plate (delta_phi = pi)
wavelength = 0.530e-6
delta_n = np.abs(n_e - n_o)
thickness = wavelength / (2 * delta_n) # OPD = lambda/2

# Build Structure
hwp_layer = b4.HomogeneousLayer(material=rotated_tensor, h=thickness)
air = b4.IsotropicNonDispersiveMaterial(1.0)
structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air),
    layers=[hwp_layer],
    back=b4.IsotropicHalfSpace(air)
)

# 4. Evaluate Jones Matrices (Transmission and Reflection)
Kx = 0.0
k0 = 2 * np.pi / wavelength
evaluation = structure.evaluate(Kx, k0)
J_trans = evaluation.T_ti
J_refl  = evaluation.T_ri # Note: Using T_ri per your specific module naming

# 5. Calculate Mueller Matrices
M_trans = jones_to_mueller(J_trans)
M_refl  = jones_to_mueller(J_refl)

# Normalize for display
M_trans_norm = M_trans / M_trans[0,0]
M_refl_norm = M_refl / M_refl[0,0] if M_refl[0,0] != 0 else M_refl

print(f"--- Normalized Mueller Matrix (Transmission, {angle_deg} deg HWP) ---")
print(np.round(M_trans_norm, 3))

print(f"\n--- Normalized Mueller Matrix (Reflection, {angle_deg} deg HWP) ---")
print(np.round(M_refl_norm, 3))

# 6. Calculate Output Stokes Vector (Horizontal Input: [1, 1, 0, 0])
S_in = np.array([1, 1, 0, 0])
S_out = np.dot(M_trans_norm, S_in)

print("\n--- Output Stokes Vector (Horizontal Input) ---")
print(np.round(S_out, 3))

# 7. Energy Balance Check
total_intensity = M_trans[0,0] + M_refl[0,0]
T = M_trans[0, 0] # Total intensity transmitted
R = M_refl[0, 0]  # Total intensity reflected

# Calculate percentages
T_percent = T * 100
R_percent = R * 100
Total_Accounted = (T + R) * 100

print(f"--- Energy Balance Check ---")
print(f"Transmittance (T): {T_percent:.2f}%")
print(f"Reflectance (R):   {R_percent:.2f}%")
print(f"Total (T + R):     {Total_Accounted:.2f}%")

if abs((T + R) - 1.0) > 1e-6:
    print("WARNING: Energy is not conserved! Check your material absorption or layer stack.")
else:
    print("SUCCESS: Energy is conserved.")