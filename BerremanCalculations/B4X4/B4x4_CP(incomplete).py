import Berreman4x4 as b4
import numpy as np

# --- 1. SETUP GLOBAL OPTICAL PARAMETERS ---
wavelength = 0.530e-6  # 530 nm
Kx = 0.0               # Normal incidence
k0 = 2 * np.pi / wavelength
air_material = b4.IsotropicNonDispersiveMaterial(1.0)


# --- 2. LAYER 1: LINEAR POLARIZER (Oriented at 45 degrees) ---
# 'ne' is the transmission axis, 'no' is highly absorbing.
# We use 0.1j to remain safely within float64 matrix numerical stability limits.
pol_material = b4.UniaxialNonDispersiveMaterial(no=1.5 + 0.1j, ne=1.5)

# Tilt the transmission axis 90 deg into the X-Y plane, then rotate by 45 deg.
# We add the 90 deg (pi/2) offset to the first angle for strict coordinate alignment.
angle_deg_lp = 90
angle_rad_lp = np.deg2rad(angle_deg_lp + 90)
rotated_tensor_lp = pol_material.rotated(b4.rotation_Euler((angle_rad_lp, np.pi/2, 0)))

# Build the homogeneous linear polarizer layer (10 microns thick to absorb completely)
lp_layer = b4.HomogeneousLayer(material=rotated_tensor_lp, h=10e-6)


# --- 3. LAYER 2: QUARTER-WAVE PLATE (Oriented at 0 degrees) ---
# Pure phase retarder with real refractive indices
n_o_qwp = 1.5
n_e_qwp = 1.6
qwp_material = b4.UniaxialNonDispersiveMaterial(no=n_o_qwp, ne=n_e_qwp)

# Align the QWP at 0 degrees (Fast/Slow axes parallel to X and Y lab axes)
angle_deg_qwp = 45
angle_rad_qwp = np.deg2rad(angle_deg_qwp + 90)
rotated_tensor_qwp = qwp_material.rotated(b4.rotation_Euler((angle_rad_qwp, np.pi/2, 0)))

# Calculate the precise thickness required for a 90-degree retarder at 633 nm
delta_n = np.abs(n_e_qwp - n_o_qwp)
qwp_thickness = wavelength / (4 * delta_n)
qwp_layer = b4.HomogeneousLayer(material=rotated_tensor_qwp, h=qwp_thickness)


# --- 4. BUILD THE ARCHITECTURE ---
# Stacking order along z: Light enters Front Air -> Linear Polarizer -> QWP -> Exit Air
structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air_material),
    layers=[lp_layer, qwp_layer],
    back=b4.IsotropicHalfSpace(air_material)
)


# --- 5. EVALUATE AND CALCULATE MUELLER MATRIX ---
evaluation = structure.evaluate(Kx, k0)
J = evaluation.T_ti

A = np.array([[1, 0, 0, 1],
              [1, 0, 0, -1],
              [0, 1, 1, 0],
              [0, 1j, -1j, 0]])

J_kron = np.kron(J, np.conjugate(J))
A_inv = np.linalg.inv(A)
M = np.real(np.dot(A, np.dot(J_kron, A_inv)))

# Normalize relative to M[0,0]
if M[0, 0] != 0:
    M_norm = M / M[0, 0]
else:
    M_norm = M

# Clean up tiny floating point residuals (e.g., 1e-17)
M_norm[np.abs(M_norm) < 1e-10] = 0

print("--- Jones Transmission Matrix (Circular Polarizer) ---")
print(np.round(J, 3))
print("\n")

print("--- Normalized Mueller Matrix (Circular Polarizer) ---")
print(np.round(M_norm, 3))
print("\n")


# --- 6. CALCULATE OUTPUT STOKES VECTOR ---
# Test with Unpolarized input light: [1, 0, 0, 0]
S_in_unpolarized = np.array([1, 0, 0, 0])
S_out = np.dot(M_norm, S_in_unpolarized)

# # FIX: Use 2D indexing [0, 3] because S_out inherits a (1, 4) matrix shape
# S_out[0, 3] = -S_out[0, 3]

# Clean up microscopic float rounding noise
S_out[np.abs(S_out) < 1e-10] = 0

print("--- Output Stokes Vector (Unpolarized Input) ---")
print(np.round(S_out, 3))