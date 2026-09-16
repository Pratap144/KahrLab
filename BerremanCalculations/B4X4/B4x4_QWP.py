import Berreman4x4 as b4
import numpy as np

def jones_to_mueller(J):
    """Converts 2x2 complex Jones matrix to 4x4 real Mueller matrix."""
    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]]) / np.sqrt(2)
    return np.real(U @ np.kron(J, np.conj(J)) @ U.conj().T)

# 1. Define the birefringent material
n_o = 1.5
n_e = 1.6
qwp_material = b4.UniaxialNonDispersiveMaterial(no=n_o, ne=n_e)

# 2. Setup Geometry (45 deg QWP)
angle_deg = 45
angle_rad = np.deg2rad(angle_deg + 90)
rotated_tensor = qwp_material.rotated(b4.rotation_Euler((angle_rad, np.pi/2, 0)))

# 3. Build the structure
wavelength = 0.530e-6
thickness = wavelength / (4 * np.abs(n_e - n_o))
qwp_layer = b4.HomogeneousLayer(material=rotated_tensor, h=thickness)
air = b4.IsotropicNonDispersiveMaterial(1.0)

structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air),
    layers=[qwp_layer],
    back=b4.IsotropicHalfSpace(air)
)

# 4. Evaluate to get Jones Matrices (Both Transmission and Reflection)
Kx = 0.0
k0 = 2 * np.pi / wavelength
evaluation = structure.evaluate(Kx, k0)

J_trans = evaluation.T_ti
J_refl  = evaluation.T_ri

# 5. Convert to Mueller Matrices
M_trans = jones_to_mueller(J_trans)
M_refl  = jones_to_mueller(J_refl)

# Normalize relative to M[0,0]
M_trans_norm = M_trans / M_trans[0, 0]
M_refl_norm = M_refl / M_refl[0, 0] if M_refl[0, 0] != 0 else M_refl

print(f"--- Normalized Mueller Matrix (Transmission) ---")
print(np.round(M_trans_norm, 3))

print(f"\n--- Normalized Mueller Matrix (Reflection) ---")
print(np.round(M_refl_norm, 3))

# 6. Energy Balance Check
# In a non-absorbing system, total intensity out (Trans + Refl) = intensity in
total_intensity = M_trans[0, 0] + M_refl[0, 0]
print(f"\n--- Energy Balance (M00 Trans + M00 Refl) ---")
print(f"Total Intensity Conserved: {np.round(total_intensity, 4)}")