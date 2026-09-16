import Berreman4x4 as b4
import numpy as np

# --- 1. OPTICAL ENVIRONMENT & BREWSTER'S ANGLE SETUP ---
wavelength = 0.530e-6
k0 = 2 * np.pi / wavelength

n_air = 1.0
n_glass = 1.5
air_material = b4.IsotropicNonDispersiveMaterial(n_air)
glass_material = b4.IsotropicNonDispersiveMaterial(n_glass)

incident_angle_deg = 45
incident_angle_rad = np.deg2rad(incident_angle_deg)
Kx = n_air * np.sin(incident_angle_rad)

structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air_material),
    layers=[],
    back=b4.IsotropicHalfSpace(glass_material)
)

# --- 2. EVALUATION ---
evaluation = structure.evaluate(Kx, k0)
J_trans = evaluation.T_ti
J_refl = evaluation.T_ri

# --- 3. CONVERSION TO NATIVE MUELLER MATRICES ---
A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
A_inv = np.linalg.inv(A)

def get_native_mueller(J_matrix):
    J_kron = np.kron(J_matrix, np.conjugate(J_matrix))
    M = np.asarray(np.real(np.dot(A, np.dot(J_kron, A_inv))))
    intensity = M[0, 0]
    M_norm = M / intensity if intensity != 0 else M
    M_norm[np.abs(M_norm) < 1e-10] = 0
    return M_norm, intensity

M_trans, T_intensity = get_native_mueller(J_trans)
M_refl, R_intensity = get_native_mueller(J_refl)

# --- 4. OUTPUT DATA ---
print(f"--- PARALLEL FRESNEL ANALYSIS AT {incident_angle_deg:.4f}° ---")
print(f"Base Transmission Intensity (T00): {T_intensity * 100:.2f}%")
print(f"Base Reflection Intensity (R00):   {R_intensity * 100:.2f}%")
print(f"Energy Balance: { (T_intensity + R_intensity) * 100:.2f}% (Losses due to area projection)\n")

print("--- NATIVE MUELLER MATRICES ---")
print("Transmission M_trans:\n", np.round(M_trans, 3))
print("Reflection M_refl:\n", np.round(M_refl, 3), "\n")

# --- 5. STOKES VECTOR CALCULATIONS (Fixed Variable Names) ---
S_in_s = np.array([1, -1, 0, 0])
S_in_p = np.array([1, 1, 0, 0])

# Using the corrected variable names
S_trans_s = np.dot(M_trans, S_in_s)
S_refl_s  = np.dot(M_refl, S_in_s)

S_trans_p = np.dot(M_trans, S_in_p)
S_refl_p  = np.dot(M_refl, S_in_p)

print("--- OBSERVABLE FINAL OUTPUT STOKES VECTORS ---")
print("Vertical (S-Pol) Input:")
print("  ↳ Transmitted Beam:", np.round(S_trans_s, 3))
print("  ↳ Reflected Beam:  ", np.round(S_refl_s, 3))

print("\nHorizontal (P-Pol) Input:")
print("  ↳ Transmitted Beam:", np.round(S_trans_p, 3))
print("  ↳ Reflected Beam:  ", np.round(S_refl_p, 3))