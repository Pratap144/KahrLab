import Berreman4x4 as b4
import numpy as np

# ====================================================================
# S3 = +1 -> Right Circular Polarization (RCP)
# S3 = -1 -> Left Circular Polarization (LCP)
# ====================================================================

# --- 1. SETUP OPTICAL PARAMETERS ---
wavelength = 0.530e-6
k0 = 2 * np.pi / wavelength

no = 1.5
ne = 1.7
n_avg = (no + ne) / 2.0
pitch = wavelength / n_avg  # ~331.25 nm
num_turns = 15

# --- 2. BUILD OPTIMIZED TWISTED STRUCTURE (Right-Handed) ---
nematic = b4.BiaxialNonDispersiveMaterial(diag=(ne, no, no))

# Right-Handed helix (angle = +np.pi)
clc_half_turn = b4.TwistedMaterial(material=nematic, d=pitch / 2.0, angle=np.pi, div=35)
clc_stack = b4.RepeatedLayers([b4.InhomogeneousLayer(clc_half_turn)], n=num_turns * 2)

glass = b4.IsotropicNonDispersiveMaterial(n_avg)
structure = b4.Structure(
    front=b4.IsotropicHalfSpace(glass),
    layers=[clc_stack],
    back=b4.IsotropicHalfSpace(glass)
)

# --- 3. EVALUATE AT NORMAL INCIDENCE ---
evaluation = structure.evaluate(0.0, k0)
J_trans = evaluation.T_ti
J_refl  = evaluation.T_ri

# --- 4. CONVERT TO SOURCE-PERSPECTIVE MUELLER MATRICES ---
A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
A_inv = np.linalg.inv(A)
T_flip = np.diag([1, 1, 1, -1])  # The Source Perspective Enforcer

def get_source_perspective_mueller(J_matrix):
    J_kron = np.kron(J_matrix, np.conjugate(J_matrix))
    M_native = np.asarray(np.real(np.dot(A, np.dot(J_kron, A_inv))))
    M_source = np.dot(T_flip, np.dot(M_native, T_flip))
    intensity = M_source[0, 0]
    M_norm = M_source / intensity if intensity != 0 else M_source
    M_norm[np.abs(M_norm) < 1e-10] = 0
    return M_norm, intensity

M_trans, T_int = get_source_perspective_mueller(J_trans)
M_refl,  R_int = get_source_perspective_mueller(J_refl)

# --- 5. PRINT MUELLER MATRICES ---
print("======================================================")
print(f"       RIGHT-HANDED CLC ANALYSIS ({wavelength*1e9:.0f} nm)")
print("======================================================")
print(f"Energy Balance: Transmission ({T_int*100:.2f}%) + Reflection ({R_int*100:.2f}%) = {(T_int+R_int)*100:.2f}%\n")

print("--- SOURCE-PERSPECTIVE MUELLER MATRICES ---")
print("TRANSMISSION MATRIX (M_trans):")
print(np.round(M_trans, 2), "\n")
print("REFLECTION MATRIX (M_refl):")
print(np.round(M_refl, 2), "\n")


# --- 6. TEST WITH SOURCE-PERSPECTIVE STOKES VECTORS ---
S_in_unpol = np.array([1, 0, 0, 0])
S_in_RCP   = np.array([1, 0, 0, 1])   # +1 is Right
S_in_LCP   = np.array([1, 0, 0, -1])  # -1 is Left

print("--- OBSERVABLE TRANSMITTED & REFLECTED BEAMS ---")

print("\n1. UNPOLARIZED INPUT [1, 0, 0, 0]")
print("   -> Transmitted: ", np.round(np.dot(M_trans, S_in_unpol) * T_int, 3))
print("   -> Reflected:   ", np.round(np.dot(M_refl,  S_in_unpol) * R_int, 3))

print("\n2. RIGHT-CIRCULAR INPUT (RCP) [1, 0, 0, +1]")
print("   -> Transmitted: ", np.round(np.dot(M_trans, S_in_RCP) * T_int, 3))
print("   -> Reflected:   ", np.round(np.dot(M_refl,  S_in_RCP) * R_int, 3))

print("\n3. LEFT-CIRCULAR INPUT (LCP)  [1, 0, 0, -1]")
print("   -> Transmitted: ", np.round(np.dot(M_trans, S_in_LCP) * T_int, 3))
print("   -> Reflected:   ", np.round(np.dot(M_refl,  S_in_LCP) * R_int, 3))
print("======================================================")