import Berreman4x4 as b4
import numpy as np

# --- 1. SETUP OPTICAL PARAMETERS ---
wavelength = 0.530e-6
k0 = 2 * np.pi / wavelength

no = 1.5
ne = 1.7
n_avg = (no + ne) / 2.0

glass = b4.IsotropicNonDispersiveMaterial(n_avg)

# --- 2. BUILD THE 90-DEGREE TWISTED CELL ---
# Extraordinary axis starts on X (Horizontal)
nematic = b4.BiaxialNonDispersiveMaterial(diag=(ne, no, no))

cell_thickness = 5.0e-6  # 5 microns (thick enough for Mauguin waveguiding)
twist_angle = np.pi / 2.0  # 90 degrees

quarter_turn = b4.TwistedMaterial(
    material=nematic,
    d=cell_thickness,
    angle=twist_angle,
    div=100
)

structure = b4.Structure(
    front=b4.IsotropicHalfSpace(glass),
    layers=[b4.InhomogeneousLayer(quarter_turn)],
    back=b4.IsotropicHalfSpace(glass)
)

# --- 3. EVALUATE ---
evaluation = structure.evaluate(0.0, k0)
J_trans = evaluation.T_ti
J_refl  = evaluation.T_ri  # Added Reflection Extraction

# --- 4. CONVERT TO SOURCE-PERSPECTIVE MUELLER MATRICES ---
A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
A_inv = np.linalg.inv(A)
T_flip = np.diag([1, 1, 1, -1])

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

print("======================================================")
print(f"       90-DEGREE TN CELL ANALYSIS ({wavelength*1e9:.0f} nm)")
print("======================================================")
print(f"Energy Balance: Transmission ({T_int*100:.2f}%) + Reflection ({R_int*100:.2f}%) = {(T_int+R_int)*100:.2f}%\n")

print("TRANSMISSION MATRIX (M_trans):")
print(np.round(M_trans, 3), "\n")
print("REFLECTION MATRIX (M_refl):")
print(np.round(M_refl, 3), "\n")


# --- 5. TEST WITH SOURCE-PERSPECTIVE STOKES VECTORS ---
S_in_H     = np.array([1, 1, 0, 0])
S_in_V     = np.array([1, -1, 0, 0])
S_in_unpol = np.array([1, 0, 0, 0])

print("--- OBSERVABLE TRANSMITTED & REFLECTED BEAMS ---")

print("\n1. HORIZONTAL INPUT [1, 1, 0, 0]")
print("   -> Transmitted: ", np.round(np.dot(M_trans, S_in_H) * T_int, 3))
print("   -> Reflected:   ", np.round(np.dot(M_refl,  S_in_H) * R_int, 3))

print("\n2. VERTICAL INPUT   [1, -1, 0, 0]")
print("   -> Transmitted: ", np.round(np.dot(M_trans, S_in_V) * T_int, 3))
print("   -> Reflected:   ", np.round(np.dot(M_refl,  S_in_V) * R_int, 3))

print("\n3. UNPOLARIZED INPUT [1, 0, 0, 0]")
print("   -> Transmitted: ", np.round(np.dot(M_trans, S_in_unpol) * T_int, 3))
print("   -> Reflected:   ", np.round(np.dot(M_refl,  S_in_unpol) * R_int, 3))
print("======================================================")