import Berreman4x4 as b4
import numpy as np
import scipy.linalg

# --- 1. SETUP OPTICAL PARAMETERS ---
wavelength = 0.530e-6
k0 = 2 * np.pi / wavelength

# Non-absorbing Biaxial Indices
n_x = 1.50
n_y = 1.60
n_z = 1.55
n_avg = (n_x + n_y + n_z) / 3.0

thickness = 2.0e-6
pitch = 20.0e-6
glass = b4.IsotropicNonDispersiveMaterial(n_avg)

# Splay Parameters (Z-axis depth slicing)
nFiber = 10
Psi_max = 5.0
Psi_rad = np.radians(Psi_max)
dz = thickness / nFiber

# --- 2. HELPERS & CONVERTERS ---
A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
A_inv = np.linalg.inv(A)
T_flip = np.diag([1, 1, 1, -1])


def get_source_perspective_mueller(J_matrix):
    J_kron = np.kron(J_matrix, np.conjugate(J_matrix))
    M_native = np.real(A @ J_kron @ A_inv)
    M_source = T_flip @ M_native @ T_flip
    intensity = M_source[0, 0]
    return M_source / intensity if intensity != 0 else M_source


def rotx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --- 3. DEFINE THE TARGET LOCATIONS ---
# Targeting the specific radii corresponding to 0, 90, and 180 degree twists.
# We set Y=0 so we are moving directly along the X-axis of the slide.
target_faces = {
    "1. Flat-on (+x) [0°]": {"x": 0.0e-6, "y": 0.0},
    "2. Edge-on [90°]": {"x": 5.0e-6, "y": 0.0},
    "3. Flat-on (-x) [180°]": {"x": 10.0e-6, "y": 0.0}
}

print(f"==================================================")
print(f"   DIFFERENTIAL MUELLER ANALYSIS (WITH SPLAY)")
print(f"==================================================\n")

base_mat = b4.BiaxialNonDispersiveMaterial(diag=(n_x, n_y, n_z))

# --- 4. EVALUATE EACH FACE ---
for face_name, coords in target_faces.items():
    x_pixel = coords["x"]
    y_pixel = coords["y"]

    # Geometric Mapping
    r = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    theta = -np.arctan2(y_pixel, x_pixel)
    phi = r * (2 * np.pi / pitch)

    layers = []

    for k in range(nFiber):
        # Splay for this depth slice
        psi = np.sin(2 * phi) * Psi_rad * ((k / (nFiber - 1)) - 0.5) if nFiber > 1 else 0

        # 3D Orientation
        R_combined = rotz(theta + psi) @ rotx(phi)
        local_mat = base_mat.rotated(R_combined)
        layers.append(b4.HomogeneousLayer(local_mat, h=dz))

    structure = b4.Structure(
        front=b4.IsotropicHalfSpace(glass),
        layers=layers,
        back=b4.IsotropicHalfSpace(glass)
    )

    # Evaluate using Berreman Engine
    eval_res = structure.evaluate(0.0, k0)
    M_trans = get_source_perspective_mueller(eval_res.T_ti)

    # Differential Matrix Logarithm
    try:
        L_matrix = np.real(scipy.linalg.logm(M_trans))
        L_matrix[np.abs(L_matrix) < 1e-5] = 0  # Clean up math dust
    except Exception as e:
        L_matrix = "Matrix Logarithm Failed"

    print(f"--- LOCATION: {face_name} ---")
    print(f"Radius = {r * 1e6} µm")
    print("\nDIFFERENTIAL MUELLER MATRIX (L):")
    print(np.round(L_matrix, 3))
    print("-" * 50 + "\n")