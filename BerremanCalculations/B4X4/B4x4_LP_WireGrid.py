import numpy as np
import Berreman4x4 as b4

def jones_to_mueller(J):
    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]]) / np.sqrt(2)
    return np.real(U @ np.kron(J, np.conj(J)) @ U.conj().T)

# 1. Define the Wire-Grid Polarizer using Effective Medium Theory
# Epsilon along wires (X) is high real-valued, perpendicular (Y, Z) is low
eps_parallel = 100.0  # Represents high conductivity/reflectivity
eps_perp = 2.25       # Represents dielectric host (n=1.5)
eps_base = np.diag([eps_parallel, eps_perp, eps_perp])

# 2. Set the angle (0 deg = wires along X-axis)
angle_deg = 0
angle_rad = np.deg2rad(angle_deg)
c, s = np.cos(angle_rad), np.sin(angle_rad)
Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
rotated_tensor = Rz @ eps_base @ Rz.T

# 3. Build the structure (Thin layer to model the grid interface)
mat = b4.NonDispersiveMaterial(rotated_tensor)
wgp_layer = b4.HomogeneousLayer(material=mat, h=100e-9)
air = b4.IsotropicNonDispersiveMaterial(1.0)

structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air),
    layers=[wgp_layer],
    back=b4.IsotropicHalfSpace(air)
)

# 4. Evaluate to get Jones Matrix
Kx = 0.0
k0 = 2 * np.pi / 0.633e-6
evaluation = structure.evaluate(Kx, k0)
J = evaluation.T_ti

print(f"--- Jones Transmission Matrix ({angle_deg} deg) ---")
print(np.round(J, 3))

# 5. Convert to Mueller
M = jones_to_mueller(J)
M_norm = M / M[0, 0]

print(f"\n--- Normalized Mueller Matrix ({angle_deg} deg) ---")
print(np.round(M_norm, 3))

# 6. Calculate Output Stokes Vector (Unpolarized Input)
S_in = np.array([1, 0, 0, 0])
S_out = np.dot(M_norm, S_in)

print("\n--- Output Stokes Vector (Unpolarized Input) ---")
print(np.round(S_out, 3))