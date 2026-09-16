import Berreman4x4 as b4
import numpy as np

# 1. Define the dichroic material (Linear Polarizer)
# Use 0.1j instead of 5.0j to prevent floating-point matrix singularities
# while still providing > 10,000:1 extinction ratio.
pol_material = b4.UniaxialNonDispersiveMaterial(no=1.5 + 0.1j, ne=1.5)

# 2. Set the angle of the polarizer (Change this to 0, 45, 90, or 135 to test)
angle_deg = 0
# Add 90 degrees to offset the starting position of the Euler rotation
angle_rad = np.deg2rad(angle_deg + 90)

# FIX: Tilt the optic axis 90 deg (pi/2) into the X-Y plane first (theta),
# then rotate it by our desired 45 deg (phi).
# rotation_Euler takes (phi, theta, psi)
rotated_tensor = pol_material.rotated(b4.rotation_Euler((angle_rad, np.pi/2, 0)))

# 3. Build the structure
# 10 microns thick to absorb the orthogonal polarization completely
pol_layer = b4.HomogeneousLayer(material=rotated_tensor, h=10e-6)

# FIX: Wrap the refractive index of air in the Material class
air_material = b4.IsotropicNonDispersiveMaterial(1.0)

structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air_material),
    layers=[pol_layer],
    back=b4.IsotropicHalfSpace(air_material)
)

# 4. Evaluate to get the Jones Matrix
Kx = 0.0
# FIX: Use scientific notation for meters (633 nm = 0.633e-6 m)
k0 = 2 * np.pi / 0.633e-6
evaluation = structure.evaluate(Kx, k0)
J = evaluation.T_ti

print(f"--- Jones Transmission Matrix ({angle_deg} deg) ---")
print(np.round(J, 3))
print("\n")

# 5. Calculate Mueller Matrix
A = np.array([[1, 0, 0, 1],
              [1, 0, 0, -1],
              [0, 1, 1, 0],
              [0, 1j, -1j, 0]])

J_kron = np.kron(J, np.conjugate(J))
A_inv = np.linalg.inv(A)
M = np.dot(A, np.dot(J_kron, A_inv))
M = np.real(M)

# Normalize M relative to M[0,0] for easier comparison with theory
if M[0, 0] != 0:
    M_norm = M / M[0, 0]
else:
    M_norm = M

print(f"--- Normalized Mueller Matrix ({angle_deg} deg) ---")
print(np.round(M_norm, 3))
print("\n")

# 6. Calculate Output Stokes Vector
# Test with unpolarized input light: [1, 0, 0, 0]
S_in_unpolarized = np.array([1, 0, 0, 0])
S_out = np.dot(M_norm, S_in_unpolarized)

print("--- Output Stokes Vector (Unpolarized Input) ---")
print(np.round(S_out, 3))