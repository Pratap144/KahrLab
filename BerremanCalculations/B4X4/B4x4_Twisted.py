import Berreman4x4 as b4
import numpy as np

# 1. Define the base birefringent material for the crystal
# Substitute your actual experimental refractive indices here
# (Banded spherulites usually don't have high absorption, so we use real numbers)
base_material = b4.UniaxialNonDispersiveMaterial(no=1.5, ne=1.55)

# 2. Lay the optic axis flat in the X-Y plane (aligned to the X-axis)
# Using our +90 degree (pi/2) trick to align it perfectly to 0 degrees
planar_tensor = base_material.rotated(b4.rotation_Euler((np.pi/2, np.pi/2, 0)))

# 3. Define the continuous twist (The Banded Spherulite)
# E.g., a 10-micron thick film with a 180-degree (pi) twist (one half-pitch)
thickness = 10e-6
twist_angle = np.pi

# 'div' is the number of mathematical slices.
# 50 is an excellent balance of calculation speed and physics accuracy.
twisted_mat = b4.TwistedMaterial(material=planar_tensor, d=thickness, angle=twist_angle, div=50)

# 4. Build the Structure using InhomogeneousLayer
twist_layer = b4.InhomogeneousLayer(material=twisted_mat)
air_material = b4.IsotropicNonDispersiveMaterial(1.0)

structure = b4.Structure(
    front=b4.IsotropicHalfSpace(air_material),
    layers=[twist_layer],
    back=b4.IsotropicHalfSpace(air_material)
)

# 5. Evaluate the Jones Matrix
Kx = 0.0
k0 = 2 * np.pi / 0.633e-6 # 633 nm laser
evaluation = structure.evaluate(Kx, k0)
J = evaluation.T_ti

print("--- Jones Transmission Matrix (Twisted Crystal) ---")
print(np.round(J, 3))
print("\n")

# 6. Convert to Mueller Matrix
A = np.array([[1, 0, 0, 1],
              [1, 0, 0, -1],
              [0, 1, 1, 0],
              [0, 1j, -1j, 0]])

J_kron = np.kron(J, np.conjugate(J))
A_inv = np.linalg.inv(A)
M = np.real(np.dot(A, np.dot(J_kron, A_inv)))

# Normalize M relative to M[0,0]
if M[0, 0] != 0:
    M_norm = M / M[0, 0]
else:
    M_norm = M

print("--- Normalized Mueller Matrix (Twisted Crystal) ---")
print(np.round(M_norm, 3))
print("\n")

# 7. Calculate Output Stokes Vector
# Let's test what happens when Horizontal Light [1, 1, 0, 0] enters the spherulite
S_in_horizontal = np.array([1, 1, 0, 0])
S_out = np.dot(M_norm, S_in_horizontal)

print("--- Output Stokes Vector (Horizontal Input) ---")
print(np.round(S_out, 3))