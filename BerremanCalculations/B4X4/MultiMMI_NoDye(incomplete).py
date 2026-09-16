import numpy as np
import matplotlib.pyplot as plt


def mm_rot(theta):
    """Mueller rotation matrix for a given angle in radians."""
    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def linear_retarder(delta):
    """Mueller matrix for a linear retarder with phase shift delta."""
    c = np.cos(delta)
    s = np.sin(delta)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, c, s],
        [0, 0, -s, c]
    ])


# def spherulite_twisted(N, nBands, nLayers, deltaTotal, twistAngle):
#     """
#     Python equivalent of the MATLAB spherulite_twisted function.
#     """
#     twist_rad = np.deg2rad(twistAngle)
#     p = N / (2 * nBands)
#
#     # Initialize Zdata array: 4x4 matrix per pixel (N x N)
#     # Using row, col, x, y indexing
#     Zdata = np.zeros((4, 4, N, N))
#
#     for X in range(N):
#         for Y in range(N):
#             x = X - N / 2
#             y = Y - N / 2
#
#             # Convert to polar
#             theta = np.arctan2(y, x)
#             r = np.sqrt(x ** 2 + y ** 2)
#             phi = 2 * np.pi * r / p
#
#             # Band modulation
#             delta_pixel = deltaTotal * (0.2 + 0.8 * np.cos(phi) ** 2)
#             delta_layer = delta_pixel / nLayers
#
#             M = np.eye(4)
#
#             for k in range(nLayers):
#                 # Layer twist
#                 alpha = theta + twist_rad * k / (nLayers - 1)
#
#                 R = mm_rot(alpha)
#
#                 # Cascade: Mk = R * Retarder * R_transpose
#                 Mk = R @ linear_retarder(delta_layer) @ R.T
#
#                 # Multiply: M_new = Mk * M_old
#                 M = Mk @ M
#
#             Zdata[:, :, X, Y] = M
#
#     return Zdata

def spherulite_twisted_vectorized(N, nBands, nLayers, deltaTotal, twistAngle):
    twist_rad = np.deg2rad(twistAngle)
    p = N / (2 * nBands)

    # Create coordinate grids
    x = np.arange(1, N + 1) - N / 2
    y = np.arange(1, N + 1) - N / 2
    X, Y = np.meshgrid(x, y, indexing='ij')

    theta = np.arctan2(Y, X)
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = 2 * np.pi * r / p

    # Pre-calculate delta for all pixels
    delta_pixel = deltaTotal * (0.2 + 0.8 * np.cos(phi) ** 2)
    delta_layer = delta_pixel / nLayers

    # Start with Identity matrices for all pixels: shape (4, 4, N, N)
    M = np.tile(np.eye(4)[:, :, np.newaxis, np.newaxis], (1, 1, N, N))

    for k in range(nLayers):
        # Calculate alpha for all pixels at once
        alpha = theta + twist_rad * k / max(1, nLayers - 1)

        # Calculate rotation matrices for all pixels: (4, 4, N, N)
        c = np.cos(2 * alpha)
        s = np.sin(2 * alpha)

        # Build Rotation array
        R = np.zeros((4, 4, N, N))
        R[0, 0] = 1;
        R[3, 3] = 1
        R[1, 1] = R[2, 2] = c
        R[1, 2] = -s
        R[2, 1] = s

        # Build Retarder array
        ret = np.zeros((4, 4, N, N))
        ret[0, 0] = 1;
        ret[1, 1] = 1
        ret[2, 2] = ret[3, 3] = np.cos(delta_layer)
        ret[2, 3] = np.sin(delta_layer)
        ret[3, 2] = -np.sin(delta_layer)

        # Matrix multiplication using einsum (fast)
        # Multiply R * Retarder * R.T
        Mk = np.einsum('ijxy, jkxy, lkxy -> ilxy', R, ret, R)

        # Update M: M_new = Mk * M_old
        M = np.einsum('ijxy, jkxy -> ikxy', Mk, M)

    return M


# --- Execution ---
N = 300
# Change the function name here to call the vectorized version:
Zdata = spherulite_twisted_vectorized(N, nBands=4, nLayers=40, deltaTotal=2 * np.pi, twistAngle=180)
# --- Visualization ---
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for row in range(4):
    for col in range(4):
        ax = axes[row, col]
        img = ax.imshow(Zdata[row, col, :, :])
        ax.axis('off')
        ax.set_title(f'M_{{{row + 1},{col + 1}}}')
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()