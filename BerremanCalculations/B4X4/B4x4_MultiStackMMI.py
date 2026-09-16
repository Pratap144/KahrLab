import numpy as np
import matplotlib.pyplot as plt
import Berreman4x4 as b4  # Ensure Berreman4x4.py is in your directory


def jones_to_mueller(J):
    """Converts a 2x2 complex Jones matrix to a 4x4 real Mueller matrix."""
    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]]) / np.sqrt(2)
    # Inverse of unitary matrix U is its conjugate transpose
    U_inv = U.conj().T
    M = U @ np.kron(J, np.conj(J)) @ U_inv
    return np.real(M)


def build_berreman_stack(deltaPixel, twist_rad, nLayers, lbda=550e-9):
    """
    Rigorous Maxwell physics solver for a single point in the banded spherulite.
    Returns BOTH Reflection and Transmission Jones matrices.
    """
    h_layer = 100e-9  # 100 nm slices
    no = 1.5

    # Map phase delay to physical birefringence
    deltaLayer = deltaPixel / nLayers
    delta_n = (deltaLayer * lbda) / (2 * np.pi * h_layer)
    ne = no + delta_n

    eps_base = np.diag([ne ** 2, no ** 2, no ** 2])

    layers = []
    for k in range(nLayers):
        alpha = twist_rad * k / max(1, nLayers - 1)

        c, s = np.cos(alpha), np.sin(alpha)
        Rz = np.array([[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]])
        eps_rot = Rz @ eps_base @ Rz.T

        mat = b4.NonDispersiveMaterial(eps_rot)
        layers.append(b4.HomogeneousLayer(mat, h=h_layer))

    glass = b4.IsotropicHalfSpace(b4.IsotropicNonDispersiveMaterial(n=1.5))
    structure = b4.Structure(front=glass, layers=layers, back=glass)

    k0 = 2 * np.pi / lbda
    T_refl, T_trans = structure.getJones(Kx=0, k0=k0)

    return T_refl, T_trans


def spherulite_berreman_fast(N, nBands, nLayers, deltaTotal, twistAngle):
    """
    Vectorized mapping of the Berreman matrices to your spatial grid.
    Computes BOTH transmission and reflection maps.
    """
    twist_rad = np.deg2rad(twistAngle)
    p = N / (2 * nBands)

    print("1. Pre-computing Berreman physics (Maxwell's equations)...")

    delta_min = deltaTotal * 0.2
    delta_max = deltaTotal * 1.0
    num_points = 100
    delta_vals = np.linspace(delta_min, delta_max, num_points)

    mueller_trans_lookup = []
    mueller_refl_lookup = []

    for d_val in delta_vals:
        J_refl, J_trans = build_berreman_stack(d_val, twist_rad, nLayers)
        mueller_trans_lookup.append(jones_to_mueller(J_trans))
        mueller_refl_lookup.append(jones_to_mueller(J_refl))

    mueller_trans_lookup = np.array(mueller_trans_lookup)
    mueller_refl_lookup = np.array(mueller_refl_lookup)

    print("2. Physics computed. Vectorizing and mapping to spatial grid...")

    x = np.arange(1, N + 1) - N / 2
    y = np.arange(1, N + 1) - N / 2
    X, Y = np.meshgrid(x, y, indexing='ij')

    theta = np.arctan2(Y, X)
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = 2 * np.pi * r / p

    delta_pixel = deltaTotal * (0.2 + 0.8 * np.cos(phi) ** 2)

    indices = np.clip(np.round((delta_pixel - delta_min) / (delta_max - delta_min) * (num_points - 1)), 0,
                      num_points - 1).astype(int)

    M_trans_base = mueller_trans_lookup[indices]
    M_refl_base = mueller_refl_lookup[indices]

    c = np.cos(2 * theta)
    s = np.sin(2 * theta)

    R = np.zeros((N, N, 4, 4))
    R[:, :, 0, 0] = 1;
    R[:, :, 3, 3] = 1
    R[:, :, 1, 1] = c;
    R[:, :, 2, 2] = c
    R[:, :, 1, 2] = -s
    R[:, :, 2, 1] = s

    # Calculate both Transmitted and Reflected Spatial Mueller Maps
    Zdata_trans = np.einsum('xyij, xyjk, xylk -> ilxy', R, M_trans_base, R)
    Zdata_refl = np.einsum('xyij, xyjk, xylk -> ilxy', R, M_refl_base, R)

    return Zdata_trans, Zdata_refl


# --- Execution ---
if __name__ == "__main__":
    N = 300

    # Run the physics engine
    Zdata_trans, Zdata_refl = spherulite_berreman_fast(N, nBands=4, nLayers=120, deltaTotal=2 * np.pi, twistAngle=180)

    # --- ENERGY BALANCE CALCULATION ---
    # For unpolarized light, M_00 is the total intensity (Transmittance or Reflectance)
    # Energy = Transmittance + Reflectance
    print("\n--- ENERGY BALANCE CHECK ---")
    T_total = Zdata_trans[0, 0, :, :]
    R_total = Zdata_refl[0, 0, :, :]
    Energy_Balance = T_total + R_total

    print(f"Mean Transmittance: {np.mean(T_total) * 100:.2f}%")
    print(f"Mean Reflectance:   {np.mean(R_total) * 100:.2f}%")
    print(f"Total Energy (T+R): {np.mean(Energy_Balance):.5f}  (Should perfectly equal 1.0 for a lossless system)\n")

    # --- Visualization of Transmission Mueller Matrix ---
    print("3. Generating Plots...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            img = ax.imshow(Zdata_trans[row, col, :, :])
            ax.axis('off')
            ax.set_title(f'M_{{{row + 1},{col + 1}}}')
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Transmission Mueller Matrix (Berreman)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- Visualization of Reflection Mueller Matrix ---
    print("4. Generating Reflection Plots...")
    fig_refl, axes_refl = plt.subplots(4, 4, figsize=(10, 10))
    for row in range(4):
        for col in range(4):
            ax = axes_refl[row, col]
            # We plot Zdata_refl now
            img = ax.imshow(Zdata_refl[row, col, :, :])
            ax.axis('off')
            ax.set_title(f'Refl M_{{{row + 1},{col + 1}}}')
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Reflection Mueller Matrix (Berreman)", fontsize=16)
    plt.tight_layout()
    plt.show()