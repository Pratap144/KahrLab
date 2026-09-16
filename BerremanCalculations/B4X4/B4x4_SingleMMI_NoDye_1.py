import numpy as np
import matplotlib.pyplot as plt
import Berreman4x4 as b4


def jones_to_mueller(J):
    """Convert a 2x2 Jones matrix to a 4x4 Mueller matrix."""
    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]], dtype=complex) / np.sqrt(2)
    M = U @ np.kron(J, np.conj(J)) @ U.conj().T
    return np.real_if_close(M)


def MMrot(theta):
    """Mueller rotation matrix."""
    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def build_berreman_out_of_plane_retarder(alpha, total_thickness, lbda=550e-9):
    """
    True Berreman physics: Constant intrinsic birefringence, but the optic axis
    twists out of the XY plane by angle 'alpha'.
    """
    no = 1.5
    ne = 1.55  # Fixed intrinsic birefringence

    # Base dielectric tensor (optic axis fully in-plane along X)
    eps_base = np.diag([ne ** 2, no ** 2, no ** 2])

    # 3D Rotation matrix to tilt the optic axis out of plane (around Y axis)
    # When alpha=0: optic axis is in-plane (maximum retardance)
    # When alpha=pi/2: optic axis is pointing at camera (zero in-plane retardance)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    Ry = np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ])

    # Apply the out-of-plane rotation to the dielectric tensor
    eps_rotated = Ry @ eps_base @ Ry.T

    # PERFORMANCE FIX: Single bulk layer instead of stacking 'nLayers'
    mat = b4.NonDispersiveMaterial(eps_rotated)
    layers = [b4.HomogeneousLayer(mat, h=total_thickness)]

    # Boundary conditions
    glass = b4.IsotropicHalfSpace(b4.IsotropicNonDispersiveMaterial(n=1.5))
    structure = b4.Structure(front=glass, layers=layers, back=glass)

    k0 = 2 * np.pi / lbda
    J_refl, J_trans = structure.getJones(Kx=0, k0=k0)

    return jones_to_mueller(J_refl), jones_to_mueller(J_trans)


def spherulite_berreman_true_physics(N, nTwist, total_thickness):
    """
    Calculates the spatial Mueller matrix image of a spherulite using
    true out-of-plane lamellar twisting.
    """
    p = N / (2 * nTwist)

    Zdata_trans = np.zeros((4, 4, N, N), dtype=float)
    Zdata_refl = np.zeros((4, 4, N, N), dtype=float)
    DI = np.zeros((N, N), dtype=float)

    print("1. Precomputing true Berreman out-of-plane lookup...")
    num_points = 200
    # Alpha (out-of-plane angle) needs to cover a full 2*pi rotation for lookup
    alpha_vals = np.linspace(0, 2 * np.pi, num_points)

    mueller_trans_lookup = []
    mueller_refl_lookup = []

    for a in alpha_vals:
        M_refl, M_trans = build_berreman_out_of_plane_retarder(a, total_thickness)
        mueller_refl_lookup.append(M_refl)
        mueller_trans_lookup.append(M_trans)

    mueller_refl_lookup = np.array(mueller_refl_lookup)
    mueller_trans_lookup = np.array(mueller_trans_lookup)

    print("2. Building spatial Mueller maps...")
    for X in range(N):
        for Y in range(N):
            x = (X + 1) - N / 2
            y = (Y + 1) - N / 2

            theta = np.arctan2(y, x)
            r = np.sqrt(x ** 2 + y ** 2)

            # Calculate the out-of-plane pitch angle based on radius
            alpha = 2 * np.pi * r / p

            # Wrap alpha to 0 -> 2*pi so it safely queries the lookup table
            alpha_wrapped = alpha % (2 * np.pi)
            idx = int(np.round((alpha_wrapped / (2 * np.pi)) * (num_points - 1)))

            Mlocal_trans = mueller_trans_lookup[idx]
            Mlocal_refl = mueller_refl_lookup[idx]

            # Rotate to match radial symmetry in the XY plane
            R = MMrot(theta)
            M_trans = R @ Mlocal_trans @ R.T
            M_refl = R @ Mlocal_refl @ R.T

            Zdata_trans[:, :, X, Y] = M_trans
            Zdata_refl[:, :, X, Y] = M_refl

            if M_trans[0, 0] != 0:
                Mnorm = M_trans / M_trans[0, 0]
            else:
                Mnorm = M_trans
            DI[X, Y] = np.sqrt(np.sum(Mnorm ** 2) - 1) / np.sqrt(3)

    return Zdata_trans, Zdata_refl, DI


if __name__ == "__main__":
    N = 300
    nTwist = 4
    # Replaced 120 layers of 100nm with a single 12um total thickness variable
    total_thickness = 12e-6

    Zdata_trans, Zdata_refl, DI = spherulite_berreman_true_physics(
        N=N,
        nTwist=nTwist,
        total_thickness=total_thickness
    )

    # Energy balance Check
    T_total = Zdata_trans[0, 0, :, :]
    R_total = Zdata_refl[0, 0, :, :]
    Energy_Balance = T_total + R_total

    print("\n--- ENERGY BALANCE CHECK ---")
    print(f"Mean Transmittance: {np.mean(T_total) * 100:.2f}%")
    print(f"Mean Reflectance:   {np.mean(R_total) * 100:.2f}%")
    print(f"Total Energy (T+R): {np.mean(Energy_Balance):.5f}")

    # Plotting Transmission Mueller Matrix
    print("\n3. Plotting transmission Mueller matrix...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            img = ax.imshow(Zdata_trans[row, col, :, :], origin='upper', cmap='coolwarm')
            ax.axis('off')
            ax.set_title(f'T M$_{{{row + 1},{col + 1}}}$')
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Transmission Mueller Matrix (True 3D Berreman Spherulite)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plotting Reflection Mueller Matrix
    print("4. Plotting reflection Mueller matrix...")
    fig_r, axes_r = plt.subplots(4, 4, figsize=(10, 10))
    for row in range(4):
        for col in range(4):
            ax = axes_r[row, col]
            img = ax.imshow(Zdata_refl[row, col, :, :], origin='upper', cmap='coolwarm')
            ax.axis('off')
            ax.set_title(f'R M$_{{{row + 1},{col + 1}}}$')
            plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Reflection Mueller Matrix (True 3D Berreman Spherulite)", fontsize=14)
    plt.tight_layout()
    plt.show()