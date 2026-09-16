import numpy as np
import matplotlib.pyplot as plt
import Berreman4x4 as b4


def jones_to_mueller(J):
    U = np.array([[1, 0, 0, 1],
                  [1, 0, 0, -1],
                  [0, 1, 1, 0],
                  [0, 1j, -1j, 0]], dtype=complex) / np.sqrt(2)
    M = U @ np.kron(J, np.conj(J)) @ U.conj().T
    return np.real_if_close(M)


def MMrot(theta):
    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def rotation_matrix_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa,  ca]
    ], dtype=float)


def rotation_matrix_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ], dtype=float)


def rotation_matrix_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [0,   0,  1]
    ], dtype=float)


def build_eps_from_tilt_and_azimuth(alpha, psi, no, ne):
    eps_base = np.diag([ne**2, no**2, no**2])
    R = rotation_matrix_z(psi) @ rotation_matrix_y(alpha)
    return R @ eps_base @ R.T


def build_twisted_splayed_pixel(alpha0, theta_xy, total_thickness, n_sub=60,
                                twist_total=np.pi, splay_amp=np.deg2rad(15),
                                lbda=550e-9, no=1.5, ne=1.55, n_ambient=1.5):
    dz = total_thickness / n_sub
    layers = []

    for k in range(n_sub):
        z = (k + 0.5) * dz

        # depth-dependent tilt (splay)
        alpha_k = alpha0 + splay_amp * np.sin(2 * np.pi * z / total_thickness)

        # depth-dependent azimuthal twist
        psi_k = theta_xy + twist_total * (z / total_thickness)

        eps_k = build_eps_from_tilt_and_azimuth(alpha_k, psi_k, no=no, ne=ne)
        mat = b4.NonDispersiveMaterial(eps_k)
        layers.append(b4.HomogeneousLayer(mat, h=dz))

    ambient = b4.IsotropicHalfSpace(b4.IsotropicNonDispersiveMaterial(n=n_ambient))
    structure = b4.Structure(front=ambient, layers=layers, back=ambient)

    k0 = 2 * np.pi / lbda
    J_refl, J_trans = structure.getJones(Kx=0, k0=k0)

    return jones_to_mueller(J_refl), jones_to_mueller(J_trans)


def spherulite_twisted_splayed_berreman(N=200, nTwist=4, total_thickness=10e-6,
                                        n_sub=60, twist_total=np.pi,
                                        splay_amp=np.deg2rad(15),
                                        alpha_max=np.pi/2,
                                        no=1.5, ne=1.55, n_ambient=1.5,
                                        lbda=550e-9):
    p = N / (2 * nTwist)

    Zdata_trans = np.zeros((4, 4, N, N), dtype=float)
    Zdata_refl = np.zeros((4, 4, N, N), dtype=float)
    DI = np.zeros((N, N), dtype=float)

    print("1. Building twisted+splayed Berreman spherulite...")

    for X in range(N):
        for Y in range(N):
            x = (X + 1) - N / 2
            y = (Y + 1) - N / 2

            theta_xy = np.arctan2(y, x)
            r = np.hypot(x, y)
            phi_r = 2 * np.pi * r / p

            # radial variation: flat-on -> edge-on -> flat-on
            alpha0 = 0.5 * alpha_max * (1 + np.cos(phi_r))

            M_refl, M_trans = build_twisted_splayed_pixel(
                alpha0=alpha0,
                theta_xy=theta_xy,
                total_thickness=total_thickness,
                n_sub=n_sub,
                twist_total=twist_total,
                splay_amp=splay_amp,
                lbda=lbda,
                no=no,
                ne=ne,
                n_ambient=n_ambient
            )

            Zdata_trans[:, :, X, Y] = M_trans
            Zdata_refl[:, :, X, Y] = M_refl

            if abs(M_trans[0, 0]) > 1e-12:
                Mnorm = M_trans / M_trans[0, 0]
            else:
                Mnorm = M_trans

            val = np.sum(Mnorm**2) - 1
            DI[X, Y] = np.sqrt(max(val, 0)) / np.sqrt(3)

    return Zdata_trans, Zdata_refl, DI


def plot_mueller_image(Zdata, title, prefix='', cmap='coolwarm', origin='upper'):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)

    for row in range(4):
        for col in range(4):
            ax = axes[row, col]
            img = ax.imshow(Zdata[row, col, :, :], origin=origin, cmap=cmap)
            ax.axis('off')
            ax.set_title(f'{prefix} M$_{{{row+1},{col+1}}}$')
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.show()


def plot_DI(DI, title='Depolarization Index', origin='upper'):
    plt.figure(figsize=(6, 6))
    plt.imshow(DI, origin=origin, cmap='viridis')
    plt.axis('image')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 180
    nTwist = 4
    total_thickness = 10e-6
    n_sub = 60
    twist_total = np.pi
    splay_amp = np.deg2rad(15)
    alpha_max = np.pi / 2
    no = 1.5
    ne = 1.55
    n_ambient = 1.5
    lbda = 550e-9

    Zdata_trans, Zdata_refl, DI = spherulite_twisted_splayed_berreman(
        N=N,
        nTwist=nTwist,
        total_thickness=total_thickness,
        n_sub=n_sub,
        twist_total=twist_total,
        splay_amp=splay_amp,
        alpha_max=alpha_max,
        no=no,
        ne=ne,
        n_ambient=n_ambient,
        lbda=lbda
    )

    T_total = Zdata_trans[0, 0, :, :]
    R_total = Zdata_refl[0, 0, :, :]
    Energy_Balance = T_total + R_total

    print("\n--- ENERGY BALANCE CHECK ---")
    print(f"Mean Transmittance: {np.mean(T_total) * 100:.4f}%")
    print(f"Mean Reflectance:   {np.mean(R_total) * 100:.4f}%")
    print(f"Mean Total Energy:  {np.mean(Energy_Balance):.6f}")

    print("\n2. Plotting transmission Mueller matrix...")
    plot_mueller_image(
        Zdata_trans,
        title="Transmission Mueller Matrix (Twisted + Splayed Berreman Spherulite)",
        prefix='T',
        cmap='coolwarm',
        origin='upper'
    )

    print("3. Plotting reflection Mueller matrix...")
    plot_mueller_image(
        Zdata_refl,
        title="Reflection Mueller Matrix (Twisted + Splayed Berreman Spherulite)",
        prefix='R',
        cmap='coolwarm',
        origin='upper'
    )

    print("4. Plotting DI...")
    plot_DI(
        DI,
        title="Depolarization Index (Twisted + Splayed Berreman Spherulite)",
        origin='upper'
    )