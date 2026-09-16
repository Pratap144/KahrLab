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


def build_pixel(alpha0, theta_xy, total_thickness, n_sub=60,
                twist_total=np.pi, splay_amp=np.deg2rad(15),
                lbda=550e-9, no=1.5, ne=1.55, n_ambient=1.5):
    dz = total_thickness / n_sub
    layers = []

    for k in range(n_sub):
        z = (k + 0.5) * dz

        alpha_k = alpha0 + splay_amp * np.sin(2 * np.pi * z / total_thickness)
        psi_k = theta_xy + twist_total * (z / total_thickness)

        eps_k = build_eps_from_tilt_and_azimuth(alpha_k, psi_k, no=no, ne=ne)
        mat = b4.NonDispersiveMaterial(eps_k)
        layers.append(b4.HomogeneousLayer(mat, h=dz))

    ambient = b4.IsotropicHalfSpace(b4.IsotropicNonDispersiveMaterial(n=n_ambient))
    structure = b4.Structure(front=ambient, layers=layers, back=ambient)

    k0 = 2 * np.pi / lbda
    J_refl, J_trans = structure.getJones(Kx=0, k0=k0)

    return jones_to_mueller(J_refl), jones_to_mueller(J_trans)


def build_image(N=120, nTwist=4, total_thickness=10e-6,
                n_sub=60, twist_total=np.pi,
                splay_amp=np.deg2rad(15), alpha_max=np.pi/2,
                no=1.5, ne=1.55, n_ambient=1.5, lbda=550e-9):
    p = N / (2 * nTwist)

    Zt = np.zeros((4, 4, N, N), dtype=float)
    Zr = np.zeros((4, 4, N, N), dtype=float)
    DI = np.zeros((N, N), dtype=float)
    E = np.zeros((N, N), dtype=float)

    for X in range(N):
        for Y in range(N):
            x = (X + 1) - N / 2
            y = (Y + 1) - N / 2

            theta_xy = np.arctan2(y, x)
            r = np.hypot(x, y)
            phi_r = 2 * np.pi * r / p
            alpha0 = 0.5 * alpha_max * (1 + np.cos(phi_r))

            Mr, Mt = build_pixel(
                alpha0, theta_xy, total_thickness,
                n_sub=n_sub,
                twist_total=twist_total,
                splay_amp=splay_amp,
                lbda=lbda, no=no, ne=ne,
                n_ambient=n_ambient
            )

            Zt[:, :, X, Y] = Mt
            Zr[:, :, X, Y] = Mr
            E[X, Y] = Mt[0, 0] + Mr[0, 0]

            if abs(Mt[0, 0]) > 1e-12:
                Mnorm = Mt / Mt[0, 0]
            else:
                Mnorm = Mt

            val = np.sum(Mnorm**2) - 1
            DI[X, Y] = np.sqrt(max(val, 0)) / np.sqrt(3)

    return Zt, Zr, DI, E


def summarize_case(name, Zt, Zr, DI, E):
    print(f"\n===== {name} =====")
    print(f"Mean T11: {np.mean(Zt[0,0]):.6f}")
    print(f"Mean R11: {np.mean(Zr[0,0]):.6f}")
    print(f"Mean T+R: {np.mean(E):.6f}")
    print(f"Min  T+R: {np.min(E):.6f}")
    print(f"Max  T+R: {np.max(E):.6f}")
    print(f"Mean DI : {np.mean(DI):.6f}")
    print(f"Min  DI : {np.min(DI):.6f}")
    print(f"Max  DI : {np.max(DI):.6f}")
    print(f"Max |M14|: {np.max(np.abs(Zt[0,3])):.6e}")
    print(f"Max |M41|: {np.max(np.abs(Zt[3,0])):.6e}")
    print(f"Max |M24|: {np.max(np.abs(Zt[1,3])):.6e}")
    print(f"Max |M42|: {np.max(np.abs(Zt[3,1])):.6e}")
    print(f"Max |M34|: {np.max(np.abs(Zt[2,3])):.6e}")
    print(f"Max |M43|: {np.max(np.abs(Zt[3,2])):.6e}")


def plot_selected_elements(cases, elements, title_prefix='Transmission', cmap='coolwarm'):
    nrows = len(cases)
    ncols = len(elements)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), constrained_layout=True)

    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for i, (case_name, Zdata) in enumerate(cases):
        for j, (r, c, label) in enumerate(elements):
            ax = axes[i, j]
            im = ax.imshow(Zdata[r, c], origin='upper', cmap=cmap)
            ax.set_title(f'{case_name}: {label}')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title_prefix, fontsize=14)
    plt.show()


def plot_maps(cases_maps, cmap='viridis'):
    nrows = len(cases_maps)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4*nrows), constrained_layout=True)

    if nrows == 1:
        axes = np.array([axes])

    for i, (name, DI, E) in enumerate(cases_maps):
        im1 = axes[i, 0].imshow(DI, origin='upper', cmap=cmap)
        axes[i, 0].set_title(f'{name}: DI')
        axes[i, 0].axis('off')
        fig.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)

        im2 = axes[i, 1].imshow(E, origin='upper', cmap=cmap)
        axes[i, 1].set_title(f'{name}: T11 + R11')
        axes[i, 1].axis('off')
        fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.show()


if __name__ == '__main__':
    params = dict(
        N=100,
        nTwist=4,
        total_thickness=10e-6,
        n_sub=60,
        alpha_max=np.pi/2,
        no=1.5,
        ne=1.55,
        n_ambient=1.5,
        lbda=550e-9,
    )

    case_defs = [
        ('Twist only', dict(twist_total=np.pi, splay_amp=0.0)),
        ('Splay only', dict(twist_total=0.0, splay_amp=np.deg2rad(15))),
        ('Twist + splay', dict(twist_total=np.pi, splay_amp=np.deg2rad(15))),
    ]

    results = []
    for name, extra in case_defs:
        print(f'Running {name} ...')
        Zt, Zr, DI, E = build_image(**params, **extra)
        summarize_case(name, Zt, Zr, DI, E)
        results.append((name, Zt, Zr, DI, E))

    elements = [
        (0, 0, 'M11'),
        (1, 1, 'M22'),
        (2, 2, 'M33'),
        (3, 3, 'M44'),
        (1, 3, 'M24'),
        (3, 1, 'M42'),
        (2, 3, 'M34'),
        (3, 2, 'M43'),
        (0, 3, 'M14'),
        (3, 0, 'M41'),
    ]

    plot_selected_elements(
        [(name, Zt) for name, Zt, _, _, _ in results],
        elements,
        title_prefix='Transmitted Mueller Elements Comparison'
    )

    plot_selected_elements(
        [(name, Zr) for name, _, Zr, _, _ in results],
        [(0, 0, 'M11'), (1, 3, 'M24'), (2, 3, 'M34'), (0, 3, 'M14')],
        title_prefix='Reflected Mueller Elements Comparison'
    )

    plot_maps(
        [(name, DI, E) for name, _, _, DI, E in results],
        cmap='viridis'
    )