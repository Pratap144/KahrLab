import numpy as np
import matplotlib.pyplot as plt
import Berreman4x4 as b4
import os


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


def analytic_linear_retarder_mueller(delta, theta=0.0):
    cd = np.cos(delta)
    sd = np.sin(delta)
    Mlocal = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0,  cd,  sd],
        [0, 0, -sd,  cd]
    ], dtype=float)
    R = MMrot(theta)
    return R @ Mlocal @ R.T


def build_berreman_mueller(alpha, total_thickness, lbda=550e-9,
                           no=1.5, ne=1.55, n_ambient=1.5):
    eps_base = np.diag([ne ** 2, no ** 2, no ** 2])

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    Ry = np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ])

    eps_rotated = Ry @ eps_base @ Ry.T

    mat = b4.NonDispersiveMaterial(eps_rotated)
    layers = [b4.HomogeneousLayer(mat, h=total_thickness)] if total_thickness > 0 else []

    ambient = b4.IsotropicHalfSpace(b4.IsotropicNonDispersiveMaterial(n=n_ambient))
    structure = b4.Structure(front=ambient, layers=layers, back=ambient)

    k0 = 2 * np.pi / lbda
    J_refl, J_trans = structure.getJones(Kx=0, k0=k0)

    return jones_to_mueller(J_refl), jones_to_mueller(J_trans), J_refl, J_trans


def normalize_mueller(M):
    if abs(M[0, 0]) < 1e-15:
        return M.copy()
    return M / M[0, 0]


def print_case(title, M_refl, M_trans):
    print(f"\n===== {title} =====")
    print("Reflection Mueller matrix:")
    print(np.array_str(M_refl, precision=6, suppress_small=True))
    print("\nTransmission Mueller matrix:")
    print(np.array_str(M_trans, precision=6, suppress_small=True))
    print(f"\nR M11 = {M_refl[0,0]:.8f}, T M11 = {M_trans[0,0]:.8f}, T+R = {M_refl[0,0] + M_trans[0,0]:.8f}")


def plot_mueller_matrix(M, title, filename=None, normalize=False, cmap='coolwarm'):
    if normalize:
        Mplot = normalize_mueller(M)
        vmin, vmax = -1, 1
    else:
        Mplot = M
        vmax = np.max(np.abs(Mplot))
        if vmax < 1e-12:
            vmax = 1e-12
        vmin = -vmax

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            im = ax.imshow(np.array([[Mplot[i, j]]]), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"M$_{{{i+1},{j+1}}}$", fontsize=10)
            ax.text(0, 0, f"{Mplot[i,j]:.4f}", ha='center', va='center',
                    fontsize=10, color='black')

    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches='tight')


def plot_case_separately(M_refl, M_trans, case_name, outdir=None, normalize=False):
    refl_file = None
    trans_file = None

    if outdir is not None:
        refl_file = os.path.join(outdir, f"{case_name}_reflection.png")
        trans_file = os.path.join(outdir, f"{case_name}_transmission.png")

    plot_mueller_matrix(
        M_refl,
        title=f"{case_name} - Reflection Mueller Matrix",
        filename=refl_file,
        normalize=normalize
    )

    plot_mueller_matrix(
        M_trans,
        title=f"{case_name} - Transmission Mueller Matrix",
        filename=trans_file,
        normalize=normalize
    )


def plot_analytic_vs_berreman_separately(Manalytic, Mberr, outdir=None):
    file1 = None
    file2 = None

    if outdir is not None:
        file1 = os.path.join(outdir, "case3_analytic_retarder.png")
        file2 = os.path.join(outdir, "case3_berreman_transmission.png")

    plot_mueller_matrix(
        Manalytic,
        title="CASE 3 - Analytic Linear Retarder Mueller Matrix",
        filename=file1,
        normalize=False
    )

    plot_mueller_matrix(
        normalize_mueller(Mberr),
        title="CASE 3 - Berreman Transmission Mueller Matrix (Normalized)",
        filename=file2,
        normalize=False
    )


if __name__ == '__main__':
    outdir = 'berreman_case_plots'
    os.makedirs(outdir, exist_ok=True)

    lbda = 550e-9
    total_thickness = 12e-6
    no = 1.5
    ne = 1.55
    theta = 0.0

    # CASE 1: Isotropic limit
    Mrefl1, Mtrans1, _, _ = build_berreman_mueller(
        alpha=0.0,
        total_thickness=total_thickness,
        lbda=lbda,
        no=1.5,
        ne=1.5,
        n_ambient=1.5
    )
    print_case('CASE 1: Isotropic limit (ne = no)', Mrefl1, Mtrans1)
    plot_case_separately(
        Mrefl1, Mtrans1,
        'case1_isotropic',
        outdir=outdir,
        normalize=False
    )

    # CASE 2: Zero thickness limit
    Mrefl2, Mtrans2, _, _ = build_berreman_mueller(
        alpha=0.0,
        total_thickness=0.0,
        lbda=lbda,
        no=no,
        ne=ne,
        n_ambient=1.5
    )
    print_case('CASE 2: Zero thickness limit', Mrefl2, Mtrans2)
    plot_case_separately(
        Mrefl2, Mtrans2,
        'case2_zero_thickness',
        outdir=outdir,
        normalize=False
    )

    # CASE 3: Uniform in-plane retarder
    Mrefl3, Mtrans3, _, _ = build_berreman_mueller(
        alpha=0.0,
        total_thickness=total_thickness,
        lbda=lbda,
        no=no,
        ne=ne,
        n_ambient=1.5
    )

    delta_est = 2 * np.pi * (ne - no) * total_thickness / lbda
    Manalytic = analytic_linear_retarder_mueller(delta_est, theta=theta)

    print_case('CASE 3: Uniform in-plane retarder limit (alpha = 0)', Mrefl3, Mtrans3)
    print('\nAnalytic retarder Mueller matrix (normalized, approximate):')
    print(np.array_str(Manalytic, precision=6, suppress_small=True))
    print('\nDifference: normalized Berreman transmission - analytic')
    print(np.array_str(normalize_mueller(Mtrans3) - Manalytic, precision=6, suppress_small=True))

    plot_case_separately(
        Mrefl3, Mtrans3,
        'case3_uniform_retarder',
        outdir=outdir,
        normalize=False
    )

    plot_analytic_vs_berreman_separately(
        Manalytic, Mtrans3,
        outdir=outdir
    )

    # CASE 4: Optic axis along beam direction
    Mrefl4, Mtrans4, _, _ = build_berreman_mueller(
        alpha=np.pi/2,
        total_thickness=total_thickness,
        lbda=lbda,
        no=no,
        ne=ne,
        n_ambient=1.5
    )
    print_case('CASE 4: Optic axis along beam direction (alpha = pi/2)', Mrefl4, Mtrans4)
    plot_case_separately(
        Mrefl4, Mtrans4,
        'case4_alpha_pi_over_2',
        outdir=outdir,
        normalize=False
    )

    plt.show()