import numpy as np
import matplotlib.pyplot as plt


def mmrot(theta):
    """Mueller rotation matrix."""
    c = np.cos(2 * theta)
    s = np.sin(2 * theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def linear_retarder(delta):
    """Linear retarder Mueller matrix."""
    cd = np.cos(delta)
    sd = np.sin(delta)
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0,  cd,  sd],
        [0, 0, -sd,  cd]
    ], dtype=float)


def spherulite_retarder(N, nTwist, deltaMax, deltaMin, show=True):
    """
    Simulates a banded spherulite as a radially oriented linear retarder
    whose retardance varies periodically with radius.
    """
    # NO PHYSICS CHANGED. IDENTICAL TO ORIGINAL CODE.
    p = N / (2 * nTwist)

    Zdata = np.zeros((4, 4, N, N), dtype=float)
    DI = np.zeros((N, N), dtype=float)

    for X in range(N):
        for Y in range(N):
            x = (X + 1) - N / 2
            y = (Y + 1) - N / 2

            theta = np.arctan2(y, x)
            r = np.hypot(x, y)

            phi = 2 * np.pi * r / p
            delta = deltaMin + (deltaMax - deltaMin) * (np.cos(phi) ** 2)

            Mlocal = linear_retarder(delta)
            R = mmrot(theta)
            M = R @ Mlocal @ R.T

            Zdata[:, :, X, Y] = M

            m11 = M[0, 0]
            DI[X, Y] = np.sqrt(np.sum((M / m11) ** 2) - 1) / np.sqrt(3)

    if show:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle("Mueller Matrix (Phenomenological Code 1)", fontsize=14)

        for row in range(4):
            for col in range(4):
                ax = axes[row, col]
                im = ax.imshow(Zdata[row, col, :, :], origin="upper", cmap="coolwarm")
                ax.set_title(f"M$_{{{row + 1},{col + 1}}}$")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        plt.figure(figsize=(5, 5))
        plt.imshow(DI, origin="upper", cmap="viridis")
        plt.axis("image")
        plt.colorbar()
        plt.title("Depolarization Index (DI)")
        plt.tight_layout()
        plt.show()

    return DI, Zdata


# Example usage
if __name__ == "__main__":
    # Parameters returned precisely to your original provided values
    DI, Zdata = spherulite_retarder(300, 4, 1.5, 0.2, show=True)