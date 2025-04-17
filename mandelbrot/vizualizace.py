import numpy as np
from numpy.typing import NDArray

import numba
import matplotlib.pyplot as plt


@numba.guvectorize(
    [numba.void(numba.int64[:, :], numba.int64[:], numba.float64, numba.float32[:, :])],
    "(n,m),(p),()->(n,m)",
)
def count_hue(
    iter: NDArray[np.int64],
    hist: NDArray[np.int64],
    total: float,
    hues: NDArray[np.float32],
):
    """
    Vypočte hodnotu barvy všech bodů množiny.

    :param iter: pole počtu iterací před divergencí
    :param hist: histogram iterací
    :param total: celkový počet pixelů
    :param hues: výsledné pole barev
    """

    for row in range(iter.shape[0]):
        for col in range(iter.shape[1]):
            iters = iter[row, col]
            hues[row, col] = 0
            for i in range(iters):
                # přidej barvu do pole
                hues[row, col] += hist[i] / total


def visual(
    set: NDArray[np.int32],
    center: complex,
    side_length: float,
    color_map: str = "plasma",
):
    """
    Vizualizuje pole počtu iterací pomocí histogramového barvení.

    Zdroj:
    https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

    :param set: pole počtu iterací před divergencí
    :param max_iter: maximální počet iterací
    """
    # vytvoř histrogram iterací
    max_iter = np.max(set)
    histogram, _ = np.histogram(set, bins=max_iter)
    # spočti celkový počet pixelů
    total = float(np.sum(histogram))
    # spočti pozici na paletě
    hues = count_hue(set, histogram, total)
    # vykresli množinu
    extent = side_length / 2
    plt.imshow(
        hues,
        cmap=color_map,
        extent=(
            center.real - extent,
            center.real + extent,
            center.imag - extent,
            center.imag + extent,
        ),
    )
    plt.show()

