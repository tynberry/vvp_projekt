import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.typing import NDArray


# @numba.guvectorize(
#    [numba.void(numba.int32[:, :], numba.int32[:], numba.float64, numba.float32[:, :])],
#    "(n,m),(p),()->(n,m)",
# )
@numba.njit(
    "void(int32[:, :], int64[:], float64, float32[:, :])",
    nogil=True,
    parallel=True,
    fastmath=True,
    locals={"row": numba.int32, "col": numba.int32},
)
def count_hue(
    iter: NDArray[np.int32],
    hist: NDArray[np.int32],
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

    for row in numba.prange(iter.shape[0]):
        for col in range(iter.shape[1]):
            # přidej barvu do pole
            hues[row, col] = hist[iter[row, col]] / total


def convert_set_to_color(
    set: NDArray[np.int32],
    hues: NDArray[np.float32],
    max_iter: int,
    color_map: str = "plasma",
) -> NDArray[np.float32]:
    """
    Vrátí pole barev obarvené množiny podle dané barevné mapy.

    :param set: pole iterací před divergencí
    :param hues: pole barev
    :param color_map: barevná mapa
    """
    # vytvoř histrogram iterací
    histogram, _ = np.histogram(set, bins=max_iter + 1)
    np.cumsum(histogram, out=histogram)
    # spočti celkový počet pixelů
    total = float(histogram[-1])
    # spočti pozici na paletě
    count_hue(set, histogram, total, hues)
    # vrať color mapu
    return matplotlib.colormaps[color_map](hues)


def visual(
    set: NDArray[np.int32],
    max_iter: int,
    center: complex = 0 + 0j,
    side_length: float = 1,
    color_map: str = "plasma",
):
    """
    Vizualizuje pole počtu iterací pomocí histogramového barvení.

    Zdroj:
    https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

    :param set: pole počtu iterací před divergencí
    :param max_iter: maximální počet iterací
    """
    # barevné pole
    hues = convert_set_to_color(
        set, np.zeros(set.shape, dtype=np.float32), max_iter, color_map
    )
    # vykresli množinu
    extent = side_length / 2
    plt.imshow(
        hues,
        extent=(
            center.real - extent,
            center.real + extent,
            center.imag - extent,
            center.imag + extent,
        ),
    )
    plt.show()
