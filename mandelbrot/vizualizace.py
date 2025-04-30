import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from mandelbrot.viz_c import convert_set_to_color


def visual(
    set: NDArray[np.int32],
    max_iter: int,
    center: complex = 0 + 0j,
    side_length: complex = 1 + 1j,
    color_map: str = "plasma",
):
    """
    Vizualizuje pole počtu iterací pomocí histogramového barvení.

    Zdroj:
    https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

    :param set: pole počtu iterací před divergencí
    :param max_iter: maximální počet iterací
    """
    # vypočti barevný LUP
    lup = np.linspace(0, 1, num=max_iter + 1, dtype=np.uint8)
    lup = matplotlib.colormaps[color_map](lup, bytes=True)
    # barevné pole
    hues = np.zeros((set.shape[0], set.shape[1], 3), dtype=np.uint8)
    convert_set_to_color(set, hues, max_iter, lup)
    # vykresli množinu
    extent = side_length / 2
    plt.imshow(
        hues,
        extent=(
            center.real - extent.real,
            center.real + extent.real,
            center.imag - extent.imag,
            center.imag + extent.imag,
        ),
    )
    plt.show()
