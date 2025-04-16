import numpy as np
import matplotlib.pyplot as plt
from .generace import mandelbrot, julia_set

def mandelbrot_visual(
        center: complex = -0.5,
        side_length: float = 3,
        color_map: str = "viridis",
        cells: int = 1024,
        max_iter: int = 100
    ):
    """
    Vykreslí Mandelbrotovu množinu se středem v `center` a 
    délkou hrany `zoom`.

    :param center: střed vykreslení
    :param side_length: délka strany zobrazovacího čtverce
    :param color_map: barevná mapa
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    #vypočti množinu 
    extent = side_length / 2
    mb_set = mandelbrot(
        center.real - extent,
        center.real + extent,
        center.imag - extent,
        center.imag + extent,
        cells,
        max_iter
    )
    #vykresli množinu 
    visual(mb_set.reshape((cells, cells)), center, side_length, color_map)


def julia_set_visual(
        center: complex = -0.5,
        side_length: float = 3,
        c: complex = -0.8 + 0.156j,
        color_map: str = "viridis",
        cells: int = 1024,
        max_iter: int = 100
    ):
    """
    Vykreslí Juliovu množinu se středem v `center` a 
    délkou hrany `zoom`.

    :param center: střed vykreslení
    :param side_length: délka strany zobrazovacího čtverce
    :param color_map: barevná mapa
    :param c: číslo, které se přičítá každou iteraci
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    #vypočti množinu 
    extent = side_length / 2
    mb_set = julia_set(
        center.real - extent,
        center.real + extent,
        center.imag - extent,
        center.imag + extent,
        c,
        cells,
        max_iter
    )
    #vykresli množinu 
    visual(mb_set.reshape((cells, cells)), center, side_length, color_map)


def visual(
        set: np.array,
        center: complex = -0.5,
        side_length: float = 3, 
        color_map: str = "viridis"
    ):
    """
    Vykreslí množinu s barevnou mapou.
    Přepíše osy, tak aby odpovídaly středu a délce strany.

    Množinu vykresluje mapováním počtu iterací na barevnou mapu.

    :param set: množina na vykreslení, pole počtu iterací před divergencí
    :param center: střed zobrazení 
    :param side_length: délka stran zobrazení
    :param color_map: barevná mapa
    """
    extent = side_length / 2
    plt.imshow(
        set,
        cmap=color_map,
        extent=(
            center.real - extent, 
            center.real + extent, 
            center.imag - extent, 
            center.imag + extent 
        )
    )
    plt.show()