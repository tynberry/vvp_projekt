import numba
import numpy as np
from typing import Any
from numpy.typing import NDArray


@numba.vectorize([numba.int64(numba.complex128, numba.complex128, numba.int32)])
def count_iters(
    state: complex | NDArray[np.complex128],
    c: complex | NDArray[np.complex128],
    max_iter: int,
) -> int | NDArray[np.int32]:
    """
    Spočítá počet operací než hodnota vyběhne ven.

    :param state: počáteční hodnota
    :param c: číslo přičítané každou iteraci
    :param max_iter: maximální počet iterací
    """
    for i in range(max_iter):
        state **= 2
        state += c
        if np.abs(state) > 2:
            return i + 1
    return max_iter


# @numba.jit(signature="int64[:,:](float64,float64,float64,float64,int64,int64)",
#           nopython=True, locals={"divergence", numba.np_ar})
def mandelbrot(
    center: complex, side_length: complex, cells: int, max_iter: int
) -> NDArray[np.int32]:
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    # divergenční matice
    divergence = np.zeros((cells, cells), dtype=np.int32)
    # matice stavu
    # state = np.zeros((cells,cells), dtype=np.complex128)
    # matice C komplexních čísel
    extents = side_length / 2
    real: NDArray[np.complex128] = np.linspace(
        center.real - extents.real,
        center.real + extents.real,
        cells,
        dtype=np.complex128,
    ).reshape((1, cells))
    imag: NDArray[np.complex128] = (
        np.linspace(
            center.imag - extents.imag,
            center.imag + extents.imag,
            cells,
            dtype=np.complex128,
        ).reshape((cells, 1))
        * 1j
    )
    c_matrix = real + imag
    # spočti iterace
    divergence: Any = count_iters(0, c_matrix, max_iter)
    # vrať počet operací
    return divergence


def julia_set(
    center: complex,
    side_length: complex,
    c: complex,
    cells: int,
    max_iter: int,
) -> NDArray[np.int32]:
    """
    Vygeneruj Juliovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param c: číslo, které se přičítá každou iteraci
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    # divergenční matice
    divergence = np.zeros((cells, cells), dtype=np.int32)
    # matice C komplexních čísel
    extents = side_length / 2
    real: NDArray[np.complex128] = np.linspace(
        center.real - extents.real,
        center.real + extents.real,
        cells,
        dtype=np.complex128,
    ).reshape((1, cells))
    imag: NDArray[np.complex128] = (
        np.linspace(
            center.imag - extents.imag,
            center.imag + extents.imag,
            cells,
            dtype=np.complex128,
        ).reshape((cells, 1))
        * 1j
    )
    # matice stavu
    state = real + imag
    # udělej iterace
    divergence: Any = count_iters(state, c, max_iter)
    # vrať počet operací
    return divergence
