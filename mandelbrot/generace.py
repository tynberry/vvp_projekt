import numba
import numpy as np
from typing import Any, Tuple
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
    center: complex, side_length: complex, cells: int | Tuple[int, int], max_iter: int
) -> NDArray[np.int32]:
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    # rozdělení na políčka
    if isinstance(cells, int):
        cells_real = cells
        cells_imag = cells
    elif isinstance(cells, tuple):
        cells_real = cells[0]
        cells_imag = cells[1]
    # divergenční matice
    divergence = np.zeros((cells_real, cells_imag), dtype=np.int32)
    # matice stavu
    # state = np.zeros((cells,cells), dtype=np.complex128)
    # matice C komplexních čísel
    extents = side_length / 2
    real: NDArray[np.complex128] = np.linspace(
        center.real - extents.real,
        center.real + extents.real,
        cells_real,
        dtype=np.complex128,
    ).reshape((1, cells_real))
    imag: NDArray[np.complex128] = (
        np.linspace(
            center.imag - extents.imag,
            center.imag + extents.imag,
            cells_imag,
            dtype=np.complex128,
        ).reshape((cells_imag, 1))
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
    cells: int | Tuple[int, int],
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
    # rozdělení na políčka
    if isinstance(cells, int):
        cells_real = cells
        cells_imag = cells
    elif isinstance(cells, tuple):
        cells_real = cells[0]
        cells_imag = cells[1]
    # divergenční matice
    divergence = np.zeros((cells_real, cells_imag), dtype=np.int32)
    # matice C komplexních čísel
    extents = side_length / 2
    real: NDArray[np.complex128] = np.linspace(
        center.real - extents.real,
        center.real + extents.real,
        cells_real,
        dtype=np.complex128,
    ).reshape((1, cells_real))
    imag: NDArray[np.complex128] = (
        np.linspace(
            center.imag - extents.imag,
            center.imag + extents.imag,
            cells_imag,
            dtype=np.complex128,
        ).reshape((cells_imag, 1))
        * 1j
    )
    # matice stavu
    state = real + imag
    # udělej iterace
    divergence: Any = count_iters(state, c, max_iter)
    # vrať počet operací
    return divergence
