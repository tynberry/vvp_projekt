import numba
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


# @numba.vectorize([numba.int64(numba.complex128, numba.complex128, numba.int32)])
@numba.njit(
    "void(complex128[:,:], complex128[:,:], int64, int32[:,:])",
    nogil=True,
    parallel=True,
    locals={"i": numba.int32, "row": numba.int32, "col": numba.int32},
)
def count_iters(
    state: NDArray[np.complex128],
    c: NDArray[np.complex128],
    max_iter: int,
    iters: NDArray[np.int32],
):
    """
    Spočítá počet operací než hodnota vyběhne ven.

    :param state: počáteční hodnota
    :param c: číslo přičítané každou iteraci
    :param max_iter: maximální počet iterací
    """
    for row in numba.prange(state.shape[0]):
        for col in range(state.shape[1]):
            iters[row, col] = 0
            for i in range(max_iter):
                state[row, col] **= 2
                state[row, col] += c[row, col]
                if state[row, col].real ** 2 + state[row, col].imag ** 2 > 4:
                    iters[row, col] = i + 1
                    break
            else:
                iters[row, col] = max_iter


# @numba.jit("int32[:,:](complex128,complex128,tuple(int32),int32)")
def mandelbrot(
    center: complex, side_length: complex, cells: Tuple[int, int], max_iter: int
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
    state = np.zeros(c_matrix.shape, dtype=np.complex128)
    divergence: NDArray[np.int32] = np.zeros(c_matrix.shape, dtype=np.int32)
    count_iters(state, c_matrix, max_iter, divergence)
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
    c_matrix = np.full(state.shape, c, dtype=np.complex128)
    divergence: NDArray[np.int32] = np.zeros(state.shape, dtype=np.int32)
    count_iters(state, c_matrix, max_iter, divergence)
    # vrať počet operací
    return divergence
