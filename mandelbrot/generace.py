import numba
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


@numba.njit(
    "int32(complex128, complex128, int32)",
    nogil=True,
    locals={
        "i": numba.int32,
        "z_real": numba.float64,
        "z_imag": numba.float64,
        "c_real": numba.float64,
        "c_imag": numba.float64,
        "temp_real": numba.float64,
    },
)
def count_iters(state: complex, c: complex, max_iter: int) -> int:
    """
    Spočítá počet operací než hodnota vyběhne ven.

    :param state: počáteční hodnota
    :param c: číslo přičítané každou iteraci
    :param max_iter: maximální počet iterací
    """
    z_real = state.real
    z_imag = state.imag
    c_real = c.real
    c_imag = c.imag

    for i in range(max_iter):
        temp_real = z_real * z_real - z_imag * z_imag + c_real
        z_imag = (z_real + z_real) * z_imag + c_imag
        z_real = temp_real
        if z_real * z_real + z_imag * z_imag > 4.0:
            return i + 1
    return max_iter


@numba.njit(
    numba.void(
        numba.complex128,
        numba.complex128,
        numba.types.containers.UniTuple(numba.int32, 2),
        numba.int32,
        numba.int32[:, :],
    ),
    nopython=True,
    nogil=True,
    parallel=True,
    locals={
        "real_ind": numba.int32,
        "imag_ind": numba.int32,
        "c_real": numba.float64,
        "c_imag": numba.float64,
    },
)
def mandelbrot(
    center: complex,
    side_length: complex,
    cells: Tuple[int, int],
    max_iter: int,
    divergence: NDArray[np.int32],
):
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    # projdi každý bod
    for real_ind in numba.prange(cells[0]):
        for imag_ind in range(cells[1]):
            # vypočti podmínky
            c_real = (center.real - side_length.real / 2) + (
                real_ind / cells[0]
            ) * side_length.real
            c_imag = (center.imag - side_length.imag / 2) + (
                imag_ind / cells[1]
            ) * side_length.imag
            # iteruj
            divergence[real_ind, imag_ind] = count_iters(
                0.0, c_real + c_imag * 1j, max_iter
            )


def julia_set(
    center: complex,
    side_length: complex,
    c: complex,
    cells: Tuple[int, int],
    max_iter: int,
    divergence: NDArray[np.int32],
):
    """
    Vygeneruj Juliovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param c: číslo, které se přičítá každou iteraci
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    # projdi každý bod
    for real_ind in numba.prange(cells[0]):
        for imag_ind in range(cells[1]):
            # vypočti podmínky
            z0_real = (center.real - side_length.real / 2) + (
                real_ind / cells[0]
            ) * side_length.real
            z0_imag = (center.imag - side_length.imag / 2) + (
                imag_ind / cells[1]
            ) * side_length.imag
            # iteruj
            divergence[real_ind, imag_ind] = count_iters(
                z0_real + z0_imag * 1j, c, max_iter
            )
