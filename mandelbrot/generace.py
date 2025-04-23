import numba
import numpy as np
from numpy.typing import NDArray
from .gen import count_iters


@numba.njit(
    "int32(float64, float64, float64, float64, int32)",
    nogil=True,
    fastmath=True,
    locals={
        "i": numba.int32,
        "temp_real": numba.float64,
    },
)
def count_iters(
    z_real: float,
    z_imag: float,
    c_real: float,
    c_imag: float,
    max_iter: int,
) -> int:
    """
    Spočítá počet operací než hodnota vyběhne ven.

    :param max_iter: maximální počet iterací
    """

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
        numba.int32,
        numba.int32[:, :],
    ),
    nogil=True,
    parallel=True,
    locals={
        "real_size": numba.int64,
        "imag_size": numba.int64,
        "real_ind": numba.int32,
        "imag_ind": numba.int32,
        "c_real": numba.float64,
        "c_imag": numba.float64,
    },
)
def mandelbrot(
    center: complex,
    side_length: complex,
    max_iter: int,
    divergence: NDArray[np.int32],
):
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param max_iter: maximální počet iterací
    """
    real_size = divergence.shape[0]
    imag_size = divergence.shape[1]
    # projdi každý bod
    for real_ind in numba.prange(real_size):
        for imag_ind in range(imag_size):
            # vypočti podmínky
            c_real = (center.real - side_length.real / 2) + (
                real_ind / real_size
            ) * side_length.real
            c_imag = (center.imag - side_length.imag / 2) + (
                imag_ind / imag_size
            ) * side_length.imag
            # iteruj
            divergence[real_ind, imag_ind] = count_iters(
                0.0, 0.0, c_real, c_imag, max_iter
            )


@numba.njit(
    numba.void(
        numba.complex128,
        numba.complex128,
        numba.complex128,
        numba.int32,
        numba.int32[:, :],
    ),
    nogil=True,
    parallel=True,
    locals={
        "real_size": numba.int64,
        "imag_size": numba.int64,
        "real_ind": numba.int32,
        "imag_ind": numba.int32,
        "z0_real": numba.float64,
        "z0_imag": numba.float64,
        "c_real": numba.float64,
        "c_imag": numba.float64,
    },
)
def julia_set(
    center: complex,
    side_length: complex,
    c: complex,
    max_iter: int,
    divergence: NDArray[np.int32],
):
    """
    Vygeneruj Juliovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param c: číslo, které se přičítá každou iteraci
    :param max_iter: maximální počet iterací
    """
    c_real = c.real
    c_imag = c.imag
    real_size = divergence.shape[0]
    imag_size = divergence.shape[1]
    # projdi každý bod
    for real_ind in numba.prange(real_size):
        for imag_ind in range(imag_size):
            # vypočti podmínky
            z0_real = (center.real - side_length.real / 2) + (
                real_ind / real_size
            ) * side_length.real
            z0_imag = (center.imag - side_length.imag / 2) + (
                imag_ind / imag_size
            ) * side_length.imag
            # iteruj
            divergence[real_ind, imag_ind] = count_iters(
                z0_real, z0_imag, c_real, c_imag, max_iter
            )
