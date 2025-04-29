import numpy as np 
cimport numpy as cnp 

import cython
from cython.parallel import prange
from cython.parallel import parallel

cdef int count_iters(
    double z_real,
    double z_imag,
    double c_real,
    double c_imag,
    int max_iter,
) noexcept nogil:
    """
    Spočítá počet operací než hodnota vyběhne ven.

    :param max_iter: maximální počet iterací
    """
    cdef double temp_real
    cdef int i

    for i in range(max_iter):
        temp_real = z_real * z_real - z_imag * z_imag + c_real
        z_imag = (z_real + z_real) * z_imag + c_imag
        z_real = temp_real
        if z_real * z_real + z_imag * z_imag > 4.0:
            return i + 1
    return max_iter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mandelbrot(
    double complex center,
    double complex side_length,
    int max_iter,
    int[:, :] divergence,
) noexcept:
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param max_iter: maximální počet iterací
    """
    cdef double c_real 
    cdef double c_imag 
    cdef int real_ind 
    cdef int imag_ind
    cdef int [:, :] divergence_view = divergence
    cdef int real_size = divergence.shape[0]
    cdef int imag_size = divergence.shape[1]
    cdef double center_real = center.real
    cdef double center_imag = center.imag
    cdef double side_length_real = side_length.real
    cdef double side_length_imag = side_length.imag
    # projdi každý bod
    for real_ind in prange(real_size, nogil=True, num_threads=6):
        for imag_ind in range(imag_size):
            # vypočti podmínky
            c_real = (center_real - side_length_real / 2) + (
                real_ind / real_size
            ) * side_length_real
            c_imag = (center_imag - side_length_imag / 2) + (
                imag_ind / imag_size
            ) * side_length_imag
            # iteruj
            divergence_view[real_ind,imag_ind] = count_iters(
                0.0, 0.0, c_real, c_imag, max_iter
            )

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void julia_set(
    double complex center,
    double complex side_length,
    double complex c,
    int max_iter,
    int[:, :] divergence,
) noexcept:
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param center: střed množiny
    :param side_length: délka stran množiny v každé ose
    :param max_iter: maximální počet iterací
    """
    cdef double c_real = c.real
    cdef double c_imag = c.imag
    cdef double z0_real 
    cdef double z0_imag
    cdef int real_ind 
    cdef int imag_ind
    cdef int [:, :] divergence_view = divergence
    cdef int real_size = divergence.shape[0]
    cdef int imag_size = divergence.shape[1]
    cdef double center_real = center.real
    cdef double center_imag = center.imag
    cdef double side_length_real = side_length.real
    cdef double side_length_imag = side_length.imag
    # projdi každý bod
    for real_ind in prange(real_size, nogil=True, num_threads=6):
        for imag_ind in range(imag_size):
            # vypočti podmínky
            z0_real = (center_real - side_length_real / 2) + (
                real_ind / real_size
            ) * side_length_real
            z0_imag = (center_imag - side_length_imag / 2) + (
                imag_ind / imag_size
            ) * side_length_imag
            # iteruj
            divergence_view[real_ind,imag_ind] = count_iters(
                z0_real, z0_imag, c_real, c_imag, max_iter
            )
