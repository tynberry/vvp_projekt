import numpy as np 
cimport numpy as cnp 

cimport cython
from cython.parallel import prange
from cython.parallel import parallel

@cython.boundscheck(False)
@cython.wraparound(False)
cdef count_hue(
   int[:,:] set,
   long[:] hist,
   double total,
   cnp.uint8_t [:,:,:] hues,
   cnp.uint8_t [:,:] lup,
):
    """
    Vypočte hodnotu barvy všech bodů množiny.

    :param set: pole počtu iterací před divergencí
    :param hist: histogram iterací
    :param total: celkový počet pixelů
    :param hues: výsledné pole barev
    """
    cdef int rows = set.shape[0]
    cdef int cols = set.shape[1]
    cdef cnp.uint8_t [:,:,:] hues_view = hues 
    cdef cnp.uint8_t [:,:] lup_view = lup
    cdef long[:] hist_view = hist
    cdef int[:,:] set_view = set
    cdef int lup_size = lup.shape[0] - 1
    cdef int index
    cdef int row 
    cdef int col
    for row in prange(rows, nogil=True):
        for col in range(cols):
            # přidej arvu do pole
            index = <int>((hist_view[set_view[row, col]] / total) * (lup_size))
            hues_view[row, col, 0] = lup_view[index, 0]
            hues_view[row, col, 1] = lup_view[index, 1]
            hues_view[row, col, 2] = lup_view[index, 2]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef convert_set_to_color(
    int[:,:] set,
    cnp.uint8_t[:,:,:] hues,
    int max_iter,
    cnp.uint8_t[:,:] lup,
):
    """
    Vrátí pole barev obarvené množiny podle dané barevné mapy.

    :param set: pole iterací před divergencí
    :param hues: pole barev
    :param color_map: barevná mapa
    """
    cdef int pixel
    #spočti histogram iterací 
    cdef cnp.ndarray[cnp.int64_t, ndim=1] histogram = np.zeros(max_iter + 1, dtype=np.int64)
    cdef int row 
    cdef int col 
    cdef int rows = set.shape[0]
    cdef int cols = set.shape[1]
    for row in range(rows):
        for col in range(cols):
            pixel = set[row, col]
            histogram[pixel] += 1
    # spočti komulativní sumu 
    cdef int cumsum = 0
    cdef int i
    for i in range(max_iter+1):
        cumsum += histogram[i]
        histogram[i] = cumsum
    cdef double total = <double>cumsum
    # spočti pozici na paletě
    count_hue(set, histogram, total, hues, lup)
