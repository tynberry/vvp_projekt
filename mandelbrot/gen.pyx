import numpy as np 
cimport numpy as cnp 

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef void count_iters(
        cnp.ndarray[cnp.complex128_t, ndim=2] state,
        cnp.ndarray[cnp.complex128_t, ndim=2] c,
        int max_iter,
        cnp.ndarray[cnp.int32_t, ndim=2] iters
   ):
    for row in range(state.shape[0]):
        for col in range(state.shape[1]):
            iters[row, col] = 0
            for i in range(max_iter):
                state[row, col] *= state[row, col]
                state[row, col] += c[row, col]
                if state[row, col].real * state[row, col].real + state[row, col].imag * state[row, col].imag > 4:
                    iters[row, col] = i + 1
                    break
            else:
                iters[row, col] = max_iter
