import numba 
import numpy as np

@numba.vectorize([numba.int64(numba.complex128, numba.complex128, numba.int32)])
def count_iters(state: numba.complex128, c: numba.complex128, max_iter: numba.int32) -> numba.int64:
    """
    Spočítá počet operací než hodnota vyběhne ven. 

    state je počáteční hodnota.
    c je číslo c přičítané každou iteraci
    max_iter je maximální počet iterací 
    """
    for i in range(max_iter):
        state **= 2
        state += c 
        if np.abs(state) > 2:
            return i+1
    return max_iter


#@numba.jit(signature="int64[:,:](float64,float64,float64,float64,int64,int64)",
#           nopython=True, locals={"divergence", numba.np_ar})
def mandelbrot(x_min:float, x_max:float, y_min:float, y_max:float, cells: int, max_iter: int) -> np.array:
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    x_min a x_max určují rozmezí na reálné ose.
    y_min a y_max určují rozmezí na imaginární ose.
    cells na kolik políček se rozdělí každá osa.
    max_iter určuje počet maximální počet iterací.
    """
    #divergenční matice 
    divergence = np.zeros((cells,cells), dtype=np.int32)
    #matice stavu 
    #state = np.zeros((cells,cells), dtype=np.complex128)
    #matice C komplexních čísel   
    x = np.linspace(x_min, x_max, cells, dtype=np.complex128).reshape((1,cells))
    y = np.linspace(y_min, y_max, cells, dtype=np.complex128).reshape((cells,1)) * 1j
    c_matrix = x + y,
    #spočti iterace 
    divergence = count_iters(0, c_matrix, max_iter)
    #vrať počet operací
    return divergence

@numba.njit
def julia_set(x_min:float, x_max:float, y_min:float, y_max:float, c: complex, cells: int, max_iter: int) -> np.array:
    """
    Vygeneruj Juliovu množinu.
    Vrací matici počtu iterací.

    x_min a x_max určují rozmezí na reálné ose.
    y_min a y_max určují rozmezí na imaginární ose.
    cells na kolik políček se rozdělí každá osa.
    max_iter určuje počet maximální počet iterací.
    c je komplexní parametr C
    """
    #divergenční matice 
    divergence = np.zeros((cells,cells), dtype=np.int32)
    #matice C komplexních čísel   
    x = np.linspace(x_min, x_max, cells, dtype=np.complex128).reshape((1,cells))
    y = np.linspace(y_min, y_max, cells, dtype=np.complex128).reshape((cells,1)) * 1j
    #matice stavu 
    state = x+y
    #udělej iterace 
    for i in range(max_iter):
        #udělej jednu iteaci 
        state **= 2
        state += c
        #přičti iteraci 
        divergence[np.abs(state) <= 2] += 1
    #vrať počet operací
    return divergence