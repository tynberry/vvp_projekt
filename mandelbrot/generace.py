import numba 
import numpy as np

@numba.vectorize([numba.int64(numba.complex128, numba.complex128, numba.int32)])
def count_iters(state: complex, c: complex, max_iter: int) -> int:
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
            return i+1
    return max_iter


#@numba.jit(signature="int64[:,:](float64,float64,float64,float64,int64,int64)",
#           nopython=True, locals={"divergence", numba.np_ar})
def mandelbrot(x_min:float, x_max:float, y_min:float, y_max:float, cells: int, max_iter: int) -> np.array:
    """
    Vygeneruj Mandelbrotovu množinu.
    Vrací matici počtu iterací.

    :param x_min: nejmenší číslo na reálné ose 
    :param x_max: největší číslo na reálné ose
    :param y_min: nejmenší číslo na imaginární ose 
    :param y_max: největší číslo na imaginární ose
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
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

def julia_set(x_min:float, x_max:float, y_min:float, y_max:float, c: complex, cells: int, max_iter: int) -> np.array:
    """
    Vygeneruj Juliovu množinu.
    Vrací matici počtu iterací.

    :param x_min: nejmenší číslo na reálné ose 
    :param x_max: největší číslo na reálné ose
    :param y_min: nejmenší číslo na imaginární ose 
    :param y_max: největší číslo na imaginární ose
    :param c: číslo, které se přičítá každou iteraci
    :param cells: na kolik políček se rozdělí každá osa
    :param max_iter: maximální počet iterací
    """
    #divergenční matice 
    divergence = np.zeros((cells,cells), dtype=np.int32)
    #matice C komplexních čísel   
    x = np.linspace(x_min, x_max, cells, dtype=np.complex128).reshape((1,cells))
    y = np.linspace(y_min, y_max, cells, dtype=np.complex128).reshape((cells,1)) * 1j
    #matice stavu 
    state = x+y
    #udělej iterace 
    divergence = count_iters(state, c, max_iter)
    #vrať počet operací
    return divergence