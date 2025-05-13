import numpy
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


# seznam Cython modulů
extensions = [
    Extension(
        "mandelbrot.gen",
        ["mandelbrot/gen.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "mandelbrot.viz_c",
        ["mandelbrot/viz_c.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

# setup balíčku
setup(
    name="mandelbrot",  # jméno balíku
    version="1.0.0",  # verze - pokud například distribujete balík uživatelům
    packages=find_packages(),
    install_requires=[
        # Seznam externích balíčků potřebných k provozu
        "numpy",
        "numba",
        "pygame",
        "matplotlib",
    ],
    ext_modules=cythonize(extensions),
)
