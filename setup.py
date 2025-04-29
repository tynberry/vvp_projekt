import numpy
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

# Před samotným setupem lze opět vykonávat libovolný Python kód
# Například kompilace Python balíčků pro rychlejší výpočty apod...

extensions = [
    Extension(
        "*",
        ["mandelbrot/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

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
    entry_points={
        # pokud by váš balíček poskytoval nástroje použitelné z terminálu, lze tímto přidat do cest
    },
)
