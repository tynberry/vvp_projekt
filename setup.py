from setuptools import setup, find_packages
from Cython.Build import cythonize

# Před samotným setupem lze opět vykonávat libovolný Python kód
# Například kompilace Python balíčků pro rychlejší výpočty apod...

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
    ext_modules=cythonize("./mandelbrot/gen.pyx"),
    entry_points={
        # pokud by váš balíček poskytoval nástroje použitelné z terminálu, lze tímto přidat do cest
    },
)

