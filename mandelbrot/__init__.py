"""
Knihovna generace a vizualizace Mandelbrot a Julia fraktálů.

Generace používá numpy a Numbu.
Vizualizace se provádí v PyGame.
"""

from .generace import *

__all__ = ["mandelbrot", "julia_set"]