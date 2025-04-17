"""
Knihovna generace a vizualizace Mandelbrot a Julia fraktálů.

Generace používá numpy a Numbu.
Vizualizace se provádí v PyGame.
"""

from .generace import mandelbrot, julia_set
from .vizualizace import visual

__all__ = ["mandelbrot", "julia_set", "visual"]

