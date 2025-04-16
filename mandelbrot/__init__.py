"""
Knihovna generace a vizualizace Mandelbrot a Julia fraktálů.

Generace používá numpy a Numbu.
Vizualizace se provádí v PyGame.
"""

from .generace import mandelbrot, julia_set
from .vizualizace import mandelbrot_visual, julia_set_visual

__all__ = ["mandelbrot", "julia_set", "mandelbrot_visual", "julia_set_visual"]