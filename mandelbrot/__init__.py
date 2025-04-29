"""
Knihovna generace a vizualizace Mandelbrot a Julia fraktálů.

Generace používá numpy a Numbu.
Vizualizace se provádí v PyGame.
"""

from .gen import mandelbrot, julia_set
from .vizualizace import visual
from .aplikace import init_app

__all__ = ["mandelbrot", "julia_set", "visual", "init_app"]
