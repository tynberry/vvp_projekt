import pygame as pg
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from pygame import Surface

# from mandelbrot.generace import julia_set, mandelbrot
from mandelbrot.gen import mandelbrot, julia_set
from mandelbrot.vizualizace import convert_set_to_color

import matplotlib


def relative_close(a: complex, b: complex, epsilon: float):
    """
    Vrátí True pokud jsou komplexní čísla `a` a `b` relativně dále než `epsilon`.
    """
    delta: complex = b - a
    return abs(delta) >= epsilon * abs(a)


class Cache:
    """
    Cache množiny.
    """

    center: complex
    side_length: complex
    iterations: int
    cells: Tuple[int, int]
    c_value: None | complex
    color_lup: None | NDArray[np.uint8]
    set: None | NDArray[np.int32]
    hues: None | NDArray[np.uint8]
    surface: None | pg.Surface
    scale_surface: None | pg.Surface
    color_map: str

    def __init__(self):
        """
        Vytvoří prázdnou cachi.

        :param center: střed pohledu
        :param side_length: délka strany pohledu
        :param iterations: počet iterací při generování
        :param cells: rozdělení pohledu na buňky
        :param c_value: Hodnota C v rovnici, Při none se vygeneruje Mandelbrotova množina, jinak Juliova
        """
        self.center = 0 + 0j
        self.side_length = 0 + 0j
        self.iterations = 0
        self.cells = (0, 0)
        self.c_value = None
        self.surface = None
        self.color_lup = None
        self.hues = None
        self.set = None
        self.scale_surface = None
        self.color_map = "viridis"
        pass

    def update(
        self,
        center: complex,
        side_length: complex,
        iterations: int,
        cells: Tuple[int, int],
        c_value: None | complex,
        color_map: str = "viridis",
    ):
        """
        Aktualizuje cachi.

        Provede aktualizaci pouze pokud se parametry příliš mění od uložených.

        :param center: střed pohledu
        :param side_length: délka strany pohledu
        :param iterations: počet iterací při generování
        :param cells: rozdělení pohledu na buňky
        :param c_value: Hodnota C v rovnici, Při none se vygeneruje Mandelbrotova množina, jinak Juliova
        :param color_map: barevná mapa
        """
        # vytvoř barevnou LUT
        if (
            self.color_lup is None
            or self.iterations != iterations
            or self.color_map != color_map
        ):
            lup = np.linspace(0, 1, num=iterations + 1, dtype=np.float32)
            self.color_lup = matplotlib.colormaps[color_map](lup, bytes=True)
        # vytvoř množiny
        if self.set is None:
            self.set = np.zeros(cells, dtype=np.int32)
        if self.hues is None:
            self.hues = np.zeros((*cells, 3), dtype=np.uint8)
        # kontrola velikosti množiny
        if self.set.shape != cells:
            self.set = np.zeros(cells, dtype=np.int32)
        if self.hues.shape != cells:
            self.hues = np.zeros((*cells, 3), dtype=np.uint8)
        # aktualizuj se
        if c_value is not None:
            julia_set(center, side_length, c_value, iterations, self.set)
        else:
            mandelbrot(center, side_length, iterations, self.set)
        # převeď na barvu
        convert_set_to_color(self.set, self.hues, iterations, self.color_lup)

        if self.surface is not None and self.cells == cells:
            # nahraj do povrchu
            pg.pixelcopy.array_to_surface(self.surface, self.hues[:, :, :])
        else:
            # vytvoř nový povrch
            self.surface = pg.surfarray.make_surface(self.hues[:, :, :])

        # ulož parametry posledního generování
        self.center = center
        self.side_length = side_length
        self.iterations = iterations
        self.cells = cells
        self.c_value = c_value
        self.color_map = color_map

    def render(self, surface: Surface, center: complex, side_length: complex):
        """
        Vykreslí obsah cache relativně vůči aktivnímu středu a zoomu.

        :param surface: Povrch na který se má cache vykreslit.
        :param center: aktivní střed
        :param side_length: délka strany aktivního pohledu
        """
        # máme co na vykreslení?
        if not isinstance(self.surface, Surface):
            return

        # vypočti velikost pixelů v komplexní rovině
        pixel_width = side_length.real / surface.get_width()
        pixel_height = side_length.imag / surface.get_height()
        # vypočti pozici vykreslení cache
        pixel_center_real = surface.get_width() / 2
        pixel_center_imag = surface.get_height() / 2
        off_real = (self.center.real - center.real) / pixel_width
        off_imag = (self.center.imag - center.imag) / pixel_height
        width = (self.side_length.real / side_length.real) * self.surface.get_width()
        height = (self.side_length.imag / side_length.imag) * self.surface.get_height()

        # vykresli povrch
        if self.scale_surface is None or self.scale_surface.get_size() != (
            width,
            height,
        ):
            self.scale_surface = pg.transform.scale(self.surface, (width, height))
        else:
            pg.transform.scale(
                self.surface, (width, height), dest_surface=self.scale_surface
            )

        surface.blit(
            self.scale_surface,
            pg.Rect(
                pixel_center_real - width / 2 + off_real,
                pixel_center_imag - height / 2 + off_imag,
                width,
                height,
            ),
        )
