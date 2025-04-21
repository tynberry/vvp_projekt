import pygame as pg
import numpy as np
from typing import Tuple

from pygame import Surface


def relative_close(a: complex, b: complex, epsilon: float):
    """
    Vrátí True pokud jsou komplexní čísla `a` a `b` relativně dále než `epsilon`.
    """
    delta: complex = b - a
    return abs(delta) / abs(a) >= epsilon


class Cache:
    """
    Cache množiny.
    """

    center: complex
    real_side_length: complex
    iterations: int
    cells: Tuple[int, int]
    c_value: None | complex
    surface: None | pg.Surface

    def __init__(
        self,
        center: complex,
        real_side_length: complex,
        iterations: int,
        cells: Tuple[int, int],
        c_value: None | complex,
    ):
        """
        Vytvoří prázdnou cachi.

        :param center: střed pohledu
        :param real_side_length: délka strany pohledu
        :param iterations: počet iterací při generování
        :param cells: rozdělení pohledu na buňky
        :param c_value: Hodnota C v rovnici, Při none se vygeneruje Mandelbrotova množina, jinak Juliova
        """
        self.center = center
        self.real_side_length = real_side_length
        self.iterations = iterations
        self.cells = cells
        self.c_value = c_value
        self.surface = None
        pass

    def should_update(
        self,
        center: complex,
        real_side_length: complex,
        iterations: int,
        cells: Tuple[int, int],
        c_value: None | complex,
    ):
        """
        Určí zda jsou parametry dostačující na aktualizaci.

        :param center: střed pohledu
        :param real_side_length: délka strany pohledu
        :param iterations: počet iterací při generování
        :param cells: rozdělení pohledu na buňky
        :param c_value: Hodnota C v rovnici, Při none se vygeneruje Mandelbrotova množina, jinak Juliova
        """
        # nezměnil se moc střed?
        epsilon = 1e-3
        if not relative_close(self.center, center, epsilon):
            return True
        # nezměnil se moc zoom?
        if not relative_close(self.real_side_length, real_side_length, epsilon):
            return True
        # nezměnily se iterace?
        if self.iterations != iterations:
            return True
        # nezměnily se buňky?
        if self.cells != cells:
            return True
        # nezměnily se typ množiny?
        if self.c_value is None and c_value is not None:
            return True
        if self.c_value is not None and c_value is None:
            return True
        # nezměnilo se moc C číslo?
        if self.c_value is not None and c_value is not None:
            if relative_close(self.c_value, c_value, epsilon):
                return True
        # žádná změna nalezena
        return False

    def update(
        self,
        center: complex,
        real_side_length: complex,
        iterations: int,
        cells: Tuple[int, int],
        c_value: None | complex,
        force: bool = False,
    ):
        """
        Aktualizuje cachi.

        Provede aktualizaci pouze pokud se parametry příliš mění od uložených.

        :param center: střed pohledu
        :param real_side_length: délka strany pohledu
        :param iterations: počet iterací při generování
        :param cells: rozdělení pohledu na buňky
        :param c_value: Hodnota C v rovnici, Při none se vygeneruje Mandelbrotova množina, jinak Juliova
        :param force: donuť aktualizace
        """
        # je aktualizace potřeba?
        if not force and not self.should_update(
            center, real_side_length, iterations, cells, c_value
        ):
            return
        # aktualizuj se

        pass

    def render(self):
        # máme co na vykreslení?
        if not isinstance(self.surface, Surface):
            return
        pass
