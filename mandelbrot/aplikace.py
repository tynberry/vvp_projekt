import pygame as pg
from typing import Tuple

from pygame.time import Clock

from .cache import Cache


def init_app():
    # incializace PyGame
    pg.init()
    screen: pg.Surface = pg.display.set_mode((1280, 720))
    clock: Clock = pg.time.Clock()
    running: bool = True

    # parametry pohledu
    center: complex = 0.0 + 0.0j
    side_length: complex = 3.0 + 3.0j
    iterations: int = 100
    cells: Tuple[int, int] = (1280, 720)
    c_value: None = None

    cache: Cache = Cache()
    cache.update(center, side_length, iterations, cells, c_value)

    # hlavní smyčka
    while running:
        # zpracuj události
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        # vyplň plochu
        screen.fill("midnightblue")

        # vykreslení
        cache.render(screen, center, side_length)

        # ukonči snímek
        pg.display.flip()
        clock.tick(60)  # 60 FPS

    # ukonči pygame
    pg.quit()
