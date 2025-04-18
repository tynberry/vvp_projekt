import pygame as pg
import numpy as np

from .generace import mandelbrot, julia_set
from .vizualizace import convert_set_to_color


def init_app():
    # incializace PyGame
    pg.init()
    screen = pg.display.set_mode((1280, 720))
    clock = pg.time.Clock()
    running = True

    # hlavní smyčka
    while running:
        # zpracuj události
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        # vyplň plochu
        screen.fill("midnightblue")

        # vykreslení

        # ukonči snímek
        pg.display.flip()
        clock.tick(60)  # 60 FPS

    # ukonči pygame
    pg.quit()
