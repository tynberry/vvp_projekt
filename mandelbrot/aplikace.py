import pygame as pg
from typing import Tuple

from pygame.time import Clock

from .cache import Cache

MOVE_SPEED = 0.3
ZOOM_FACTOR = 1.05


def side_length_from_zoom(surface: pg.Surface, zoom: int) -> complex:
    """
    Vypočte délku strany pohledu podle zoomu a velikosti povrchu.

    :param surface: povrch obrazovky
    :param zoom: zoom
    """
    aspect: float = surface.get_height() / surface.get_width()
    zoom_fl = ZOOM_FACTOR**zoom
    return 3 * zoom_fl + (3 * aspect * zoom_fl) * 1j


def init_app():
    # incializace PyGame
    pg.init()
    screen: pg.Surface = pg.display.set_mode((1280, 720))
    clock: Clock = pg.time.Clock()
    running: bool = True

    # parametry pohledu
    center: complex = -0.5 + 0.0j
    zoom: int = 0
    iterations: int = 100
    cells: Tuple[int, int] = (1280, 720)
    c_value: None = None

    rough_cache: Cache = Cache()
    rough_cache.update(
        center,
        side_length_from_zoom(screen, zoom),
        100,
        cells,
        c_value,
    )
    cache: Cache = Cache()
    cache.update(
        center, side_length_from_zoom(screen, zoom), iterations, cells, c_value
    )

    # hlavní smyčka
    while running:
        # zpracuj události
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_a:
                    zoom -= 1
                elif event.key == pg.K_b:
                    zoom += 1
                elif event.key == pg.K_r:
                    cache.update(
                        center,
                        side_length_from_zoom(screen, zoom),
                        iterations,
                        cells,
                        c_value,
                        force=True,
                    )
                elif event.key == pg.K_o:
                    center: complex = -0.5 + 0.0j
                    zoom: int = 0

        # pohyb pohledem
        dt = clock.get_time() / 1000
        zoom_fl = ZOOM_FACTOR**zoom
        if pg.key.get_pressed()[pg.K_LEFT]:
            center -= MOVE_SPEED * zoom_fl * dt
        if pg.key.get_pressed()[pg.K_RIGHT]:
            center += MOVE_SPEED * zoom_fl * dt
        if pg.key.get_pressed()[pg.K_UP]:
            center -= MOVE_SPEED * zoom_fl * dt * 1j
        if pg.key.get_pressed()[pg.K_DOWN]:
            center += MOVE_SPEED * zoom_fl * dt * 1j

        # vyplň plochu
        screen.fill("midnightblue")

        # vykreslení
        rough_cache.render(screen, center, side_length_from_zoom(screen, zoom))
        cache.render(screen, center, side_length_from_zoom(screen, zoom))

        # ukonči snímek
        pg.display.flip()
        clock.tick(60)  # 60 FPS

    # ukonči pygame
    pg.quit()
