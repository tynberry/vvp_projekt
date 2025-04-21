import pygame as pg
from typing import Tuple

from pygame.time import Clock

from .cache import Cache

MOVE_SPEED = 0.3
ZOOM_FACTOR = 1.5
COLOR_MAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "jet", "turbo"]


def side_length_from_zoom(surface: pg.Surface, zoom: int) -> complex:
    """
    Vypočte délku strany pohledu podle zoomu a velikosti povrchu.

    :param surface: povrch obrazovky
    :param zoom: zoom
    """
    aspect: float = surface.get_height() / surface.get_width()
    zoom_fl = ZOOM_FACTOR**zoom
    return 3 * zoom_fl + (3 * aspect * zoom_fl) * 1j


def color_map(ind: int) -> str:
    """
    Vybere barevnou mapu z globálního seznamu map.
    """
    return COLOR_MAPS[ind]


def init_app():
    # incializace PyGame
    pg.init()
    pg.font.init()
    default_font = pg.font.Font(pg.font.get_default_font(), 32)

    screen: pg.Surface = pg.display.set_mode((1280, 720))
    clock: Clock = pg.time.Clock()
    running: bool = True

    # parametry pohledu
    center: complex = -0.5 + 0.0j
    zoom: int = 0
    iterations: int = 100
    cells: Tuple[int, int] = (1280, 720)
    c_value: None = None
    color_ind: int = 0

    cache: Cache = Cache()
    cache.update(
        center,
        side_length_from_zoom(screen, zoom),
        iterations,
        cells,
        c_value,
        color_map(color_ind),
    )

    auto_refresh = True

    # hlavní smyčka
    while running:
        # zpracuj události
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_q:
                    zoom -= 1
                elif event.key == pg.K_e:
                    zoom += 1
                elif event.key == pg.K_r:
                    cache.update(
                        center,
                        side_length_from_zoom(screen, zoom),
                        iterations,
                        cells,
                        c_value,
                        color_map(color_ind),
                        force=True,
                    )
                elif event.key == pg.K_o:
                    center: complex = -0.5 + 0.0j
                    zoom: int = 0
                elif event.key == pg.K_d:
                    color_ind += 1
                    color_ind %= len(COLOR_MAPS)
                elif event.key == pg.K_a:
                    color_ind -= 1
                    if color_ind < 0:
                        color_ind += len(COLOR_MAPS)
                elif event.key == pg.K_t:
                    auto_refresh = not auto_refresh

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

        # aktualizuj množinu
        if auto_refresh:
            cache.update(
                center,
                side_length_from_zoom(screen, zoom),
                iterations,
                cells,
                c_value,
                color_map(color_ind),
            )
        # vyplň plochu
        screen.fill("midnightblue")

        # vykreslení
        cache.render(screen, center, side_length_from_zoom(screen, zoom))

        # vykreslení textů
        center_text = default_font.render(f"Center: {center:.3}", False, "black")
        zoom_text = default_font.render(f"Zoom: {zoom}", False, "black")
        side_text = default_font.render(
            f"Side length: {side_length_from_zoom(screen, zoom):.3}", False, "black"
        )
        iteration_text = default_font.render(
            f"Iterations: {iterations}", False, "black"
        )
        c_text = default_font.render(f"C Value: {c_value}", False, "black")
        colormap_text = default_font.render(
            f"Color map: {color_map(color_ind)}", False, "black"
        )
        if auto_refresh:
            auto_refresh_text = default_font.render("Autorefresh ON", False, "black")
        else:
            auto_refresh_text = default_font.render("Autorefresh OFF", False, "black")

        screen.blit(center_text, (0, 0))
        screen.blit(zoom_text, (0, 32))
        screen.blit(side_text, (0, 64))
        screen.blit(iteration_text, (0, 96))
        screen.blit(c_text, (0, 128))
        screen.blit(colormap_text, (0, 160))
        screen.blit(auto_refresh_text, (0, 192))

        # ukonči snímek
        pg.display.flip()
        clock.tick(60)  # 60 FPS

    # ukonči pygame
    pg.quit()
