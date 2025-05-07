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

    :param ind: index barevné mapy
    """
    return COLOR_MAPS[ind]


def init_app():
    """
    Spustí aplikaci pro vizualizaci množin.
    """
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
    cached_c_value: complex = -0.8 + 0.156j
    c_value: complex | None = None
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

    # flagy aplikace
    auto_refresh: bool = True
    should_refresh: bool = False
    dragging: bool = False

    # hlavní smyčka
    while running:
        # zpracuj události
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEWHEEL:
                if event.y != 0:
                    zoom -= max(min(event.y, 1), -1)
                    should_refresh = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    cache.update(
                        center,
                        side_length_from_zoom(screen, zoom),
                        iterations,
                        cells,
                        c_value,
                        color_map(color_ind),
                    )
                elif event.key == pg.K_o:
                    should_refresh = True
                    center: complex = -0.5 + 0.0j
                    zoom: int = 0
                elif event.key == pg.K_d:
                    should_refresh = True
                    color_ind += 1
                    color_ind %= len(COLOR_MAPS)
                elif event.key == pg.K_a:
                    should_refresh = True
                    color_ind -= 1
                    if color_ind < 0:
                        color_ind += len(COLOR_MAPS)
                elif event.key == pg.K_t:
                    auto_refresh = not auto_refresh
                elif event.key == pg.K_j:
                    if c_value is None:
                        c_value = cached_c_value
                    else:
                        cached_c_value = c_value
                        c_value = None
                    should_refresh = True
                elif event.key == pg.K_h:
                    iterations += 25
                    should_refresh = True
                elif event.key == pg.K_g:
                    iterations -= 25
                    if iterations < 25:
                        iterations = 25
                    should_refresh = True

        # pohyb pohledem pomocí myši
        if pg.mouse.get_pressed()[0] and not dragging:
            dragging = True
            # očisti deltu myši
            _ = pg.mouse.get_rel()
        if not pg.mouse.get_pressed()[0] and dragging:
            dragging = False
            should_refresh = True
        if dragging:
            # pohyb myši v pixelech
            delta: Tuple[int, int] = pg.mouse.get_rel()
            # pohyb myši vůči celé obrazovce
            dx: float = delta[0] / screen.get_width()
            dy: float = delta[1] / screen.get_height()
            # škálování pohybu podle zoomu
            extents: complex = side_length_from_zoom(screen, zoom)
            dx = dx * extents.real
            dy = dy * extents.imag
            # změna položky
            if pg.key.get_mods() & pg.KMOD_LSHIFT:
                if c_value is not None:
                    c_value -= dx + dy * 1j
            else:
                center -= dx + dy * 1j

        # aktualizuj množinu
        if auto_refresh and should_refresh:
            cache.update(
                center,
                side_length_from_zoom(screen, zoom),
                iterations,
                cells,
                c_value,
                color_map(color_ind),
            )
            should_refresh = False
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
        fps_text = default_font.render(f"FPS: {clock.get_fps():.4}", False, "black")
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
        screen.blit(fps_text, (0, 224))

        # ukonči snímek
        pg.display.flip()
        clock.tick(60)  # 60 FPS

    # ukonči pygame
    pg.quit()
