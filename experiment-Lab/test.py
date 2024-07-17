from psychopy import visual, core, event, logging, monitors, gui
import numpy as np


def win_flip(win, cursor=False, bg_color=(0, 0, 0)):
    win.winHandle.set_mouse_visible(cursor)
    win.mouseVisible = cursor
    win.winHandle.set_mouse_position(-1, -1)
    win.fillColor = bg_color
    win.flip()


def show_msg(
    win: visual.window.Window, text: str, kwargs: dict = None, keys: list = None
):
    """Show message on psychopy window"""
    win_flip(win)
    kwargs = {} if kwargs is None else kwargs
    msg = visual.TextStim(win, text, **kwargs)
    # TODO : color=txt_color, wrapWidth=scn_width / 2)
    msg.draw()
    win_flip(win)

    if keys:
        pressed_key = event.waitKeys(keyList=keys)
        win_flip(win)
        return pressed_key


win_size = 800, 600


win = visual.Window(
    size=win_size,
    winType="pyglet",
    fullscr=False,
    screen=0,
    color=[0, 0, 0],
    units="pix",
)
win_flip(win)

max_items = 8
img_size = [80, 80]

blank_space_pct = 0.05
resolution = win_size


def prepare_images(resolution, max_items, blank_space_pct=0.05):
    # from pathlib import Path
    # from PIL import Image

    # wd = Path(__file__).parent
    # img_dir = wd.parent / "config/images/original"
    # img_files = [Image.open(img) for img in img_dir.glob("*.png")]
    # input_folder = img_dir
    # #####

    def count_black_pixels(img):
        """
        img: PIL image object, mode: RGBA, alpha channel set to 0 for transparent pixels
        """
        img_array = np.array(img)
        return np.sum(img_array[:, :, 3] > 0)

    width, height = resolution
    max_img_width = int(np.floor(width / max_items * (1 - blank_space_pct)))
    max_img_height = int(np.floor(height / 3))
    target_size = [min(max_img_width, max_img_height)] * 2

    wd = Path(__file__).parent
    img_dir = wd.parent / "config/images/original"
    img_files = [Image.open(img) for img in img_dir.glob("*.png")]
    bl_pix_counts = {img.stem: count_black_pixels(Image.open(img))for img in img_dir.glob("*.png")}
    max_black_pxl = max(bl_pix_counts.values())
    bl_pix_counts = {k: v / max_black_pxl for k, v in bl_pix_counts.items()}
    
    bl_pix_counts = {k: v for k, v in bl_pix_counts.items() if v >= 0.85}

    bl_pix_counts = {k: v for k, v in sorted(bl_pix_counts.items(), key=lambda item: item[1])}


def get_x_pos(img_size, win_size, n_items):
    total_width = img_size[0] * n_items
    empty_space_width = win_size[0] - total_width

    if empty_space_width < 0:
        raise ValueError("Image size too big for window")

    sep_space_width = empty_space_width / (n_items + 1)
    x_shift = img_size[0] + sep_space_width
    start_pos = sep_space_width + img_size[0] / 2
    x_pos = np.arange(start_pos, win_size[0], x_shift) - win_size[0] / 2

    return x_pos


n_items = max_items

x_pos = get_x_pos(img_size, win_size, n_items)

rectangles = [
    visual.rect.Rect(
        win,
        lineWidth=10,
        lineColor="green",
        fillColor=None,
        pos=(pos, 0),
        size=max_img_size,
        units="pix",
    )
    for pos in x_pos
]

for r in rectangles:
    r.draw()
win.flip()

# win.mouseVisible = False
# win.winHandle.set_mouse_visible(False)
# win_flip(win, cursor=True)
core.wait(2)

show_msg(win, "WELCOME, press enter to continue", keys=["return"])
win_flip(win, cursor=True)

for xpos in range(0, 900, 100):
    core.wait(1)
    win.winHandle.set_mouse_position(-1, -1)

key = event.waitKeys()

show_msg(win, "START")
core.wait(2)
win_flip(win)

show_msg(win, "SCREEN 1")
core.wait(2)
win.flip(win)

show_msg(win, "PRESS ANY KEY TO QUIT")
key = event.waitKeys()

win.close()
core.quit()
sys.exit()
