from typing import Tuple
from collections import deque

import vizdoom as vzd

import numpy as np
from skimage import transform
from skimage.color import rgb2gray

from dddqn import DDDQN
from memory import Memory

from fire import Fire


def preprocess(
    frame: np.ndarray,
    resize: Tuple=(100, 120),
    to_gray: bool=False
) -> np.ndarray:
    out = np.copy(frame)

    if to_gray:
        out = rgb2gray(out)

    return transform.resize(out / 255.0, resize)


def make_state(stack: deque, frame: np.ndarray) -> np.ndarray:
    stack.append(frame)
    return np.stack(stack, axis=2)


def pretrain(
    stack: deque,
    memory: object,
    actions: np.ndarray,
    n_pretraining_eps: int
):
    for ep in range(n_pretraining_eps):
       pass


def main(
    config: str,
    stack_size: int=4,
    to_gray: bool=False
) -> None:
    game = vzd.DoomGame()
    game.load_config(config)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_render_hud(True)

    game.init()
    game.new_episode()
    state = game.get_state()
    frame = state.screen_buffer
    preprocessed = preprocess(frame, to_gray=to_gray)
    stack = deque([preprocessed for _ in range(stack_size)], maxlen=stack_size)

    game.close()


if __name__ == '__main__':
    Fire(main)
