import sys

from fire import Fire
from vizdoom import DoomGame


class Environment:
    def __init__(self, path_to_cfg: str, path_to_wad: str):
        self.path_to_cfg = path_to_cfg
        self.path_to_wad = path_to_wad

        self.game = DoomGame()
        self.game.load_config(path_to_cfg)
        self.game.set_doom_scenario_path(path_to_wad)
        self.game.init()

    def test(self):
        self.game = DoomGame()
        self.game.load_config(self.path_to_cfg)
        self.game.set_doom_scenario_path(self.path_to_wad)
        self.game.init()


if __name__ == "__main__":
    Fire(Environment)

