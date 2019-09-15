import vizdoom as vzd


if __name__ == '__main__':
    game = vzd.DoomGame()
    game.add_game_args('+freelook 1')
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

