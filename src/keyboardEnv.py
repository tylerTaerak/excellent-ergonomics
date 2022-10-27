# import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
# from gym.error import DependencyNotInstalled


SCORES = [                                         [0.85, 0.95],
[0.5, 0.25, 0.25, 0.1, 0.12, 0.12, 0.1, 0.25, 0.3, 0.5, 0.7, 0.8, 0.9],
  [0.1, 0.075, 0.01, 0.0, 0.05, 0.05, 0.0, 0.01, 0.075, 0.1, 0.2],
    [0.6, 0.4, 0.3, 0.2, 0.25, 0.2, 0.3, 0.4, 0.6, 0.8]
]

KEYS = "abcdefghijklmnopqrstuvwxyz-=[]\\;',./"


# The file from which text will be drawn
FILE = "./frankenstein.txt"


# borrowed from https://stackoverflow.com/a/49752733
def generate_text(handle, size=10000):
    block = []
    for line in handle:
        for c in line:
            block.append(c)
        if len(block) >= size:
            yield block
            block = []

    if block:
        yield block


def type_through_keymap(keymap, text, reward):
    for c in text:
        if c in KEYS:
            index = (0, 0)
            for i in range(len(keymap)):
                for j in range(i):
                    if keymap[i][j] == c:
                        index = (i, j)
                        break
            reward[c] += SCORES[index[0]][index[1]]
    return reward


class KeyboardEnv(gym.Env):
    """
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, fname: Optional[str] = FILE):
        self.action_space = spaces.Tuple((
            spaces.Box(low=ord('\''), high=ord('z'), shape=(2,), dtype=np.uint),
            spaces.Box(low=ord('\''), high=ord('z'), shape=(13,), dtype=np.uint),
            spaces.Box(low=ord('\''), high=ord('z'), shape=(11,), dtype=np.uint),
            spaces.Box(low=ord('\''), high=ord('z'), shape=(10,), dtype=np.uint)
             ))

        # all 26 letters, plus 10 punctuation keys on the right side
        self.observation_space = spaces.Dict(
            {
                    'a': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'b': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'c': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'd': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'e': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'f': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'g': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'h': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'i': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'j': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'k': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'l': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'm': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'n': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'o': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'p': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'q': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'r': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    's': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    't': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'u': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'v': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'w': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'x': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'y': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    'z': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '-': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '=': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '[': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    ']': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '\\': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    ';': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '\'': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    ',': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '.': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float),
                    '/': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float)
                }
            )

        self.keymap = [                                 ['-', '='],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
          ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
        ]

        self.textgen = generate_text(open(fname))
        self.render_mode = render_mode

    def step(self, action):
        self.keymap = action
        self.keyscores = self._reset_scores()
        text = ''

        terminated = False
        try:
            text = next(self.textgen)
        except StopIteration:
            terminated = True

        obs = type_through_keymap(self.keymap, text, self.keyscores)

        reward = sum(obs.values())

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.keyscores = self._reset_scores()

        text = next(self.textgen)

        obs = type_through_keymap(self.keymap, text, self.keyscores)

        return obs, {}

    def _reset_scores(self):
        return {
            'a': 0.0,
            'b': 0.0,
            'c': 0.0,
            'd': 0.0,
            'e': 0.0,
            'f': 0.0,
            'g': 0.0,
            'h': 0.0,
            'i': 0.0,
            'j': 0.0,
            'k': 0.0,
            'l': 0.0,
            'm': 0.0,
            'n': 0.0,
            'o': 0.0,
            'p': 0.0,
            'q': 0.0,
            'r': 0.0,
            's': 0.0,
            't': 0.0,
            'u': 0.0,
            'v': 0.0,
            'w': 0.0,
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            '-': 0.0,
            '=': 0.0,
            '[': 0.0,
            ']': 0.0,
            '\\': 0.0,
            ';': 0.0,
            '\'': 0.0,
            ',': 0.0,
            '.': 0.0,
            '/': 0.0,
        }

    def render(self):
        pass
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Minecraft.ttf"), screen_height // 15
        )
        dealer_text = small_font.render(
            "Dealer: " + str(dealer_card_value), True, white
        )
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"{self.dealer_top_card_suit}{self.dealer_top_card_value_str}.png",
                )
            )
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(
            player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing)
        )

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        """

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)
