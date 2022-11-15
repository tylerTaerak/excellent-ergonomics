from typing import Optional

import numpy as np
from os import listdir
import random

import gym
from gym import spaces


SCORES = [                                         [0.85, 0.95],
[0.5, 0.25, 0.25, 0.1, 0.12, 0.12, 0.1, 0.25, 0.3, 0.5, 0.7, 0.8, 0.9],
  [0.1, 0.075, 0.01, 0.0, 0.05, 0.05, 0.0, 0.01, 0.075, 0.1, 0.2],
    [0.6, 0.4, 0.3, 0.2, 0.25, 0.2, 0.3, 0.4, 0.6, 0.8]
]

KEYS = "abcdefghijklmnopqrstuvwxyz-=[]\\;',./"


# The files from which text will be drawn
files = [f for f in listdir("./texts")]
random.shuffle(files)


# borrowed from https://stackoverflow.com/a/49752733
def generate_text(handle, size=5000):
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
                for j in range(len(keymap[i])):
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

    def __init__(self, render_mode: Optional[str] = None, fname: Optional[str] = None):
        self.action_space = spaces.Tuple((
            spaces.Box(low=ord('\''), high=ord('z'), shape=(2,), dtype=np.uint),
            spaces.Box(low=ord('\''), high=ord('z'), shape=(13,), dtype=np.uint),
            spaces.Box(low=ord('\''), high=ord('z'), shape=(11,), dtype=np.uint),
            spaces.Box(low=ord('\''), high=ord('z'), shape=(10,), dtype=np.uint)
             ))

        # all 26 letters, plus 10 punctuation keys on the right side self.observation_space = spaces.Dict(
        self.observation_space = spaces.Dict(
                {
                    'a': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'b': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'c': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'd': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'e': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'f': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'g': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'h': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'i': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'j': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'k': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'l': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'm': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'n': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'o': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'p': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'q': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'r': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    's': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    't': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'u': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'v': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'w': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'x': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'y': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    'z': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '-': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '=': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '[': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    ']': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '\\': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    ';': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '\'': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    ',': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '.': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint),
                    '/': spaces.Box(low=0, high=len(KEYS), shape=(1,), dtype=np.uint)
                }
            )

        self.keymap = [                                 ['-', '='],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
          ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\''],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
        ]

        self.render_mode = render_mode
        self.fname = fname

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
        count = 0
        for i in range(len(self.keymap)):
            for j in range(len(self.keymap[i])):
                obs[self.keymap[i][j]] = count
                count += 1

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        fname = self.fname
        if self.fname is None:
            fname = "./texts/" + files[0]
            files.pop(0)

        self.textgen = generate_text(open(fname))
        self.keyscores = self._reset_scores()

        text = next(self.textgen)

        obs = type_through_keymap(self.keymap, text, self.keyscores)
        count = 0
        for i in range(len(self.keymap)):
            for j in range(len(self.keymap[i])):
                obs[self.keymap[i][j]] = count
                count += 1

        return obs, {}

    def _reset_scores(self):
        return {
            'a': 0,
            'b': 0,
            'c': 0,
            'd': 0,
            'e': 0,
            'f': 0,
            'g': 0,
            'h': 0,
            'i': 0,
            'j': 0,
            'k': 0,
            'l': 0,
            'm': 0,
            'n': 0,
            'o': 0,
            'p': 0,
            'q': 0,
            'r': 0,
            's': 0,
            't': 0,
            'u': 0,
            'v': 0,
            'w': 0,
            'x': 0,
            'y': 0,
            'z': 0,
            '-': 0,
            '=': 0,
            '[': 0,
            ']': 0,
            '\\': 0,
            ';': 0,
            '\'': 0,
            ',': 0,
            '.': 0,
            '/': 0,
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
