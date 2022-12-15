import torch
import numpy as np
import random

import keyboardEnv


keys = "abcdefghijklmnopqrstuvwxyz-=[]\\;',./"

class KeyboardAgent:
    def __init__(self, environment):
        # keymaps will be dynamically added to both state and action qTable entries
        self.qTable = np.zeros(shape=(1000, 1000))

        self.state_visited = {"act_count": 0, "act_visited": dict()}
        self.state_count = 0

        self.randomness = 1.0 # chance to act randomly
        self.decay = 0.8 # decay to multiply randomness by

    def act(self, state):
        if random.random() < self.randomness:
            return self.act_random(state)

        return self.act_greedy(state)

    def act_random(self, state):
        if self.state_count >= 1000:
            indices, states = zip(*self.state_visited.items())
            state_index = states.index(state)

            state = indices[state_index]

            keymap = state
            while keymap == state:
                keymap = random.randrange(1000)

            keymap = self.state_visited[keymap]

            toReturn = ['' for i in range(36)]

            for i in keymap.keys():
                toReturn[keymap[i]] = i

            return self.generate_keymap(toReturn)

        keylist = list(keys)
        random.shuffle(keylist)
        toReturn = self.generate_keymap(keylist) # generate random keymap
        return toReturn

    def act_greedy(self, state):
        indices, states = zip(*self.state_visited.items())
        state_index = states.index(state)

        state = indices[state_index]

        # from https://stackoverflow.com/a/45002906
        nonzero = self.qTable[np.nonzero(self.qTable[state])]
        if len(nonzero) > 0:
            i,j = np.where( self.qTable==np.min(nonzero))
        else:
            return self.act_random(state)

        keymap = self.state_visited["act_visited"][i[0]]

        toReturn = []

        for i in range(len(keymap)):
            for j in range(len(keymap[i])):
                toReturn.append(keymap[i][j])

        return self.generate_keymap(toReturn)

    def generate_keymap(self, order):
        keymap = [                                  [0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
        for i in range(len(keymap)):
            for j in range(len(keymap[i])):
                keymap[i][j] = order.pop(0)

        return keymap

    def update(self, state, next_state, action, reward):

        # handle indexing of newly visited states/actions
        if state not in self.state_visited.values():
            self.state_visited[self.state_count] = state
            self.state_count += 1

        if next_state not in self.state_visited.values():
            self.state_visited[self.state_count] = next_state
            self.state_count += 1
        
        if action not in self.state_visited["act_visited"].values():
            self.state_visited["act_visited"][self.state_visited["act_count"]] = action
            self.state_visited["act_count"] += 1

        indices, states = zip(*self.state_visited.items())
        state_index = states.index(state)
        ns_index = states.index(next_state)

        state = indices[state_index]

        next_state = indices[ns_index]

        indices, actions = zip(*self.state_visited["act_visited"].items())
        action_index = actions.index(action)

        action = indices[action_index]

        nonzero = self.qTable[np.nonzero(self.qTable[next_state])]
        if len(nonzero) > 0:
            i, j = np.where(self.qTable==np.min(nonzero))
            target = reward + i[0]
        else:
            target = reward
        
        self.qTable[(state,) + (action,)] += (target - self.qTable[(state,) + (action,)]) * .0001

    def update_randomness(self):
        self.randomness *= self.decay

    def get_best(self):
        return self.state_visited[np.argmin(self.qTable)]

