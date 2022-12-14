import torch
import numpy as np
import random

import keyboardEnv


keys = "abcdefghijklmnopqrstuvwxyz-=[]\\;',./"


class KeyboardNN(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(KeyboardNN, self).__init__()

        self.layer1 = torch.nn.Linear(observation_space, 4)
        self.layer2 = torch.nn.Linear(4, 36)
        self.layer3 = torch.nn.Linear(36, action_space)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1)
        x = torch.nn.functional.relu(self.layer2)
        x = self.layer3(x)

        return x


class KeyboardAgent:
    def __init__(self, environment):
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = KeyboardNN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        """

# keymaps will be dynamically added to both state and action qTable entries
        self.qTable = np.zeros(shape=(10000, 10000))
        """
        self.qTable = {
                'a': {x: np.inf for x in range(len(keys))},
                'b': {x: np.inf for x in range(len(keys))},
                'c': {x: np.inf for x in range(len(keys))},
                'd': {x: np.inf for x in range(len(keys))},
                'e': {x: np.inf for x in range(len(keys))},
                'f': {x: np.inf for x in range(len(keys))},
                'g': {x: np.inf for x in range(len(keys))},
                'h': {x: np.inf for x in range(len(keys))},
                'i': {x: np.inf for x in range(len(keys))},
                'j': {x: np.inf for x in range(len(keys))},
                'k': {x: np.inf for x in range(len(keys))},
                'l': {x: np.inf for x in range(len(keys))},
                'm': {x: np.inf for x in range(len(keys))},
                'n': {x: np.inf for x in range(len(keys))},
                'o': {x: np.inf for x in range(len(keys))},
                'p': {x: np.inf for x in range(len(keys))},
                'q': {x: np.inf for x in range(len(keys))},
                'r': {x: np.inf for x in range(len(keys))},
                's': {x: np.inf for x in range(len(keys))},
                't': {x: np.inf for x in range(len(keys))},
                'u': {x: np.inf for x in range(len(keys))},
                'v': {x: np.inf for x in range(len(keys))},
                'w': {x: np.inf for x in range(len(keys))},
                'x': {x: np.inf for x in range(len(keys))},
                'y': {x: np.inf for x in range(len(keys))},
                'z': {x: np.inf for x in range(len(keys))},
                '-': {x: np.inf for x in range(len(keys))},
                '=': {x: np.inf for x in range(len(keys))},
                '[': {x: np.inf for x in range(len(keys))},
                ']': {x: np.inf for x in range(len(keys))},
                '\\': {x: np.inf for x in range(len(keys))},
                ';': {x: np.inf for x in range(len(keys))},
                '\'': {x: np.inf for x in range(len(keys))},
                ',': {x: np.inf for x in range(len(keys))},
                '.': {x: np.inf for x in range(len(keys))},
                '/': {x: np.inf for x in range(len(keys))},
                }
        """

        self.state_visited = {}
        self.state_count = 0
        self.act_visited = {}
        self.act_count = 0

        self.randomness = 1.0 # chance to act randomly
        self.decay = 0.8 # decay to multiply randomness by

    def act(self, state):
        if random.random() < self.randomness:
            keylist = list(keys)
            random.shuffle(keylist)
            toReturn = self.generate_keymap(keylist) # generate random keymap
            return toReturn

        toReturn = self.act_greedy(state)
        return toReturn # self.act_greedy(state)

    # act_greedy will never be the first action called, so we don't need to worry about an empty dictionary
    def act_greedy(self, state):
        if state not in self.state_visited.values():
            self.state_visited[self.state_count] = state
            state_count += 1

        indices, states = zip(*self.state_visited.items())
        state_index = states.index(state)

        state = indices[state_index]



        # from https://stackoverflow.com/a/45002906
        i,j = np.where( self.qTable==np.min(self.qTable[np.nonzero(self.qTable)]))

        keymap = self.act_visited[i[0]]

        toReturn = []

        for i in range(len(keymap)):
            for j in range(len(keymap[i])):
                toReturn.append(keymap[i][j])

        return self.generate_keymap(toReturn)

        """
        order = [-1 for i in range(len(keys))]
        keymap = list(self.qTable.keys());
        random.shuffle(keymap);
        for i in keymap:
            index = self.qTable[i]
            subDict = dict(sorted(index.items(), key=lambda x:x[1]))
            k = 0
            j = list(subDict.keys())[k] # minimum reward for key location
            
            while order[j] != -1:
                k += 1
                j = list(subDict.keys())[k] # minimum reward for key location
                
            order[j] = i

        return self.generate_keymap(order)
        """

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
        
        if action not in self.act_visited.values():
            self.act_visited[self.act_count] = action
            self.act_count += 1

# FIX THIS - BACKWARDS

        indices, states = zip(*self.state_visited.items())
        state_index = states.index(state)
        ns_index = states.index(next_state)

        state = indices[state_index]

        next_state = indices[ns_index]

        indices, actions = zip(*self.act_visited.items())
        action_index = actions.index(action)

        action = indices[action_index]

        target = reward + np.max(self.qTable[next_state])
        self.qTable[(state,) + (action,)] += (target - self.qTable[(state,) + (action,)]) * .0001

        """
        for i in self.qTable.keys():
            if state[i] in self.qTable[i]:
                comparator = self.qTable[i][state[i]]
            else:
                comparator = 0

            if next_state[i] in self.qTable[i]:
                qVal = min(self.qTable[i].items(), key=lambda x:x[1])
                if qVal != np.inf:
                    target = reward + qVal[1]
                else:
                    target = reward
            else:
                target = reward
            
            self.qTable[i][state[i]] = comparator + (target - comparator) * 3e-5
        """

    def update_randomness(self):
        self.randomness *= self.decay

    def getQ(self, state):
        q = {}
        for i in state.keys():
            q[i] = self.qTable[i][state[i]]

        return q


    """
    def act(self, state):
        # move the state to a Torch Tensor
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # find the probabilities of taking each action
        probs = self.model(state).cpu()

        # creates a categorical distribution parameterized by probs 
        m = torch.distributions.Categorical(probs)

        # choose an action from the probability distribution
        action = m.sample()

        # return that action, and the chance we chose that action
        return action.item(), m.log_prob(action)
    """

