import torch

import keyboardEnv


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = KeyboardNN(environment.observation_space.shape[0], environment.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

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


env = keyboardEnv.KeyboardEnv()
net = KeyboardNN(len(env.observation_space.spaces), len(env.action_space.spaces))
print(net)
