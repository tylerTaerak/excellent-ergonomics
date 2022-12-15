from keyboardEnv import KeyboardEnv as kEnv
from agent import KeyboardAgent as Agent
import numpy as np
import matplotlib.pyplot as plt


"""
    The below code is to showcase the keyboard environment
env = kEnv()

state, _ = env.reset()

print('--- Key Scores with QWERTY ---')
for i in state:
    print(f'{i}: {round(state[i], 2)}')

action = [                                         ['a', 'b'],
['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
  ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
    ['-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/']
]

state, reward, done, *_ = env.step(action)

print('\n--- Key Scores with Alphabetical Order ---')
for i in state:
    print(f'{i}: {round(state[i], 2)}')
print(f'Reward for Alphabetical: {reward}')

action = [                                         ['-', '='],
['/', ',', '.', 'p', 'y', 'f', 'g', 'c', 'r', 'l', '[', ']', '\\'],
  ['a', 'o', 'e', 'u', 'i', 'd', 'h', 't', 'n', 's', '\''],
    [';', 'q', 'j', 'k', 'x', 'b', 'm', 'w', 'v', 'z']
]

state, reward, done, *_ = env.step(action)

print('\n--- Key Scores with Dvorak ---')
for i in state:
    print(f'{i}: {round(state[i], 2)}')
print(f'Reward for Alphabetical: {reward}')
"""

if __name__ == "__main__":
    MAX_ITERATIONS = 25
    LOG_ITERATION = 5

    total_count = 0
    
    env = kEnv()
    
    agent = Agent(env)
    episode_count = 0
    learning = []

    for iteration in range(1, MAX_ITERATIONS+1):
        episode_count += 1
        done = False

        state, _ = env.reset()

        total_reward = 0
        steps = 0
        while not done:
            total_count += 1
            steps += 1
            action = agent.act(state)

            next_state, reward, done, *_ = env.step(action)
            agent.update(state, next_state, action, reward)
            state = next_state
            total_reward += reward


        agent.update_randomness()
        learning.append(total_reward/steps)

        print(f"Iteration: {iteration} | Average reward: {(total_reward/steps):.4f}")


    print(agent.get_best())

    x = np.arange(1, len(learning) + 1)
    y = learning

    plt.plot(x, y)
    plt.title("Average Score per 2000 characters")
    plt.xlabel("Iteration")
    plt.ylabel("Average Score")
    plt.show()

