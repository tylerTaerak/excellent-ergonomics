from keyboardEnv import KeyboardEnv as kEnv


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
  ['a', 'o', 'e', 'u', 'i', 'd', 'v', 't', 'n', 's', '\''],
    [';', 'q', 'j', 'k', 'x', 'b', 'm', 'w', 'v', 'z']
]

state, reward, done, *_ = env.step(action)

print('\n--- Key Scores with Dvorak ---')
for i in state:
    print(f'{i}: {round(state[i], 2)}')
print(f'Reward for Alphabetical: {reward}')
