from keyboardEnv import KeyboardEnv as kEnv


env = kEnv()

state, _ = env.reset()

print(state)

action = [                                         ['a', 'b'],
['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
  ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
    ['-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/']
]

state, reward, done, *_ = env.step(action)

print(state)
print(reward)
