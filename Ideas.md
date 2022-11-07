# Ideas for Final Project

I haven't really had any super great ideas for the final project
for this class yet. Well, I kind of have, but they can't be just
easily done by the simple reinforcement learning we've done in 
class so far. Here is a list of ideas and implementations thereof.

## Optimal Keyboard Mapping

The idea here is for the agent to generate keymaps that will
eventually converge on an "optimal" keymap, which would theoretically
be the keymap with the least amount of required finger movement to type. 

### The Environment

The environment will consist of a set of text to be "typed" through 
using the keymap given by the agent. It will return a reward based on
the amount of finger movement required by the text for the given keymap.

### The Agent

The agent will generate a keymap based on its current reward. I don't really know
how I am to implement this.

### Questions

    * The state is not particularly utilized, therefore, the agent can't really be reactionary
    * 

### Notes

    * Look at OpenGym
    * Or any other sort of simulator
    * Send a note tomorrow

## Arrow-Catching Robot

This is a robot that would be trained to catch an arrow (or similarly shafted projectile).
This one is considerably less exciting than the keyboard project to me, but it is also
a lot less exciting to me. 

### The Environment

The environment initializes when the projectile is shot. Each step is one time-tick. The
environment returns a reward of 0 while the projectile travels, 1 if the projectile is successfully caught,
and -1 if the projectile is missed.

### The Agent

The agent in this scenario simply chooses when to catch. 

### Notes

    * Further implementations could include image processing
    * There would likely need to be a number of global variables to facilitate velocity among other things

## Conclusion

After some consideration and talking with Dr. Flann, I have decided to move forward
with my keyboard optimization idea. I wrote down some notes for what to look for in
preparation for beginning my project, and we'll just figure that one out.

[Here](https://dl.acm.org/doi/10.1145/3163080.3163117) is a paper on typing with reinforcement
learning that I may be able to utilize.

Note: would it be more practical to just use a Q-learning table for the learning mechanism?
