![gym-omegle](http://i.imgur.com/XQOIR1B.gif)

# gym-omegle: An OpenAI Gym environment for Omegle

Defines an [OpenAI Gym](https://gym.openai.com/) environment (`omegle-v0`) for training agents to chat with strangers on Omegle. 

Uses Selenium with Chrome to perform the actions on the Omegle website.

### Environment

__Observation space__: last message received by the stranger; each word is encoded using the [GloVe](http://nlp.stanford.edu/projects/glove/) algorithm trained on the Common Crawl corpus.

__Action space__: 98 actions in total; 95 actions for each of the [printable ASCII characters](http://www.ascii-code.com/), plus the following 3 actions:

1. send current message

2. clear current message

3. wait for 1 second

__Reward__: 1 if the stranger has written a new message; 0 otherwise. 

### Installation

```
$ pip install .
```

### Usage

```python
import gym
import gym_omegle

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

env = gym.make("omegle-v0")
agent = RandomAgent(env.action_space)

reward = 0
done = False
episode_count = 3

for i in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break

env.close()
```
