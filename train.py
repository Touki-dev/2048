import numpy as np
from game_env import GameEnv
from DQN_agent import DQNAgent
import time

env = GameEnv(grid_size=4)
state_size = env.grid_size * env.grid_size
action_size = 4  # 0: up, 1: down, 2: left, 3: right
agent = DQNAgent(state_size, action_size)
episodes = 1000
batch_size = 32

t0 = time.time()

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for i in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Score: {env.score}, Epsilon: {agent.epsilon}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 50 == 0:
        t1 = time.time()
        print(t1 - t0)

        with open("time.txt", "a") as fichier:
            fichier.write(f"Episodes {e} : {t1 - t0} \n")

        agent.save(f"models/2048-dqn-{e}.weights.h5")
        

t2 = time.time()
print(t2 - t0)

with open("time.txt", "a") as fichier:
    fichier.write(f"Total duration : {t2 - t0}")