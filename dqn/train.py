import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from sklearn.decomposition import PCA
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_size):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.layer1 = nn.Linear(n_inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        value = f.relu(self.layer1(state))
        value = self.layer2(value)
        return value

import pykitml as pk
import mss

EPISODES = 10
EPISODES_SECS = 300
N_STEPS = 18000

N_INPUTS = 64
N_ACTIONS = 3
HIDDEN_SIZE = 16

GAMMA = 0.99

from joblib import load
pca = load('../lstm/pca.joblib')
model = DQN(N_INPUTS, N_ACTIONS, HIDDEN_SIZE).float()
# model.actor.load_state_dict(torch.load('../lstm/lstm.pt'))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

ep_rewards = []
monitor = {"top": 224, "left": 256-116, "width":256, "height":256}

import cv2
import multiprocessing as mp
import time

# Values shared between processess
A_val = mp.Value('d', 0)
left_val = mp.Value('d', 0)
opp_hp = mp.Value('d', 0)
restart = mp.Value('d', 0)

def on_frame(server, frame, A_val, left_val): 
    if restart.value == 1:
        server.reset()
        server.frame_advance()
        restart.value = 0
    # Toggle start button to start rounds
    if(frame%10 < 5): start = True
    else: start = False

    # Set joypad
    server.set_joypad(A=A_val.value==1, left=left_val.value==1, start=start)

    opp_hp.value = server.read_mem(398)

    # Continue emulation
    server.frame_advance()

# Initialize and start server
def start_server(A_val, left_val):
    server = pk.FCEUXServer(lambda server, frame: on_frame(server, frame, A_val, left_val))
    print(server.info)
    server.start()

p = mp.Process(target=start_server, args=(A_val, left_val))
p.start()
for episode in range(EPISODES):
    print(f"EPISODE: {episode}")
    ep_reward = 0
    with mss.mss() as sct:
        start_time = time.time()
        running = True
        while running:
            cur_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))
            # Convert to gray scale
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # Resize image
            img = cv2.resize(src=img, dsize=(64, 64))
            # Reshape
            img = img.reshape(4096)
            # Normalize
            img = img/255

            # PCA
            img = pca.transform([img])

            value = model(torch.from_numpy(np.array([img])).float())
            value = value.flatten()
            dist = Categorical(f.softmax(value, dim=0))
            action = dist.sample()
            model.saved_actions.append(SavedAction(dist.log_prob(action), value[action]))
            # print(action)
            A_val.value = (action==0).item()
            left_val.value = (action==1).item()
            # server.frame_advance()
            # reward = server.read_mem(391) - server.read_mem(398)
            reward = -1-opp_hp.value
            model.rewards.append(reward)
            ep_reward += reward
            if cur_time-start_time >= EPISODES_SECS:
                running = False
    ep_rewards.append(ep_reward)
    print(f"REWARD: {ep_reward}")
    # loss is calculated differently than usual, so this
    # is not really a conventional DQN, but rather a 
    # A2C with a Q network as both the critic and an
    # actor?
    R = 0
    policy_losses = []
    value_losses = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + GAMMA*R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std()
    for (log_prob, value), R in zip(model.saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob*advantage)
        value_losses.append(f.smooth_l1_loss(value, torch.tensor([R])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
    restart.value = 1

torch.save(model.state_dict(), 'dqn.pt')
import matplotlib.pyplot as plt
plt.plot(range(EPISODES), ep_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()