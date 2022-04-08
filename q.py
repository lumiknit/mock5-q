import signal
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda'

from mock5 import Mock5
from mock5.agent_random import agent as agent_rnd
from mock5.agent_analysis_based import agent as agent_ab

MODEL_PATH = "./Q_MODEL"

HEIGHT = 12
WIDTH = 12


#-- Helpers

def game_to_input(game):
  t = game.tensor(one_hot_encoding=True, rank=3).view(1, 3, game.height, game.width).to(device)
  return t

def cut_fouls(game, qvs):
  q = qvs.clone().detach()
  q[game.empty_tensor(rank=1, empty=False, non_empty=True, dtype=torch.bool)] = float('-inf')
  return q

#-- DQN

import collections

class FlattenLayer(torch.nn.Module):
  def forward(self, x):
    return x.flatten(1, -1)

class SqueezeLayer(torch.nn.Module):
  def forward(self, x):
    return x.squeeze()

model = nn.Sequential(
    collections.OrderedDict([
      ("conv1", nn.Conv2d(3, 8, 5, padding='same')),
      #("pool1", nn.AvgPool2d(2)),
      ("relu1", nn.ReLU()),
      ("conv3", nn.Conv2d(8, 16, 3, padding='same')),
      ("relu3", nn.ReLU()),
      ("flt", FlattenLayer()),
      ("lin1", nn.Linear(16 * HEIGHT * WIDTH, 2 * HEIGHT * WIDTH)),
      ("relu4", nn.ReLU()),
      ("lin2", nn.Linear(2 * HEIGHT * WIDTH, HEIGHT * WIDTH)),
      ("sqz", SqueezeLayer()),
      ])
).to(device)

loss_fn = nn.MSELoss()

gamma = 0.98

win_reward = HEIGHT * WIDTH

def calc_q(game, idx):
  if not game.place_stone_at_index(idx):
    return -win_reward, game.player
  else:
    w = game.check_win()
    if w == None:
      with torch.no_grad():
        qvs2 = cut_fouls(game, model(game_to_input(game)))
      return (0 - gamma * torch.max(qvs2).item()), w
    elif w == 0:
      return 0.0, w
    elif w == 3 - game.player:
      return win_reward, w
    else:
      return -win_reward, w

learning_running = False

def learning_signal_handler(sig, frame):
  global learning_running
  print("Leraning will be stopped!")
  signal.signal(signal.SIGINT, signal.SIG_DFL)
  learning_running = False

def learn():
  global learning_running

  learning_rate = 4e-3

  optimizer = optim.SGD(
      model.parameters(),
      lr=learning_rate,
      momentum=0.9,
      weight_decay=1e-5)
  
  epsilon = 0.8

  n_epoch = 100
  dec_ep_epoch = 400
  ep_dim = 0.999
  ep_lb = 0.2
  print_interval = 10

  n_loss = 0
  acc_loss = 0

  signal.signal(signal.SIGINT, learning_signal_handler)
  print("Press ^C to stop learning")

  learning_running = True
  for epoch in range(n_epoch):
    if not learning_running: break
    game = Mock5(HEIGHT, WIDTH)

    winner = None
    with torch.no_grad():
      while True:
        idx = None
        rnd = np.random.rand()
        if rnd < epsilon:
          r, c = agent_ab(game)
          if type(r) is int:
            idx = game._reduce_index(r, c)
          else: idx = 0
          #print("ab", idx)
        else:
          qvs = model(game_to_input(game))
          idx = torch.argmax(cut_fouls(game, qvs.clone().detach())).item()
          #print("model", idx)
        if not game.place_stone_at_index(idx): break
        winner = game.check_win()
        if winner is not None: break
    reward = win_reward

    for _i in range(len(game.history)):
      last_idx = game.history[-1]
      game.undo()

      qvs = model(game_to_input(game))
      Y = qvs.clone().detach()
      # nq, w = calc_q(game, idx)
      Y[last_idx] = reward

      optimizer.zero_grad()
      loss = loss_fn(qvs, Y)
      loss.backward()
      optimizer.step()

      acc_loss += loss.item()
      n_loss += 1
      
      reward = -reward

    if epoch % print_interval == print_interval - 1:
      print("{:6d}: e={:.4f}; L={:8.4f}"\
          .format(epoch + 1, epsilon, acc_loss / n_loss))
      acc_loss = 0
      n_loss = 0
    if epoch > dec_ep_epoch:
      epsilon *= ep_dim
      if epsilon < ep_lb: epsilon = ep_lb

def model_action(game):
  global model
  with torch.no_grad():
    qv = cut_fouls(game, model(game_to_input(game)))
  idx = torch.argmax(qv).item()
  return game._expand_index(idx)

if __name__ == "__main__":
  msg = "(q)uit/(s)ave/(l)oad/(L)earn/vs (a)i/vs (A)lgo/algo (t)est/(p)lay"
  while True:
    print('[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]')
    x = input(msg + "\n > ").strip()
    if x == 'q': break
    elif x == 's':
      torch.save(model, MODEL_PATH)
    elif x == 'l':
      model = torch.load(MODEL_PATH)
      model.eval()
    elif x == 'L':
      learn()
    elif x == 'a':
      Mock5(HEIGHT, WIDTH).play(model_action, model_action)
    elif x == 'A':
      Mock5(HEIGHT, WIDTH).play(model_action, agent_ab)
    elif x == 'p':
      Mock5(HEIGHT, WIDTH).play(model_action, input2=None)
    elif x == 'T':
      total = 100
      count = 0
      win = 0
      draw = 0
      lose = 0
      for i in range(total):
        count += 1
        res = Mock5(HEIGHT, WIDTH).play(model_action, agent_ab, print_intermediate_state=False)
        if res == 1: win += 1
        elif res == 0: draw += 1
        else: lose += 1
      print("Model win rate {:.4f} (w {} / d {} / l {})".format(
        win / count, win, draw, lose))
