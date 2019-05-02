import ast
import math
import random
import os

def safelog(x):
  return math.log(x) if x > .0 else -math.inf

def logsumexp(X):
  X_max = X[0]
  for x in X:
    if x > X_max:
      X_max = x
  acc = .0
  for x in X:
    acc += math.exp(x - X_max)
  return safelog(acc) + X_max

class hmm_model:
  def __init__(self, race):
    self.vocabulary = 28
    self.components = 32
    with open(os.path.join(os.path.dirname(__file__), race + "/startprob"), "r") as f:
      self.startprob = ast.literal_eval(f.read())
    with open(os.path.join(os.path.dirname(__file__), race + "/transmat"), "r") as f:
      self.transmat = ast.literal_eval(f.read())
    with open(os.path.join(os.path.dirname(__file__), race + "/emissionprob"), "r") as f:
      self.emitprob = ast.literal_eval(f.read())

class hmm_gen:
  def __init__(self, model):
    self.model = model

  def sample(self):
    start_dice = random.random()
    state = 0
    for i in range(self.model.components):
      if start_dice < self.model.startprob[i]:
        break
      state = state + 1
      start_dice = start_dice - self.model.startprob[i]
    name = ""
    for l in range(100):
      char_dice = random.random()
      character = 0
      for i in range(self.model.vocabulary):
        if char_dice < self.model.emitprob[state][i]:
          break
        character = character + 1
        char_dice = char_dice - self.model.emitprob[state][i]
      if character == 27:
        break
      elif character == 26:
        name += "'"
      else:
        name += chr(character + ord('a'))
      trans_dice = random.random()
      trans = 0
      for i in range(self.model.components):
        if trans_dice < self.model.transmat[state][i]:
          break
        trans = trans + 1
        trans_dice = trans_dice - self.model.transmat[state][i]
      state = trans
    return name

  def score(self, name):
    S = []
    for c in name:
      if c == "'":
        S.append(26)
      else:
        S.append(ord(c.lower()) - ord('a'))
    S.append(27)
    framelogprob = []
    fwdlattice = []
    for i in range(self.model.components):
      framelogprob.append([safelog(self.model.emitprob[i][c]) for c in S])
      fwdlattice.append([0] * len(S))
    work_buffer = [0] * self.model.components
    for i in range(self.model.components):
      fwdlattice[i][0] = safelog(self.model.startprob[i]) + framelogprob[i][0]
    for t in range(1, len(S)):
      for j in range(self.model.components):
        for i in range(self.model.components):
          work_buffer[i] = fwdlattice[i][t - 1] + safelog(self.model.transmat[i][j])
        fwdlattice[j][t] = logsumexp(work_buffer) + framelogprob[j][t]
    return logsumexp([fwdlattice[i][len(S) - 1] for i in range(self.model.components)]) / len(S) - safelog(1.0/self.model.vocabulary)

if __name__ == "__main__":
  nef = hmm_model("nef")
  gen = hmm_gen(nef)
  for i in range(100):
    name = gen.sample()
    print(name, gen.score(name))
