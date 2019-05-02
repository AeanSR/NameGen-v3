from hmmlearn import hmm
import numpy as np
import math

seq = []
lens = []
f = open("nem.txt", "r", encoding="latin-1")
for line in f:
  sline = line.strip()
  if len(sline):
    lens.append(len(sline) + 1)
    for c in sline:
      if c.isalpha():
        seq.append([ord(c.upper())-ord('A')])
      else:
        seq.append([26])
    seq.append([27])

best_model = None
best_score = -math.inf

for restart in range(100):
  model = hmm.MultinomialHMM(n_components=32, random_state=restart, n_iter=10000).fit(seq, lens)
  score = model.score(seq)
  print("model {} scored {}".format(restart, score))
  if score > best_score:
    best_model = model
    best_score = score
    with open("nem/transmat", "w") as fo:
      print(model.transmat_.tolist(), file=fo)
    with open("nem/startprob", "w") as fo:
      print(model.startprob_.tolist(), file=fo)
    with open("nem/emissionprob", "w") as fo:
      print(model.emissionprob_.tolist(), file=fo)
    print("new best model! test outputs:")
    for i in range(10):
      x = model.sample(100)
      name = ""
      for c in x[0]:
        if c[0] == 27:
          break
        if c[0] == 26:
          name+='\''
        else:
          name+=chr(c[0]+ord('a'))
      evals = []
      for c in name:
        if c.isalpha():
          evals.append([ord(c.upper())-ord('A')])
        else:
          evals.append([26])
      evals.append([27])
      score = model.score(evals) / len(evals) - math.log(1.0/28.0)
      print(name, score)
