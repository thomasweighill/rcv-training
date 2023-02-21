'''
Predict outcomes for minority groups under ranked choice voting
using four different models of voter behavior.

Enter basic input parameters under Global variables, then run the
code in order to simulate elections and output expected number of poc
candidates elected under each model and model choice.
'''

import numpy as np
from IPython.display import display
from itertools import product, permutations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, sys
import pandas as pd
import compute_winners as cw
from vote_transfers import cincinnati_transfer
from model_details import Cambridge_ballot_type, BABABA, luce_dirichlet, bradley_terry_dirichlet


###-------- Global variables -- change these
poc_share = 0.18
poc_support_for_poc_candidates = 0.85
white_support_for_poc_candidates = 0.06
num_ballots = 100
num_simulations = 10
seats_open = 5
num_poc_candidates = 3
num_white_candidates = 5
output_file = sys.argv[0].rstrip('.py') + '.csv' #default name for output, you can change this or leave it
##-----------------------------------------


#Setup
poc_support_for_white_candidates = 1.0-poc_support_for_poc_candidates+1e-6
white_support_for_white_candidates = 1.0-white_support_for_poc_candidates+1e-6
poc_support_for_poc_candidates -= 1e-6
white_support_for_poc_candidates -= 1e-6
### alphas: >> 1 means very similar supports, <<1 means most support goes to 1-2 cands
# inner list goes [poc_for_poc, poc_for_white, white_for_poc, white_for_white]
concentration_list = [[0.5]*4, [2,0.5,0.5,0.5], [2,2,2,2], [0.5,0.5,2,2], [1.0]*4]
scenarios_list = ["A", "B", "C", "D"]
results = {s:[] for s in scenarios_list}
results['E'] = []  



#Plackett-Luce
print('Running PL...')
poc_elected_luce_dirichlet = []
for i, concentrations in enumerate(concentration_list):
  poc_elected_rcv,_ = luce_dirichlet(
      poc_share = poc_share,
      poc_support_for_poc_candidates = poc_support_for_poc_candidates,
      poc_support_for_white_candidates = poc_support_for_white_candidates,
      white_support_for_white_candidates = white_support_for_white_candidates,
      white_support_for_poc_candidates = white_support_for_poc_candidates,
      num_ballots = num_ballots,
      num_simulations = num_simulations,
      seats_open = seats_open,
      num_poc_candidates = num_poc_candidates,
      num_white_candidates = num_white_candidates,
      concentrations = concentrations
  )
  poc_elected_luce_dirichlet.append(poc_elected_rcv)
for i, s in enumerate(results):
  results[s].append(np.mean(poc_elected_luce_dirichlet[i]))

#Bradley-Terry
print('Running BT...')
poc_elected_bradley_terry_dirichlet = []
for i, concentrations in enumerate(concentration_list):
  poc_elected_rcv,_ = bradley_terry_dirichlet(
      poc_share = poc_share,
      poc_support_for_poc_candidates = poc_support_for_poc_candidates,
      poc_support_for_white_candidates = poc_support_for_white_candidates,
      white_support_for_white_candidates = white_support_for_white_candidates,
      white_support_for_poc_candidates = white_support_for_poc_candidates,
      num_ballots = num_ballots,
      num_simulations = num_simulations,
      seats_open = seats_open,
      num_poc_candidates = num_poc_candidates,
      num_white_candidates = num_white_candidates,
      concentrations = concentrations,
  )
  poc_elected_bradley_terry_dirichlet.append(poc_elected_rcv)
for i, s in enumerate(results):
  results[s].append(np.mean(poc_elected_luce_dirichlet[i]))


#Alternating crossover model
print('Running AC...')
poc_elected_bababa,_  = BABABA(
    poc_share = poc_share,
    poc_support_for_poc_candidates = poc_support_for_poc_candidates,
    poc_support_for_white_candidates = poc_support_for_white_candidates,
    white_support_for_white_candidates = white_support_for_white_candidates,
    white_support_for_poc_candidates = white_support_for_poc_candidates,
    num_ballots = num_ballots,
    num_simulations = num_simulations,
    seats_open = seats_open,
    num_poc_candidates = num_poc_candidates,
    num_white_candidates = num_white_candidates,
    scenarios_to_run = scenarios_list,
    verbose=False,
)
for i, s in enumerate(scenarios_list):
  results[s].append(np.mean(poc_elected_bababa[s]))
results['E'] = np.mean([np.mean(poc_elected_bababa[c]) for c in scenarios_list])


#Cambridge Sampler
print('Running CS...')
poc_elected_Cambridge,_  = Cambridge_ballot_type(
    poc_share = poc_share,
    poc_support_for_poc_candidates = poc_support_for_poc_candidates,
    poc_support_for_white_candidates = poc_support_for_white_candidates,
    white_support_for_white_candidates = white_support_for_white_candidates,
    white_support_for_poc_candidates = white_support_for_poc_candidates,
    num_ballots = num_ballots,
    num_simulations = num_simulations,
    seats_open = seats_open,
    num_poc_candidates = num_poc_candidates,
    num_white_candidates = num_white_candidates,
    scenarios_to_run = scenarios_list,
)
for i, s in enumerate(scenarios_list):
  results[s].append(np.mean(poc_elected_Cambridge[s]))
results['E'] = np.mean([np.mean(poc_elected_Cambridge[c]) for c in scenarios_list])

results['model'] = ['PL', 'BT', 'AC', 'CS']
df = pd.DataFrame(results)
df = df.set_index('model')
display(df)
df.to_csv(output_file, index_label='model')
