# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:28:14 2020

@author: darac
"""
import pandas as pd
import numpy as np
from numpy.random import choice
import itertools
import random
import matplotlib.pyplot as plt

def paired_comparison_mcmc(num_ballots,
                           mean_support_by_race,
                           std_support_by_race,
                           cand_list,
                           vote_portion_by_race,
                           race_list,
                           seeds=None,
                           sample_interval=10,
                           verbose = True):
    #Sample from probability distribution for each race using MCMC - don't explicitly
    #compute probability of each ballot in advance
    #Draw from each race's prob distribution (number of ballots per race dtmd by cvap share)
    ordered_cand_pairs = list(itertools.permutations(cand_list,2))
    ballots_list = []

    for race in race_list:
        #make dictionairy of paired comparisons: i.e. prob i>j for all ordered pairs of candidates
        #keys are ordered pair of candidates, values are prob i>j in pair of candidates
        paired_compare_dict = {k: mean_support_by_race[race][k[0]]/(mean_support_by_race[race][k[0]]+mean_support_by_race[race][k[1]]) for k in ordered_cand_pairs}
        #starting ballot for mcmc
        start_ballot = list(np.random.permutation(cand_list))
        #function for evaluating single ballot in MCMC
        #don't need normalization term here! Exact probability of a particular ballot would be
        #output of this fnction divided by normalization term that MCMC allows us to avoid
        track_ballot_prob = []
        def ballot_prob(ballot):
            pairs_list_ballot = list(itertools.combinations(ballot,2))
            paired_compare_trunc = {k: paired_compare_dict[k] for k in pairs_list_ballot}
            ballot_prob = np.product(list(paired_compare_trunc.values()))
            return ballot_prob

        #start MCMC with 'start_ballot'
        num_ballots_race = int(num_ballots*vote_portion_by_race[race])
        race_ballot_list = []
        step = 0
        accept = 0
        while len(race_ballot_list) < num_ballots_race: #range(num_ballots_race):
            #proposed new ballot is a random switch of two elements in ballot before
            proposed_ballot = start_ballot.copy()
            j1,j2 = random.sample(range(len(start_ballot)),2)
            proposed_ballot[j1], proposed_ballot[j2] = proposed_ballot[j2], proposed_ballot[j1]

            #acceptance ratio: (note - symmetric proposal function!)
            accept_ratio = min(ballot_prob(proposed_ballot)/ballot_prob(start_ballot),1)
            #accept or reject proposal
            if random.random() < accept_ratio:
                start_ballot = proposed_ballot
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot)
                accept += 1
            else:
                if step % sample_interval == 0:
                    race_ballot_list.append(start_ballot)
            step += 1
        ballots_list = ballots_list + race_ballot_list
        if verbose:
            if step > 0:
                print("Acceptance ratio for {} voters: ".format(race), accept/step)
       # plt.plot(track_ballot_prob)
    return ballots_list
