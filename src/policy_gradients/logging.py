import torch as ch
import numpy as np
from .torch_utils import *
from torch.nn.utils import parameters_to_vector as flatten
from torch.nn.utils import vector_to_parameters as assign
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


#####
# Understanding TRPO approximations for KL constraint
#####

def paper_constraints_logging(agent, saps, old_pds, table):
    new_pds = agent.policy_model(saps.states)
    new_log_ps = agent.policy_model.get_loglikelihood(new_pds,
                                                    saps.actions)

    ratios = ch.exp(new_log_ps - saps.action_log_probs)
    max_rat = ratios.max()

    kls = agent.policy_model.calc_kl(old_pds, new_pds)
    avg_kl = kls.mean()

    row = {
        'avg_kl':avg_kl,
        'max_ratio':max_rat,
        'opt_step':agent.n_steps,
    }

    for k in row:
        if k != 'opt_step':
            row[k] = float(row[k])

    agent.store.log_table_and_tb(table, row)
