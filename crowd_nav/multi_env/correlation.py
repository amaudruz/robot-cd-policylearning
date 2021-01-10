import torch 
import numpy as np

import random

from crowd_nav.imitate import test_model

if __name__ == "__main__":
    model_path = 'data/full_data_tests/trajpred/imitate-trajpred-2.50-weight-1to4-length-traj/policy_net_129.pth'
    traj_path = 'data/full_data_tests/trajpred/imitate-trajpred-2.50-weight-1to4-length-traj/prediction_head_129.pth'

    test_grid_orca = {'safety_space' : np.linspace(0.01, 1, 100), 'neighbor_dist' : np.linspace(0, 6, 30), 
                'time_horizon' : np.linspace(0, 10, 10)}

    n_tests=2000
    results = []
    for i in  range(n_tests):
        if i % 100 == 0 :
            torch.save(results, 'data/outlier/results.pth')
        env_mods = {'safety_space' : random.choice(np.linspace(0.01, 0.8, 100)), 
                    'neighbor_dist': random.choice(np.linspace(0, 6, 30)),
                    'time_horizon' : random.choice(np.linspace(0, 10, 10))}
        
        results.append((env_mods, test_model(m_p=model_path, model_type='sail_traj_simple', visible=False, n_episodes=100,\
            env_type='orca', traj_path=traj_path, env_mods=env_mods, notebook=False, suffix='{} :'.format(i))))
    