from numpy.lib.index_tricks import nd_grid
import torch 
import os
import random
import shutil
import argparse
import configparser

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.tests import test_model
from crowd_nav.multi_env.correlation import pick_best_model

def model_from_path(model_path) :
    policy_p = 'configs/policy.config'
    policy = policy_factory['sail_traj_simple']()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_p)
    policy.configure(policy_config)
    
    model = policy.model
    model.load_state_dict(torch.load(model_path))
    return model

def results_filter(min, max, results, n_res=40) :
    res = []
    for r in results['perfs'] :
        if r[1]['collision'] <= max and r[1]['collision'] >= min :
            res.append(r)
        if len(res) == n_res :
            results['perfs'] = res
            return results
    
def main() : 
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--outlier_path', type=str, default='data/multi_env/imitate-trajpred-2.50-weight-1to4-length-traj')  

    parser.add_argument('--n_environments', type=int, default=40)
    parser.add_argument('--min_collisions', type=int, default=0.10)
    parser.add_argument('--max_collisions', type=int, default=0.24)
    parser.add_argument('--simulation_episodes', type=int, default=1000)
    
    parser.add_argument('--output_dir', type=str, default='data/improvement')
    
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_dir, args.outlier_path.split('/')[-1])
    if os.path.exists(output_dir) :
        key = input('Path already exists, overwrite ? (y/n) : ')
        if key == 'y' :
            shutil.rmtree(output_dir)
        else :
            print('Aborting')
            return 
    os.mkdir(output_dir)
    

    
    results = torch.load(os.path.join(args.outlier_path, 'results.pth'))
    model = model_from_path(results['model_path'])

    results = results_filter(args.min_collisions, args.max_collisions, results, n_res=args.n_environments)

    environement_dirs = []
    for i, r in enumerate(results['perfs']) :
        temp_dir = os.path.join(output_dir, 'environement_{}'.format(i))
        environement_dirs.append(temp_dir)
    
        os.mkdir(temp_dir)
        env_mods = r[0]
        res, experience = test_model(m_p=None, model_type='sail_traj_simple', visible=False, n_episodes=args.simulation_episodes,\
           env_type='orca', traj_path=results['traj_path'], env_mods=env_mods, model=model, return_exp=True, notebook=False)
        torch.save({'experience' : experience, 'results' : res, 'env_mods' : env_mods, 'default_res' : r[1], 'model_path' : results['model_path'],\
            'traj_path' : results['traj_path']}, os.path.join(temp_dir, 'data.pth'))

if __name__ == "__main__":
    main()