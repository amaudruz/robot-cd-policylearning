import torch 
import os
import random
import shutil
import argparse
import configparser

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.tests import test_model

def model_from_path(model_path) :
    policy_p = 'configs/policy.config'
    # configure policy
    policy = policy_factory['sail_traj_simple']()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_p)
    policy.configure(policy_config)
    
    model = policy.model
    model.load_state_dict(torch.load(model_path))
    return model

def results_filter(min, max, results) :
    res = []
    for r in results :
        #print(r[1]['collision'])
        if r[1]['collision'] <= max and r[1]['collision'] >= min :
            res.append(r)
    return res


def main() :
    init_model_path = 'data/full_data_tests/trajpred/imitate-trajpred-2.50-weight-1to4-length-traj/policy_net_129.pth'
    init_traj_path = 'data/full_data_tests/trajpred/imitate-trajpred-2.50-weight-1to4-length-traj/prediction_head_129.pth'   
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy_net_path', type=str, default=init_model_path)
    parser.add_argument('--traj_fext_path', type=str, default=init_traj_path)
    parser.add_argument('--outlier_path', type=str, default='data/outlier/results.pth')  

    parser.add_argument('--n_enviroements', type=int, default=40)
    parser.add_argument('--min_collisions', type=int, default=0.10)
    parser.add_argument('--max_collisions', type=int, default=0.24)
    parser.add_argument('--simulation_episodes', type=int, default=1000)
    parser.add_argument('--episodes_per_epoch', type=int, default=200)
    
    parser.add_argument('--output_dir', type=str, default='data/improvement/temp')
    
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir) :
        key = input('Path already exists, overwrite ? (y/n) : ')
        if key == 'y' :
            shutil.rmtree(args.output_dir)
        else :
            print('Aborting')
            return 
    os.mkdir(args.output_dir)

    model = model_from_path(args.policy_net_path)
    results = results_filter(args.min_collisions, args.max_collisions, torch.load(args.outlier_path))

    environement_dirs = []
    for i in range(args.n_enviroements) :
        temp_dir = os.path.join(args.output_dir, 'environement_{}'.format(i))
        environement_dirs.append(temp_dir)
        
        os.mkdir(temp_dir)
        env_mods, _ = random.choice(results)
        res, experience = test_model(m_p=None, model_type='sail_traj_simple', visible=False, n_episodes=args.simulation_episodes,\
           env_type='orca', traj_path=args.traj_fext_path, env_mods=env_mods, model=model, return_exp=True, notebook=False)
        torch.save({'experience' : experience, 'results' : res, 'env_mods' : env_mods}, os.path.join(temp_dir, 'data.pth'))

if __name__ == "__main__":
    main()