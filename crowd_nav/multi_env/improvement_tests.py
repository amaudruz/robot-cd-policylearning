import torch 
import os
import argparse
from crowd_nav.multi_env.improvement import model_from_path
from crowd_nav.policy.sail_traj_mult import mult_to_simple
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.tests import test_model

def main() :
    init_model_path = 'data/full_data_tests/trajpred/imitate-trajpred-2.50-weight-1to4-length-traj/policy_net_129.pth'
    init_traj_path = 'data/full_data_tests/trajpred/imitate-trajpred-2.50-weight-1to4-length-traj/prediction_head_129.pth'   
    
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy_net_path', type=str, default=init_model_path)
    parser.add_argument('--mult', default=False, action='store_true')

    # testing
    parser.add_argument('--episodes_per_epoch', type=int, default=200)
    
    parser.add_argument('--data_dir', type=str, default='data/improvement/temp')
    
    args = parser.parse_args()

    environement_dirs = os.listdir(args.data_dir)
    for i, env_dir in enumerate(environement_dirs) :
        
        env_dir = os.path.join(args.data_dir, env_dir)

        data = torch.load(os.path.join(env_dir, 'data.pth'))
        mods = data['env_mods']
        experience = data['experience']
        results = data['results']
        print('Environement {} : init reward : {:.3f}, init collision {:.3f}'.format(i, results['reward'], results['collision']))

        # t_fex = model_from_path(args.policy_net_path).trajectory_fext.to('cpu')
        # pred_head = torch.load(args.traj_fext_path).to('cpu')
        suffix = '  '
        if not args.mult :
            model = model_from_path(args.policy_net_path)
        else :
            model = mult_to_simple(torch.load(args.policy_net_path))[0].model
        
        models = torch.load(os.path.join(env_dir, 'models.pth'))
        results_per_epoch = []
        for i, (tf, ph) in enumerate(models) :
            #if i % 3 == 0 :
            n_suffix = suffix + 'Epoch {} : '.format(i)
            res = test_model(m_p=None, model_type='sail_traj_simple', visible=False, n_episodes=args.episodes_per_epoch,\
                        env_type='orca', traj_path=init_traj_path, env_mods=mods, model=model, t_fex=tf, pred_head=ph, suffix=n_suffix, notebook=False)
            results_per_epoch.append((i, res))        
            
        final_results = {'env_mods' : mods, 'init_results' : results, 'result_per_epoch' : results_per_epoch}
        torch.save(final_results, os.path.join(env_dir, 'results.pth'))    



if __name__ == "__main__":
    main()