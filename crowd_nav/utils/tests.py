import configparser
import torch
import gym

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.crowd_sim_sf import CrowdSim_SF
from crowd_nav.utils.explorer import ExplorerDs
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.multi_envs import modify_env_params

def test(model_type='sail', visible=False, n_episodes=100, model_path=None, itera=range(4, 199, 5), env_mods=None) :
    res_list = []
    if model_path is None :
        if model_type != 'sail' :
            model_path = 'data/output/imitate-baseline-data-0.5-traj/'
        else :
            model_path = 'data/output/imitate-baseline-data-0.5-notraj/'
    for i in itera :
        s = str(i) if i > 10 else '{}'.format(i)
        res = test_model(model_path + 'policy_net_{}.pth'.format(s), model_type=model_type, visible=visible, n_episodes=n_episodes, env_mods=env_mods)
        res_list.append((i, res))
    return res_list

def test_model(m_p, model_type, visible=False, n_episodes=5000, model=None, env_type='orca', traj_path=None, traj_length=5, env_mods=None,
                layout='circle_crossing', return_exp=False, t_fex=None, pred_head=None, notebook=True, suffix=None) :
    policy_p = 'configs/policy.config'
    if not visible : 
        env_p = 'configs/env.config'
    else :
        env_p = 'configs/env_visible.config'
    policy_type = model_type
    
    # configure policy
    policy = policy_factory[policy_type]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_p)
    policy.configure(policy_config)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_p)
    if env_type == 'orca' :
        env = gym.make('CrowdSim-v0')
    else :
        assert(env_type == 'socialforce')
        env = CrowdSim_SF()
    env.configure(env_config)
    modify_env_params(env, layout=layout, mods=env_mods)

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    policy.set_env(env)

    policy.set_env(env)
    policy.set_phase('val')
    policy.set_device('cpu')

    prediction_head = None
    if traj_path is not None :
        #prediction_head = ProjHead(feat_dim=32, hidden_dim=16*max(int(traj_length/2), 1), head_dim=2*traj_length)
        prediction_head = torch.load(traj_path).to('cpu')
    if pred_head is not None :
        prediction_head = pred_head

    if model is None :
        policy.model.load_state_dict(torch.load(m_p))
    else :
        policy.model = model

    if t_fex is not None :
        policy.model.trajectory_fext = t_fex

    if model_type == 'sail' :
        explorer = ExplorerDs(env, robot, 'cpu', 1, gamma=0.9)
    else :
        explorer = ExplorerDs(env, robot, 'cpu', 5, gamma=0.9)
    explorer.robot = robot

    res, exp = explorer.run_k_episodes(n_episodes, 'test', progressbar=True, 
                                       output_info=True, notebook=notebook, print_info=True, 
                                       traj_head=prediction_head, return_states=True, suffix=suffix)
    if return_exp :
        return res, exp 
    return res