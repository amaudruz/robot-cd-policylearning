
import torch
import sys
sys.path.append("..")
sys.path.append('/home/louis/Documents/Master_project/social-nce')

import configparser
import os
import torch

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.crowd_sim_sf import CrowdSim_SF
from crowd_sim.envs.utils.robot import Robot


layout_names = ['circle_crossing', 'square_crossing', 'cm_hall', 'cm_hall_oneway', 'line', 'line-td', 'tri-td',  'mixed']


# def create_env(gpu=False, type='sarl_mem', weights_path=None, policy_type='orca', expl=False) :
#     device = 'cuda' if gpu else 'cpu'
#     configs_path = '../configs/'

#     # configure policy
#     policy = policy_factory[type]()
#     policy_config = configparser.RawConfigParser()
#     policy_config.read(configs_path + 'policy.config')
#     policy.configure(policy_config)
#     policy.set_device(device)

#     if weights_path :
#         # getting trained weights
#         policy.model.load_state_dict(torch.load(weights_path))

#     # domain/env config (initially same as used in training)
#     env_config = configparser.RawConfigParser()
#     env_config.read(configs_path + 'env.config')
#     if policy_type == 'orca' : 
#        env = CrowdSim()
#     else : 
#         env = CrowdSim_SF()
#     env.configure(env_config)
#     robot = Robot(env_config, 'robot') if  type=='sarl' else RobotMEM(env_config, 'robot') 
#     robot.set_policy(policy)
#     env.set_robot(robot)

#     memory = ReplayMemory(100000)
#     explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy) if type=='sarl' else ExplorerMEM(env, robot, device, memory, policy.gamma, target_policy=policy)

#     policy.set_env(env)
#     robot.set_policy(policy)

#     if expl :
#         return env, robot, policy, explorer
    
#     return env, robot, policy

def modify_env_params(env, n_humans=None, layout=None ,mods=None) :
    if n_humans :
        env.human_num = n_humans
    if layout :
        env.test_sim = layout
    if mods :
        env.modify_domain(mods)


