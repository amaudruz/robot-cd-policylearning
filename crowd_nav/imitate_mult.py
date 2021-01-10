from posix import listdir
from statistics import mode
import sys
from typing import Container 
sys.path.append('..')
sys.path.append('home/louis/Documents/Master_project/social-nce')
import logging
import argparse
import configparser
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook

import torch 
import os
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pickle
from utils.dataset import ImitDataset, ImitDatasetTraj, ImitDatasetTrajMult
from policy.sail_traj import ExtendedNetworkTraj
from policy.policy_factory import policy_factory
import configparser
import gym
import sys
sys.path.append('..')
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.crowd_sim_sf import CrowdSim_SF
from crowd_nav.utils.explorer import ExplorerDs

from crowd_nav.policy.sail_traj_mult import mult_to_simple

from crowd_nav.utils.pretrain import freeze_model, trim_model_dict
from crowd_nav.utils.dataset import ImitDataset, split_dataset, ImitDatasetTraj
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.configure import config_log, config_path
from crowd_nav.snce.contrastive import SocialNCE
from crowd_nav.snce.model import ProjHead, SpatialEncoder, EventEncoder
from crowd_nav.traj.unipred import UniPred
from crowd_nav.traj.trajpred import TrajPred, TrajPredFF, TrajPredFFMult
from crowd_nav.utils.multi_envs import modify_env_params

torch.manual_seed(2020)

def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='sail')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--traj_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_obs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--visibility', type=str, default='invisible', choices=['visible', 'invisible'])
    parser.add_argument('--trajectory_weight', type=float, default=0.5)
    parser.add_argument('--trajectory_length', type=float, default=3)
    parser.add_argument('--uni_weight', type=float, default=0.5)
    parser.add_argument('--uni_length', type=float, default=3)
    parser.add_argument('--contrast_sampling', type=str, default='social')
    parser.add_argument('--contrast_weight', type=float, default=1.0)
    parser.add_argument('--contrast_horizon', type=int, default=4)
    parser.add_argument('--contrast_temperature', type=float, default=0.2)
    parser.add_argument('--contrast_range', type=float, default=2.0)
    parser.add_argument('--contrast_nboundary', type=int, default=0)
    parser.add_argument('--ratio_boundary', type=float, default=0.5)
    parser.add_argument('--percent_label', type=float, default=0.5)
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--length_pred', type=int, default=1)
    parser.add_argument('--skip_pred', type=int, default=1)
    parser.add_argument('--model_file', type=str, default="")
    parser.add_argument('--output_dir', type=str, default='data/output/imitate')
    parser.add_argument('--memory_dir', type=str, default='data/demonstrate')
    parser.add_argument('--freeze', default=False, action='store_true')
    parser.add_argument('--predict', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--data_sample', default=1, type=float)
    args = parser.parse_args()
    return args

def build_policy(args):
    """
    Build navigation policy
    """
    if args.traj_path is not None :
        assert(args.policy == 'sail_traj')
        policy = policy_factory[args.policy](args.traj_path)    
    else :
        policy = policy_factory[args.policy]()
    if not policy.trainable:
        raise Exception('Policy has to be trainable')
    if args.policy_config is None:
        raise Exception('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config, n_envs=args.n_envs)
    return policy

def set_loader(args, device):
    """
    Set Data Loader
    """
    enviorment_names = os.listdir(args.memory_dir)
    assert(len(enviorment_names) == args.n_envs)
    data = []
    logging.info('Model uses {} enviormements : {}'.format(len(enviorment_names), enviorment_names))
    for env_name in enviorment_names :
        env_path = os.path.join(args.memory_dir, env_name)
        demo_file = os.path.join(env_path, 'data_imit_mem.pt')
        logging.info('Load data from %s', demo_file)
        data_imit = torch.load(demo_file)
        data.append(data_imit)
    
    contrast = args.auxiliary_task == 'contrastive'
    if contrast :
        horizon = args.contrast_horizon
    else :
        horizon = 10
    dataset_imit = ImitDatasetTrajMult(data, None, device, horizon=horizon, sample=args.data_sample, contrast=contrast)

    validation_split = 0.04
    train_loader, valid_loader = split_dataset(dataset_imit, args.batch_size, args.percent_label, validation_split)
    return train_loader, valid_loader

def most_recent_model(args) :
    models_n = [s for s in os.listdir(args.output_dir) if s.startswith('policy_net_')]
    models_n = [(int(s.split('_')[-1].split('.')[0]), s) for s in models_n]

    ep, m = sorted(models_n, key=lambda x : x[0], reverse=True)[0]
    


def set_model(args, device, c=False):
    """
    Set policy network
    """
    policy = build_policy(args)
    policy.set_device(device)
    policy_net = policy.get_model().to(device)

    # if c :
    #     model_p, last_epoch = most_recent_model(args)

    return policy_net

def load_model(policy_net, args, device):
    """
    Load pretrained model
    """
    pretrain = torch.load(args.model_file, map_location=device)
    if 'human_encoder.0.weight' in pretrain.keys():
        info = policy_net.load_state_dict(pretrain, strict=True)
    else:
        trim_state = trim_model_dict(pretrain, 'encoder')
        info = policy_net.human_encoder.load_state_dict(trim_state, strict=True)
    print(info)
    logging.info('Load pretrained model from %s', args.model_file)

    if args.freeze:
        freeze_model(policy_net, ['human_encoder'])

def train(policy_net, train_loader, criterion, tpred, optimizer, args, notebook=False):
    """
    Jointly train the policy net and contrastive encoders
    """
    policy_net.train()
    loss_sum_all, loss_sum_task, loss_sum_nce, loss_sum_tpred, loss_sum_upred= 0, 0, 0, 0, 0

    if notebook :
        iter = tqdm_notebook(train_loader, leave=False)
        iter.set_description('Train')
    else :
        iter = tqdm(train_loader, leave=False)
        iter.set_description('Train')

    for robot_states, human_states, action_targets, pos_seeds, neg_seeds in iter:
        if 'traj' in args.policy :
            human_states = human_states[:, -args.max_obs:, :, :]
        else :
            raise NotImplementedError()

        # main task
        outputs, features = policy_net(robot_states, human_states, aux_task=args.auxiliary_task)
        loss_task = criterion(outputs, action_targets)
        
        loss = loss_task + 0
        # contrastive task
        if args.auxiliary_task == 'traj':
            #print('traj')
            #assert('traj' in args.policy and (args.traj_weight > 0))
            last_humans = human_states.clone()
            if 'traj' in args.policy : 
                last_humans = last_humans[:, -1,]
            loss_tpred = tpred.loss(features, neg_seeds, human_states[:, :, :2])
            loss_sum_tpred += loss_tpred.item()
            loss += args.traj_weight*loss_tpred
        else :
            raise NotImplementedError()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        # print(loss_task)
        loss_sum_all += loss.data.item()
        loss_sum_task += loss_task.item()

    num_batch = len(train_loader)
    return loss_sum_all / num_batch, loss_sum_task / num_batch, loss_sum_tpred / num_batch

def validate(policy_net, valid_loader, criterion, tpred, args, notebook=False):
    """
    Evaluate policy net
    """
    policy_net.eval()
    loss_sum_all, loss_sum_task, loss_sum_nce, loss_sum_tpred, loss_sum_upred = 0, 0, 0, 0, 0

    with torch.no_grad():

        if notebook :
            iter = tqdm_notebook(valid_loader, leave=False)
            iter.set_description('Eval')
        else :
            iter = tqdm(valid_loader, leave=False)
            iter.set_description('Eval')
        for robot_states, human_states, action_targets, pos_seeds, neg_seeds in iter:
            if 'traj' in args.policy :
                human_states = human_states[:, -args.max_obs:, :, :] 
            else :
                raise NotImplementedError()
            if robot_states.shape[0] < 50 : continue
            outputs, features = policy_net(robot_states, human_states, aux_task=args.auxiliary_task)
            loss_task = criterion(outputs, action_targets)
            loss = loss_task + 0
            
            if args.auxiliary_task == 'traj':
                #print('traj')
                #assert('traj' in args.policy and (args.traj_weight > 0))
                last_humans = human_states.clone()
                if 'traj' in args.policy : 
                    last_humans = last_humans[:, -1,]
                loss_tpred = tpred.loss(features, neg_seeds, last_humans[:, :, :2])
                loss_sum_tpred += loss_tpred.item()
                loss += args.traj_weight*loss_tpred
            else :
                raise NotImplementedError()
            
           # print(loss, loss_task)
            loss_sum_all += loss.data.item()
            loss_sum_task += loss_task.item()

    num_batch = len(valid_loader)
    return loss_sum_all / num_batch, loss_sum_task / num_batch, loss_sum_tpred / num_batch


def imitate(args_change=None, return_losses=True, r=False) :
    args = ARGS()
    if args_change :
        for attr, val in args_change.items() :
            setattr(args, attr, val)
    # config
    suffix = ""
    if args.auxiliary_task == 'contrastive':
        suffix += "-{}-data-{:.1f}-weight-{:.1f}-horizon-{:d}-temperature-{:.2f}-nboundary-{:d}".format(args.contrast_sampling, args.percent_label, args.contrast_weight, args.contrast_horizon, args.contrast_temperature, args.contrast_nboundary)
        if args.contrast_nboundary > 0:
            suffix += "-ratio-{:.2f}".format(args.ratio_boundary)
        if args.contrast_sampling == 'local':
            suffix += "-range-{:.2f}".format(args.contrast_range)
    elif args.auxiliary_task == 'traj' :
        suffix += "-trajpred-{:.2f}-weight-{}to{}-length".format(args.traj_weight, args.traj_start + 1,args.traj_start+ args.traj_length)
    elif args.auxiliary_task == 'uni' :
        suffix += "-unipred-{:.2f}-weight-{}-length".format(args.uni_weight, args.uni_length)
    else :
        suffix += "-baseline-data-{:.1f}".format(args.percent_label)
    suffix += "-traj" if 'traj' in args.policy else "-notraj"
    continu = config_path(args, suffix)
    config_log(args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # dataset
    train_loader, valid_loader = set_loader(args, device)

    # model
    policy_net= set_model(args, device, c=continu)
    
    prediction_head = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2).to(device)
    
    # unipred and traj_pre
    if args.auxiliary_task == 'traj' :
        prediction_head = nn.ModuleList([ProjHead(feat_dim=32, hidden_dim=16*max(int(args.traj_length/2), 1), head_dim=2*args.traj_length).to(device) for i in range(args.n_envs)])
    else :
        raise NotImplementedError()

    # pretrain
    if os.path.exists(args.model_file):
        load_model(policy_net, args, device)

    logging.info('Auxiliary task {}'.format(args.auxiliary_task))

    # optimize
    param = list(policy_net.parameters())  + list(prediction_head.parameters()) 
    optimizer = optim.Adam(param, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, threshold=0.01, cooldown=20, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()
    
    # Auxiliary task modules
    tpred = TrajPredFFMult(prediction_head, pred_length=args.traj_length, pred_start=args.traj_start)

    # loop
    train_losses = []
    val_losses = []
    for epoch in range(args.num_epoch):

        train_loss_all, train_loss_task, train_loss_tpred = train(policy_net, train_loader, criterion, tpred, optimizer, args, notebook=True)
        eval_loss_all, eval_loss_task, eval_loss_tpred = validate(policy_net, valid_loader, criterion, tpred, args, notebook=True)
        
        train_losses.append((train_loss_all, train_loss_task, train_loss_tpred))
        val_losses.append((eval_loss_all, eval_loss_task, eval_loss_tpred))

        #print(train_loss_all, train_loss_task, train_loss_nce, train_loss_tpred, train_loss_upred )

        #print(eval_loss_all, eval_loss_task)
        
        scheduler.step(train_loss_all)      # (optional) learning rate decay once training stagnates
        print("Epoch #{}: loss = ({:.4f}, {:.4f}), task = ({:.4f}, {:.4f}), tpred = ({:.4f}, {:.4f})".format(epoch, train_loss_all, eval_loss_all, train_loss_task, eval_loss_task, train_loss_tpred, eval_loss_tpred))
        
        if epoch % args.save_every == (args.save_every - 1):
            torch.save(policy_net.state_dict(), os.path.join(args.output_dir, 'policy_net_{:02d}.pth'.format(epoch)))
            if args.traj_weight > 0 :
                torch.save(prediction_head, os.path.join(args.output_dir, 'prediction_head_{:02d}.pth'.format(epoch)))
            elif args.uni_weight > 0 : 
                torch.save(prediction_head, os.path.join(args.output_dir, 'prediction_head_{:02d}.pth'.format(epoch)))

    torch.save(policy_net.state_dict(), os.path.join(args.output_dir, 'policy_net.pth'))
    torch.save((train_losses, val_losses), os.path.join(args.output_dir, 'losses.pth'))
    if return_losses :
        return train_losses, val_losses


class ARGS() :
    def __init__(self) :
        self.policy = 'sail_traj_mult'
        self.policy_config = 'configs/policy.config'
        self.traj_path = None
        self.batch_size = 128
        self.max_obs = 5
        self.lr = 1e-3
        self.visibility = 'invisible'
        self.contrast_sampling = 'event'
        self.contrast_weight = 0.4
        self.contrast_horizon = 4
        self.contrast_temperature= 0.2
        self.contrast_range = 2.0
        self.contrast_nboundary = 0
        self.ratio_boundary = 0.5
        self.percent_label = 0.5
        self.num_epoch = 200
        self.save_every = 5
        self.length_pred = 1
        self.skip_pred = 1
        self.model_file = ""
        self.output_dir = 'data/output_mul/imitate'
        self.memory_dir = 'data/demonstrate_mul'
        self.freeze = False
        self.predict = False
        self.gpu = False
        self.data_sample = 1
        
        self.traj_weight = 2.5
        self.traj_length = 4
        self.traj_start = 0
        self.uni_weight = 0.4
        self.uni_length = 1

        self.auxiliary_task='traj'

        self.max_ds_size = 100000
        self.n_envs = 2

def test(model_path, visible=False, n_episodes=100, itera=range(4, 199, 5)) :
    res_list = []
    for i in itera :
        res = test_model(model_path + 'policy_net_{:02d}.pth'.format(i), visible=visible, n_episodes=n_episodes)
        res_list.append((i, res))
        print('\n')
    return res_list

def test_model(m_p, visible=False, n_episodes=5000, model=None, env_type='orca', traj_path=None, traj_length=5, env_mods=None,
                layout='circle_crossing', return_exp=False, t_fex=None, pred_head=None, notebook=True, suffix=None) :
    policy_p = 'configs/policy.config'
    if not visible : 
        env_p = 'configs/env.config'
    else :
        env_p = 'configs/env_visible.config'
    
    results, experience = [], []
    policies = mult_to_simple(torch.load(m_p))
    #env_mods = {'safety_space' : 0.03, 'time_horizon' : 1}
    for i, policy in enumerate(policies) :
        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(env_p)
        if env_type == 'orca' :
            env = gym.make('CrowdSim-v0')
        else :
            assert(env_type == 'socialforce')
            env = CrowdSim_SF()
        env.configure(env_config)
        
        env_mods=None
        if i == 1 :
            env_mods = {'safety_space' : 0.03, 'time_horizon' : 1}
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

        # if model is None :
        #     policy.model.load_state_dict(torch.load(m_p))
        # else :
        #     policy.model = model

        if t_fex is not None :
            policy.model.trajectory_fext = t_fex
        
        explorer = ExplorerDs(env, robot, 'cpu', 5, gamma=0.9)
        explorer.robot = robot

        res, exp = explorer.run_k_episodes(n_episodes, 'test', progressbar=True, 
                                        output_info=True, notebook=notebook, print_info=True, 
                                        traj_head=prediction_head, return_states=True, suffix=suffix)
        results.append(res)
        experience.append(exp)
    
    if return_exp :
        return results, experience 
    return results



# def main():
#     args = parse_arguments()
#     print(args)

#     # config
#     suffix = ""
#     if args.contrast_weight > 0:
#         suffix += "-{}-data-{:.1f}-weight-{:.1f}-horizon-{:d}-temperature-{:.2f}-nboundary-{:d}".format(args.contrast_sampling, args.percent_label, args.contrast_weight, args.contrast_horizon, args.contrast_temperature, args.contrast_nboundary)
#         if args.contrast_nboundary > 0:
#             suffix += "-ratio-{:.2f}".format(args.ratio_boundary)
#         if args.contrast_sampling == 'local':
#             suffix += "-range-{:.2f}".format(args.contrast_range)
#     if args.trajectory_weight > 0 :
#         suffix += "-trajpred-{}-weight-{}-length".format(args.trajectory_weight, args.trajectory_length)
#     if args.uni_weight > 0 :
#         suffix += "-unipred-{}-weight-{}-length".format(args.uni_weight, args.uni_length)
#     if args.contrast_weight == 0 and args.trajectory_weight == 0 and args.uni_weight == 0 :
#         suffix += "-baseline-data-{:.1f}".format(args.percent_label)
#     suffix += "-traj" if 'traj' in args.policy else "-notraj"
#     config_path(args, suffix)
#     config_log(args)

#     device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
#     logging.info('Using device: %s', device)

#     # dataset
#     train_loader, valid_loader = set_loader(args, device)

#     # model
#     policy_net = set_model(args, device)

#     # contrastive
#     projection_head = ProjHead(feat_dim=64, hidden_dim=16, head_dim=8).to(device)
#     if args.contrast_sampling == 'event':
#         encoder_sample = EventEncoder(hidden_dim=8, head_dim=8).to(device)
#     else:
#         encoder_sample = SpatialEncoder(hidden_dim=8, head_dim=8).to(device)
    
#     # unipred
#     prediction_head = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2).to(device)
    
#     # traj_pred
#     # position_head = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2*args.trajectory_length).to(device)
#     position_head_traj = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2).to(device)
#     position_emb = ProjHead(feat_dim=2, hidden_dim=16, head_dim=24).to(device)
#     decoder = torch.nn.LSTMCell(24, 32).to(device)

#     # pretrain
#     if os.path.exists(args.model_file):
#         load_model(policy_net, args, device)

#     # optimize
#     # param = list(policy_net.parameters()) + list(projection_head.parameters()) + list(encoder_sample.parameters()) + list(prediction_head.parameters()) + \
#     #     list(position_head.parameters())
#     param = list(policy_net.parameters()) + list(projection_head.parameters()) + list(encoder_sample.parameters()) + list(prediction_head.parameters()) + \
#         list(position_head_traj.parameters()) + list(position_emb.parameters()) + list(decoder.parameters())
#     optimizer = optim.Adam(param, lr=args.lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, threshold=0.01, cooldown=20, min_lr=1e-5, verbose=True)
#     criterion = nn.MSELoss()
    
#     # Auxiliary task modules
#     nce = SocialNCE(projection_head, encoder_sample, args.contrast_sampling, args.contrast_horizon, args.contrast_nboundary, args.contrast_temperature, args.contrast_range, args.ratio_boundary)
#     upred = UniPred(prediction_head, args.uni_length)
#     # tpred = TrajPred(position_head, pred_length=args.trajectory_length)
#     tpred = TrajPred(position_head_traj, decoder, position_emb, pred_length=args.trajectory_length)

#     # loop
#     for epoch in range(args.num_epoch):

#         train_loss_all, train_loss_task, train_loss_nce, train_loss_tpred, train_loss_upred = train(policy_net, projection_head, encoder_sample, train_loader, criterion, nce, tpred, upred, optimizer, args)
#         eval_loss_all, eval_loss_task, eval_loss_nce, eval_loss_tpred, eval_loss_upred = validate(policy_net, projection_head, encoder_sample, valid_loader, criterion, nce, tpred, upred, args)

#         scheduler.step(train_loss_all)      # (optional) learning rate decay once training stagnates

#         if epoch % args.save_every == (args.save_every - 1):
#             logging.info("Epoch #%02d: loss = (%.4f, %.4f), task = (%.4f, %.4f), nce = (%.4f, %.4f), tpred = (%.4f, %.4f), upred = (%.4f, %.4f) ", epoch, train_loss_all, eval_loss_all, train_loss_task, eval_loss_task, train_loss_nce, eval_loss_nce, train_loss_tpred, eval_loss_tpred, train_loss_upred, eval_loss_upred)
#             torch.save(policy_net.state_dict(), os.path.join(args.output_dir, 'policy_net_{:02d}.pth'.format(epoch)))

#     torch.save(policy_net.state_dict(), os.path.join(args.output_dir, 'policy_net.pth'))

# if __name__ == '__main__':
#     try :
#         main()
#     except :
#         import pdb, traceback, sys
#         extype, value, tb = sys.exc_info()
#         traceback.print_exc()
#         pdb.post_mortem(tb)
#     #main()

