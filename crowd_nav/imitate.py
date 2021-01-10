import logging
import argparse
import configparser
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook

import gym
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.crowd_sim_sf import CrowdSim_SF
from crowd_nav.utils.explorer import ExplorerDs

from crowd_nav.utils.pretrain import freeze_model, trim_model_dict
from crowd_nav.utils.dataset import ImitDataset, split_dataset, ImitDatasetTraj
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.configure import config_log, config_path
from crowd_nav.snce.contrastive import SocialNCE
from crowd_nav.snce.model import ProjHead, SpatialEncoder, EventEncoder
from crowd_nav.traj.unipred import UniPred
from crowd_nav.traj.trajpred import TrajPred, TrajPredFF
from crowd_nav.utils.multi_envs import modify_env_params
from crowd_nav.utils.imitate_utils import ARGS

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
    

    parser.add_argument('--auxiliary_task', type=str, choices=['contrastive', 'traj', 'uni'])

    # Human trajectory prediction 
    parser.add_argument('--traj_weight', type=float, default=0.5)
    parser.add_argument('--traj_length', type=float, default=3)
    parser.add_argument('--traj_start', type=float, default=0)

    # Human future position prediction
    parser.add_argument('--uni_weight', type=float, default=0.5)
    parser.add_argument('--uni_length', type=float, default=3)

    # Contrastive learning
    parser.add_argument('--contrast_sampling', type=str, default='social')
    parser.add_argument('--contrast_weight', type=float, default=1.0)
    parser.add_argument('--contrast_horizon', type=int, default=4)
    parser.add_argument('--contrast_temperature', type=float, default=0.2)
    parser.add_argument('--contrast_range', type=float, default=2.0)
    parser.add_argument('--contrast_nboundary', type=int, default=0)

    # Data
    parser.add_argument('--ratio_boundary', type=float, default=0.5)
    parser.add_argument('--percent_label', type=float, default=0.5)

    # Training
    parser.add_argument('--num_epoch', type=int, default=200)
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
    policy.configure(policy_config)
    return policy

def set_loader(args, device):
    """
    Set Data Loader
    """
    # if 'traj' in args.policy :
    #     demo_file = os.path.join(args.memory_dir, 'data_imit_mem.pt')
    # else :
    #     demo_file = os.path.join(args.memory_dir, 'data_imit.pt')
    demo_file = os.path.join(args.memory_dir, 'data_imit_mem.pt')
    logging.info('Load data from %s', demo_file)
    data_imit = torch.load(demo_file)

    # if 'traj' in args.policy :
    #     dataset_imit = ImitDatasetTraj(data_imit, None, device, horizon=args.contrast_horizon, sample=args.data_sample)
    #     #dataset_imit = ImitDatasetTraj.from_mult(data_imit, device, horizon=args.contrast_horizon)
    # else :
    #     dataset_imit = ImitDataset(data_imit, None, device, horizon=args.contrast_horizon, sample=args.data_sample)
    #     #dataset_imit = ImitDataset.from_mult(data_imit, device, horizon=args.contrast_horizon)

    contrast = args.auxiliary_task == 'contrastive'
    if contrast :
        horizon = args.contrast_horizon
    else :
        horizon = 10
    dataset_imit = ImitDatasetTraj(data_imit, None, device, horizon=horizon, sample=args.data_sample, contrast=contrast)

    validation_split = 0.02
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

def train(policy_net, projection_head, encoder_sample, train_loader, criterion, nce, tpred, upred, optimizer, args, notebook=False):
    """
    Jointly train the policy net and contrastive encoders
    """
    policy_net.train()
    projection_head.train()
    encoder_sample.train()
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
            human_states = human_states[:, -1, :, :]

        # main task
        outputs, features = policy_net(robot_states, human_states, aux_task=args.auxiliary_task)
        loss_task = criterion(outputs, action_targets)
        
        loss = loss_task + 0
        # contrastive task
        if args.auxiliary_task == 'contrastive' :
            assert(args.contrast_weight > 0)
            human_states_nce = human_states.clone()
            if 'traj' in args.policy : 
                human_states_nce = human_states_nce[:, -1, :, :]
            loss_nce = nce.loss(robot_states, human_states_nce, pos_seeds, neg_seeds, features)
            loss_sum_nce += loss_nce.item()
            loss += loss_nce * args.contrast_weight
        if args.auxiliary_task == 'traj':
            #print('traj')
            #assert('traj' in args.policy and (args.traj_weight > 0))
            last_humans = human_states.clone()
            if 'traj' in args.policy : 
                last_humans = last_humans[:, -1,]
            loss_tpred = tpred.loss(features, neg_seeds, human_states[:, :, :2])
            loss_sum_tpred += loss_tpred.item()
            loss += args.traj_weight*loss_tpred
        if args.auxiliary_task == 'uni':
            #print('uni')
            #assert('traj' in args.policy and (args.uni_weight > 0))
            loss_upred = upred.loss(features, neg_seeds)
            loss_sum_upred += loss_upred.item()
            loss += args.uni_weight*loss_upred

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        # print(loss_task)
        loss_sum_all += loss.data.item()
        loss_sum_task += loss_task.item()

    num_batch = len(train_loader)
    return loss_sum_all / num_batch, loss_sum_task / num_batch, loss_sum_nce / num_batch, loss_sum_tpred / num_batch, loss_sum_upred / num_batch

def validate(policy_net, projection_head, encoder_sample, valid_loader, criterion, nce, tpred, upred, args, notebook=False):
    """
    Evaluate policy net
    """
    policy_net.eval()
    projection_head.eval()
    encoder_sample.eval()
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
                human_states = human_states[:, -1, :, :]
            if robot_states.shape[0] < 50 : continue
            outputs, features = policy_net(robot_states, human_states, aux_task=args.auxiliary_task)
            loss_task = criterion(outputs, action_targets)
            loss = loss_task + 0
            
            if args.auxiliary_task == 'contrastive' :
                assert(args.contrast_weight > 0)
                loss_nce = nce.loss(robot_states, human_states, pos_seeds, neg_seeds, features)
                loss += loss_nce * args.contrast_weight
                loss_sum_nce += loss_nce.item()
            if args.auxiliary_task == 'traj':
                #print('traj')
                #assert('traj' in args.policy and (args.traj_weight > 0))
                last_humans = human_states.clone()
                if 'traj' in args.policy : 
                    last_humans = last_humans[:, -1,]
                loss_tpred = tpred.loss(features, neg_seeds, last_humans[:, :, :2])
                loss_sum_tpred += loss_tpred.item()
                loss += args.traj_weight*loss_tpred
            if args.auxiliary_task == 'uni':
                #print('uni')
                #assert('traj' in args.policy and (args.uni_weight > 0))
                loss_upred = upred.loss(features, neg_seeds)
                loss_sum_upred += loss_upred.item()
                loss += args.uni_weight*loss_upred
            
           # print(loss, loss_task)
            loss_sum_all += loss.data.item()
            loss_sum_task += loss_task.item()

    num_batch = len(valid_loader)
    return loss_sum_all / num_batch, loss_sum_task / num_batch, loss_sum_nce / num_batch, loss_sum_tpred / num_batch, loss_sum_upred / num_batch


def imitate(args_change=None, return_losses=True, r=False, notebook=True) :
    
    if notebook :
        args = ARGS()
        if args_change :
            for attr, val in args_change.items() :
                setattr(args, attr, val)
    else :
        args = parse_arguments()

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
    
    # contrastive
    projection_head = ProjHead(feat_dim=64, hidden_dim=16, head_dim=8).to(device)
    if args.contrast_sampling == 'event':
        encoder_sample = EventEncoder(hidden_dim=8, head_dim=8).to(device)
    else:
        encoder_sample = SpatialEncoder(hidden_dim=8, head_dim=8).to(device)
    
    prediction_head = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2).to(device)
    # unipred and traj_pre
    if args.auxiliary_task == 'uni' :
        prediction_head = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2).to(device)
    elif args.auxiliary_task == 'traj' :
        prediction_head = ProjHead(feat_dim=32, hidden_dim=16*max(int(args.traj_length/2), 1), head_dim=2*args.traj_length).to(device)
    # position_head_traj = ProjHead(feat_dim=32, hidden_dim=16, head_dim=2).to(device)
    # position_emb = ProjHead(feat_dim=2, hidden_dim=16, head_dim=24).to(device)
    # decoder = torch.nn.LSTMCell(24, 32).to(device)

    # pretrain
    if os.path.exists(args.model_file):
        load_model(policy_net, args, device)

    logging.info('Auxiliary task {}'.format(args.auxiliary_task))

    # optimize
    param = list(policy_net.parameters()) + list(projection_head.parameters()) + list(encoder_sample.parameters()) + list(prediction_head.parameters()) 
    # param = list(policy_net.parameters()) + list(projection_head.parameters()) + list(encoder_sample.parameters()) + list(prediction_head.parameters()) + \
    #     list(position_head_traj.parameters()) + list(position_emb.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(param, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, threshold=0.01, cooldown=20, min_lr=1e-5, verbose=True)
    criterion = nn.MSELoss()
    
    # Auxiliary task modules
    nce = SocialNCE(projection_head, encoder_sample, args.contrast_sampling, args.contrast_horizon, args.contrast_nboundary, args.contrast_temperature, args.contrast_range, args.ratio_boundary)
    upred = UniPred(prediction_head, args.uni_length)
    tpred = TrajPredFF(prediction_head, pred_length=args.traj_length, pred_start=args.traj_start)
    # tpred = TrajPred(position_head_traj, decoder, position_emb, pred_length=args.trajectory_length)


    # loop
    train_losses = []
    val_losses = []
    for epoch in range(args.num_epoch):

        train_loss_all, train_loss_task, train_loss_nce, train_loss_tpred, train_loss_upred = train(policy_net, projection_head, encoder_sample, train_loader, criterion, nce, tpred, upred, optimizer, args, notebook=notebook)
        eval_loss_all, eval_loss_task, eval_loss_nce, eval_loss_tpred, eval_loss_upred = validate(policy_net, projection_head, encoder_sample, valid_loader, criterion, nce, tpred, upred, args, notebook=notebook)
        
        train_losses.append((train_loss_all, train_loss_task, train_loss_nce, train_loss_tpred, train_loss_upred))
        val_losses.append((eval_loss_all, eval_loss_task, eval_loss_nce, eval_loss_tpred, eval_loss_upred))

        #print(train_loss_all, train_loss_task, train_loss_nce, train_loss_tpred, train_loss_upred )

        #print(eval_loss_all, eval_loss_task)
        
        scheduler.step(train_loss_all)      # (optional) learning rate decay once training stagnates
        logging.info("Epoch #{}: loss = ({:.4f}, {:.4f}), task = ({:.4f}, {:.4f}), nce = ({:.4f}, {:.4f}), tpred = ({:.4f}, {:.4f}), upred = ({:.4f}, {:.4f})".format(epoch, train_loss_all, eval_loss_all, train_loss_task, eval_loss_task, train_loss_nce, eval_loss_nce, train_loss_tpred, eval_loss_tpred, train_loss_upred, eval_loss_upred))
        
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

if __name__ == "__main__":
    imitate(notebook=False)


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