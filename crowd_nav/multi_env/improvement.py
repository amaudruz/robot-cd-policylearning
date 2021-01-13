import torch 
import os
import argparse
import configparser
import copy
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, tqdm_notebook

from crowd_nav.policy.sail_traj_mult import mult_to_simple
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.sail_traj_mult import get_heads, mult_to_simple
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

def test_traj_acc(ds, trajectory_fext, pred_head, batch_size=20, device='cpu', notebook=True) :
    dl = DataLoader(ds, batch_size=128, shuffle=True)
    if notebook :
        ite = tqdm_notebook(dl, leave=False)
    else :
        ite = tqdm(dl, leave=False)
    loss_sum = 0
    for human_states, future_pos in ite :
        with torch.no_grad() :
            human_states = human_states.to(device)
            future_pos = future_pos.to(device)

            human_feats = trajectory_fext(human_states)
            pred_pos = pred_head(human_feats)
            loss = ((pred_pos - future_pos)**2).mean()

            loss_sum += loss.item()
    epoch_loss = loss_sum / len(dl)
    return epoch_loss

def train_traj(ds, trajectory_fext, pred_head, n_epochs=1, batch_size=128, device='cpu', notebook=True, suffix='',
                print_every=5) :
    params = list(trajectory_fext.parameters()) + list(pred_head.parameters())
    opt = optim.Adam(params)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    models = []
    for e in range(n_epochs) :
        if notebook :
            ite = tqdm_notebook(dl, leave=False)
        else :
            ite = tqdm(dl, leave=False)
        
        loss_sum = 0
        for human_states, future_pos in ite :
            human_states = human_states.to(device)
            future_pos = future_pos.to(device)
            
            opt.zero_grad()
            human_feats = trajectory_fext(human_states)
            pred_pos = pred_head(human_feats)
            
            loss = ((pred_pos - future_pos)**2).mean()
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
        models.append((copy.deepcopy(trajectory_fext), copy.deepcopy(pred_head)))
        epoch_loss = loss_sum / len(dl)
        if e % print_every == 0 :
            print(suffix + 'Epoch {} loss {:.4f}'.format(e, epoch_loss))
    return models

def exp_to_ds(experience) :
    human_states_per_ep = [[h for (_, h), _ in ep] for ep in experience]
    ds = []
    for ep in human_states_per_ep :
        for i, h_state in enumerate(ep) :
            next_pos = []
            for j in range(4) :
                next_pos.append(ep[min(len(ep)-1, i+j+1)][-1, :, :2])
            next_pos = torch.cat(next_pos, dim=1)
            ds.append((h_state, next_pos))
    return ds

def main() :
   
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--mult', default=False, action='store_true')
    parser.add_argument('--compute_init_resutls', default=False, action='store_true')

    # training 
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--data_dir', type=str, default='data/improvement/imitate-trajpred-2.50-weight-1to4-length-traj')
    
    args = parser.parse_args()
    
    environement_dirs = os.listdir(args.data_dir)
    for i, env_dir in enumerate(environement_dirs) :
       
        env_dir = os.path.join(args.data_dir, env_dir)

        data = torch.load(os.path.join(env_dir, 'data.pth'))
        mods = data['env_mods']
        experience = data['experience']
        results = data['results']
        
        print('Environement {} : init reward : {:.4f}, init collision {:.4f}'.format(i, results['reward'], results['collision']))

        if not args.mult :
            t_fex = model_from_path(data['model_path']).trajectory_fext.to('cpu')
            pred_head = torch.load(data['traj_path']).to('cpu')
        else :
            policy = mult_to_simple(torch.load(args.policy_net_path))[0]
            t_fex = policy.model.trajectory_fext.to('cpu')
            pred_head = next(iter(get_heads(args.traj_fext_path)))[1].to('cpu')

        if args.compute_init_resutls and args.mult:
            raise NotImplementedError
        
        suffix = '  '
        ds = exp_to_ds(experience)
        init_traj_loss = test_traj_acc(ds, trajectory_fext=t_fex, pred_head=pred_head, batch_size=args.batch_size, notebook=False)
        print(suffix + 'init traj loss : {:.4f}'.format(init_traj_loss))
        models = train_traj(ds, t_fex, pred_head, device='cpu', n_epochs=args.num_epoch, batch_size=args.batch_size, notebook=False,
                            suffix=suffix)
        losses = [test_traj_acc(ds, tf, pre, notebook=False) for tf, pre in models]
        loss = {'init_traj_loss' : init_traj_loss, 'losses' : losses}
        torch.save(models, os.path.join(env_dir, 'models.pth'))    
        torch.save(loss, os.path.join(env_dir, 'losses.pth'))  


if __name__ == "__main__":
    main()