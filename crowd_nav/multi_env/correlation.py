from configparser import Error
import torch 
import numpy as np
import argparse
import os
import random
import pandas as pd
import seaborn as sns
import shutil

from crowd_nav.utils.tests import test_model

def pick_best_model(path) :
    results = torch.load(os.path.join(path, 'results.pth'))
    epoch, default_perf = max(results, key=lambda x : x[1]['reward'])
    model_path = os.path.join(path, 'policy_net_{:02d}.pth'.format(epoch))
    traj_path = os.path.join(path, 'prediction_head_{:02d}.pth'.format(epoch))
    return model_path, traj_path, default_perf

def plot_figs(df, output_dir) :
    figs_path = os.path.join(output_dir, 'figs')
    os.mkdir(figs_path)
    sns.set_style("darkgrid")

    plot = sns.scatterplot(
    data=df,
    x="trajectory_loss", y="collision"
    )
    plot.set(xlabel='Trajectory Loss', ylabel='Collision (%)')
    fig = plot.get_figure()
    fig.set_size_inches(15, 10)
    fig.savefig(os.path.join(figs_path, 'traj_coll_corr.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    sns.set(font_scale=2)
    plot = sns.heatmap(df.corr())
    fig = plot.get_figure()
    fig.set_size_inches(15, 10)
    fig.savefig(os.path.join(figs_path, 'corr.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='sail_traj')
    parser.add_argument('--output_dir', type=str, default='data/multi_env')
    parser.add_argument('--model_path', type=str, default='data/output/imitate-trajpred-2.50-weight-1to4-length-traj')
    parser.add_argument('--figs', default=False, action='store_true')
    
    parser.add_argument('--n_envs', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=200)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir) :
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, args.model_path.split('/')[-1])
    if os.path.exists(output_dir) :
        key = input('Path already exists, Overwrite ? (y/n) : ')
        if key == 'n' :
            raise Error
        else :
            shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    model_path, traj_path, default_perf = pick_best_model(args.model_path)

    n_tests=args.n_envs
    results = {'perfs' : [], 'model_path' : model_path, 'traj_path' : traj_path, 'default_perf' : default_perf}
    for i in  range(n_tests):
        if i % args.save_interval == 0 :
            torch.save(results, os.path.join(output_dir, 'results.pth'))
        env_mods = {'safety_space' : random.choice(np.linspace(0.01, 1, 100)), 
                    'neighbor_dist': random.choice(np.linspace(0, 6, 100)),
                    'time_horizon' : random.choice(np.linspace(0, 10, 10))}
        
        results['perfs'].append((env_mods, test_model(m_p=model_path, model_type=args.policy, visible=False, n_episodes=100,\
            env_type='orca', traj_path=traj_path, env_mods=env_mods, notebook=False, suffix='{} :'.format(i))))
    torch.save(results, os.path.join(output_dir, 'results.pth'))
    
    results = results['perfs']
    traj_accuracy = [r[1]['traj accuracy'] for r in results]
    collisions = np.asarray([1 - r[1]['success'] for r in results]) * 100
    env_m = {k : [] for k in results[0][0].keys()}
    for res in results :
        for k in env_m.keys() :
            env_m[k].append(res[0][k])
    env_m['collision'] = collisions
    env_m['trajectory_loss'] = traj_accuracy
    df = pd.DataFrame(env_m)
    torch.save(df, os.path.join(output_dir, 'results.csv'))

    if args.figs :
        plot_figs(df, output_dir)
        


    