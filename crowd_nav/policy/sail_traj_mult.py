import torch
import torch.nn as nn
import configparser

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.multi_human_rl import MultiHumanPolicy
from crowd_nav.policy.sail_traj import TrajFeatureExtractor

from collections import OrderedDict

from crowd_nav.utils.transform import MultiAgentTransform

class SAILTRAJMULT(MultiHumanPolicy):
    def __init__(self, traj_model_path=None):
        super().__init__()
        self.name ='SAILTRAJMULT'
        self.traj_model_path = traj_model_path

    def configure(self, config, max_obs=5, n_envs=2):
        self.set_common_parameters(config)
        self.multiagent_training = config.getboolean('sspn', 'multiagent_training')
        self.model = ExtendedNetworkTrajMult(config.getint('sspn', 'human_num'), max_obs=max_obs, n_envs=n_envs)
        if self.traj_model_path is not None :
             self.model.forecast_network.load_state_dict(torch.load(self.traj_model_path)['state_dict'])

    def predict(self, state):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        self.last_state = self.transform(state)
        #print(self.last_state)

        action, feats = self.model(self.last_state[0].unsqueeze(0), self.last_state[1].unsqueeze(0))

        action = action.squeeze()
        action = ActionXY(action[0].item(), action[1].item()) if self.kinematics == 'holonomic' else ActionRot(action[0].item(), action[1].item())

        self.last_state = self.last_state, feats
        return action

    def transform(self, state):
        """ Transform state object to tensor input of RNN policy
        """

        robot_state = torch.Tensor([state.self_state.px, state.self_state.py, state.self_state.vx, state.self_state.vy, state.self_state.gx, state.self_state.gy])

        num_frames = len(state.human_states)
        num_human = len(state.human_states[0])
        human_state = torch.empty([num_frames,num_human, 4])
        for i in range(num_frames) :
            for k in range(num_human):
                human_state[i,k,0] = state.human_states[i][k].px
                human_state[i,k,1] = state.human_states[i][k].py
                human_state[i,k,2] = state.human_states[i][k].vx
                human_state[i,k,3] = state.human_states[i][k].vy
        #print(human_state, robot_state)

        return [robot_state, human_state]

from crowd_nav.policy.policy_factory import policy_factory

def mult_to_simple(state_dict) :
    policy_p = 'configs/policy.config'
    policies = []
    for i in range(2) :
        policy = policy_factory['sail_traj_simple']()
        policy_config = configparser.RawConfigParser()
        policy_config.read(policy_p)
        policy.configure(policy_config)
            
        o = OrderedDict() 
        for name, tens in state_dict.items() :
            if not name.startswith('trajectory_fext') :
                o[name] = tens
            else :
                ns = name.split('.')
                if int(ns[1]) == i :
                    o['.'.join(['trajectory_fext'] + ns[2:])] = tens
        policy.model.load_state_dict(o)
        policies.append(policy)
    return policies
def get_heads(path) :
    return torch.load(path).named_children()
    
     
class ExtendedNetworkTrajMult(nn.Module):
    def __init__(self, num_human, embedding_dim=64, hidden_dim=64, local_dim=32, forecast_hidden_dim=32, forecast_emb_dim=16, max_obs=5,
                n_envs=2):
        super().__init__()
        self.num_human = num_human
        self.transform = MultiAgentTransform(num_human)
        self.max_obs=max_obs
        self.n_envs = n_envs

        self.robot_encoder = nn.Sequential(
            nn.Linear(4, local_dim),
            nn.ReLU(inplace=True),
            nn.Linear(local_dim, local_dim),
            nn.ReLU(inplace=True)
        )

        
        self.human_encoder = nn.Sequential(
            nn.Linear(4*self.num_human, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.human_head = nn.Sequential(
            nn.Linear(hidden_dim, local_dim),
            nn.ReLU(inplace=True)
        )

        self.joint_embedding = nn.Sequential(
            nn.Linear(local_dim + local_dim, embedding_dim),

            nn.ReLU(inplace=True)
        )

        self.pairwise = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1)
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True)
        )

        trajectory_fexts = [TrajFeatureExtractor(self.transform, self.human_encoder, self.human_head, n_heads=4, max_obs=self.max_obs) for i in range(self.n_envs)]
        self.trajectory_fexts = nn.ModuleList(trajectory_fexts)
        self.planner = nn.Linear(hidden_dim, 2)

    def forward(self, robot_state, crowd_obsv, aux_task=''):

        bs, n_envs, _, n_humans, _ = crowd_obsv.shape

        assert(n_envs==self.n_envs)

        if len(robot_state.shape) < 2: 
            robot_state = robot_state.unsqueeze(0)
            crowd_obsv = crowd_obsv.unsqueeze(0)

        emb_human = torch.stack([tf(crowd_obsv[:, i]) for i, tf in enumerate(self.trajectory_fexts)], dim=1)

        emb_human = emb_human.reshape(bs*n_envs, n_humans, -1)
        robot_state = robot_state.reshape(bs*n_envs, -1)

        # preprocessing
        emb_robot = self.robot_encoder(robot_state[:,:4])
        
        # emb_human = torch.cat([emb_human, emb_human_traj], dim=-1)
        emb_concat = torch.cat([emb_robot.unsqueeze(1).repeat(1,self.num_human,1), emb_human], axis=2)

        # embedding
        emb_pairwise = self.joint_embedding(emb_concat)

        # pairwise
        feat_pairwise = self.pairwise(emb_pairwise)

        # attention
        logit_pairwise = self.attention(emb_pairwise)
        score_pairwise = nn.functional.softmax(logit_pairwise, dim=1)

        # crowd
        feat_crowd = torch.sum(feat_pairwise * score_pairwise, dim=1)

        # planning
        reparam_robot_state = torch.cat([robot_state[:,-2:] - robot_state[:,:2], robot_state[:,2:4]], axis=1)
        feat_task = self.task_encoder(reparam_robot_state)

        feat_joint = self.joint_encoder(torch.cat([feat_task, feat_crowd], axis=1))
        action = self.planner(feat_joint)

        action = action.reshape(bs, n_envs, -1)
        emb_human = emb_human.reshape(bs, n_envs, n_humans, -1)

        if aux_task == 'contrastive' :
            return action, feat_joint
        return action, emb_human




 # bs, n_frames, n_humans, dim = crowd_obsv.shape

        
        # last_frame = copy.deepcopy(crowd_obsv[:, -1, :, :])
        # human_state_last_frame = self.transform.transform_frame(last_frame)
        # feat_human_last_frame = self.human_encoder(human_state_last_frame)
        # emb_human_last_frame = self.human_head(feat_human_last_frame)

        # #print(emb_human_last_frame)
        # #print('---------------------------------')
        

        # human_state_frames = self.transform.transform_frame(crowd_obsv.reshape(n_frames*bs, n_humans, dim))
        # feat_human_frames = self.human_encoder(human_state_frames)
        # #feat_human_frames = feat_human_frames.reshape(n_frames, bs, n_humans, -1).reshape(n_frames, bs*n_humans, -1)
        # feat_human_frames = feat_human_frames.reshape(bs, n_frames, n_humans, -1).transpose(1, 2).reshape(bs*n_humans, n_frames, -1)
        


        # # print('---------')
        # # print(feat_human_frames)
        # device = 'cpu'
        # if crowd_obsv.is_cuda :
        #     device = 'cuda'
        # position_embeddings = torch.tensor(range(n_frames))[None, :, None].repeat(bs*n_humans, 1, 1).to(device)
        # feat_human_frames_pos = torch.cat([feat_human_frames, position_embeddings], dim=-1)


        # logit_memory = self.history_attention(feat_human_frames_pos)
        # score_memory = nn.functional.softmax(logit_memory, dim=1)
        # print(score_memory)
        # # score_memory = torch.zeros_like(score_memory)
        # # for i in range(score_memory.shape[0]) :
        # #     score_memory[i, -1] = 1
        # emb_human = self.human_head(torch.sum(feat_human_frames * score_memory, dim=1).reshape(bs, n_humans, -1))
        