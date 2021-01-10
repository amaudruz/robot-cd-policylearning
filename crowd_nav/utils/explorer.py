import logging
import copy
import statistics
import torch
from crowd_nav.utils.frames import FrameStack, latest_frame
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import JointState
from tqdm import tqdm, tqdm_notebook


def obs_to_frame(ob):
    frame = [[human.px, human.py, human.vx, human.vy] for human in ob]
    return frame

class Explorer(object):
    def __init__(self, env, robot, device, num_frames, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.num_frames = num_frames
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False, noise_explore=0.0, progressbar=True, notebook=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        if imitation_learning: human_obsv = []

        iter = range(k)
        if progressbar :
            if notebook :
                iter = tqdm_notebook(iter, leave=False)
            else :   
                iter = tqdm(iter, leave=False)    
        
        for i in iter:
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            scene = []

            #print(self.robot.policy)

            while not done:
                
                # infer action from policy
                if 'RNN' in self.robot.policy.name:
                    action = self.robot.act(ob)
                elif self.num_frames == 1:
                    action = self.robot.act(ob)
                else:
                    last_ob = latest_frame(ob, self.num_frames)
                    action = self.robot.act(last_ob)
                actions.append(action)
                #print(self.target_policy.name)
                # state append
                if imitation_learning:
                    if 'RNN' in self.target_policy.name:
                        # TODO: enumerate and test differenct combinations of policies
                        joint_state = JointState(self.robot.get_full_state(), ob)
                    elif 'SAIL' in self.target_policy.name or 'SARL' in self.target_policy.name :
                        
                        joint_state = JointState(self.robot.get_full_state(), ob)
                    else:
                        joint_state = self.robot.policy.last_state
                    last_state = self.target_policy.transform(joint_state)
                    states.append(last_state)
                    scene.append(obs_to_frame(ob))
                    #print(last_state)
                else:
                    states.append(self.robot.policy.last_state)

                # env step
                if imitation_learning and noise_explore > 0.0:
                    noise_magnitude = noise_explore * 2.0
                    action = ActionXY(action[0] + torch.rand(1).sub(0.5) * noise_magnitude, action[1] + torch.rand(1).sub(0.5) * noise_magnitude) if self.robot.policy.kinematics == 'holonomic' else ActionRot(action[0] + torch.rand(1).sub(0.5) * noise_magnitude, action[1] + torch.rand(1).sub(0.5) * noise_magnitude)

                ob, reward, done, info = self.env.step(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if imitation_learning: human_obsv.append(torch.FloatTensor(scene))

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                self.update_memory(states, actions, rewards, imitation_learning)

            # episode result
            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            # if i % 100 == 0 and i > 0:
            #     logging.info("{:<5} Explore #{:d} / {:d} episodes".format(phase.upper(), i, k))

        # episodes statistics
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {} success: {:.2f}, collision: {:.2f}, nav time: {:.2f}, reward: {:.4f} +- {:.4f}'.
                 format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, statistics.mean(cumulative_rewards), (statistics.stdev(cumulative_rewards) if len(cumulative_rewards) > 1 else 0.0)))
        print('{:<5} {} success: {:.2f}, collision: {:.2f}, nav time: {:.2f}, reward: {:.4f} +- {:.4f}'.
                 format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, statistics.mean(cumulative_rewards), (statistics.stdev(cumulative_rewards) if len(cumulative_rewards) > 1 else 0.0)))

        if phase in ['val', 'test']:
            total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
            logging.info('Frequency of being in danger: %.2f', too_close / total_time)

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        if imitation_learning:
            return human_obsv

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):

            cumulative_reward = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                     * (1 if t >= i else 0) for t, reward in enumerate(rewards)]) 
            value = torch.tensor([cumulative_reward], dtype=torch.float32).to(self.device)

            action = torch.tensor([actions[i][0], actions[i][1]], dtype=torch.float32).to(self.device)

            self.memory.push((state, action, value))


class ExplorerDs(object):
    def __init__(self, env, robot, device, num_frames, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.num_frames = num_frames
        self.target_model = None

        if self.num_frames > 1 :
            self.env = FrameStack(self.env, self.num_frames)

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)
    
    @staticmethod
    def transform_mult(state):
        """ Transform state object to tensor input of RNN policy
        """

        robot_state = torch.Tensor([state.self_state.px, state.self_state.py, state.self_state.vx, state.self_state.vy, state.self_state.gx, state.self_state.gy])

        num_human = len(state.human_states[0])
        memory_size = len(state.human_states)
        human_state = torch.empty([memory_size, num_human, 4])
        for k in range(num_human):
            for t in range(memory_size) :    
                human_state[t,k,0] = state.human_states[t][k].px
                human_state[t,k,1] = state.human_states[t][k].py
                human_state[t,k,2] = state.human_states[t][k].vx
                human_state[t,k,3] = state.human_states[t][k].vy

        return [robot_state, human_state]

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, traj_head=None,
                       print_failure=False, noise_explore=0.0, progressbar=False, output_info=False, notebook=False, 
                       print_info=False, return_states=False, suffix=None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        
        states_s = []

        if imitation_learning: human_obsv = []

        iter = range(k)
        if progressbar :
            if notebook :
                iter = tqdm_notebook(iter, leave=False)
            else :   
                iter = tqdm(iter, leave=False)      
        traj_acc_sum = 0
        for i in iter:
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            scene = []

            while not done:
                
                # infer action from policy
                if 'TRAJ' in self.robot.policy.name:
                    action = self.robot.act(ob)
                elif self.num_frames == 1:
                    action = self.robot.act(ob)
                else:
                    last_ob = latest_frame(ob, self.num_frames)
                    action = self.robot.act(last_ob)
                actions.append(action)

                # state append
                if imitation_learning:
                    # if 'RNN' in self.target_policy.name:
                    #     # TODO: enumerate and test differenct combinations of policies
                    #     joint_state = JointState(self.robot.get_full_state(), ob)
                    # elif 'SAIL' in self.target_policy.name:
                    #     joint_state = JointState(self.robot.get_full_state(), ob)
                    # else:
                    #     joint_state = self.robot.policy.last_state

                    joint_state = JointState(self.robot.get_full_state(), ob)
                    last_state = ExplorerDs.transform_mult(joint_state)
                    states.append(last_state)
                    scene.append(obs_to_frame(ob))
                else:
                    states.append(self.robot.policy.last_state)

                # env step
                if imitation_learning and noise_explore > 0.0:
                    noise_magnitude = noise_explore * 2.0
                    action = ActionXY(action[0] + torch.rand(1).sub(0.5) * noise_magnitude, action[1] + torch.rand(1).sub(0.5) * noise_magnitude) if self.robot.policy.kinematics == 'holonomic' else ActionRot(action[0] + torch.rand(1).sub(0.5) * noise_magnitude, action[1] + torch.rand(1).sub(0.5) * noise_magnitude)

                ob, reward, done, info = self.env.step(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if imitation_learning: human_obsv.append(torch.FloatTensor(scene))

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')
            traj_accuracy = 0
            if traj_head is not None :
                for i, ([rob, hum], human_feats) in enumerate(states) :
                    #next_human_pos = torch.cat([states[max(i, len(states)-1)][0][1] for i in range(len(states))], dim=1)
                    next_human_pos = []
                    for j in range(4) :
                        n_p = states[min(i+j+1, len(states)-1)][0][1][-1, :, :2]
                        #print(states[min(i+j+1, len(states)-1)][0][1])
                        next_human_pos.append(n_p)
                    with torch.no_grad() :
                        next_human_pos = torch.cat(next_human_pos, dim=1) 
                        next_human_pos_pred = traj_head(human_feats)

                        # print(hum[-1])
                        # print(next_human_pos)
                        # print(next_human_pos_pred)
                        # print('\n')
                        accuracy = ((next_human_pos - next_human_pos_pred)**2).mean().item()
                        traj_accuracy += accuracy
                    #print(traj_accuracy)
            traj_acc_sum += traj_accuracy

            if update_memory:
                self.update_memory(states, actions, rewards, imitation_learning)
            
            states_s.append(states)

            # episode result
            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            # if i % 100 == 0 and i > 0:
            #     logging.info("{:<5} Explore #{:d} / {:d} episodes".format(phase.upper(), i, k))
        traj_acc_sum = traj_acc_sum / sum(len(s) for s in states_s)
        # episodes statistics
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        if not print_info :
            extra_info = '' if episode is None else 'in episode {} '.format(episode)
            logging.info('{:<5} {} success: {:.2f}, collision: {:.2f}, nav time: {:.2f}, reward: {:.4f} +- {:.4f}'.
                    format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, statistics.mean(cumulative_rewards), (statistics.stdev(cumulative_rewards) if len(cumulative_rewards) > 1 else 0.0)))
        info = {'success' : success_rate, 'collision' : collision_rate, 'nav time' : avg_nav_time, 'reward': statistics.mean(cumulative_rewards)}
        if print_info :
            info = 'success: {:.4f},collision: {:.4f},nav time: {:.2f},reward: {:.4f}, timeout : {:.4f}'.\
            format(success_rate, collision_rate, avg_nav_time, statistics.mean(cumulative_rewards), timeout_rate)
            if traj_head is not None :
                info += ', traj accuracy : {:.4f}'.format(traj_acc_sum)
            if suffix is not None :
                info = suffix + info
            print(info)
        # if phase in ['val', 'test']:
        #     total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
        #     logging.info('Frequency of being in danger: %.2f', too_close / total_time)

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
        
        info_dic = {'success' : success_rate, 'collision' :collision_rate, 'nav time' : avg_nav_time,
                    'reward' :statistics.mean(cumulative_rewards), 'traj accuracy' : traj_acc_sum} 
        
        if output_info :
            if return_states :
                return info_dic, states_s
            else :
                return info_dic
        
        if imitation_learning:
            return human_obsv

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):

            cumulative_reward = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                     * (1 if t >= i else 0) for t, reward in enumerate(rewards)]) 
            value = torch.tensor([cumulative_reward], dtype=torch.float32).to(self.device)

            action = torch.tensor([actions[i][0], actions[i][1]], dtype=torch.float32).to(self.device)

            self.memory.push((state, action, value))
