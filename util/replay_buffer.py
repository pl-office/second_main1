import copy
import torch
from util.utils import GetValueInputs, GetPolicyInputs

class ReplayBuffer():
   def __init__(self, params):
       self.mec_num = params.mec_num
       self.train_freq = params.train_freq
       self.train_time_slots = params.train_time_slots
       self.state_dim = params.state_dim
       self.obs_dim = params.obs_dim
       self.action_dim = params.action_dim
       self.gamma = params.gamma
       self.lamda = params.lamda
       
       self.ps = [0, 0]
       self.mec_obss = [[None for j in range(self.train_time_slots + 1)]
                                 for i in range(self.train_freq)]
       self.mec_acts = [[None for j in range(self.train_time_slots + 1)]
                                      for i in range(self.train_freq)]
       self.mec_act_logprobs = [[None for j in range(self.train_time_slots + 1)]
                                         for i in range(self.train_freq)]
       self.joint_reward = [[None for j in range(self.train_time_slots + 1)]
                                  for i in range(self.train_freq)]
       
   def store(self,mec_obss, mec_acts, mec_act_logprobs, joint_reward):
       # store sampled data
       self.mec_obss[self.ps[0]][self.ps[1]] = copy.copy(mec_obss)
       self.mec_acts[self.ps[0]][self.ps[1]] = copy.copy(mec_acts)
       self.mec_act_logprobs[self.ps[0]][self.ps[1]] = copy.copy(mec_act_logprobs)
       self.joint_reward[self.ps[0]][self.ps[1]] = copy.copy(joint_reward)
       
       # update positions
       if self.ps[1] == self.train_time_slots:
           self.ps[0] = (self.ps[0] + 1) % (self.train_freq)
       self.ps[1] = (self.ps[1] + 1) % (self.train_time_slots + 1)
       
   def get_training_data(self, value_net):
       '''GAE'''
       v_inputs = torch.zeros([self.train_freq, self.train_time_slots + 1, self.state_dim])
       for i in range(self.train_freq):
           for j in range(self.train_time_slots + 1):
               mec_obss = copy.copy(self.mec_obss[i][j])#15个任务
               inputs = GetValueInputs(mec_obss)
               v_inputs[i, j] = inputs
       
       with torch.no_grad():
           vs = value_net(v_inputs.reshape([-1, self.state_dim]))
       vs = vs.reshape([self.train_freq, self.train_time_slots + 1, 1])
       
       # [train_freq, train_time_slots, 1]
       rewards = torch.tensor(self.joint_reward, dtype = torch.float) \
                 [:, 0: self.train_time_slots].unsqueeze(-1)
       
       # [train_episodes, train_time_slots, 1]
       deltas = rewards + self.gamma * vs[:, 1: self.train_time_slots + 1] - \
                vs[:, 0: self.train_time_slots]
       gae = 0
       advs = torch.zeros([self.train_freq, self.train_time_slots, 1])
       for t in reversed(range(self.train_time_slots)):
           gae = deltas[:, t] + self.lamda * self.gamma * gae
           advs[:, t] = gae
       # [train_episodes, train_time_slots, 1]
       # 价值标签
       v_tags = advs + vs[:, 0: self.train_time_slots]
       # normalization
       advs = (advs - advs.mean()) / (advs.std() + 1e-5)
       
       '''training data - value network'''
       # [train_freq x train_time_slots, state_dim]
       #价值网络输入
       v_inputs = v_inputs[:, 0: self.train_time_slots].reshape([-1, self.state_dim])
       # [train_freq x train_time_slots, 1]
       #价值标签
       v_tags = v_tags.reshape([-1, 1])
       
       '''training data - policy networks'''
       p_inputs = torch.zeros([self.train_freq, self.train_time_slots, 
                               self.mec_num, self.obs_dim])
       acts = torch.zeros([self.train_freq, self.train_time_slots, 
                           self.mec_num, self.action_dim])
       act_logprobs = torch.zeros([self.train_freq, self.train_time_slots, 
                                   self.mec_num, 1])
       for i in range(self.train_freq):
           for j in range(self.train_time_slots):
               for k in range(self.mec_num):
                    obs = copy.copy(self.mec_obss[i][j][k])
                    inputs = GetPolicyInputs(obs)
                        

                    p_inputs[i, j, k] = inputs
                        # 将所有动作列表合并为一个列表
                        # 如果 self.mec_acts[i][j][k] 是列表，则合并
                   
                    acts[i, j, k] = torch.tensor(self.mec_acts[i][j][k],
                                                dtype = torch.float)
                   
                    act_logprobs[i, j, k] = torch.tensor(self.mec_act_logprobs[i][j][k],
                                                            dtype = torch.float)
       # [train_freq x train_time_slots, mec_num, obs_dim]
       #策略输入
       p_inputs = p_inputs.reshape([-1, self.mec_num, self.obs_dim])
       # [train_freq x train_time_slots, mec_num, action_dim]
       #动作
       acts = acts.reshape([-1, self.mec_num, self.action_dim])
       # [train_freq x train_time_slots, mec_num, 1]
       # 动作对数概率
       act_logprobs = act_logprobs.reshape([-1, self.mec_num, 1])
       #优势函数
       advs = advs.reshape([-1, 1])
       
       return v_inputs, v_tags, p_inputs, acts, act_logprobs, advs