import numpy as np
import torch
from config.params import get_params

# params
params = get_params()

'''calculate mean and std dynamically'''
class RunningMeanStd():
    def __init__(self, shape, epsilon=1e-8):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.ones(shape)  # 初始化为1，而不是0
        self.epsilon = epsilon
        
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
            

class ObsScaling():
    def __init__(self):
        self.mec_num = params.mec_num
        # 初始化云观测值的运行均值和标准差
        self.e_running_ms = RunningMeanStd(1)
        # 每个MEC设备的第一个子列表有17个元素，所以维度为17 * mec_num
        self.d_running_ms = RunningMeanStd((2*(self.mec_num+1)+self.mec_num) * self.mec_num)
        
        # 设置最大数据大小
        self.max_data_size = params.max_data_size
        # 设置最大计算密度
        self.max_comp_dens = params.max_comp_dens
        # 计算最大延迟约束
        self.max_dly_cons = self.max_data_size * pow(10, 6) * self.max_comp_dens \
                            / params.std_comp_freq

    def __call__(self,cld_obs,mec_obss, evaluate):
        # 将mec_obss中的元组转换为列表
        mec_obss = [list(obs) for obs in mec_obss]
        # 提取所有设备的第一个子列表用于归一化
        nor_cld_obs = cld_obs[0]#云端队列
        nor_mec_obss = []
        for obs in mec_obss:#每个智能体的第一个子列表
            nor_mec_obss.extend(obs[0])#每个智能体的第一个子列表的所有元素
        
        if not evaluate:
            self.e_running_ms.update(nor_cld_obs)
            self.d_running_ms.update(nor_mec_obss)
        nor_cld_obs = (nor_cld_obs - self.e_running_ms.mean) / \
                                                 (self.e_running_ms.std + 1e-8)
        nor_mec_obss = (nor_mec_obss - self.d_running_ms.mean) / \
                                                 (self.d_running_ms.std + 1e-8)
        cld_obs[0] = float(nor_cld_obs)
        
        # 更新mec_obss
        num_elements_per_mec = 2 * (self.mec_num + 1) + self.mec_num
        for i in range(params.mec_num):
            start_index = i * num_elements_per_mec
            for j in range(num_elements_per_mec):
                mec_obss[i][0][j] = float(nor_mec_obss[start_index + j])
       
        # 处理任务信息（第二个子列表）- 按原来的方式处理
       
            for j in range(params.device_num*params.task_num):
                mec_obss[i][1][j][0] /= self.max_data_size  # data_size
                mec_obss[i][1][j][1] /= self.max_comp_dens  # comp_dens
                mec_obss[i][1][j][2] /= self.max_dly_cons   # dly_cons
        
        


                
class RewardScaling():
    def __init__(self):
        # discount factor
        self.gamma = params.gamma
        self.R = 0
        self.r_running_ms = RunningMeanStd(1)

    def __call__(self, reward):
        self.R = self.gamma * self.R + reward
        self.r_running_ms.update(self.R)
        reward = float(reward / (self.r_running_ms.std + 1e-8))
        
        return reward
        
    # reset 'R' when an episode is done 
    def reset(self):  
        self.R = 0
        
def GetValueInputs(mec_obss):
    inputs = []
    for i in range(len(mec_obss)):
        inputs += mec_obss[i]
    inputs = torch.tensor(inputs, dtype = torch.float).reshape([1, -1])
   
    return inputs

def GetPolicyInputs(obs):
    # 假设obs已经是一个批量数据，或者可以转换为批量形式
    if isinstance(obs, list):
        obs = np.array(obs)
    inputs = torch.tensor(obs, dtype=torch.float)
    # 如果obs是单样本，确保有batch维度
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    return inputs