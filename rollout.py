import pickle  # 用于保存和加载模型、数据等
import copy  # 用于深拷贝对象
import numpy as np  # 导入 numpy，常用于数值计算
import torch  # 导入 PyTorch，深度学习框架
import pandas as pd  # 导入 pandas，数据分析库
from env.cloud_env import CloudEnv  # 导入自定义的云环境类
from agent.cld_agent import CldAgent  # 导入云端智能体
from agent.mec_agent import MecAgent  # 导入 MEC 智能体
from util.replay_buffer import ReplayBuffer  # 导入经验回放缓冲区
from util.utils import ObsScaling, RewardScaling  # 导入观测和奖励归一化工具
import pickle  # 再次导入 pickle（重复了，其实只需一次）
import os  # 导入 os，用于文件路径操作


class Rollout:  # 定义 Rollout 类，负责训练和评估流程
    def __init__(self, params):  # 初始化方法，传入参数对象
        self.evaluate = params.evaluate  # 是否为评估模式
        # fix random seed
        if self.evaluate == False:  # 如果是训练模式
            torch.manual_seed(params.train_seed)  # 设置 PyTorch 随机种子
            np.random.seed(params.train_seed)  # 设置 numpy 随机种子
        else:  # 如果是评估模式
            torch.manual_seed(params.eval_seed)
            np.random.seed(params.eval_seed)
    
        self.mec_num = params.mec_num  # MEC 设备数量
        self.task_num = params.task_num  # 任务数量
        self.device_num = params.device_num

        # train
        self.train_time_slots = params.train_time_slots  # 训练时隙数
        self.train_freq = params.train_freq  # 训练频率
        self.save_freq = params.save_freq  # 保存频率
        self.results_dir = params.results_dir  # 结果保存目录
        self.weights_dir = params.weights_dir  # 权重保存目录
      

        # obs scaling
        self.use_obs_scaling = params.use_obs_scaling  # 是否使用观测归一化
        if self.use_obs_scaling:
            self.obs_scaling = ObsScaling()  # 创建观测归一化对象
            if params.load_scales:
                self.load_scales()  # 加载归一化参数
        # reward scaling
        self.use_reward_scaling = params.use_reward_scaling  # 是否使用奖励归一化
        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling()  # 创建奖励归一化对象

        # evaluate
        self.eval_mode = params.eval_mode  # 评估模式
        self.eval_time_slots = params.eval_time_slots  # 评估时隙数

        # cloud env
        self.cloud_env = CloudEnv(params)  # 创建云环境对象
        if not self.evaluate:
            # cld agent
            self.cld_agent = CldAgent(params)  # 创建云端智能体
            # replay buffer
            self.replay_buffer = ReplayBuffer(params)  # 创建经验回放缓冲区
        # mec agents
        self.mec_agents = []  # 保存所有 MEC 智能体
        for i in range(self.mec_num):
            self.mec_agents.append(MecAgent(i, params))  # 创建并添加 MEC 智能体

        # initialize agents' policy networks
        if not self.evaluate:
            for i in range(self.mec_num):
                self.mec_agents[i].update_net(self.cld_agent.p_nets[i].state_dict())  # 用云端智能体的参数初始化 MEC 智能体
        else:
            if self.eval_mode == "mappo":
                for i in range(self.mec_num):
                    path = self.weights_dir + "p_net_params_" + str(i) + ".pkl"  # 权重文件路径
                    self.mec_agents[i].load_net(path)  # 加载权重

        # criteria
        self.joint_reward = None  # 联合奖励
        self.mec_rewards = None  # MEC 奖励
        self.new_mec_acts = None  # MEC 动作
        self.joint_cost = None  # 联合代价
        self.mec_costs = None  # MEC 代价
        self.mec_comp_qls = None  # MEC 计算队列长度
        self.mec_comp_dlys = None  # MEC 计算延迟
        self.mec_csum_engys = None  # MEC 能耗
        self.mec_comp_expns = None  # MEC 计算开销
        self.device_overtime_nums = None  # 设备超时数量
        self.action_dim = params.action_dim
    def reset(self):  # 重置统计指标
        self.joint_reward = 0  # 联合奖励归零
        self.mec_rewards = np.zeros([self.mec_num], dtype = np.float32)  # MEC 奖励归零
        self.mec_acts  = np.zeros([self.mec_num], dtype = np.float32)  # MEC 动作归零
        self.mec_acts = np.zeros([self.mec_num, self.device_num*self.task_num, self.action_dim], dtype=np.float32)
        self.joint_cost = 0  # 联合代价归零
        self.mec_costs = np.zeros([self.mec_num], dtype = np.float32)  # MEC 代价归零
        self.mec_comp_qls = np.zeros([self.mec_num], dtype = np.float32)  # MEC 队列长度归零
        self.mec_comp_dlys = np.zeros([self.mec_num], dtype = np.float32)  # MEC 延迟归零
        self.mec_csum_engys = np.zeros([self.mec_num], dtype = np.float32)  # MEC 能耗归零
        self.mec_comp_expns = np.zeros([self.mec_num], dtype = np.float32)  # MEC 计算开销归零
        self.mec_overtime_nums = np.zeros([self.mec_num], dtype = np.float32)  # MEC 超时数量归零

    def run(self, e_id):  # 运行一轮训练或评估
        # reset
        self.reset()  # 重置统计指标
        # 检查是否有检查点可以加载（只在训练模式下，且仅第一轮）
        if not self.evaluate and self.use_reward_scaling:
            self.reward_scaling.reset()
        
        cld_obs,mec_obss= self.cloud_env.reset()  # 重置环境，获取初始观测
        mec_comp = mec_obss[0][0]  # 获取第一个 MEC 的计算信息
        #mec_comp_qls = [mec_comp[i] for i in [0,2,4,6,8]]  # 提取队列长度
        cld_comp_ql =cld_obs[0]
        mec_comp_qls = [mec_comp[i] for i in [0,2,4]]  # 提取队列长度

        # obs scaling
        if self.use_obs_scaling:
            self.obs_scaling(cld_obs,mec_obss, self.evaluate)  # 对观测进行归一化

        # rollout
        time_slots = self.train_time_slots + 1 if not self.evaluate else self.eval_time_slots  # 决定时隙数
        for t_id in range(1, time_slots + 1):
            #print("-------------time slot: " + str(t_id) + "-------------")  # 打印当前时隙

            mec_acts = [None for i in range(self.mec_num)]  # 初始化动作列表
            mec_act_logprobs = [None for i in range(self.mec_num)]  # 初始化动作对数概率列表

            # 遍历设备 i
            for i in range(self.mec_num): # 长度 1
                mec_obss[i][0][2*self.mec_num] = cld_comp_ql  # 将 cld_comp_ql 添加到 mec_obss[i][0] 的末尾
                node = mec_obss[i][0]          # 长度 4
                tasks = mec_obss[i][1]         # 长度 2
                # 重新组装
                mec_obss[i] = [
                    node + tasks[t]               # t=0,1
                    for t in range(self.device_num*self.task_num)#3个任务
                ]
            # choose action
            for i in range(self.mec_num):
                act, act_logprob = self.mec_agents[i].choose_action(i,mec_obss[i],
                                                                    evaluate=self.evaluate)  # 选择动作
                mec_acts[i] = act  # 保存动作
                mec_act_logprobs[i] = act_logprob  # 保存动作对数概率
            
            #step
            joint_reward, mec_rewards, \
            joint_cost, mec_costs, \
            mec_comp_dlys, mec_csum_engys, \
            mec_comp_expns, mec_overtime_nums, \
            next_mec_obss = self.cloud_env.step(mec_acts)  # 执行动作，获得环境反馈

                
            self.average(t_id, joint_reward, mec_rewards,mec_acts,
                               joint_cost, mec_costs,mec_comp_qls,
                               mec_comp_dlys, mec_csum_engys,
                               mec_comp_expns, mec_overtime_nums)  # 更新统计指标

            if not self.evaluate:
                if self.use_reward_scaling:
                    joint_reward = self.reward_scaling(joint_reward)  # 奖励归一化

               
                new_mec_obss = [item for sublist in mec_obss for item in sublist]  # 展平成一维
                new_mec_acts = [item for sublist in mec_acts for item in sublist]
                new_mec_act_logprobs = [item for sublist in mec_act_logprobs for item in sublist]
                # store sampled data
                self.replay_buffer.store(
                                         new_mec_obss,
                                         new_mec_acts,
                                         new_mec_act_logprobs,

                                         joint_reward)#要放单任务奖励还是联合奖励？
            # update computing-queue lengths

            mec_comp = mec_obss[0][0]
            #mec_comp_qls = [mec_comp[i] for i in [0,2,4,6,8]]
            mec_comp_qls = [mec_comp[i] for i in [0,2,4]]


            mec_obss = next_mec_obss
        # 检查是否有检查点可以加载（只在训练模式下）
        if not self.evaluate:
            # train
            if e_id % self.train_freq == 0:
                self.cld_agent.train_nets(self.replay_buffer)  # 训练云端智能体

                for i in range(self.mec_num):
                    self.mec_agents[i].update_net(self.cld_agent.p_nets[i].state_dict())  # 更新 MEC 智能体参数

            # save
            if e_id % self.save_freq== 0: 
                self.cld_agent.save_nets(e_id)  # 保存模型参数
             

      

        joint_reward = copy.copy(self.joint_reward)  # 拷贝联合奖励
        mec_acts = copy.copy(self.mec_acts )
        mec_rewards = copy.copy(self.mec_rewards)  # 拷贝 MEC 奖励
        joint_cost = copy.copy(self.joint_cost)  # 拷贝联合代价
        mec_costs = copy.copy(self.mec_costs)  # 拷贝 MEC 代价
        mec_comp_qls = copy.copy(self.mec_comp_qls)  # 拷贝队列长度
        mec_comp_dlys = copy.copy(self.mec_comp_dlys)  # 拷贝延迟
        mec_csum_engys = copy.copy(self.mec_csum_engys)  # 拷贝能耗
        mec_comp_expns = copy.copy(self.mec_comp_expns)  # 拷贝开销
        mec_overtime_nums = copy.copy(self.mec_overtime_nums)  # 拷贝超时数量
      
        return joint_reward, mec_rewards,mec_acts, \
               joint_cost, mec_costs, mec_comp_qls, \
               mec_comp_dlys, mec_csum_engys, \
               mec_comp_expns, mec_overtime_nums

    def average(self, t_id, joint_reward, mec_rewards,mec_acts,
                            joint_cost, mec_costs, mec_comp_qls,
                            mec_comp_dlys, mec_csum_engys,
                            mec_comp_expns, mec_overtime_nums):
        self.joint_reward += 1 / t_id * (joint_reward - self.joint_reward)  # 更新联合奖励均值
        self.mec_rewards += 1 / t_id * (mec_rewards - self.mec_rewards)  # 更新 MEC 奖励均值
        self.mec_acts  += 1 / t_id * (mec_acts - self.mec_acts)  # 更新 MEC 动作均值
        self.joint_cost += 1 / t_id * (joint_cost - self.joint_cost)  # 更新联合代价均值
        self.mec_costs += 1 / t_id * (mec_costs - self.mec_costs)  # 更新 MEC 代价均值
        self.mec_comp_qls += 1 / t_id * (mec_comp_qls - self.mec_comp_qls)  # 更新队列长度均值
        self.mec_comp_dlys += 1 / t_id * (mec_comp_dlys - self.mec_comp_dlys)  # 更新延迟均值
        self.mec_csum_engys += 1 / t_id * (mec_csum_engys - self.mec_csum_engys)  # 更新能耗均值
        self.mec_comp_expns += 1 / t_id * (mec_comp_expns - self.mec_comp_expns)  # 更新开销均值
        self.mec_overtime_nums += mec_overtime_nums  # 累加超时数量

    def save_scales(self, e_id):  # 保存归一化参数
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        path = self.results_dir + "obs_scales_" + str(e_id) + ".pkl"  # 文件路径
        with open(path, "wb") as f:
            e_mean = self.obs_scaling.e_running_ms.mean
            pickle.dump(e_mean, f)
            e_std = self.obs_scaling.e_running_ms.std
            pickle.dump(e_std, f)
            d_mean = self.obs_scaling.d_running_ms.mean
            pickle.dump(d_mean, f)
            d_std = self.obs_scaling.d_running_ms.std
            pickle.dump(d_std, f)

    def load_scales(self):  # 加载归一化参数
        path = self.results_dir + "obs_scales.pkl"
        with open(path, "rb") as f:
            e_mean = pickle.load(f)
            e_std = pickle.load(f)
            d_mean = pickle.load(f)
            d_std = pickle.load(f)
        self.obs_scaling.e_running_ms.mean = e_mean
        self.obs_scaling.e_running_ms.std = e_std
        self.obs_scaling.d_running_ms.mean = d_mean
        self.obs_scaling.d_running_ms.std = d_std
