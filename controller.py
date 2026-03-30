import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from rollout import Rollout

class Controller:
    def __init__(self, params):
        self.mec_num = params.mec_num
        
        # train
        self.train_episodes = params.train_episodes
        self.visu_freq = params.visu_freq
        self.results_dir = params.results_dir
        self.weights_dir = params.weights_dir
        # train results
        self.joint_reward_col = []
        self.mec_rewards_col = []
        self.actions_col = []
        self.joint_cost_col = []
        self.mec_costs_col = []
        self.mec_comp_qls_col = []
        self.mec_comp_dlys_col = []
        self.mec_csum_engys_col = []
        self.mec_comp_expns_col = []
        self.mec_overtime_nums_col = []
        self.num=params.num
        # evaluate
        self.eval_episodes = params.eval_episodes
        
        # rollout
        self.rollout = Rollout(params)
        
    def _get_checkpoint_path(self):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        return os.path.join(self.weights_dir, "training_checkpoint.pt")

    def _save_checkpoint(self, e_id):
        checkpoint_path = self._get_checkpoint_path()
        checkpoint = {
            "next_episode": e_id + 1,
            "joint_reward_col": self.joint_reward_col,
            "mec_rewards_col": self.mec_rewards_col,
            "actions_col": self.actions_col,
            "joint_cost_col": self.joint_cost_col,
            "mec_costs_col": self.mec_costs_col,
            "mec_comp_qls_col": self.mec_comp_qls_col,
            "mec_comp_dlys_col": self.mec_comp_dlys_col,
            "mec_csum_engys_col": self.mec_csum_engys_col,
            "mec_comp_expns_col": self.mec_comp_expns_col,
            "mec_overtime_nums_col": self.mec_overtime_nums_col,
            "v_state_dict": self.rollout.cld_agent.v_net.state_dict(),
            "p_state_dicts": [p_net.state_dict() for p_net in self.rollout.cld_agent.p_nets],
        }
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self):
        checkpoint_path = self._get_checkpoint_path()
        if not os.path.exists(checkpoint_path):
            return 1
        # PyTorch 2.6 之后 torch.load 默认 weights_only=True，会限制反序列化内容，
        # 这里我们加载的是自己保存的完整 checkpoint，显式关闭 weights_only 限制。
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_episode = checkpoint.get("next_episode", 1)
        self.joint_reward_col = checkpoint.get("joint_reward_col", [])
        self.mec_rewards_col = checkpoint.get("mec_rewards_col", [])
        self.actions_col = checkpoint.get("actions_col", [])
        self.joint_cost_col = checkpoint.get("joint_cost_col", [])
        self.mec_costs_col = checkpoint.get("mec_costs_col", [])
        self.mec_comp_qls_col = checkpoint.get("mec_comp_qls_col", [])
        self.mec_comp_dlys_col = checkpoint.get("mec_comp_dlys_col", [])
        self.mec_csum_engys_col = checkpoint.get("mec_csum_engys_col", [])
        self.mec_comp_expns_col = checkpoint.get("mec_comp_expns_col", [])
        self.mec_overtime_nums_col = checkpoint.get("mec_overtime_nums_col", [])

        v_state_dict = checkpoint.get("v_state_dict")
        if v_state_dict is not None:
            self.rollout.cld_agent.v_net.load_state_dict(v_state_dict)

        p_state_dicts = checkpoint.get("p_state_dicts")
        if p_state_dicts is not None:
            for i, state in enumerate(p_state_dicts):
                if i < len(self.rollout.cld_agent.p_nets):
                    self.rollout.cld_agent.p_nets[i].load_state_dict(state)
                    if i < len(self.rollout.mec_agents):
                        self.rollout.mec_agents[i].update_net(state)

        return start_episode
        
    def train(self):
        start_episode = self._load_checkpoint() if not self.rollout.evaluate else 1
        for e_id in range(start_episode, self.train_episodes + 1): 
            print("------------------train episode: " + str(e_id) + "------------------")
            
            joint_reward, mec_rewards,actions, \
            joint_cost, mec_costs, mec_comp_qls, \
            mec_comp_dlys, mec_csum_engys, \
            mec_comp_expns, mec_overtime_nums = self.rollout.run(e_id)
            
            # collect
            self.joint_reward_col.append(joint_reward)
            self.mec_rewards_col.append(mec_rewards)
            self.actions_col.append(actions)
            self.joint_cost_col.append(joint_cost)
            self.mec_costs_col.append(mec_costs)
            self.mec_comp_qls_col.append(mec_comp_qls)
            self.mec_comp_dlys_col.append(mec_comp_dlys)
            self.mec_csum_engys_col.append(mec_csum_engys)
            self.mec_comp_expns_col.append(mec_comp_expns)
            self.mec_overtime_nums_col.append(mec_overtime_nums)
            
            if e_id % self.visu_freq  == 0:
                episode_num = len(self.joint_reward_col)
                self.visualize(episode_num)  
                #self.rollout.cld_agent.plot_losses(episode_num)

            if e_id %1000 == 0:
                if not os.path.exists(self.results_dir):
                    os.makedirs(self.results_dir)
                num = 1
               
                # 保存为csv文件
                if e_id % 1000 == 0:  # 每1000轮保存一次数据
                    num = self.num  # num为str类型
                    if not os.path.exists(self.results_dir):
                        os.makedirs(self.results_dir)
                    with open(self.results_dir + f"main_result{num}joint_rewards_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.joint_reward_col, f)
                    with open(self.results_dir + f"main_result{num}mec_rewards_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_rewards_col, f)
                    with open(self.results_dir + f"main_result{num}joint_costs_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.joint_cost_col, f)
                    with open(self.results_dir + f"main_result{num}mec_costs_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_costs_col, f)
                    with open(self.results_dir + f"main_result{num}mec_comp_qls_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_comp_qls_col, f)
                    with open(self.results_dir + f"main_result{num}mec_comp_dlys_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_comp_dlys_col, f)
                    with open(self.results_dir + f"main_result{num}mec_csum_engys_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_csum_engys_col, f)
                    with open(self.results_dir + f"main_result{num}mec_comp_expns_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_comp_expns_col, f)
                    with open(self.results_dir + f"main_result{num}mec_overtime_nums_" + str(e_id) + ".pkl", "wb") as f:
                        pickle.dump(self.mec_overtime_nums_col, f)
            if not self.rollout.evaluate and (e_id % self.rollout.save_freq == 0):
                self._save_checkpoint(e_id)
        # 在每个训练epoch结束后添加：
           
    def evaluate(self):
        joint_reward = 0
        mec_rewards = np.zeros([self.mec_num], dtype = np.float32)
        joint_cost = 0
        mec_costs = np.zeros([self.mec_num], dtype = np.float32)
        mec_comp_qls = np.zeros([self.mec_num], dtype = np.float32)
        mec_comp_dlys = np.zeros([self.mec_num], dtype = np.float32)
        mec_csum_engys = np.zeros([self.mec_num], dtype = np.float32)
        mec_comp_expns = np.zeros([self.mec_num], dtype = np.float32)
        mec_overtime_nums = np.zeros([self.mec_num], dtype = np.float32)
        
        for e_id in range(1, self.eval_episodes + 1):
            joint_reward_, mec_rewards_, \
            joint_cost_, mec_costs_, mec_comp_qls_, \
            mec_comp_dlys_, mec_csum_engys_, \
            mec_comp_expns_, mec_overtime_nums_ = self.rollout.run(e_id)
            
            joint_reward += joint_reward_
            mec_rewards += mec_rewards_
            joint_cost += joint_cost_
            mec_costs += mec_costs_
            mec_comp_qls += mec_comp_qls_
            mec_comp_dlys += mec_comp_dlys_
            mec_csum_engys += mec_csum_engys_
            mec_comp_expns += mec_comp_expns_
            mec_overtime_nums += mec_overtime_nums_
            
        # average
        joint_reward /= self.eval_episodes
        mec_rewards /= self.eval_episodes
        joint_cost /= self.eval_episodes
        mec_costs /= self.eval_episodes
        mec_comp_qls /= self.eval_episodes
        mec_comp_dlys /= self.eval_episodes
        mec_csum_engys /= self.eval_episodes
        mec_comp_expns /= self.eval_episodes
        mec_overtime_nums /= self.eval_episodes
        
        return joint_reward, mec_rewards, \
               joint_cost, mec_costs, mec_comp_qls, \
               mec_comp_dlys, mec_csum_engys, \
               mec_comp_expns, mec_overtime_nums
    def moving_average(self, data, window_size=20):
        """对数据进行滑动平均处理"""
        if len(data) < window_size:
            return np.array(data)
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    def visualize(self, episode_num):  # 可视化方法
        # x-axis
        x = np.array(range(episode_num))  # 横坐标为轮数
      

        window_size = 20  # 滑动窗口大小，可根据需要调整

        # 创建或清空txt文件
        txt_file_path = self.results_dir + f"/visualization_data_{episode_num}.txt"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        with open(txt_file_path, "w") as f:
            f.write(f"Episode: {episode_num}\n")
            f.write("="*20 + "\n")

        def save_to_txt(file_handle, title, data):
            file_handle.write(f"{title}:\n")
            np.savetxt(file_handle, data, fmt="%.6f", delimiter=", ")
            file_handle.write("\n" + "="*20 + "\n")

        # joint reward
        fig1 = plt.figure(figsize = (12, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        y = np.array(self.joint_reward_col)
        y_ma = self.moving_average(y, window_size)
        x_ma = np.arange(len(y_ma))
        ax1.plot(x_ma, y_ma, marker = ".", linewidth = 3, label="Joint Reward")
        ax1.grid(True)
        ax1.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax1.set_ylabel(ylabel = "Joint Reward", fontsize = 18)
        ax1.tick_params(axis = 'both', labelsize = 18)
        ax1.legend()
        with open(txt_file_path, "a") as f:
            save_to_txt(f, "Joint Reward (Smoothed)", y_ma)

        # mec reward
        fig2 = plt.figure(figsize = (12, 8))
        ax2 = fig2.add_subplot(1, 1, 1)
        y = np.array(self.mec_rewards_col)
        # 均值曲线
        mean_y = np.mean(y, axis=1)
        mean_y_ma = self.moving_average(mean_y, window_size)
        x_mean_ma = np.arange(len(mean_y_ma))
        ax2.plot(x_mean_ma, mean_y_ma, marker=".", linewidth=3, color="black", label="Mean Reward")
        with open(txt_file_path, "a") as f:
            save_to_txt(f, "MEC Reward (Mean, Smoothed)", mean_y_ma)
        # 单智能体曲线
        for i in range(self.mec_num):
            y_ma = self.moving_average(y[:, i], window_size)
            x_ma = np.arange(len(y_ma))
            ax2.plot(x_ma, y_ma, marker = ".", linewidth = 3, label=f"MEC {i}")
            with open(txt_file_path, "a") as f:
                save_to_txt(f, f"MEC {i} Reward (Smoothed)", y_ma)
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax2.set_ylabel(ylabel = "mec Reward", fontsize = 18)
        ax2.tick_params(axis = 'both', labelsize = 18)

        # mec computing-queue length
        fig6 = plt.figure(figsize = (12, 8))
        ax6 = fig6.add_subplot(1, 1, 1)
        y = np.array(self.mec_comp_qls_col)
        mean_y = np.mean(y, axis=1)
        mean_y_ma = self.moving_average(mean_y, window_size)
        x_mean_ma = np.arange(len(mean_y_ma))
        ax6.plot(x_mean_ma, mean_y_ma, marker=".", linewidth=3, color="black", label="Mean Queue Length")
        with open(txt_file_path, "a") as f:
            save_to_txt(f, "MEC Computing Queue Length (Mean, Smoothed)", mean_y_ma)
        for i in range(self.mec_num):
            y_ma = self.moving_average(y[:, i], window_size)
            x_ma = np.arange(len(y_ma))
            ax6.plot(x_ma, y_ma, marker = ".", linewidth = 3, label=f"MEC {i}")
            with open(txt_file_path, "a") as f:
                save_to_txt(f, f"MEC {i} Computing Queue Length (Smoothed)", y_ma)
        ax6.legend()
        ax6.grid(True)
        ax6.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax6.set_ylabel(ylabel = "mec Computation-Queue Length (Gcycles)", fontsize = 18)
        ax6.tick_params(axis = 'both', labelsize = 18)

        # mec cmputation-delay
        fig7 = plt.figure(figsize = (12, 8))
        ax7 = fig7.add_subplot(1, 1, 1)
        y = np.array(self.mec_comp_dlys_col) * pow(10, 3)
        mean_y = np.mean(y, axis=1)
        mean_y_ma = self.moving_average(mean_y, window_size)
        x_mean_ma = np.arange(len(mean_y_ma))
        ax7.plot(x_mean_ma, mean_y_ma, marker=".", linewidth=3, color="black", label="Mean Delay")
        with open(txt_file_path, "a") as f:
            save_to_txt(f, "MEC Computation Delay (Mean, Smoothed, ms)", mean_y_ma)
        for i in range(self.mec_num):
            y_ma = self.moving_average(y[:, i], window_size)
            x_ma = np.arange(len(y_ma))
            ax7.plot(x_ma, y_ma, marker = ".", linewidth = 3, label=f"MEC {i}")
            with open(txt_file_path, "a") as f:
                save_to_txt(f, f"MEC {i} Computation Delay (Smoothed, ms)", y_ma)
        ax7.legend()
        ax7.grid(True)
        ax7.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax7.set_ylabel(ylabel = "Task Computation-Delay (ms)", fontsize = 18)
        ax7.tick_params(axis = 'both', labelsize = 18)

        # mec consumed-energy
        fig8 = plt.figure(figsize = (12, 8))
        ax8 = fig8.add_subplot(1, 1, 1)
        y = np.array(self.mec_csum_engys_col) * pow(10, 3)
        mean_y = np.mean(y, axis=1)
        mean_y_ma = self.moving_average(mean_y, window_size)
        x_mean_ma = np.arange(len(mean_y_ma))
        ax8.plot(x_mean_ma, mean_y_ma, marker=".", linewidth=3, color="black", label="Mean Energy")
        with open(txt_file_path, "a") as f:
            save_to_txt(f, "MEC Consumed Energy (Mean, Smoothed, mJ)", mean_y_ma)
        for i in range(self.mec_num):
            y_ma = self.moving_average(y[:, i], window_size)
            x_ma = np.arange(len(y_ma))
            ax8.plot(x_ma, y_ma, marker = ".", linewidth = 3, label=f"MEC {i}")
            with open(txt_file_path, "a") as f:
                save_to_txt(f, f"MEC {i} Consumed Energy (Smoothed, mJ)", y_ma)
        ax8.legend()
        ax8.grid(True)
        ax8.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax8.set_ylabel(ylabel = "Task Energy-Consumption (mJ)", fontsize = 18)
        ax8.tick_params(axis = 'both', labelsize = 18)

        # mec overtime-number
        fig10 = plt.figure(figsize = (12, 8))
        ax10 = fig10.add_subplot(1, 1, 1)
        y = np.array(self.mec_overtime_nums_col)
        mean_y = np.mean(y, axis=1)
        mean_y_ma = self.moving_average(mean_y, window_size)
        x_mean_ma = np.arange(len(mean_y_ma))
        ax10.plot(x_mean_ma, mean_y_ma, marker=".", linewidth=3, color="black", label="Mean Overtime Num")
        with open(txt_file_path, "a") as f:
            save_to_txt(f, "MEC Overtime Number (Mean, Smoothed)", mean_y_ma)
        for i in range(self.mec_num):
            y_ma = self.moving_average(y[:, i], window_size)
            x_ma = np.arange(len(y_ma))
            ax10.plot(x_ma, y_ma, marker = ".", linewidth = 3, label=f"MEC {i}")
            with open(txt_file_path, "a") as f:
                save_to_txt(f, f"MEC {i} Overtime Number (Smoothed)", y_ma)
        ax10.legend()
        ax10.grid(True)
        ax10.set_xlabel(xlabel = "Episodes", fontsize = 18)
        ax10.set_ylabel(ylabel = "mec Overtime-Number", fontsize = 18)
        ax10.tick_params(axis = 'both', labelsize = 18)
        
      
        # plt.show()  # 显示所有图像（可选）
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        fig1.savefig(self.results_dir 
                     + "/joint_reward_" + str(episode_num) + ".png")  # 保存联合奖励图
        fig2.savefig(self.results_dir 
                     + "/mec_rewards_" + str(episode_num) + ".png")  # 保存MEC奖励图
        # fig3.savefig(self.results_dir 
        #              + "/joint_cost_" + str(episode_num) + ".png")
        # fig4.savefig(self.results_dir 
        #              + "/mec_costs_" + str(episode_num) + ".png")
        fig6.savefig(self.results_dir 
                     + "/mec_comp_qls_" + str(episode_num) + ".png")  # 保存队列长度图
        fig7.savefig(self.results_dir 
                     + "/mec_comp_dlys_" + str(episode_num) + ".png")  # 保存延迟图
        fig8.savefig(self.results_dir 
                     + "/mec_csum_engys_" + str(episode_num) + ".png")  # 保存能耗图
        # fig9.savefig(self.results_dir 
        #              + "/mec_comp_expns_" + str(episode_num) + ".png")
        fig10.savefig(self.results_dir 
                     + "/mec_overtime_nums_" + str(episode_num) + ".png")  # 保存超时数量图
        

        plt.close()  # 关闭所有画布，释放内存