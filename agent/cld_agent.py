import os
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from network.policy_net import PolicyNet
from network.value_net import ValueNet
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt


class CldAgent():
    def __init__(self, params):
        self.mec_num = params.mec_num
        
        # train
        self.train_freq = params.train_freq
        self.train_time_slots = params.train_time_slots
        self.v_batch_size = params.v_batch_size
        self.p_batch_size = params.p_batch_size
        self.v_epochs = params.v_epochs
        self.p_epochs = params.p_epochs
        self.p_clip = params.p_clip
        
        self.use_enty_coef_clip = params.use_enty_coef_clip
        self.enty_coef = params.enty_coef
        self.enty_coef_min = params.enty_coef_min
        self.enty_coef_decay =  params.enty_coef_decay
        self.v_lr = params.v_lr
        self.p_lr = params.p_lr
        self.gamma = params.gamma
        # gradient clip
        self.use_grad_clip = params.use_grad_clip
        self.v_grad_clip = params.v_grad_clip
        self.p_grad_clip = params.p_grad_clip
        self.weights_dir = params.weights_dir
        # learning-rate decay
        self.use_lr_decay = params.use_lr_decay
        self.min_v_lr = params.min_v_lr
        self.min_p_lr = params.min_p_lr
        self.decay_fac = params.decay_fac
        # 损失存储
        self.value_losses = []  # 存储value network损失
        self.value_means = []   # 存储每轮值函数均值
        self.policy_losses = []  # 训练轮次 x 节点，每轮保存所有节点的平均损失
        self.policy_entropies = []  # 存储每轮所有agent的平均熵
        self.gate_means = []  # 每轮门控均值（注意力融合稳定性监控）
        self.results_dir = params.results_dir
        # value network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.v_net = ValueNet(params).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(),
                                            lr = self.v_lr)
        # policy networks
        self.p_nets = []
        self.p_optimizers = []
        for i in range(self.mec_num):
            p_net = PolicyNet(params).to(self.device)
            self.p_nets.append(p_net)
            p_optimizer = torch.optim.Adam(p_net.parameters(),
                                          lr = self.p_lr)
            self.p_optimizers.append(p_optimizer)
           
            
        # load networks' weights 
        if params.load_weights:
            v_path = self.weights_dir + "v_net_params.pkl"
            self.v_net.load_state_dict(torch.load(v_path, map_location=self.device))
            for i in range(self.mec_num):
                p_path = self.weights_dir + "p_net_params_" + str(i) + ".pkl"
                self.p_nets[i].load_state_dict(torch.load(p_path, map_location=self.device))
            
    def train_nets(self, replay_buffer):
        '''training data'''
        # v_inputs: [train_freq x train_time_slots, state_dim]
        # v_tags: [train_freq x train_time_slots, 1]
        # p_inputs: [train_freq x train_time_slots, mec_num, obs_dim]
        # acts: [train_freq x train_time_slots, mec_num, action_dim]
        # act_logprobs: [train_freq x train_time_slots, mec_num, 1]
        # advs: [train_freq x train_time_slots, 1]
        v_inputs, v_tags, p_inputs, acts, act_logprobs, advs = \
                                    replay_buffer.get_training_data(self.v_net)
                                    
        # 训练 ValueNet 并记录该轮的 V 值均值（用于绘图）
        v_mean = self.train_value_net(v_inputs, v_tags)
        self.value_means.append(v_mean)
        round_policy_losses = []
        round_policy_entropies = []
        for i in range(self.mec_num):
            avg_loss, avg_enty = self.train_policy_net(i, p_inputs[:, i], acts[:, i], act_logprobs[:, i], advs)
            round_policy_losses.append(avg_loss)
            round_policy_entropies.append(avg_enty)
        self.policy_losses.append(round_policy_losses)
        self.policy_entropies.append(round_policy_entropies)
        if self.use_lr_decay:
            self.decay_lr()
        if self.use_enty_coef_clip:
            self.decay_enty_coef()
    
    def train_value_net(self, v_inputs, v_tags):
        v_total_size = self.train_freq * self.train_time_slots
        v_inputs = v_inputs.to(self.device)
        v_tags = v_tags.to(self.device)
        value_loss = []
        value_mean = []
        gate_mean = []
        for e in range(self.v_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(v_total_size)),
                                    self.v_batch_size, False):
                vs = self.v_net(v_inputs[ids])
                loss = F.mse_loss(v_tags[ids], vs)

                gate = getattr(self.v_net, "last_gate", None)
                if gate is not None:
                    gate_mean.append(gate.mean().item())

                # 辅助信用分配损失：用注意力权重 a_i 引导个体价值 V_i，使加权和 v_credit 拟合联合价值标签
                credit_coef = float(getattr(self.v_net, "credit_coef", 0.0))
                v_credit = getattr(self.v_net, "last_v_credit", None)
                if credit_coef > 0 and v_credit is not None:
                    loss = loss + credit_coef * F.mse_loss(v_tags[ids], v_credit)
                value_loss.append(loss.item())  # 保存损失
                value_mean.append(vs.mean().item())  # 保存均值
                self.v_optimizer.zero_grad()
                loss.backward()
                # gradient clip
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), 
                                                self.v_grad_clip)
                self.v_optimizer.step()
            self.value_losses.append(sum(value_loss)/len(value_loss))
            self.gate_means.append(sum(gate_mean) / len(gate_mean) if gate_mean else 0)
        return sum(value_mean)/len(value_mean) if value_mean else 0
        
    def train_policy_net(self, agent_id, p_inputs, acts, act_logprobs, advs):
        p_total_size = self.train_freq * self.train_time_slots
        p_inputs = p_inputs.to(self.device)
        acts = acts.to(self.device)
        act_logprobs = act_logprobs.to(self.device)
        advs = advs.to(self.device)
        policy_loss = []
        policy_enty = []
        
        for e in range(self.p_epochs):
            for ids in BatchSampler(SubsetRandomSampler(range(p_total_size)),
                                    self.p_batch_size, False):
                # 策略网络输出：服务器选择 logits + 卸载比例高斯分布参数
                server_logits, ratio_means, ratio_logstds = self.p_nets[agent_id](p_inputs[ids])
                # server_logits: [bs, action_dim]
                # ratio_means:   [bs, action_dim]
                # ratio_logstds: [bs, action_dim]
                bs = server_logits.shape[0]
                batch_idx = torch.arange(bs, device=self.device)
                acts_batch = acts[ids]  # [bs, action_dim]

                # -------- 从存储的动作重建选中的服务器索引 --------
                # 动作格式: acts[selected_server]=offload_ratio, acts[agent_id]=1-offload_ratio
                # 将本地位置(agent_id)置0后，非零位置即为选中的远端服务器
                acts_masked = acts_batch.clone()
                acts_masked[:, agent_id] = 0.0
                non_local = acts_masked.abs().max(dim=-1).values > 1e-6  # [bs] bool
                selected_servers = torch.where(
                    non_local,
                    acts_masked.argmax(dim=-1),
                    torch.full((bs,), agent_id, dtype=torch.long, device=self.device)
                )  # [bs]

                # -------- 第一部分：服务器选择（Categorical）--------
                server_dist = Categorical(logits=server_logits)
                new_server_logprobs = server_dist.log_prob(selected_servers)  # [bs]

                # -------- 第二部分：卸载比例（Normal + sigmoid逆变换）--------
                offload_ratios = acts_batch[batch_idx, selected_servers]  # [bs]
                # 逆 sigmoid 恢复采样前的 z：z = logit(r)
                offload_ratios_c = torch.clamp(offload_ratios, 1e-6, 1 - 1e-6)
                sampled_z = torch.log(offload_ratios_c / (1 - offload_ratios_c))

                sel_ratio_means = ratio_means[batch_idx, selected_servers]           # [bs]
                sel_ratio_stds  = torch.exp(ratio_logstds[batch_idx, selected_servers])  # [bs]
                ratio_dist = Normal(sel_ratio_means, sel_ratio_stds)
                new_ratio_logprobs = ratio_dist.log_prob(sampled_z)  # [bs]

                # 本地选择时不加比例的 log_prob
                new_act_logprobs = new_server_logprobs + torch.where(
                    non_local, new_ratio_logprobs, torch.zeros_like(new_ratio_logprobs)
                )

                # -------- PPO 剪切目标 --------
                old_act_logprobs_batch = act_logprobs[ids].reshape([-1])
                ratios_ppo = torch.exp(new_act_logprobs - old_act_logprobs_batch)
                batch_advs = advs[ids].reshape([-1])
                surr1 = ratios_ppo * batch_advs
                surr2 = torch.clamp(ratios_ppo, 1 - self.p_clip, 1 + self.p_clip) * batch_advs

                # -------- 熵鼓励探索 --------
                server_enty = server_dist.entropy()  # [bs] Categorical熵
                ratio_enty  = ratio_dist.entropy()   # [bs] Normal熵（对应选中服务器）
                total_enty  = server_enty + torch.where(
                    non_local, ratio_enty, torch.zeros_like(ratio_enty)
                )

                policy_enty.append(total_enty.mean().item())

                # 计算损失
                loss = -(torch.min(surr1, surr2) + self.enty_coef * total_enty)
                policy_loss.append(loss.mean().item())

                # 优化步骤
                self.p_optimizers[agent_id].zero_grad()
                loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.p_nets[agent_id].parameters(),
                                                   self.p_grad_clip)
                self.p_optimizers[agent_id].step()

        avg_loss = sum(policy_loss) / len(policy_loss) if policy_loss else 0
        avg_enty = sum(policy_enty) / len(policy_enty) if policy_enty else 0
        return avg_loss, avg_enty


    def decay_lr(self):
        self.v_lr = max(self.v_lr * self.decay_fac, self.min_v_lr)
        for params in self.v_optimizer.param_groups:
            params['lr'] = self.v_lr
                
        self.p_lr = max(self.p_lr * self.decay_fac, self.min_p_lr)
        for i in range(self.mec_num):
            for params in self.p_optimizers[i].param_groups:
                params['lr'] = self.p_lr
        
    def save_nets(self, e_id):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        torch.save(self.v_net.state_dict(),
                    self.weights_dir + "v_net_params_" + str(e_id) + ".pkl")
        for i in range(self.mec_num):
            torch.save(self.p_nets[i].state_dict(),
                        self.weights_dir + "p_net_params_" + str(i) + "_" + str(e_id) + ".pkl")
       

    
    def plot_losses(self, episode_num):
        # 每2个train episode保存一次，所以x轴为 [2, 4, 6, ...]
        x = [2 * (i + 1) for i in range(len(self.policy_losses))]
        fig = plt.figure(figsize=(16, 8))
        # 1. Value Network Loss
        plt.subplot(2, 2, 1)
        plt.plot(x, self.value_losses, marker='o')
        plt.title('Value Network Loss')
        plt.xlabel('Train Episode')
        plt.ylabel('Loss')
        # 2. Value Function Mean
        plt.subplot(2, 2, 2)
        plt.plot(x, self.value_means, marker='o', color='green')
        plt.title('Value Function Mean')
        plt.xlabel('Train Episode')
        plt.ylabel('Mean Value')
        # 3. Policy Network Losses
        plt.subplot(2, 2, 3)
        policy_losses_arr = list(zip(*self.policy_losses))  # 转置，按智能体分组
        for i, agent_losses in enumerate(policy_losses_arr):
            plt.plot(x, agent_losses, marker='o', label=f'Agent {i+1}')
        plt.title('Policy Network Losses')
        plt.xlabel('Train Episode')
        plt.ylabel('Loss')
        plt.legend()
        # 4. Policy Entropy
        plt.subplot(2, 2, 4)
        policy_enty_arr = list(zip(*self.policy_entropies))  # 转置，按智能体分组
        for i, agent_entys in enumerate(policy_enty_arr):
            plt.plot(x, agent_entys, marker='o', label=f'Agent {i+1}')
        plt.title('Policy Entropy')
        plt.xlabel('Train Episode')
        plt.ylabel('Entropy')
        plt.legend()
        plt.tight_layout()
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        fig.savefig(self.results_dir + "/Loss_" + str(episode_num) + ".png")


    def decay_enty_coef(self):
        self.enty_coef = max(self.enty_coef * self.enty_coef_decay, self.enty_coef_min)