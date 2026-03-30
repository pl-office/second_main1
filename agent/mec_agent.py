from torch.distributions import Categorical, Normal  
import numpy as np  
import torch  
from network.policy_net import PolicyNet  
from util.utils import GetPolicyInputs 

class MecAgent():  
    """
    两阶段决策MEC智能体：
    第一阶段：从action_dim个服务器中选择一个（离散动作，使用多项分布）
    第二阶段：确定该服务器的卸载比例（连续动作，使用高斯分布）
    """
    def __init__(self, agent_id, params):  
        self.agent_id = agent_id  
        self.device_num = params.device_num  
        self.task_num = params.task_num  
        
        # policy network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.p_net = PolicyNet(params).to(self.device)  
        self.action_dim = params.action_dim  
        self.eval_mode = params.eval_mode  
      

    def choose_action(self, i,obs, evaluate):  
        """
        两阶段动作选择（支持批量输入）
        
        参数：
            obs: 批量观测，shape (batch_size, obs_dim) 或单样本 (obs_dim,)
        返回：
            acts: 批量动作列表，每个元素为动作向量（只有选中的服务器位置非零）
            act_logprobs: 批量对数概率列表（训练模式返回浮点数，其余为None）
        """
        # 推断批量大小（用于非网络分支）
        obs_arr = np.array(obs)
        batch_size = obs_arr.shape[0] if obs_arr.ndim > 1 else 1

        if not evaluate or (evaluate and self.eval_mode == "mappo"):  
            with torch.no_grad():
                p_inputs = GetPolicyInputs(obs)
                server_logits, ratio_means, ratio_logstds = self.p_net(p_inputs)
            # server_logits: (batch_size, action_dim)
            # ratio_means:   (batch_size, action_dim)
            # ratio_logstds: (batch_size, action_dim)
            batch_size = server_logits.shape[0]
            idx = torch.arange(batch_size)

            if evaluate:  # 评估模式 - 使用期望值
                # 第一阶段：贪心选择概率最高的服务器
                server_probs = torch.softmax(server_logits, dim=-1)
                selected_servers = torch.argmax(server_probs, dim=-1)  # (batch_size,)
                
                # 第二阶段：使用均值作为卸载比例
                offload_ratios = ratio_means[idx, selected_servers]  # (batch_size,)
                
                # 构造动作向量批量
                # 当选中服务器恰为本地设备i时，视为全本地计算
                non_local = selected_servers != i          # (batch_size,) bool
                acts_tensor = torch.zeros(batch_size, self.action_dim)
                acts_tensor[idx[non_local], selected_servers[non_local]] = offload_ratios[non_local]
                acts_tensor[non_local, i] = 1 - offload_ratios[non_local]
                acts_tensor[~non_local, i] = 1.0           # 选中本地：全本地计算
                acts = acts_tensor.tolist()
                act_logprobs = [None] * batch_size
                
            else:  # 训练模式 - 采样
                # 第一阶段：从多项分布采样选择服务器
                server_dist = Categorical(logits=server_logits)
                selected_servers = server_dist.sample()  # (batch_size,)
                
                # 获取各样本对应服务器的高斯分布参数
                selected_ratio_means = ratio_means[idx, selected_servers]       # (batch_size,)
                selected_ratio_logstds = ratio_logstds[idx, selected_servers]   # (batch_size,)
                selected_ratio_stds = torch.exp(selected_ratio_logstds)         # (batch_size,)
                
                # 第二阶段：从高斯分布采样卸载比例
                ratio_dist = Normal(selected_ratio_means, selected_ratio_stds)
                sampled_ratios = ratio_dist.rsample()  # (batch_size,) 重参数化采样
                
                # 限制在[0, 1]范围内（sigmoid映射）
                offload_ratios = torch.sigmoid(sampled_ratios)  # (batch_size,)
                
                # 构造动作向量批量
                # 当选中服务器恰为本地设备i时，视为全本地计算
                non_local = selected_servers != i          # (batch_size,) bool
                acts_tensor = torch.zeros(batch_size, self.action_dim)
                acts_tensor[idx[non_local], selected_servers[non_local]] = offload_ratios[non_local]
                acts_tensor[non_local, i] = 1 - offload_ratios[non_local]
                acts_tensor[~non_local, i] = 1.0           # 选中本地：全本地计算
                acts = acts_tensor.tolist()
                
                # 计算对数概率
                server_logprobs = server_dist.log_prob(selected_servers)   # (batch_size,)
                ratio_logprobs = ratio_dist.log_prob(sampled_ratios)       # (batch_size,)
                act_logprobs = (server_logprobs + ratio_logprobs).tolist()
                
        elif self.eval_mode == "local_comp":  # 本地计算 - 不卸载
            acts = [[0.0] * self.action_dim for _ in range(batch_size)]
            for act in acts:
                act[i] = 1.0  # 1 - 0.0
            act_logprobs = [None] * batch_size
        elif self.eval_mode == "edge_comp":  # 边缘计算 - 全卸载到第一个服务器
            acts = [[1.0] + [0.0] * (self.action_dim - 1) for _ in range(batch_size)]
            for act in acts:
                act[i] = 0.0  # 1 - 1.0
            act_logprobs = [None] * batch_size
        else:  # 默认：随机选择
            acts = []
            for _ in range(batch_size):
                selected_server = np.random.randint(0, self.action_dim)
                offload_ratio = np.random.uniform(0, 1)
                act = [0.0] * self.action_dim
                if selected_server == i:  # 选中本地：全本地计算
                    act[i] = 1.0
                else:
                    act[selected_server] = float(offload_ratio)
                    act[i] = float(1 - offload_ratio)
                acts.append(act)
            act_logprobs = [None] * batch_size
            
        return acts, act_logprobs  
    
    def load_net(self, path):  
        self.p_net.load_state_dict(torch.load(path))  
    
    def update_net(self, params):  
        self.p_net.load_state_dict(params)




