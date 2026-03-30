
import torch
import torch.nn as nn

# 两阶段决策：先从N个服务器中选择一个（离散动作），再确定卸载比例（连续动作）
class PolicyNet(nn.Module):
    def __init__(self, params):
        super(PolicyNet, self).__init__()
        
        def orthogonal_init(layer, gain=1.0):
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, gain=gain)
        
        self.fc1 = nn.Linear(params.obs_dim, params.p_hid_dims[0])
        self.fc2 = nn.Linear(params.p_hid_dims[0], params.p_hid_dims[1])
        
        # 第一阶段：服务器选择（使用多项分布）
        # 输出action_dim个logits，对应每个服务器的选择概率
        self.fc_server_logits = nn.Linear(params.p_hid_dims[1], params.action_dim)
        
        # 第二阶段：卸载比例（使用高斯分布）
        # 为每个服务器输出卸载比例的均值和标准差
        self.fc_ratio_mean = nn.Linear(params.p_hid_dims[1], params.action_dim)
        self.fc_ratio_logstd = nn.Linear(params.p_hid_dims[1], params.action_dim)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()  # 将均值限制在[0,1]
        self.softplus = nn.Softplus()  # 确保logstd为正
         
        if params.use_orthogonal:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_server_logits, gain=0.01)
            orthogonal_init(self.fc_ratio_mean, gain=0.01)
            orthogonal_init(self.fc_ratio_logstd, gain=0.01)
        
    def forward(self, obs):
        """
        前向传播，输出两个阶段的决策参数
        
        返回：
            server_logits: (batch_size, action_dim) - 服务器选择的logits
            ratio_means: (batch_size, action_dim) - 每个服务器卸载比例的均值
            ratio_logstds: (batch_size, action_dim) - 每个服务器卸载比例的logstd
        """
        x = self.tanh(self.fc1(obs))
        x = self.tanh(self.fc2(x))
        
        # 第一阶段：服务器选择编码（使用softmax作为多项分布）
        server_logits = self.fc_server_logits(x)
        
        # 第二阶段：卸载比例编码
        ratio_means = self.sigmoid(self.fc_ratio_mean(x))  # 限制在[0,1]
        ratio_logstds = self.fc_ratio_logstd(x)  # 标准差的log值
        
        return server_logits, ratio_means, ratio_logstds
    