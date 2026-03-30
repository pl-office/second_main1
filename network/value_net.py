import torch
import torch.nn as nn
from network.attention_module import JointValueAttention


class ValueNet(nn.Module):
    """联合价值网络，可在原始 MLP 与注意力版本之间切换。

    说明：
    - 外部接口保持不变：输入 state 为 (batch_size, state_dim) 的全局状态向量，
      兼容 ReplayBuffer 和 CldAgent 当前的调用方式。
    - 当 params.use_attention_value=True 时，按 "state 在经验池按 agent 堆叠" 的约定，
      内部将 state reshape 为 (batch_size, mec_num, state_dim_per_agent)，使用注意力聚合；
      否则，使用原始的三层 MLP 结构。
    """

    def __init__(self, params):
        super(ValueNet, self).__init__()

        def orthogonal_init(layer, gain=1.0):
            for name, p in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(p, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(p, gain=gain)

        self.state_dim = params.state_dim
        self.use_attention_value = getattr(params, "use_attention_value", False)
        self.use_credit_assignment = getattr(params, "use_credit_assignment", False)
        self.credit_coef = float(getattr(params, "credit_coef", 0.0))
        self.tanh = nn.Tanh()

        # ====== 原始 MLP 版 ValueNet（不使用注意力） ======
        self.fc_plain1 = nn.Linear(self.state_dim, params.v_hid_dims[0])
        self.fc_plain2 = nn.Linear(params.v_hid_dims[0], params.v_hid_dims[1])
        self.fc_plain3 = nn.Linear(params.v_hid_dims[1], 1)

        # ====== 注意力版 ValueNet（多智能体联合价值估计） ======
        # 仅在需要时使用，但仍提前构造，方便直接切换
        self.mec_num = params.mec_num
        assert self.state_dim % self.mec_num == 0, "state_dim 必须能被 mec_num 整除，以便按 agent 切分"
        self.agent_state_dim = self.state_dim // self.mec_num

        self.attn = JointValueAttention(
            state_dim=self.agent_state_dim,
            embed_dim=params.v_attn_embed_dim,
            num_heads=params.v_attn_heads,
            use_layer_norm=True,
            use_gated_fusion=getattr(params, "use_gated_attn_fusion", False),
            gate_hidden_dim=int(getattr(params, "gate_hidden_dim", 64)),
            gate_init_bias=float(getattr(params, "gate_init_bias", -2.0)),
        )

        self.fc_attn1 = nn.Linear(params.v_attn_embed_dim, params.v_hid_dims[0])
        self.fc_attn2 = nn.Linear(params.v_hid_dims[0], params.v_hid_dims[1])
        self.fc_attn3 = nn.Linear(params.v_hid_dims[1], 1)

        # 信用分配增强：个体价值头 V_i（由 agent 嵌入估计）
        self.credit_fc1 = nn.Linear(params.v_attn_embed_dim, params.v_hid_dims[1])
        self.credit_fc2 = nn.Linear(params.v_hid_dims[1], 1)

        # 供训练阶段读取的缓存（每次 forward 会更新）
        self.last_attn_weights = None  # (B, N)
        self.last_v_individual = None  # (B, N, 1)
        self.last_v_credit = None      # (B, 1)

        # trick: orthogonal initialization
        if params.use_orthogonal:
            orthogonal_init(self.fc_plain1)
            orthogonal_init(self.fc_plain2)
            orthogonal_init(self.fc_plain3)
            orthogonal_init(self.fc_attn1)
            orthogonal_init(self.fc_attn2)
            orthogonal_init(self.fc_attn3)
            orthogonal_init(self.credit_fc1)
            orthogonal_init(self.credit_fc2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch_size, state_dim)。

        - use_attention_value=False: 直接走原来的 MLP 结构；
        - use_attention_value=True: 按 agent 切分后做注意力聚合再送入 MLP。
        """
        # 清理上次 forward 的缓存
        self.last_attn_weights = None
        self.last_v_individual = None
        self.last_v_credit = None

        if not self.use_attention_value:
            # 原始三层 MLP 版本
            x = self.tanh(self.fc_plain1(state))
            x = self.tanh(self.fc_plain2(x))
            v = self.fc_plain3(x)
            return v

        # 注意力版本
        batch_size = state.shape[0]
        # (B, state_dim) -> (B, mec_num, agent_state_dim)
        agent_state = state.view(batch_size, self.mec_num, self.agent_state_dim)
        # 多智能体注意力，得到联合表示 (B, v_attn_embed_dim)
        # 同时取出注意力权重用于信用分配增强
        context, attn = self.attn(agent_state, return_attn=True)
        # 再通过 MLP 得到价值
        x = self.tanh(self.fc_attn1(context))
        x = self.tanh(self.fc_attn2(x))
        v = self.fc_attn3(x)

        # ---- 基于注意力权重的信用分配增强（辅助分支）----
        # a_i: 由多头注意力权重在 head 维上取平均得到 (B, N)
        # V_i: 由 agent 嵌入估计得到 (B, N, 1)
        # v_credit = sum_i a_i * V_i
        if self.use_credit_assignment and self.credit_coef > 0:
            # a: (B, N) 注意力权重在 head 维上取平均
            a = attn.mean(dim=1)  # (B, N)
            # 使用原始按 agent 切分后的状态，经同一 input_proj 得到每个 agent 的嵌入
            # e: (B, N, D_embed)
            e = self.attn.input_proj(agent_state)
            # 个体价值 V_i: (B, N, 1)
            v_ind = self.credit_fc2(self.tanh(self.credit_fc1(e)))  # (B, N, 1)
            # 加权求和得到信用分配增强后的全局价值分量
            v_credit = (a.unsqueeze(-1) * v_ind).sum(dim=1)  # (B, 1)

            self.last_attn_weights = a
            self.last_v_individual = v_ind
            self.last_v_credit = v_credit
        return v