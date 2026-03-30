import torch
import torch.nn as nn
from typing import Optional
import math


class MultiAgentAttention(nn.Module):
    """\
    多智能体自注意力模块（可独立使用）。

    输入:  state 形状为 (batch_size, n_agents, state_dim)
    输出:  context 形状为 (batch_size, embed_dim)，为聚合后的联合特征

    参数:
        state_dim: 每个智能体局部状态的维度
        embed_dim: 注意力内部使用的嵌入维度
        num_heads: Multi-Head 数量
        use_layer_norm: 是否在注意力输出后使用 LayerNorm
    """

    def __init__(self, state_dim: int, embed_dim: int, num_heads: int = 4, use_layer_norm: bool = True):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 将原始 state 映射到注意力空间
        self.input_proj = nn.Linear(state_dim, embed_dim)

        # 使用 PyTorch 内置的多头自注意力
        # batch_first=True: 输入输出形状都为 (batch, seq_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          batch_first=True)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。

        参数:
            state: (batch_size, n_agents, state_dim)
            mask:  (batch_size, n_agents)，无效智能体位置为 True/1（与 pytorch attn 的 key_padding_mask 语义一致）

        返回:
            context: (batch_size, embed_dim)
        """
        # 线性映射到嵌入空间
        # (B, N, D_state) -> (B, N, D_embed)
        x = self.input_proj(state)

        # MultiheadAttention 的 key_padding_mask 形状为 (B, N)，True 表示需要被 mask 的位置
        key_padding_mask = mask if mask is not None else None

        # 自注意力: Q = K = V = x
        # attn_out: (B, N, D_embed)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)

        if self.use_layer_norm:
            attn_out = self.layer_norm(attn_out)

        # 聚合所有智能体的特征，得到联合表示
        # 这里采用平均池化，你也可以按需要改为加权池化或取特定 agent
        context = attn_out.mean(dim=1)  # (B, D_embed)
        return context


class JointValueAttention(nn.Module):
    """\
    基于注意力的联合价值表示模块（更贴近“联合价值估计”的注意力聚合写法）。

    与 MultiAgentAttention 的区别：
    - MultiAgentAttention: agent 之间做 self-attention，然后简单池化。
    - JointValueAttention: 先构造一个全局 query（由所有 agent 表示聚合得到），
      再对各 agent 的 key/value 做注意力加权求和，得到联合上下文向量。

    输入:
        state: (batch_size, n_agents, state_dim)
        mask:  (batch_size, n_agents)，True 表示该 agent 位置无效/需要被 mask

    输出:
        context: (batch_size, embed_dim)

    备注:
        该模块返回的注意力权重可解释为“当前时刻各 agent 对联合价值贡献的重要性权重”（按 head 分开）。
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        use_layer_norm: bool = True,
        use_gated_fusion: bool = False,
        gate_hidden_dim: int = 64,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_proj = nn.Linear(state_dim, embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 门控注意力融合：g * AttentionOutput + (1-g) * ResidualInput
        # g 由轻量 MLP + Sigmoid 生成，范围 (0,1)
        self.use_gated_fusion = use_gated_fusion
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_init_bias = gate_init_bias
        if self.use_gated_fusion:
            self.gate_fc1 = nn.Linear(embed_dim * 2, gate_hidden_dim)
            self.gate_fc2 = nn.Linear(gate_hidden_dim, 1)
            self.gate_act = nn.ReLU()
            self.gate_sigmoid = nn.Sigmoid()

            # 稳定化初始化：默认更依赖残差输入（小 g），避免早期噪声注意力污染价值估计
            nn.init.constant_(self.gate_fc1.weight, 0.0)
            nn.init.constant_(self.gate_fc1.bias, 0.0)
            nn.init.constant_(self.gate_fc2.weight, 0.0)
            nn.init.constant_(self.gate_fc2.bias, float(gate_init_bias))

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """前向传播。

        参数:
            state: (B, N, D_state)
            mask:  (B, N)  True 表示需要被 mask
            return_attn: 是否返回注意力权重

        返回:
            context: (B, D_embed)
            attn(optional): (B, H, N)
        """
        bsz, n_agents, _ = state.shape

        # 1) agent 表示嵌入
        # E: (B, N, D_embed)
        e = self.input_proj(state)

        # 2) 构造全局 query：使用平均池化得到全局摘要，再映射成 query
        # g: (B, D_embed)
        g = e.mean(dim=1)

        # Q: (B, D_embed) -> (B, H, D_head)
        q = self.q_proj(g).view(bsz, self.num_heads, self.head_dim)

        # K,V: (B, N, D_embed) -> (B, H, N, D_head)
        k = self.k_proj(e).view(bsz, n_agents, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(e).view(bsz, n_agents, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 注意力打分与归一化
        # scores: (B, H, N)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).to(dtype=torch.bool), float("-inf"))

        attn = torch.softmax(scores, dim=-1)

        # 4) 加权聚合得到联合上下文
        # context_h: (B, H, D_head)
        context_h = torch.matmul(attn.unsqueeze(2), v).squeeze(2)

        # context_attn: (B, D_embed)
        context_attn = context_h.reshape(bsz, self.embed_dim)
        context_attn = self.out_proj(context_attn)

        # 残差输入：全局摘要（由 agent 嵌入平均得到）
        residual = g  # (B, D_embed)

        if self.use_gated_fusion:
            gate_inp = torch.cat([context_attn, residual], dim=-1)
            gate_logit = self.gate_fc2(self.gate_act(self.gate_fc1(gate_inp)))
            gate = self.gate_sigmoid(gate_logit)  # (B, 1)
            context = gate * context_attn + (1.0 - gate) * residual
        else:
            context = context_attn

        if self.use_layer_norm:
            context = self.layer_norm(context)

        if return_attn:
            return context, attn
        return context


# class AttentionValueNet(nn.Module):
#     """带注意力机制的联合价值网络示例。

#     用法示例：
#         net = AttentionValueNet(state_dim=obs_dim_per_agent,
#                                 embed_dim=128,
#                                 num_heads=4,
#                                 v_hid_dims=[128, 64])
#         v = net(state)  # state: (B, N, state_dim)
#     """

#     def __init__(self,
#                  state_dim: int,
#                  embed_dim: int,
#                  num_heads: int,
#                  v_hid_dims,
#                  use_orthogonal: bool = False):
#         super().__init__()

#         def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
#             for name, p in layer.named_parameters():
#                 if "bias" in name:
#                     nn.init.constant_(p, 0.0)
#                 elif "weight" in name:
#                     nn.init.orthogonal_(p, gain=gain)

#         self.attn = MultiAgentAttention(state_dim=state_dim,
#                                         embed_dim=embed_dim,
#                                         num_heads=num_heads,
#                                         use_layer_norm=True)

#         # 注意力后的 context 作为 Value MLP 的输入
#         self.fc1 = nn.Linear(embed_dim, v_hid_dims[0])
#         self.fc2 = nn.Linear(v_hid_dims[0], v_hid_dims[1])
#         self.fc3 = nn.Linear(v_hid_dims[1], 1)
#         self.tanh = nn.Tanh()

#         if use_orthogonal:
#             orthogonal_init(self.fc1)
#             orthogonal_init(self.fc2)
#             orthogonal_init(self.fc3)

#     def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """state: (batch_size, n_agents, state_dim)"""
#         # 先通过注意力进行联合表示
#         context = self.attn(state, mask=mask)  # (B, embed_dim)

#         # 再通过多层感知机得到联合价值
#         x = self.tanh(self.fc1(context))
#         x = self.tanh(self.fc2(x))
#         v = self.fc3(x)  # (B, 1)
#         return v
