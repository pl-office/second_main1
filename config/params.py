
import argparse  # 导入 argparse 模块用于解析命令行参数

"""
algorithm params
模块级文档字符串：描述该模块用于定义算法运行所需的参数
"""

def get_params():
    """构建并返回解析后的参数对象
    返回：argparse.Namespace，包含所有配置参数
    """
    parser = argparse.ArgumentParser(description = "algorithm params")  # 创建参数解析器，描述信息
    parser.add_argument("--results_dir", type = str, default =
                        "结果统计//MEC/3-24/6-main_result",
                        help = "the dir for saving training results")  # 结果保存目录
    parser.add_argument("--mec_num", type = int, default =3,
                        help = "the number of mecs")   # MEC 的数量
    parser.add_argument("--task_num", type = int, default =2, 
                        help = "the number of arrival tasks at each time-slot")  # 每个时间片到达的任务数
    parser.add_argument("--num", type = str, default ="MEC2", 
                        help = "the number of arrival tasks at each time-slot")  # 每个时间片到达的任务数
    parser.add_argument("--action_dim", type = int, default =4,
                        help = "the dimension of agents' actions")  # action 空间维度
    parser.add_argument("--device_num", type = int, default = 3,
                        help = "the number of devices")  # 设备数量（默认 3）   
    parser.add_argument("--obs_dim", type = int, default = 14,#12个节点状态，5个网络状态，3个任务状态
                        help = "the dimension of agents' observations")  # 单个 agent 的观测维度
    #每个智能体的任务量
    parser.add_argument("--state_dim", type = int, default = 252,#任务数*智能体数*设备数量*14
                        help = "the dimension of global states")  # 全局状态维度
    parser.add_argument("--total_bandwidth", type = float, 
                        default = 10 * pow(10, 6),
                        help = "the total bandwidth (Hz)")  # 总带宽，单位：Hz，默认 10e6
    
   
   
   
   
   
   
    parser.add_argument("--evaluate", type = bool, default = False,
                        help = "evaluate or train")  # 是否在评估模式下运行（True）
    
   
   
    parser.add_argument("--spec_dens", type = float, 
                        default = pow(10, -174 / 10), #服务器噪声功率谱密度，单位 mW/Hz
                        help = "the spectral density of noise power (mW/Hz)")  # 噪声功率谱密度，单位 mW/Hz
    parser.add_argument("--delta", type = float, default = 0.5, 
                        help = "the duration of each time-slot (s)")  # 每个时间片的持续时间（秒）
    parser.add_argument("--std_comp_freq", type = float, default = 2, 
                        help = "the standard computation frequency (Gcycles/s)")  # 标准计算频率，单位 Gcycles/s
    
   
    # 边缘环境（MECs）相关参数
    # parser.add_argument("--Address", type = list, 
    #                     default = [(0,0), (300, 400), (800, 200), (500, 700),(700, 500), (3000, 3000)], 
    #                     help = "the address of mecs")  # 各 MEC 的地理坐标列表（x,y）
    parser.add_argument("--Address", type = list, 
                        default = [(500, 700), (300, 400), (800, 200),(0, 0)], 
                        help = "the address of mecs")  # 各 MEC 的地理坐标列表（x,y）
    
     
   
   
    parser.add_argument("--mec_trans_powers", type = list, 
                        default = [251, 206, 126, 227, 186], #发射功率
                        help = "the transmission powers of mecs (mW)")  # 各 MEC 的发射功率列表（mW）
    
    parser.add_argument("--mec_path_loss", type = list, 
                        default =[[7.174415050824367e-12, 1.17711701112218e-12, 2.7280000057872574e-13],
[7.174415050824367e-12, 1.5874196938561604e-12, 2.0983251388373213e-12],
[1.17711701112218e-12, 1.5874196938561604e-12, 3.1980352331473656e-13] , 
],
                        help = "the path loss of mecs")
    
    parser.add_argument("--mec_comp_freqs", type = list, # 工作频率
                        default = [20, 20, 20, 20, 20], 
                        help = "the computation frequencies of mecs (Gcycles/s)")  # 各 MEC 的计算频率（Gcycles/s）
    
    parser.add_argument("--mec_engy_facs", type = list, 
                        default = [1, 1, 1, 1, 1], 
                        help = "the energy factors of mec (J/Gcycles)")  # 各 MEC 的能量系数（J/Gcycles）
    parser.add_argument("--mec_service_price", type = float, default = 0.1, 
                        help = "the service price of cloud server ($/Gcycles)")  # MEC 提供服务的单价（$/Gcycles），此处作为云服务器价格示例
    
    
    # 任务参数
    
    parser.add_argument("--data_size_inls", type = int, 
                        default = 200, 
                        help = "the data-size intervals of tasks (KB)")  # 每个 MEC 对应任务数据大小范围（KB）
    
    parser.add_argument("--comp_dens_inls", type = int, 
                        default = [[20, 2000], [20, 2000], [20, 2000],  
                                   [20, 2000], [20, 2000]], 
                        help = "the computation-density intervals of tasks (cycles/bit)")  # 计算密度区间（cycles/bit）
    parser.add_argument("--max_data_size", type = float, 
                        default = 200 * 1024 * 8 * pow(10, -6), 
                        help = "the max data-size (Mb)")  # 最大数据大小，单位 Mb（通过 KB->Mb 的换算给出）
    
    parser.add_argument("--max_comp_dens", type = float, 
                        default =2000 * pow(10, -9), 
                        help = "the max computation density (Gcycles/bit)")  # 最大计算密度，单位 Gcycles/bit
    
    # 云环境（Cloud）相关参数
    parser.add_argument("--cld_comp_freq", type = float, default = 50, 
                        help = "the computation frequency of cloud server (Gcycles/s)")  # 云服务器的计算频率（Gcycles/s）
    
    parser.add_argument("--cld_service_price", type = float, default = 0.3, 
                        help = "the service price of cloud server ($/Gcycles)")  # 云服务器的服务价格（$/Gcycles）
  
    parser.add_argument("--cld_energy_weights", type = list, 
                        default = [0.8, 0.8, 0.8, 0.8, 0.8],
                        help = "the weights of tasks' energy consumption")  # 任务能耗权重（列表，用于不同 MEC）
    
    parser.add_argument("--cld_expense_weights", type = list, 
                        default = [0.2, 0.2, 0.2, 0.2, 0.2], 
                        help = "the weights of tasks' cld computation expense")  # 云计算开销权重
    parser.add_argument("--cld_engy_facs", type = float, default = 0.5, 
                        help = "the energy factors of devices (J/Gcycles)")  # 设备在云端计算时的能量系数（J/Gcycles）
   
    # 设备环境（devices）相关参数
    parser.add_argument("--device_engy_facs", type = list, 
                        default = [1, 1, 1, 1, 1], 
                        help = "the energy factors of mec (J/Gcycles)")  # 设备能量系数列表（J/Gcycles）
    parser.add_argument("--device_trans_powers", type = list, 
                        default = [251, 206, 126, 227, 186], 
                        help = "the transmission powers of devices (mW)")  # 设备发射功率列表（mW）
    
    parser.add_argument("--device_path_loss", type = list, 
                        default = [2.4e-10, 7.6e-11, 8.0e-11, 4.0e-10, 5.3e-10], 
                        help = "the path loss of devices")  # 设备路径损耗系数列表
    parser.add_argument("--device_comp_freqs", type = list, 
                        default = [2.1, 2.5, 2.8, 2.2, 2.4], 
                        help = "the computation frequencies of devices (Gcycles/s)")  # 设备计算频率列表（Gcycles/s）
   
    
    
    # 网络结构（神经网络）相关参数2*(mec_num+1)+mec_num+3
   
    #每个智能体的任务量
   
    
    
    
    parser.add_argument("--v_hid_dims", type = list, default = [400, 400],   
                        help = "the dimension of value network's hidden layers")  # value 网络隐藏层维度列表

    parser.add_argument("--p_hid_dims", type = list, default = [400, 400],
                        help = "the dimension of policy network's hidden layers")  # policy 网络隐藏层维度列表

    # 注意力联合价值估计相关参数
    parser.add_argument("--use_attention_value", type = bool, default = True,
                        help = "whether to use attention-based joint value estimation")
    parser.add_argument("--use_credit_assignment", type = bool, default = True,
                        help = "whether to enhance credit assignment via attention weights")
    parser.add_argument("--credit_coef", type = float, default = 0.1,
                        help = "the coefficient of auxiliary credit assignment loss")
    parser.add_argument("--credit_target", type = str, default = "v_joint",
                        help = "aux credit loss target: 'v_joint' (distill, stable) or 'v_tag' (TD target)")
    parser.add_argument("--credit_detach_attn", type = bool, default = True,
                        help = "whether to detach attention weights in credit branch to avoid interfering with attention")
    parser.add_argument("--credit_detach_embed", type = bool, default = True,
                        help = "whether to detach shared embeddings in credit branch to avoid interfering with joint value")
    parser.add_argument("--v_attn_embed_dim", type = int, default = 128,
                        help = "the embedding dimension used in attention-based value network")
    parser.add_argument("--v_attn_heads", type = int, default = 4,
                        help = "the number of attention heads in value network")

    # 门控注意力融合模块（用于鲁棒联合价值估计）
    parser.add_argument("--use_gated_attn_fusion", type = bool, default = True,
                        help = "whether to use gated attention fusion in joint value attention")
    parser.add_argument("--gate_hidden_dim", type = int, default = 64,
                        help = "hidden dimension of the gating MLP")
    parser.add_argument("--gate_init_bias", type = float, default = -2.0,
                        help = "initial bias of gate logit; negative means relying more on residual at start")
    
    parser.add_argument("--use_orthogonal", type = bool, default = True, 
                        help = "whether to use orthogonal-initialization")  # 是否使用正交初始化
    
    # 训练参数
    parser.add_argument("--train_seed", type = int, default = 3456,
                        help = "the training random-seed")  # 训练随机种子
    
    parser.add_argument("--train_episodes", type = int, default =3000,
                        help = "the number of training episodes")  # 训练轮数（episodes）

    parser.add_argument("--train_time_slots", type = int, default = 200,
                        help = "the number of training time-slots")  # 每个 episode 的时间片数量
    
    parser.add_argument("--train_freq", type = int, default = 4,      
                        help = "the training frequency")  # 训练频率（例如每多少步更新一次）
    
    # 网络训练参数
    parser.add_argument("--v_batch_size", type = int, default = 512,
                        help = "the batch-size of value network")  # value 网络的批量大小
    
    parser.add_argument("--p_batch_size", type = int, default = 512, 
                        help = "the batch-size of policy network")  # policy 网络的批量大小
    
    parser.add_argument("--v_epochs", type = int, default = 3,
                        help = "the number of epoches of value network")  # value 网络训练的 epoch 数
    
    parser.add_argument("--p_epochs", type = int, default = 3,  
                        help = "the number of epoches of policy network")  # policy 网络训练的 epoch 数
    
    parser.add_argument("--gamma", type = float, default = 0.99, 
                        help = "the discount factor of reward")  # 折扣因子 gamma
    
    parser.add_argument("--lamda", type = float, default = 0.90,
                        help = "the param about GAE")  # GAE 的 lambda 参数
    
    parser.add_argument("--v_lr", type = float, default = 3e-4,
                        help = "the learning-rate of value network")  # value 网络学习率
    
    parser.add_argument("--p_lr", type = float, default = 1.5e-4,
                        help = "the learning-rate of policy network")  # policy 网络学习率
    
    parser.add_argument("--adam_eps", type = float, default = 1e-5,
                        help = "the param about Adam optimizer")  # Adam 优化器的 eps 参数
    
    parser.add_argument("--use_lr_decay", type = bool, default = True, 
                        help = "whether to use learning-rate decay")  # 是否使用学习率衰减
    
    parser.add_argument("--min_v_lr", type = float, default = 1e-5,            
                        help = "the minimal learning-rate of value network")  # value 网络最小学习率
    
    parser.add_argument("--min_p_lr", type = float, default = 5e-6,        
                        help = "the minimal learning-rate of policy network")  # policy 网络最小学习率
    
    parser.add_argument("--decay_fac", type = float, default = 0.998,  
                        help = "the param about learning-rate decay")  # 学习率衰减因子
    
    parser.add_argument("--use_obs_scaling", type = bool, default = True, 
                        help = "whether to use observation scaling")  # 是否对观测进行缩放
    
    parser.add_argument("--load_scales", type = bool, default = False, 
                        help = "whether to load observation scaling params")  # 是否加载已保存的观测缩放参数
    
    parser.add_argument("--use_reward_scaling", type = bool, default = False, 
                        help = "whether to use reward scaling")  # 是否使用奖励缩放
    
    parser.add_argument("--use_grad_clip", type = bool, default = True, 
                        help = "whether to use gradient clip")  # 是否对梯度进行裁剪
    
    parser.add_argument("--v_grad_clip", type = float, default =5,  
                        help = "the param about value network's gradient clip")  # value 网络梯度裁剪阈值
    
    parser.add_argument("--p_grad_clip", type = float, default = 5,  
                        help = "the param about policy network's gradient clip")  # policy 网络梯度裁剪阈值
    
    parser.add_argument("--use_enty_coef_clip", type = bool, default = True, 
                        help = "whether to use gradient clip") 
    parser.add_argument("--p_clip", type = float, default = 0.1,  
                        help = "the param about ppo clip")  # PPO 的裁剪参数
    parser.add_argument("--enty_coef", type = float, default = 0.008,   
                        help = "the coefficient about policy's entropy")  # 策略熵项系数
    parser.add_argument("--enty_coef_min", type = float, default = 0.002,   
                        help = "the minimal coefficient about policy's entropy")  # 策略熵项最小系数
    parser.add_argument("--enty_coef_decay", type = float, default = 0.990,   
                        help = "the decay factor about policy's entropy coefficient")  # 策略熵项系数衰减因子
   
   
    parser.add_argument("--save_freq", type = int, default = 50, 
                        help = "the save frequency of networks")  # 网络参数保存频率（步数或 episode）
    
    parser.add_argument("--visu_freq", type = int, default = 20, 
                        help = "the visualization frequency")  # 可视化输出频率
    
    parser.add_argument("--load_weights", type = bool, default = False, 
                        help = "whether to load network params")  # 是否加载已保存的模型权重
    
    parser.add_argument("--weights_dir", type = str, default = "weight/", 
                        help = "the dir for saving network params")  # 权重保存目录
    
   
    
    # 评估参数
    parser.add_argument("--eval_seed", type = int, default = 2345,
                        help = "random seed")  # 评估时使用的随机种子
    
    parser.add_argument("--eval_mode", type = str, default = "mappo",
                        help = "evaluation mode")  # 评估模式名称
    
    parser.add_argument("--eval_episodes", type = int, default = 25, 
                        help = "the number of sample-episodes for evaluation")  # 评估时抽样的 episode 数
    
    parser.add_argument("--eval_time_slots", type = int, default = 200,
                        help = "the number of time-slots for evaluation")  # 评估每个 episode 的时间片数量
    
    params = parser.parse_args()  # 解析命令行参数并返回 Namespace 对象

    return params  # 返回解析后的参数对象