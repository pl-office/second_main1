import numpy as np
import math
from config.params import get_params

class NodeStatusMonitor:
    """节点状态监控器 - 监控工作节点的状态"""
    
    def __init__(self):
      
        #self.node_performance = {f'工作节点{i+1}': 0 for i in range(node_count)}#初始化节点性能node'的运行性能
        self.edge_nodes = []  # 存储所有边缘节点（MecEnv实例）
        self.mec_num = get_params().mec_num
        self.address = get_params().Address
        
        
    def add_node(self, node):
        """添加边缘节点"""
        self.edge_nodes.append(node)

    def get_node_status(self):
        """收集所有节点的状态（用于调度）"""
        status_dict = {}  # 使用字典而不是列表，键为节点ID
        node_status = {}
        for node in self.edge_nodes:
            # 重置信道增益（保持原逻辑）
            node.channel_gain = [pl * np.random.exponential(1) for pl in node.mec_path_loss]
            cgnp_rto = [cg / node.mec_noise_power for cg in node.channel_gain]
            # 将cgnp_rto与distance_list的值一一对应相乘（假设顺序一致）
           
            #计算距离
            distance_list=self.calculate_distance(node)
            # 将距离的-2次方用于衰减（对列表进行逐元素运算）
            # distance_list 是一个字典
            cgnp_rto1 = {k: cgnp_rto[i]   for i, (k, v) in enumerate(distance_list.items())}#1.76
            #cgnp_rto1 = {k: cgnp_rto[i] * v ** -2 for i, (k, v) in enumerate(distance_list.items())}#1.76

            trans_rate = {
                k: node.bandwidth * math.log(1 + node.mec_trans_powers * v, 2) * 1e-6
                for k, v in cgnp_rto1.items()
            }
            # 将字典最后一个键的值赋值为40
           
            last_key = list(trans_rate.keys())[-1]
            trans_rate[last_key] = 40
            # 创建当前节点的状态详情
            node_detail = {
                #边缘服务器状态
                "comp": node.mec_comp_freq ,
                "task_queue_len": node.comp_ql,
                #网络状态
                #"bandwidth": node.bandwidth,
                "trans_rate":trans_rate,
                "cgnp_rto_mm":  cgnp_rto1,#当前节点的信道增益和信道增益/噪声功率
                "node_status": node.node_status,
                
            }
            
            # 直接将节点ID作为键添加到主字典中
            status_dict[node.mec_id] = node_detail
            node_status[node.mec_id] = node.node_status
       
       
        return status_dict
    def calculate_distance(self, node):
        """计算两点之间的距离"""
     
        distance1 = {}
        for item in range(self.mec_num+1):
            coord1=self.address[item]
            if node.address == coord1:
                continue
            else:
                x1, y1 = node.address
                x2, y2 = coord1
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 1e-3
                distance1[item+1] = distance
                

        return distance1
               