import copy
import math
import numpy as np
from env.device_env import DeviceEnv



class MecEnv():
    def __init__(self, mec_id, params, node_manager=None):
        # 初始化代码保持不变（已移除日志记录）
        self.node_manager = node_manager
        self.delta = params.delta
        self.train_time_slots = params.train_time_slots
        self.eval_time_slots = params.eval_time_slots

        # 网络参数
        self.bandwidth = params.total_bandwidth / params.mec_num#带宽
        self.mec_channel_gain = None
        self.mec_noise_power = params.spec_dens * self.bandwidth#噪声功率
        
    
         #边缘节点
        self.address = params.Address[mec_id-1]#位置
        self.mec_path_loss = params.mec_path_loss[mec_id-1]#路径损失
        self.mec_trans_powers = params.mec_trans_powers[mec_id-1]#发射功率
        self.mec_num = params.mec_num
        self.mec_id = mec_id
        self.mec_comp_freq = params.mec_comp_freqs[mec_id-1]#工作频率
        self.mec_service_price = params.mec_service_price
        self.mec_engy_facs = params.mec_engy_facs[mec_id-1]
        self.task_num= params.task_num
    
        self.comp_ql = 0
        self.current_comp_dly=None
        self.mec_sched_tasks = []
        self.device_num = params.device_num
        self.device_envs = []
        self.mec_sched_tasks = []
        self.temporary_storage = []

        
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i+1, params))
            task_msgs = self.device_envs[i].generate_task()
            self.temporary_storage.append(task_msgs)
        for sublist in self.temporary_storage:
            self.mec_sched_tasks.extend(sublist)


       
        self.local_comps = {}
        self.node_status = 'available'
        self.task_counter = 0
        self.re_performance = None
        
        
    def reset(self, node_performance):

        self.comp_ql = 0
        self.mec_channel_gain =  [pl * np.random.exponential(1) for pl in self.mec_path_loss]
        node_state, task_msgs = self.get_obs(node_performance)
        return node_state, task_msgs
    
        
    def get_obs(self, node_performance):
        # comp_ql = self.comp_ql
        # cgnp_rto = self.mec_channel_gain / self.mec_noise_power
        task_msgs = []
     
        #任务数据
        for i in range(len(self.mec_sched_tasks)):
            data_size = self.mec_sched_tasks[i].data_size
            comp_dens = self.mec_sched_tasks[i].comp_dens
            dly_cons = self.mec_sched_tasks[i].dly_cons
            taskms=[data_size, comp_dens, dly_cons]
            task_msgs.append(taskms)
           
        
        #节点数据
        node_state=[]
        
        # 提取所有节点的状态
        for node_id in range(1, self.mec_num + 1):#按顺序提取
                node_data = node_performance[node_id]
                node_state.append(node_data["task_queue_len"])
                node_state.append(node_data["comp"])
        node_state.append(0)#云端队列长度
        node_state.append(50)#云端计算能力
        #当前云端传输信噪比增益
        
        #提取网络状态
        network_state=[]
        for node_id in range(1, self.mec_num + 1):
            if node_id == self.mec_id:  # 本地节点
                node_data = node_performance[node_id]
                for key, val in node_data["trans_rate"].items():
                    network_state.append(val)
                node_state.extend(network_state)
        return node_state, task_msgs  # node_state:17个值， task_msgs:3个列表，每个列表3个值

    def compute(self, act, status_monitor, node_manager):
       
        # 优化：批量初始化各节点的卸载数据量字典
        node_count = 3  # 节点数（不含云端）
        offl_dzs = [{} for _ in range(node_count)]
        offl_dzs_cloud = {}

        # 计算每个任务的卸载数据量，并存储到对应节点
        for i, task in enumerate(self.mec_sched_tasks):
            task.scource_mec_id = self.mec_id
            for j in range(node_count):
                offl_dzs[j][i] = task.data_size * act[i][j]
            offl_dzs_cloud[i] = task.data_size * act[i][-1]
            # 记录每个任务的各节点卸载量
            task.offl_dz = [offl_dzs[j][i] for j in range(node_count)] + [offl_dzs_cloud[i]]

        # 对每个节点的卸载量排序
        offl_dzs_sorted = [sorted(d.items(), key=lambda x: x[1]) for d in offl_dzs]
        offl_dzs_cloud_sorted = sorted(offl_dzs_cloud.items(), key=lambda x: x[1])
        all_offl_dzs = offl_dzs_sorted + [offl_dzs_cloud_sorted]

        # 传输功率与速率
        trans_power = self.mec_trans_powers
        node_data = node_manager[self.mec_id]
        trans_rate = node_data["trans_rate"]

        # 计算每个节点最大可传输数据量
        total_trans_dz = {k: self.delta * v for k, v in trans_rate.items()}
        total_offl_dz = {k: 0 for k in trans_rate.keys()}

        # 依次处理本地、边缘、云端
        for node_idx, node_offl_dzs in enumerate(all_offl_dzs, 1):
            total_local_comp = self.comp_ql
            for task_id, offl_dz in node_offl_dzs:
                sub_task = self.mec_sched_tasks[task_id]
                # 本地计算
                if node_idx == self.mec_id:
                    if sub_task.offl_dz[node_idx-1] == 0:
                        sub_task.trans_time[node_idx-1] = 0
                        sub_task.trans_engy[node_idx-1] = 0
                        sub_task.comp_dly[node_idx-1] = 0
                        sub_task.comp_engy[node_idx-1] = 0
                    # 计算量累加
                    comp_cycles = offl_dz * pow(10, 6) * sub_task.comp_dens
                    total_local_comp += comp_cycles
                    sub_task.trans_time[node_idx-1] = 0
                    sub_task.trans_engy[node_idx-1] = 0
                    sub_task.comp_dly[node_idx-1] = total_local_comp / self.mec_comp_freq
                    sub_task.comp_engy[node_idx-1] = 1.2 * sub_task.trans_time[node_idx-1] + 0.2 * sub_task.comp_dly[node_idx-1]
                    sub_task.comp_expn[node_idx-1] = total_local_comp * pow(10, 6) * sub_task.comp_dens * self.mec_service_price
                # 云端
                elif node_idx == len(all_offl_dzs):
                    if sub_task.offl_dz[-1] == 0:
                        sub_task.trans_time[-1] = 0
                        sub_task.trans_engy[-1] = 0
                        sub_task.comp_dly[-1] = 0
                        sub_task.comp_engy[-1] = 0
                    # 限制最大传输量
                    offl_dz = min(offl_dz, total_trans_dz.get(node_idx, 0))
                    total_trans_dz[node_idx] = total_trans_dz.get(node_idx, 0) - offl_dz
                    total_offl_dz[node_idx] = total_offl_dz.get(node_idx, 0) + offl_dz
                    if offl_dz == 0:
                        sub_task.trans_time[node_idx-1] = 0
                        sub_task.trans_engy[node_idx-1] = 0
                        sub_task.comp_expn[node_idx-1] = 0
                        sub_task.comp_engy[node_idx-1] = 0
                    else:
                        sub_task.trans_time[node_idx-1] = total_offl_dz[node_idx] / trans_rate.get(node_idx, 1)
                        sub_task.trans_engy[node_idx-1] = trans_power * pow(10, -3) * total_offl_dz[node_idx] / trans_rate.get(node_idx, 1)
                        sub_task.comp_engy[node_idx-1] = None
                        sub_task.comp_dly[node_idx-1] = None
                        sub_task.comp_expn[node_idx-1] = 100
                # 其他边缘节点
                else:
                    if sub_task.offl_dz[-1] == 0:
                        sub_task.trans_time[-1] = 0
                        sub_task.trans_engy[-1] = 0
                        sub_task.comp_dly[-1] = 0
                        sub_task.comp_engy[-1] = 0
                    offl_dz = min(offl_dz, total_trans_dz.get(node_idx, 0))
                    total_trans_dz[node_idx] = total_trans_dz.get(node_idx, 0) - offl_dz
                    total_offl_dz[node_idx] = total_offl_dz.get(node_idx, 0) + offl_dz
                    if offl_dz == 0:
                        sub_task.trans_time[node_idx-1] = 0
                        sub_task.trans_engy[node_idx-1] = 0
                    else:
                        sub_task.trans_time[node_idx-1] = total_offl_dz[node_idx] / trans_rate.get(node_idx, 1)
                        sub_task.trans_engy[node_idx-1] = trans_power * pow(10, -3) * total_offl_dz[node_idx] / trans_rate.get(node_idx, 1)
                    # 调用目标节点的本地计算
                    target_env = status_monitor.edge_nodes[node_idx-1]
                    target_env.sub_local_compute(node_idx, sub_task, offl_dz, sub_task.trans_time[node_idx-1])

        # 更新本地计算队列长度
        self.comp_ql = max(0, total_local_comp - self.mec_comp_freq * self.delta)
        # 更新信道增益
        self.channel_gain = [pl * np.random.exponential(1) for pl in self.mec_path_loss]

        # 生成新任务
        mec_sched_tasks = copy.copy(self.mec_sched_tasks)
        self.temporary_storage.clear()
        for device_env in self.device_envs:
            task_msgs = device_env.generate_task()
            self.temporary_storage.append(task_msgs)
        self.mec_sched_tasks.clear()
        for sublist in self.temporary_storage:
            self.mec_sched_tasks.extend(sublist)
        return mec_sched_tasks
    


    def sub_local_compute(self, id, sub_task, offl_dz, trans_time):
        
        # 优化：本地节点处理卸载任务
        offl_dz_cycles = offl_dz * pow(10, 6) * sub_task.comp_dens
        self.current_comp_dly = self.comp_ql / self.mec_comp_freq
        self.comp_ql = max(0, self.comp_ql - self.mec_comp_freq * self.delta)
        # 计算延迟
        if trans_time == 0:
            sub_task.comp_dly[id-1] = 0
        else:
            sub_task.comp_dly[id-1] = max(self.current_comp_dly, trans_time) + offl_dz_cycles / self.mec_comp_freq
        sub_task.comp_expn[id-1] = offl_dz_cycles * self.mec_service_price
        sub_task.comp_engy[id-1] = 1.2 * sub_task.trans_time[id-1] + 0.2 * sub_task.comp_dly[id-1]
        # 更新队列长度
        available_processing = self.mec_comp_freq * max(0, self.delta - max(self.current_comp_dly, trans_time))
        self.comp_ql += max(0, offl_dz_cycles - available_processing)
                
      