import copy
import math
import random
import numpy as np
from typing import List, Dict, Any


# 单个任务
class Task():
    def __init__(self, data_size, comp_dens, mec_num):
        """单个任务的数据结构，不再在内部调用 get_params，而是由外部传入 mec_num。"""
        # unit: Mb
        self.mec_num = mec_num
        self.data_size = data_size
        # unit: Gcycles/bit
        self.comp_dens = comp_dens
        # unit: s
        self.dly_cons = None
     
        #传输时间
       
        self.task_counter = 0
        #归一化指标
        self.norm_csum_engy = None # 标准化能耗
        self.norm_comp_expn = None # 标准化计算成本
        self.norm_comp_dly = None # 标准化延迟基准
     
        self.comp_expn=[None for i in range(self.mec_num+1)]#边缘计算费用
        self.scource_mec=None#任务来源的边缘节点
        self.offl_dz=None#每个任务分配到每个节点的数据量
        #本地到边缘传输
        self.transtime_le =None# 传输时间（本地到边缘）
        self.le_csum_engy=None#本地到边缘传输能耗
        #边缘之间传输
        self.trans_time=[None for i in range(self.mec_num+1)]#每个任务分配到每个节点的传输时间(边缘之间传输时延)
        self.trans_engy=[None for i in range(self.mec_num+1)]#边缘传输能耗（边缘之间传输能耗）
        #边缘本地计算
        self.comp_engy=[None for i in range(self.mec_num+1)]#边缘本地计算能耗（边缘本地计算能耗）
        self.comp_dly=[None for i in range(self.mec_num+1)]#计算延迟
     
       
        
        
       
    

       
        


    def __str__(self):
        return "data_size: " + str(self.data_size) + \
               "\ncomp_dens: " + str(self.comp_dens) + \
               "\ndly_cons: " + str(self.dly_cons) + \
               "\nl_comp_dly: " + str(self.l_comp_dly) + \
               "\nl_csum_engy: " + str(self.l_csum_engy) + \
               "\noffl_data_size: " + str(self.offl_dz) + \
               "\ntrans_time: " + str(self.trans_time) + \
               "\ne_comp_dly: " + str(self.e_comp_dly) + \
               "\ne_csum_engy: " + str(self.trans_time_engy) + \
               "\ncomp_expn: " + str(self.comp_expn) + \
               "\nnorm_csum_engy: " + str(self.norm_csum_engy) + \
               "\nnorm_comp_expn: " + str(self.norm_comp_expn)
        
class DeviceEnv():
    def __init__(self, env_id, params):
        self.bandwidth = params.total_bandwidth / params.device_num
        self.channel_gain = None
        self.noise_power = params.spec_dens * self.bandwidth   
        self.std_comp_freq = params.std_comp_freq
      

        # 设备环境
        self.env_id = env_id
        self.device_trans_powers = params.device_trans_powers[env_id-1]
        self.device_path_loss = params.device_path_loss[env_id-1]
        self.device_engy_fac = params.device_engy_facs[env_id-1]
        self.mec_num = params.mec_num
       
        #任务
        self.task_num = params.task_num
        self.comp_dens_inl = params.comp_dens_inls[env_id-1]
        self.service_price = params.mec_service_price+params.cld_service_price
        self.data_size_inl = params.data_size_inls
        self.sched_tasks=[]

    
    def generate_task(self):
        task_msgs=[]
        # reset channel gain
        self.channel_gain = self.device_path_loss * np.random.exponential(1)
        
        # reset scheduling tasks
      
        for i in range(self.task_num):
          
            data_size = np.random.uniform(self.data_size_inl)
            data_size = data_size * 1024 * 8 * pow(10, -6)#Mb用于传输速率
            
            # unit: Gcycles/bit
            comp_dens = np.random.uniform(self.comp_dens_inl[0], self.comp_dens_inl[1])
            comp_dens = comp_dens * pow(10, -9)#Gcycles/bit用于计算延迟
            comp = data_size * pow(10, 6) * comp_dens# Gcycles/s（即 GHz）用于计算延迟



             # 归一化指标
            task = Task(data_size, comp_dens, self.mec_num)
            task.norm_csum_engy = data_size * self.device_engy_fac
            task.norm_comp_expn = data_size * self.service_price
            task.norm_comp_dly = data_size * pow(10, 6) * self.comp_dens_inl[1] * pow(10, -9) / 20  # 最大密度任务在单MEC的理想延迟
            # task.norm_csum_engy= norm_csum_engy
            # task.norm_comp_expn= norm_comp_expn
            task.transtime_le, task.le_csum_engy = self.tr_compute(task)
            task.dly_cons = comp/self.std_comp_freq #延迟约束
            # task.dly_cons = dly_cons
  
            self.sched_tasks.append(task)

            task_msgs.append(task)

        return task_msgs


   
    

    #计算传输时间      
    def tr_compute(self,task):
        
        # transmission-power ratio
    
        trans_power = self.device_trans_powers 
        # unit: Mb/s
        trans_rate = self.bandwidth * math.log(1 + trans_power * self.channel_gain / 
                                               self.noise_power, 2) * pow(10, -6)
      
        
        
        trans_time_le = task.data_size / trans_rate
        le_csum_engy = trans_power * pow(10, -3) * task.data_size / trans_rate
       
        
        # update channel gain
        self.channel_gain = self.device_path_loss * np.random.exponential(1)
  

        return  trans_time_le , le_csum_engy