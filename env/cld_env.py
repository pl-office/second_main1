class CldEnv():
    def __init__(self, params):
        # unit: s
        self.delta = params.delta
        # unit: Gcycles/s
        self.cld_comp_freq = params.cld_comp_freq
        self.cld_engy_facs = params.cld_engy_facs  # J/Gcycles
        self.cld_service_price = params.cld_service_price  # $/Gcycles
        
        # unit: Gcycles
        self.comp_ql = 0
        
    def reset(self):
        # reset computation-queue length
        self.comp_ql = 0
        
        # obs
        obs = self.get_obs()
        
        return obs
        
    def get_obs(self):
        obs = [self.comp_ql]
        
        return obs
        
    def compute(self, mec_sched_tasks):
        """处理来自所有边缘节点的任务"""
        # 收集所有卸载到云端的任务
        mec_sched_tasks_ = []
      
        for sched_tasks in mec_sched_tasks:
            mec_sched_tasks_ += sched_tasks
        mec_sched_tasks_ = sorted(mec_sched_tasks_, 
                                     key = lambda x: x.trans_time)
        # 计算云端任务的完成时间和能耗
        comp_dly = self.comp_ql / self.cld_comp_freq
        self.comp_ql = max(0, self.comp_ql - self.cld_comp_freq * self.delta)
        for task in mec_sched_tasks_:
            if task.trans_time[-1] == 0:
                task.comp_dly[-1] = 0
            else:
                task.comp_dly[-1] = max(comp_dly, task.trans_time[-1]) + task.offl_dz[-1] * \
                                  pow(10, 6) * task.comp_dens / self.cld_comp_freq
                self.comp_ql += max(0, task.offl_dz[-1] * pow(10, 6) * task.comp_dens - 
                                    self.cld_comp_freq * max(0, self.delta - 
                                                              max(comp_dly, task.trans_time[-1])))
                task.comp_engy[-1] =  1.2*task.trans_time[-1]+0.2*task.comp_dly[-1]#计算能耗  # 计算能耗
                comp_dly = task.comp_dly[-1]
