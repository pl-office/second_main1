from env.edge_env import MecEnv
from env.cld_env import CldEnv
from network.node_manager import NodeStatusMonitor


class CloudEnv():
    def __init__(self, params):
        """
        初始化云环境类
        
        参数:
            params: 包含环境配置参数的对象
        """
        self.task_num = params.task_num
        self.device_num = params.device_num
        self.mec_num = params.mec_num  # MEC(多接入边缘计算)节点的数量
       

        # cld env
        self.cld_env = CldEnv(params)
        # mec envs
        self.mec_envs =[]

        #初始化MEC环境
        for i in range(self.mec_num):
            self.mec_envs.append(MecEnv(i+1, params))

        # 初始化节点管理器并添加MEC节点
        self.node_manager=NodeStatusMonitor()  # 创建NodeManager实例
        for mec_env in self.mec_envs:
            self.node_manager.add_node(mec_env)  # 将每个MEC节点添加到管理器
        # reward
        self.cld_expense_weights = params.cld_expense_weights
        self.cld_energy_weights = params.cld_energy_weights

        
    def reset(self):
            cld_obs = self.cld_env.reset()
            mec_obss = [None for i in range(self.mec_num)]
            node_performance = self.node_manager.get_node_status()
            for i in range(self.mec_num):
                mec_obss[i]= self.mec_envs[i].reset(node_performance)
            return cld_obs,mec_obss
      


        
    def step(self, mec_acts):
        node_performance = self.node_manager.get_node_status()
        mec_sched_tasks = [None for i in range(self.mec_num)]
        for i in range(self.mec_num):
            sched_tasks = self.mec_envs[i].compute(mec_acts[i], self.node_manager, node_performance)
            mec_sched_tasks[i] = sched_tasks
        self.cld_env.compute(mec_sched_tasks)
        mec_rewards = [0 for i in range(self.mec_num)]
        mec_costs = [0 for i in range(self.mec_num)]
        mec_comp_dlys = [0 for i in range(self.mec_num)]
        mec_csum_engys = [0 for i in range(self.mec_num)]
        mec_comp_expns = [0 for i in range(self.mec_num)]
        mec_overtime_nums = [0 for i in range(self.mec_num)]

        for i in range(self.mec_num):
            sched_tasks = mec_sched_tasks[i]
            task_num = len(sched_tasks)
            for j in range(task_num):
                task = sched_tasks[j]
                # 时延 = max(各节点计算完成时间 + 结果回传时延)
                # 本地节点 trans_time=0 不受影响，远程节点需加上回传时延
                comp_dly =  max(
                    task.comp_dly[idx] + (task.trans_time[idx] if task.trans_time[idx] else 0)
                    for idx in range(len(task.comp_dly))
                )
                #时间(本地到边缘传输时间+边缘本地计算时间+边缘之间传输时间+结果回传时间)
                mec_comp_dlys[i] += 1 / (j + 1) * (comp_dly - mec_comp_dlys[i])
                #能耗（本地到边缘传输能耗+边缘本地计算能耗+边缘之间传输能耗）
                csum_engy = sum(task.comp_engy) 
                # use incremental averaging consistent with others (avoid raw sum over tasks)
                mec_csum_engys[i] += 1 / (j + 1) * (csum_engy - mec_csum_engys[i])
                #mec_csum_engys[i] += csum_engy
                #费用
                total_expn = sum(task.comp_expn)
                mec_comp_expns[i] += 1 / (j + 1) * (total_expn - mec_comp_expns[i])
                mec_costs[i] += self.cld_energy_weights[i] * csum_engy + \
                                self.cld_expense_weights[i] * total_expn
                if comp_dly > task.dly_cons:
                    mec_overtime_nums[i] += 1
                    #mec_rewards[i] += -5000
                norm_csum_engy = task.norm_csum_engy
                norm_comp_expn = task.norm_comp_expn
                # mec_rewards[i] += -10 * (self.cld_energy_weights[i] *
                #                         csum_engy / norm_csum_engy +
                #                         self.cld_expense_weights[i] *
                #                         total_expn / norm_comp_expn)
                #self.cld_energy_weights=0.8, self.cld_expense_weights=0.2
                mec_rewards[i] += -1000 * (0.5 *
                                        csum_engy / norm_csum_engy +
                                        0.5 * comp_dly / task.norm_comp_dly)
        joint_reward = sum(mec_rewards)
        joint_cost= sum(mec_costs)

        next_mec_obss = [None for i in range(self.mec_num)]
        node_performance = self.node_manager.get_node_status()
        for i in range(self.mec_num):
            next_mec_obss[i] = self.mec_envs[i].get_obs(node_performance)

        return joint_reward, mec_rewards, \
            joint_cost, mec_costs, \
            mec_comp_dlys, mec_csum_engys, \
            mec_comp_expns, mec_overtime_nums, \
            next_mec_obss
