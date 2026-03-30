from config.params import get_params
from controller import Controller
import os

# os.makedirs('result8', exist_ok=True)
if __name__ == '__main__':
    # params
    params = get_params() 
    
    ctr = Controller(params)
    
   
    
    if params.evaluate == False:
        
        ctr.train()
     
    else:
        '''
        if eval_mode = "mappo", open the switches (load_weights = True, load_scales = True)
        '''
        joint_reward, mec_rewards, \
        joint_cost, mec_costs, \
        cld_comp_ql, mec_comp_qls, \
        mec_comp_dlys, mec_csum_engys, \
        mec_comp_expns, mec_overtime_nums = ctr.evaluate()
        
        print("joint_reward:\n", joint_reward)
        print("mec_rewards:\n", mec_rewards)
        print("joint_cost:\n", joint_cost)
        print("mec_costs:\n", mec_costs)
        print("cld_comp_ql:\n", cld_comp_ql)
        print("mec_comp_qls:\n", mec_comp_qls)
        print("mec_comp_dlys:\n", mec_comp_dlys)
        print("mec_csum_engys:\n", mec_csum_engys)
        print("mec_comp_expns:\n", mec_comp_expns)
        print("mec_overtime_nums:\n", mec_overtime_nums)
