import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime
from copy import deepcopy


alg_name =  'TwoLLMsOb'     

alpha = .5
a1_to_a2_ratio = .5 # stepsize of the first layer
RMSPw = .99
RMSPc = .99

beta_m = 0.0

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/_test3.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for a1_to_a2_ratio in [.5]:
                for RMSP in [.99]:
                    sweeps.append({
                                    'file_path':os.path.abspath(__file__),
                                    'alpha':alpha,
                                    'a1_to_a2_ratio':a1_to_a2_ratio,
                                    'RMSPw':RMSP,
                                    'RMSPc':RMSP,
                                    
                                    'bias':'True'
                                    })
                    
    return sweeps
    #command_list = ['python3 IDBD.py ' + ' '.join([f'--{param}={config[param]}' for param in config]) for config in list_param_sweeps]
    #return command_list

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("utils/configurator.py").read())  # overrides from command line or config file

bias = False if bias in [False,'False',0,'0'] else True
alg_config={
            'dataset':dataset,
            'alg_name':f'{alg_name}',
            'params':{
                    'alpha':alpha,
                    'a1_to_a2_ratio':a1_to_a2_ratio,
                    'RMSPw':RMSPw,
                    'RMSPc':RMSPc,
                    },
            'bias':'True',
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,0,None))



if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)

    trace_grad_before_delta_w = 0.0
    trace_grad_before_delta_c = 0.0
    trace_abs_x = 0.0
    normalizer_w = 1.0
    normalizer_c = 1.0

    
    
    wb = np.zeros(d+1)
    c = np.zeros(d)  # centers or the bias weights of the first layer
    m_c = 0
    m_w = 0

    MSE_list = []
    t = 0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        if 1:
            x,z = env.next()

            if t==1:
                c = x+0.0
                init_c = c+0.0
            
            xb = np.append(x-c,1)
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)

            trace_grad_before_delta_w = RMSPw*trace_grad_before_delta_w + (1-RMSPw)* ((xb)**2)
            normalizer_w = np.sqrt(trace_grad_before_delta_w/(1-RMSPw**t)+1e-16) 
            trace_grad_before_delta_c = RMSPc*trace_grad_before_delta_c + (1-RMSPc)* ((wb[:-1])**2)
            normalizer_c = np.sqrt(trace_grad_before_delta_c/(1-RMSPc**t)+1e-16) 
            

            obn_coeff = 1/((xb*xb / normalizer_w).sum() + (a1_to_a2_ratio * wb[:-1] * wb[:-1] / normalizer_c).sum())
            
            c = c  - obn_coeff * alpha * a1_to_a2_ratio * delta * wb[:-1] / normalizer_c # This is the gradient update of the first layer weights
            wb = wb +  obn_coeff * alpha *  delta * xb / normalizer_w

            if np.isnan(wb).any(): break
        else:
            break
        
    #     if iteration>1_100_000-10:
    #          print(x[:5])
    # print()
    # print(init_c[:5])
    # print(c[:5])
    # print(wb[:5])
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    