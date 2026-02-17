import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime
from copy import deepcopy


alg_name =  'TwoLLMsL1C_MGEN'     # first layer has gradient based centering (weight updates are scaled by RMSProp of inputs)

alpha0 = 1e-3
RMSPw = .99
RMSPw_meta = .99
RMSPc_meta = .99
meta_stepsize = 1e-3
alpha_min = 1e-4

h_decay = .9 # '1-alpha'  or  '1-alpha*x2'

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/_test3.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for meta_stepsize in [1e-4, 1e-3, 1e-2, 1e-1]:
        for h_decay in ['1-alpha']:
                for RMSP in [.999]:
                    for RMSP_meta in [.99]:
                        sweeps.append({
                                        'file_path':os.path.abspath(__file__),
                                        'meta_stepsize':meta_stepsize,
                                        'h_decay':h_decay,
                                        'RMSPw':RMSP,
                                        'RMSPw_meta':RMSP_meta,
                                        'RMSPc_meta':RMSP_meta,
                                        'alpha_min':alpha_min,
                                        'alpha0':alpha0,
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
                    'metastep':meta_stepsize,
                    'h_decay':h_decay,
                    'RMSPw':RMSPw,
                    'RMSPw_meta':RMSPw_meta,
                    'RMSPc_meta':RMSPc_meta,
                    'alpha_min':alpha_min,
                    'alpha0':alpha0,
                    },
            'bias':'True',
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,0,None))

def clip0(a):
    return np.clip(a, a_min=0, a_max=None)


if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)

    try:
        h_decay = float(h_decay)
    except:
        pass

    v_w = 0.0
    v_c = 0.0
    normalizer_w = 1.0
    normalizer_c = 1.0

    h_w = 0.0
    h_c = 0.0
         
    
    wb = np.zeros(d+1)
    c = np.zeros(d)  # centers or the bias weights of the first layer
    
    beta_w = np.log(alpha0) * np.ones_like(wb)
    beta_c = np.log(alpha0) * np.ones_like(c)
    alpha_w = alpha0 * np.ones_like(wb)
    alpha_c = alpha0 * np.ones_like(c)

    MSE_list = []
    t = 0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()

            if t==1:
                c = x+0.0
                init_c = c+0.0
            
            xb = np.append(x-c,1)
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)

            v_w = RMSPw*v_w + (1-RMSPw)* ((xb)**2)
            normalizer_w = np.sqrt(v_w/(1-RMSPw**t)+1e-16) 
            
            alpha_w = np.exp(beta_w)
            alpha_c = np.exp(beta_c)

            #obn_coeff_for_w = min(1, 1/(alpha*xb*xb / normalizer_w).sum())
            obn_coeff_for_w = 1.0

            meta_grad_w = - delta * h_w * xb
            meta_grad_c = -xb[:-1] * h_c 

            v_w_meta = RMSPw_meta*v_w + (1-RMSPw_meta)* (meta_grad_w**2)
            v_c_meta = RMSPc_meta*v_c + (1-RMSPc_meta)* (meta_grad_c**2)

            beta_w = beta_w - meta_stepsize * meta_grad_w / np.sqrt(v_w_meta + 1e-16)
            beta_c = beta_c - meta_stepsize * meta_grad_c / np.sqrt(v_c_meta + 1e-16)
            
            beta_w = np.clip(beta_w, a_min=np.log(alpha_min/d), a_max=None)
            beta_c = np.clip(beta_c, a_min=np.log(alpha_min/d), a_max=None)
            

            h_decay_coeff_w, h_decay_coeff_c = (
                            (h_decay, h_decay)  if isinstance(h_decay, float) else
                            (1-alpha_w, 1-alpha_c) if h_decay=='1-alpha' else
                            (clip0(1-alpha_w*xb*xb/normalizer_w), clip0(1-alpha_c*wb[:-1]*wb[:-1]/normalizer_c))  if h_decay=='1-alpha*x2' else
                            0/0)
            h_w = h_decay_coeff_w * h_w + delta*xb/normalizer_w
            h_c = h_decay_coeff_c * h_c + xb[:-1]
            
            c = c + alpha_c * xb[:-1]  # This is the gradient update of the first layer weights
            wb = wb + obn_coeff_for_w * alpha_w *  delta * xb / normalizer_w

            #print(t, beta_w.sum(), beta_c.sum(), alpha_w.sum(), alpha_c.sum(),  wb.sum(), c.sum())
            if np.isnan(wb).any(): break
        except:
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
    