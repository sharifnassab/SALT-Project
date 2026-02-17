import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime
from copy import deepcopy


alg_name =  'LMS_GC'      # goal directed centering

alpha = 1e-5
alpha_mu = "alpha/2" # stepsize of the first layer


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/_test3.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for alpha_mu in ['alpha/10', 'alpha/2']:
                sweeps.append({
                                'file_path':os.path.abspath(__file__),
                                'alpha':alpha,
                                'alpha_mu':alpha_mu,
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
                    'alpha_mu':alpha_mu,
                    },
            'bias':'True',
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,0,None))




class GC():
    def __init__(self, alpha_mu):
        self.t = 0
        self.mu = 0.0
        self.alpha_mu = alpha_mu

    def step(self, x, z, w, b):
        self.t+=1
        x_tilde = x-self.mu
        y = (w*x_tilde).sum() + b
        delta = z  - y
        self.mu = self.mu - self.alpha_mu * delta * w
        return x_tilde, self.mu, None   


if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)

    if alpha_mu=='alpha':
         alpha_mu = alpha
    elif isinstance(alpha_mu,str) and alpha_mu.split('/')[0]=='alpha':
        alpha_mu =alpha/float(alpha_mu.split('/')[1])
         
    
    wb = np.zeros(d+1)
    input_normalizer = GC(alpha_mu=alpha_mu)
    MSE_list = []
    t = 0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()
            x_tilde, mu, sigma = input_normalizer.step(x, z, wb[:d], b=wb[d] if bias else 0.0)
            
            xb = np.append(x_tilde,1)
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)
            
            #obn_coeff_for_w = min(1, 1/(alpha*xb*xb).sum())
            wb = wb + alpha *  delta * xb

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
    