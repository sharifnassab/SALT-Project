import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime
from copy import deepcopy
from algorithms.LMS_SAN import SAN
from algorithms.LMS_EAN import EAN
from algorithms.LMS_MDN import MDN
from algorithms.LMS_MDNa import MDNa
from algorithms.LMS_MDNPN import MDNPN
from algorithms.LMS_MDNPN_KC import MDNPN_KC


alg_name =  'Normalized_IDBD' 
base_alg = 'LMS_MDN'
base_alg_params = {'theta_MDN':.001, 'eta0':.001} # {} {'theta_MDN':.001, 'eta':.001}

alpha0 = 1e-4
meta_stepsize = 1e-3
gamma = 1.0


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS7-20-1100000.txt' 
save_to='csvs/_test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
        for gamma in [1.0]:
            for base_alg, base_alg_params_list, alpha0_list in [
                                    ('LMS', [{}],[1e-7]),
                                    #('LMS_SAN', [{}], [1e-5]),
                                    ('LMS_EAN', [{'eta':.001}], [1e-5]),
                                    ('LMS_MDN', [{'theta_MDN':.001, 'eta0':.001}], [1e-5]),
                                    ('LMS_MDNa', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    ('LMS_MDNPN', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    ('LMS_MDNPN_KC', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    ]:
                for base_alg_params in base_alg_params_list:
                    for meta_stepsize in [1e-4,1e-3,1e-2]:
                        for alpha0 in alpha0_list:
                            sweeps.append({
                                            'file_path':os.path.abspath(__file__),
                                            'base_alg':base_alg,
                                            'base_alg_params':base_alg_params,
                                            'alpha0':alpha0,
                                            'meta_stepsize':meta_stepsize,
                                            'gamma':gamma,
                                            'bias':bias,
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
            'alg_name':f'{base_alg}^{alg_name}',
            'params':{
                    'metastep':meta_stepsize,
                    'alpha0':alpha0,
                    'base':base_alg_params,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,None,None))

def softmax(x):
    exp_z = np.exp(x-np.median(x))  # median is substracted for numerical statibility and avoiding exp blow-up
    return exp_z/exp_z.sum()

if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)
    
    
    wb = np.zeros([d+1 if bias else d])
    input_normalizer = (
                        identity if base_alg=='LMS' else
                        SAN() if base_alg=='LMS_SAN' else
                        EAN(eta=base_alg_params['eta']) if base_alg=='LMS_EAN' else
                        MDN(theta=base_alg_params['theta_MDN'], eta0=base_alg_params['eta0']) if base_alg=='LMS_MDN' else
                        MDNa(theta=base_alg_params['theta_MDN'], eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNa' else
                        MDNPN(theta=base_alg_params['theta_MDN'], eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN' else
                        MDNPN_KC(theta=base_alg_params['theta_MDN'], eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN_KC' else
                        0/0
                        )
    
    MSE_list = []
    logging_list=[]
    t = 0

    # initialize meta:
    beta = np.zeros(wb.size+1) 
    beta[-1] = np.log(wb.size)
    alpha = softmax(beta)
    alpha[-1] = 0
    h = 0.0
    clip0 = lambda x: np.clip(x, a_min=0.0, a_max=None) 
    clip_eps = lambda x: np.clip(x, a_min=1e-8, a_max=None) 
    N=0.0
    meta_eta=1.0
    prod_last_alpha=1.0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()
            x_tilde, _, _ = input_normalizer.step(x)
            xb = np.append(x_tilde,1) if bias else x_tilde
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)

            # meta update
            N = N + (1-alpha[-1]) * (delta**2-N) 
            prod_last_alpha *= alpha[-1]
            N_corrected = N/(1-prod_last_alpha)   # this is absent in the original Normalized-IDBD
            delta_tilde = delta / clip_eps(np.sqrt(N_corrected))
            beta[:-1] = beta[:-1] + meta_stepsize * delta_tilde*xb*h * np.sqrt(2*alpha[:-1]-alpha[:-1]**2)
            alpha = softmax(beta)
            h = h * clip0(1-alpha[:-1]* xb**2) + delta_tilde*xb
            
            # base update
            wb = wb + alpha[:-1] * delta_tilde*xb

            if np.isnan(wb).any(): break
        except:
            break
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    