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


alg_name =  'IDBD_MGENC' 
base_alg = 'LMS_MDN'
base_alg_params = {'theta_MDN':.001, 'eta0':.001} # {} {'theta_MDN':.001, 'eta':.001}

alpha0 = 1e-4
meta_stepsize = 1e-3
meta_eta = 1e-3
gamma = 1.0
alpha_min = 1e-8


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS7-20-1100000.txt' 
save_to='csvs/_test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
        for gamma in [1.0]:
            for base_alg, base_alg_params_list, alpha0_list in [
                                    # ('LMS', [{}],[1e-7]),
                                    # #('LMS_SAN', [{}], [1e-5]),
                                    # ('LMS_EAN', [{'eta':.001}], [1e-5]),
                                    #('LMS_MDN', [{'theta_MDN':.001, 'eta0':.001}], [1e-5]),
                                    ('LMS_MDN', [{'theta_MDN':'theta_meta', 'eta0':.001}], [1e-5]),
                                    ('LMS_MDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001}], [1e-5]),
                                    # ('LMS_MDNa', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    # ('LMS_MDNPN', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    #('LMS_MDNPN_KC', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    ]:
                for base_alg_params in base_alg_params_list:
                    for meta_eta in [1e-3]:
                        for meta_stepsize in [1e-4,1e-3,1e-2]:
                            for alpha0 in alpha0_list:
                                sweeps.append({
                                                'file_path':os.path.abspath(__file__),
                                                'base_alg':base_alg,
                                                'base_alg_params':base_alg_params,
                                                'alpha0':alpha0,
                                                'meta_stepsize':meta_stepsize,
                                                'meta_eta':meta_eta,
                                                'gamma':gamma,
                                                'bias':bias,
                                                'alpha_min':alpha_min,
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
                    'meta_eta':meta_eta,
                    'alpha0':alpha0,
                    'base':base_alg_params,
                    'alpha_min':alpha_min,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,None,None))



if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)
    
    wb = np.zeros([d+1 if bias else d])
    if 'theta_MDN' in base_alg_params:
        if  isinstance(base_alg_params['theta_MDN'], str):
            tmp = base_alg_params['theta_MDN'].split('/')
            if tmp[0] == 'theta_meta':
                if len(tmp) == 1:
                    theta_MDN = meta_stepsize+0.0
                elif len(tmp) == 2:
                    theta_MDN = (meta_stepsize)/float(tmp[1])
            else: 
                raise(ValueError)
        else:
            theta_MDN = base_alg_params['theta_MDN'] + 0.0
    input_normalizer = (
                        identity if base_alg=='LMS' else
                        SAN() if base_alg=='LMS_SAN' else
                        EAN(eta=base_alg_params['eta']) if base_alg=='LMS_EAN' else
                        MDN(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDN' else
                        MDNa(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNa' else
                        MDNPN(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN' else
                        MDNPN_KC(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN_KC' else
                        0/0
                        )
    
    MSE_list = []
    logging_list=[]
    t = 0

    # initialize meta:
    beta = np.log(alpha0) * np.ones_like(wb)
    alpha = alpha0 * np.ones_like(wb)
    h = 0.0
    clip = lambda H: np.clip(H, a_min=0.0, a_max=None) 
    meta_nu=0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()
            x_tilde, mu, nu_sqrt = input_normalizer.step(x)
            xb = np.append(x_tilde,1) if bias else x_tilde
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)

            # meta update
            coeff_meta_nu = meta_eta/(1-(1-meta_eta)**t)
            meta_grad = -delta*xb*h
            meta_nu += coeff_meta_nu*(meta_grad**2-meta_nu)
            beta = beta - meta_stepsize* meta_grad / (np.sqrt(meta_nu)+1e-8)
            beta = np.clip(beta, a_min=np.log(alpha_min), a_max=None)
            alpha = np.exp(beta)
            h = h * (1-alpha*xb*xb) + alpha * delta*xb

            # base update
            wb = wb + alpha * delta*xb

            if np.isnan(wb).any(): break
        except:
            break
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    