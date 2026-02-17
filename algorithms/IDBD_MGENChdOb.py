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
from algorithms.LMS_MDNEAN import MDNEAN
from algorithms.centering import center_, center_and_augment


alg_name =  'IDBD_MGENChdOb'   # hd stands for h decay
base_alg = 'LMS_MDN'
base_alg_params = {'theta_MDN':.001, 'eta0':.001} # {} {'theta_MDN':.001, 'eta':.001}

alpha0 = 1e-4
meta_stepsize = 1e-3
meta_eta = 5e-3
gamma = 1.0
alpha_min = 1e-3 # this will be divided by d before being applied
h_decay = "1-alpha"  # float or '1-alpha' or '1-alpha*x2'
os_decay = .99


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
                                    #('LMS_EAN', [{'eta':.001}], [1e-5]),
                                    #('LMS_EAN', [{'eta':.001, 'cente_only':'center'}], [1e-5]),
                                    #('LMS_EAN', [{'eta':.001, 'cente_only':'center_and_augment'}], [1e-5]),
                                    #('LMS_EAN_eta_eq_alpha', [{'eta/alpha':1}], [1e-5]),
                                    #('LMS_EAN_eta_eq_alpha', [{'eta/alpha':.1}], [1e-5]),
                                    #('LMS_MDN', [{'theta_MDN':.001, 'eta0':.001}], [1e-5]),
                                    #('LMS_MDN', [{'theta_MDN':'theta_meta', 'eta0':.001}], [1e-5]),
                                    #('LMS_MDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001}], [1e-4]),
                                    #('LMS_MDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'cente_only':'center'}], [1e-5]),
                                    #('LMS_MDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'cente_only':'center_and_augment'}], [1e-5]),
                                    # ('LMS_MDNa', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    # ('LMS_MDNPN', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    #('LMS_MDNPN_KC', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    #('LMS_MDNEAN', [{'theta_MDN':0.001, 'eta_v':.01},{'theta_MDN':0.001, 'eta_v':.001}], [1e-3]),
                                    ('LMS_MDNEAN', [{'theta_MDN':0.001, 'eta_v':.001}], [1e-3]),
                                    #('LMS_MDNEAN', [{'theta_MDN':'theta_meta/10', 'eta_v':.01}], [1e-3]),
                                    ]:
                for base_alg_params in base_alg_params_list:
                    for meta_eta in [1e-3]:
                        for meta_stepsize in [1e-4,1e-3,1e-2]:
                            for alpha0 in alpha0_list:
                                for h_decay in [
                                                # 0.0,
                                                # 0.9,
                                                # 0.99,
                                                # 0.999,
                                                '1-alpha'
                                                # '1-alpha*x2'
                                                ]:
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
                                                    'h_decay': h_decay,
                                                    'os_decay':os_decay,
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
                    'h_decay':h_decay,
                    'os_decay':os_decay,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,None,None))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
                        EAN(eta=.001) if base_alg=='LMS_EAN_eta_eq_alpha' else
                        MDN(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDN' else
                        MDNa(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNa' else
                        MDNPN(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN' else
                        MDNPN_KC(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN_KC' else
                        MDNEAN(theta_mu=theta_MDN, eta_v=base_alg_params['eta_v']) if base_alg=='LMS_MDNEAN' else
                        0/0
                        )
    if 'cente_only' in base_alg_params:
        if base_alg_params['cente_only']=='center':
            input_normalizer = center_(input_normalizer)
        elif base_alg_params['cente_only']=='center_and_augment':
            input_normalizer = center_and_augment(input_normalizer)
            wb = np.zeros([2*d+1 if bias else 2*d])
        else:
            raise(ValueError)
    
    MSE_list = []
    logging_list=[]
    t = 0
    try:
        h_decay=float(h_decay)
    except:
        pass

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
            if base_alg in ['LMS_EAN_eta_eq_alpha']: input_normalizer.eta = base_alg_params['eta/alpha'] * (alpha[:-1] if bias else alpha)
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
            beta = np.clip(beta, a_min=np.log(alpha_min/wb.size), a_max=None)
            alpha = np.exp(beta)
            h_decay_coeff = (
                            h_decay if isinstance(h_decay, float) else
                            (1-alpha) if h_decay=='1-alpha' else
                            clip(1-alpha*xb*xb) if h_decay=='1-alpha*x2' else
                            0/0)
            h = h * h_decay_coeff + delta*xb

            update_size = (alpha*xb*xb).sum()
            overshoot_protection = min(1, 1/(update_size+1e-10))
            if update_size>1:
                beta = beta +np.log(os_decay)

            # base update
            wb = wb + overshoot_protection * alpha * delta*xb

            if np.isnan(wb).any(): break
        except:
            break
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    