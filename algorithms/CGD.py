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
from algorithms.LMS_GMDN import GMDN
from algorithms.LMS_MDNa import MDNa
from algorithms.LMS_MDNPN import MDNPN
from algorithms.LMS_MDNPN_KC import MDNPN_KC


alg_name =  'CGD'     # Centered stochastic Gradient Descent   (y is computed based on )
normalizer = 'EAN'
normalizer_params = {'eta':.001} #{'theta_MDN':.001, 'eta0':.001} # {} {'theta_MDN':.001, 'eta':.001}

alpha = 1e-3


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/_test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
        for normalizer, normalizer_params_list in [
                                # ('none', [{}]),
                                # #('SAN'),
                                # ('EAN', [{'eta':.001}, {'eta':.0001}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:none,post:none'}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:none,post:v@0.01'}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:v@0.01'}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:none,post:v@0.001'}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:v@0.001'}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:none,post:both@0.01@0.01'}]),
                                #('GMDN', [{'theta_MDN':'theta_meta', 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}]),
                                #('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}]),
                                # ('GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:both@0.001@0.001'}]),
                                ('MDN', [{'theta_MDN':.001, 'eta0':.001}]),
                                #('MDN', [{'theta_MDN':'theta_meta', 'eta0':.001}]),
                                #('MDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001}]),
                                # ('MDNa', [{'theta_MDN':.01, 'eta0':.001}]),
                                # ('MDNPN', [{'theta_MDN':.01, 'eta0':.001}]),
                                #('MDNPN_KC', [{'theta_MDN':.01, 'eta0':.001}]),
                                ]:
            for normalizer_params in normalizer_params_list:
                for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    sweeps.append({
                                    'file_path':os.path.abspath(__file__),
                                    'normalizer':normalizer,
                                    'normalizer_params':normalizer_params,
                                    'alpha':alpha,
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
            'alg_name':f'{alg_name}_{normalizer}',
            'params':{
                    'alpha':alpha,
                    'normalizer':normalizer,
                    'normalizer_params':normalizer_params,
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
    input_normalizer = (
                        identity if normalizer=='none' else
                        SAN() if normalizer=='SAN' else
                        EAN(eta=normalizer_params['eta']) if normalizer=='EAN' else
                        MDN(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0']) if normalizer=='MDN' else
                        GMDN(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0'], transform=normalizer_params['transform']) if normalizer=='GMDN' else
                        MDNa(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0']) if normalizer=='MDNa' else
                        MDNPN(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0']) if normalizer=='MDNPN' else
                        MDNPN_KC(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0']) if normalizer=='MDNPN_KC' else
                        0/0
                        )
    
    MSE_list = []
    logging_list=[]
    t = 0

    # initialize meta:
    h = 0.0
    clip = lambda H: np.clip(H, a_min=0.0, a_max=None) 
    meta_nu=0

    alpha_vec = np.ones_like(wb)
    if bias:
        alpha_vec[:-1] = alpha/(2*d)
        alpha_vec[-1] = alpha/2
    else:
        alpha_vec[:] = alpha/d

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()
            if normalizer in ['GMDN']: 
                x_tilde, mu, nu_sqrt = input_normalizer.step(x, z, wb[:d], b=wb[d] if bias else 0.0)
            else:
                x_tilde, mu, nu_sqrt = input_normalizer.step(x)
            xb = np.append(x,1) if bias else x
            xb_centered = np.append(x-mu,1) if bias else x-mu
            
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)
            
            wb = wb +  delta * alpha_vec * xb_centered

            if np.isnan(wb).any(): break
        except:
            break
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    