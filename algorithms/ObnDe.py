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
from algorithms.LMS_MDNEAN import MDNEAN
from algorithms.LMS_GMDN import GMDN
from algorithms.LMS_GMDNcenter import GMDNcenter
from algorithms.LMS_MDNa import MDNa
from algorithms.LMS_MDNPN import MDNPN
from algorithms.LMS_MDNPN_KC import MDNPN_KC


alg_name =  'ObnDe'     # Obn with disentanglement (of normalization and weight updates)
normalizer = 'EAN2'
normalizer_params = {'eta':.001, 'eta_v':.001} #{'theta_MDN':.001, 'eta0':.001} # {} {'theta_MDN':.001, 'eta':.001}

alpha = 1e-1
eta_norm_x = 0.001


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/_test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
        for normalizer, normalizer_params_list in [
                                #('none', [{}]),
                                # #('SAN'),
                                #('EAN', [{'eta':.01}]),
                                #('EAN2', [{'eta':.1, 'eta_v':.01}]),
                                #('EAN2', [{'eta':.001, 'eta_v':.01}]),
                                #('EAN2', [{'eta':.01, 'eta_v':.001}]),
                                #('EAN2', [{'eta':.001, 'eta_v':.0001}]),
                                #('MDNEAN', [{'theta_MDN':0.001, 'eta_v':.001}]),
                                # ('GMDN', [{'theta_MDN':0.001, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}]),
                                # ('GMDN', [{'theta_MDN':0.0001, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}]),
                                # ('GMDN', [{'theta_MDN':0.001, 'eta0':.001, 'transform':'pre:signW,post:both@0.001@0.001'}]),
                                ('GMDNcenter', [{'theta_MDN':0.001, 'eta0':.001, 'transform':'pre:signW,post:0.01'}]),
                                #('MDN', [{'theta_MDN':.001, 'eta0':.001}]),
                                # ('MDNa', [{'theta_MDN':.01, 'eta0':.001}]),
                                # ('MDNPN', [{'theta_MDN':.01, 'eta0':.001}]),
                                #('MDNPN_KC', [{'theta_MDN':.01, 'eta0':.001}]),
                                ]:
            for normalizer_params in normalizer_params_list:
                for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    for eta_norm_x in [
                                        1.0,   # normalizes only to ||x_t||^2
                                        #0.01,
                                        #0.001,
                                        #0.0,   # this is effectively like 1
                    ]:
                        sweeps.append({
                                        'file_path':os.path.abspath(__file__),
                                        'eta_norm_x': eta_norm_x,
                                        'alpha':alpha,
                                        'bias':bias,
                                        'normalizer':normalizer,
                                        'normalizer_params':normalizer_params,
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
                    'eta_norm_x': eta_norm_x,
                    'normalizer':normalizer,
                    'normalizer_params':normalizer_params,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------

class identity: step = staticmethod(lambda x: (x,0,None))

class disentangler():
    def __init__(self):
        self.mu = 0.0
        self.mu_old = 0.0
        self.nu_sqrt = 100.0
    def step(self, w, mu, nu_sqrt):
        w_disentangled = np.zeros_like(w)

        # w_disentangled[:-1] = w[:-1]* nu_sqrt/self.nu_sqrt
        # w_disentangled[-1] = w[-1] #+ (w[:-1]*(mu - self.mu)/nu_sqrt).sum()

        w_disentangled[:-1] = w[:-1]* nu_sqrt/self.nu_sqrt
        w_disentangled[-1] = w[-1] + (w[:-1]*(self.mu - self.mu_old)/self.nu_sqrt).sum()

        self.mu_old = self.mu + 0.0
        self.mu = mu
        self.nu_sqrt = nu_sqrt
        return w_disentangled



if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)
    
    wb = np.zeros([d+1 if bias else d])
    input_normalizer = (
                        identity if normalizer=='none' else
                        SAN() if normalizer=='SAN' else
                        EAN(eta=normalizer_params['eta']) if normalizer=='EAN' else
                        EAN(eta=normalizer_params['eta'], eta_v=normalizer_params['eta_v']) if normalizer=='EAN2' else
                        MDN(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0']) if normalizer=='MDN' else
                        MDNEAN(theta_mu=normalizer_params['theta_MDN'], eta_v=normalizer_params['eta_v']) if normalizer=='MDNEAN' else
                        GMDN(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0'], transform=normalizer_params['transform']) if normalizer=='GMDN' else
                        GMDNcenter(theta=normalizer_params['theta_MDN'], eta0=normalizer_params['eta0'], transform=normalizer_params['transform']) if normalizer=='GMDNcenter' else
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
    norm_x_trace = 1.0

    disentangle = disentangler()


    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        try:
            x,z = env.next()
            if normalizer in ['GMDN', 'GMDNcenter']: 
                x_tilde, mu, nu_sqrt = input_normalizer.step(x, z, wb[:d], b=wb[d] if bias else 0.0)
            else:
                x_tilde, mu, nu_sqrt = input_normalizer.step(x)
            xb = np.append(x_tilde,1) if bias else x_tilde
            wb_old = wb + 0.0
            wb = disentangle.step(wb, mu, nu_sqrt)
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append(((wb_old*xb).sum()-z)**2)
            
            norm_x = (xb*xb).sum()
            norm_x_trace = norm_x_trace +  eta_norm_x * (norm_x - norm_x_trace)

            wb = wb + (alpha/ (max([norm_x,np.sqrt(norm_x*norm_x_trace)]))) *  delta * xb 

            if np.isnan(wb).any(): break
        except:
            break
            

    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    