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
from algorithms.LMS_GMDNcenter import GMDNcenter
from algorithms.LMS_MDNcenter import MDNcenter
from algorithms.LMS_MDNa import MDNa
from algorithms.LMS_MDNPN import MDNPN
from algorithms.LMS_MDNPN_KC import MDNPN_KC
from algorithms.LMS_MDCGN import MDCGN
from algorithms.LMS_2DN import _2DN
from algorithms.LMS_2DN_ import _2DN_
from algorithms.LMS_GC import GC
from algorithms.LMS_GCobn import GCobn
from algorithms.LMS_GCsignObn import GCsignObn
from algorithms.LMS_GCsidbd import GCsidbd
from algorithms.LMS_GCsidbdObn import GCsidbdObn
from algorithms.LMS_GCsidbdSignObn import GCsidbdSignObn


alg_name =  'SIDBD_Obn'  # Scalar IDBD on top of Obn algorithm
base_alg = 'LMS_MDN'
base_alg_params = {'theta_MDN':.001, 'eta0':.001} # {'theta_MDN':'theta_meta/10', 'eta0':.001}  {} {'theta_MDN':.001, 'eta':.001}

alpha0 = 1e-4
meta_stepsize = 1e-3
beta_inc_clip = "maxmin_0.1" 
beta_clip = "maxmin_10"
meta_post_norm = 0.99 # float


bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS7-20-1100000.txt' 
save_to='csvs/_test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

def param_sweeps():
    sweeps=[]
    for bias in ['True']:
            for base_alg, base_alg_params_list, alpha0_list in [
                                    # #('LMS_SAN', [{}], [1e-5]),
                                    # ('LMS_EAN', [{'eta':.001}], [1e-5]),
                                    #('LMS_GMDN', [{'theta_MDN':'theta_meta', 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:none,post:none'}], [1e-5]),
                                    # ('LMS_GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:v@0.01'}], [1e-5]),
                                    #('LMS_GMDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_GMDNcenter', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:0.01'}], [1e-5]),
                                    #('LMS_MDNcenter', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'post:0.01'}], [1e-5]),
                                    # ('LMS_MDN', [{'theta_MDN':.001, 'eta0':.001}], [1e-5]),
                                    # ('LMS_MDN', [{'theta_MDN':'theta_meta', 'eta0':.001}], [1e-5]),
                                    #('LMS_MDN', [{'theta_MDN':'theta_meta/10', 'eta0':.001}], [1e-5]),
                                    #('LMS_MDCGN', [{'theta_MDN':'theta_meta/10', 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS', [{}],[1e-7]),
                                    # ('LMS_MDNa', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    # ('LMS_MDNPN', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    #('LMS_MDNPN_KC', [{'theta_MDN':.01, 'eta0':.001}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':.1, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':.2, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':.5, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':-.2, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':.1, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':.2, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':.5, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':-.2, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':-.1, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':-.5, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta/10', 'weight_of_MDN':-1.0, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':-.1, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    # ('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':-.5, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_2DN', [{'theta_MDN':'theta_meta', 'weight_of_MDN':-1.0, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]),
                                    #('LMS_2DN_', [{'theta_MDN':'theta_meta', 'weight_of_MDN':.2, 'eta0':.001, 'transform':'pre:signW,post:both@0.01@0.01'}], [1e-5]), # best in class
                                    ('LMS_GC', [{'alpha_mu':1e-5}], [1e-3]),
                                    ('LMS_GC', [{'alpha_mu':1e-6}], [1e-3]),
                                    ('LMS_GCobn', [{'alpha_mu':0.01}], [1e-3]),
                                    ('LMS_GCobn', [{'alpha_mu':0.001}], [1e-3]),
                                    ('LMS_GCobn', [{'alpha_mu':'alpha/2'}], [1e-3]),
                                    ('LMS_GCobn', [{'alpha_mu':'alpha/10'}], [1e-3]),
                                    ('LMS_GCsignObn', [{'alpha_mu':0.01}], [1e-3]),
                                    ('LMS_GCsignObn', [{'alpha_mu':0.001}], [1e-3]),
                                    ('LMS_GCsignObn', [{'alpha_mu':'alpha/2'}], [1e-3]),
                                    ('LMS_GCsignObn', [{'alpha_mu':'alpha/10'}], [1e-3]),
                                    ('LMS_GCsidbd', [{'meta_stepsize_mu':1e-3, 'meta_post_norm_mu':0.99, 'alpha0_mu':1e-3}], [1e-3]),
                                    ('LMS_GCsidbd', [{'meta_stepsize_mu':'theta_meta', 'meta_post_norm_mu':0.99, 'alpha0_mu':1e-3}], [1e-3]),
                                    ('LMS_GCsidbdObn', [{'meta_stepsize_mu':1e-3, 'meta_post_norm_mu':0.99, 'alpha0_mu':1e-3}], [1e-3]),
                                    ('LMS_GCsidbdObn', [{'meta_stepsize_mu':'theta_meta', 'meta_post_norm_mu':0.99, 'alpha0_mu':1e-3}], [1e-3]),
                                    ('LMS_GCsidbdSignObn', [{'meta_stepsize_mu':1e-3, 'meta_post_norm_mu':0.99, 'alpha0_mu':1e-3}], [1e-3]),
                                    ('LMS_GCsidbdSignObn', [{'meta_stepsize_mu':'theta_meta', 'meta_post_norm_mu':0.99, 'alpha0_mu':1e-3}], [1e-3]),
                                    ]:
                for base_alg_params in base_alg_params_list:
                    for meta_stepsize in [1e-4,1e-3,1e-2]:
                        for beta_inc_clip in [ 
                                                  #"none", 
                                                  #"max_0.1",
                                                  #"maxmin_0.1",
                                                  "maxmin_0.5"
                                                ]:
                            for beta_clip in [ 
                                                  #"none", 
                                                  "maxmin_10"
                                                ]:
                                for meta_post_norm in [
                                                        0.99,
                                                        ]:                                        
                                    for alpha0 in alpha0_list:
                                        sweeps.append({
                                                        'file_path':os.path.abspath(__file__),
                                                        'base_alg':base_alg,
                                                        'base_alg_params':base_alg_params,
                                                        'alpha0':alpha0,
                                                        'meta_stepsize':meta_stepsize,
                                                        'beta_inc_clip':beta_inc_clip,
                                                        'beta_clip':beta_clip,
                                                        'meta_post_norm':meta_post_norm,
                                                        'bias':bias,
                                                        })
    return sweeps

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
                    'beta_inc_clip':beta_inc_clip,
                    'beta_clip':beta_clip,
                    'meta_post_norm':meta_post_norm,
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
def act_exp(x): 
    return np.exp(x)
def act_sigmoid(x): 
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    env, num_steps, d = data_loader(dataset)

    try:
        meta_post_norm = float(meta_post_norm)
    except:
        raise(ValueError)
    

    beta_inc_clip_max = 1e8
    beta_inc_clip_min = -1e8
    if 'max' in beta_inc_clip:
        beta_inc_clip_max = float(beta_inc_clip.split('_')[1])
    if 'min' in beta_inc_clip:
        beta_inc_clip_min = -float(beta_inc_clip.split('_')[1])

    
    beta_clip_max = 1e8
    beta_clip_min = -1e8
    if 'max' in beta_clip:
        beta_clip_max = float(beta_clip.split('_')[1])
    if 'min' in beta_clip:
        beta_clip_min = -float(beta_clip.split('_')[1])

    
    
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
    
    if 'meta_stepsize_mu' in base_alg_params:
        if  isinstance(base_alg_params['meta_stepsize_mu'], str):
            tmp = base_alg_params['meta_stepsize_mu'].split('/')
            if tmp[0] == 'theta_meta':
                if len(tmp) == 1:
                    meta_stepsize_mu = meta_stepsize+0.0
                elif len(tmp) == 2:
                    meta_stepsize_mu = (meta_stepsize)/float(tmp[1])
            else: 
                raise(ValueError)
        else:
            meta_stepsize_mu = base_alg_params['meta_stepsize_mu'] + 0.0

    if 'alpha_mu' in base_alg_params:
        if  isinstance(base_alg_params['alpha_mu'], str):
            tmp = base_alg_params['alpha_mu'].split('/')
            if tmp[0] == 'alpha':
                if len(tmp) == 1:
                    alpha_mu_multiplier_of_alpha = 1.0
                elif len(tmp) == 2:
                    alpha_mu_multiplier_of_alpha = 1.0/float(tmp[1])
            else: 
                raise(ValueError)

    input_normalizer = (
                        identity if base_alg=='LMS' else
                        SAN() if base_alg=='LMS_SAN' else
                        EAN(eta=base_alg_params['eta']) if base_alg=='LMS_EAN' else
                        MDN(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDN' else
                        GMDN(theta=theta_MDN, eta0=base_alg_params['eta0'], transform=base_alg_params['transform']) if base_alg=='LMS_GMDN' else
                        MDCGN(theta=theta_MDN, eta0=base_alg_params['eta0'], transform=base_alg_params['transform']) if base_alg=='LMS_MDCGN' else
                        GMDNcenter(theta=theta_MDN, eta0=base_alg_params['eta0'], transform=base_alg_params['transform']) if base_alg=='LMS_GMDNcenter' else
                        MDNcenter(theta=theta_MDN, eta0=base_alg_params['eta0'], transform=base_alg_params['transform']) if base_alg=='LMS_MDNcenter' else
                        MDNa(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNa' else
                        MDNPN(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN' else
                        MDNPN_KC(theta=theta_MDN, eta0=base_alg_params['eta0']) if base_alg=='LMS_MDNPN_KC' else
                        _2DN(theta=theta_MDN, eta0=base_alg_params['eta0'], transform=base_alg_params['transform'], weight_of_MDN=base_alg_params['weight_of_MDN']) if base_alg=='LMS_2DN' else
                        _2DN_(theta=theta_MDN, eta0=base_alg_params['eta0'], transform=base_alg_params['transform'], weight_of_MDN=base_alg_params['weight_of_MDN']) if base_alg=='LMS_2DN_' else
                        GC(alpha_mu=base_alg_params['alpha_mu']) if base_alg=='LMS_GC' else
                        GCobn(alpha_mu=base_alg_params['alpha_mu']) if base_alg=='LMS_GCobn' else
                        GCsignObn(alpha_mu=base_alg_params['alpha_mu']) if base_alg=='LMS_GCsignObn' else
                        GCsidbd(meta_stepsize_mu=meta_stepsize_mu, meta_post_norm_mu=base_alg_params['meta_post_norm_mu'], alpha_mu_0=base_alg_params['alpha0_mu']) if base_alg=='LMS_GCsidbd' else
                        GCsidbdObn(meta_stepsize_mu=meta_stepsize_mu, meta_post_norm_mu=base_alg_params['meta_post_norm_mu'], alpha_mu_0=base_alg_params['alpha0_mu']) if base_alg=='LMS_GCsidbdObn' else
                        GCsidbdSignObn(meta_stepsize_mu=meta_stepsize_mu, meta_post_norm_mu=base_alg_params['meta_post_norm_mu'], alpha_mu_0=base_alg_params['alpha0_mu']) if base_alg=='LMS_GCsidbdSignObn' else
                        0/0
                        )
    
    MSE_list = []
    logging_list=[]
    t = 0

    # initialize meta:
    beta = -np.log(wb.size)
    alpha = act_sigmoid(beta)
    h = 0.0
    clip0 = lambda x: np.clip(x, a_min=0.0, a_max=None) 
    clip_eps = lambda x: np.clip(x, a_min=1e-8, a_max=None) 
    trace_meta_post_norm=0.0

    # main loop:
    for iteration in range(num_steps):
        t+=1
        
        if 1:
            x,z = env.next()
            if base_alg in ['LMS_GC', 'LMS_GCobn', 'LMS_GCsignObn']:
                if isinstance(base_alg_params['alpha_mu'], str):
                    input_normalizer.alpha_mu = alpha_mu_multiplier_of_alpha* alpha
            if base_alg in ['LMS_GMDN', 'LMS_GMDNcenter', 'LMS_MDCGN', 'LMS_2DN', 'LMS_2DN_', 'LMS_GC', 'LMS_GCobn', 'LMS_GCsignObn', 'LMS_GCsidbd', 'LMS_GCsidbdObn', 'LMS_GCsidbdSignObn']: 
                x_tilde, _, _ = input_normalizer.step(x, z, wb[:d], b=wb[d] if bias else 0.0)
            else:
                x_tilde, _, _ = input_normalizer.step(x)

            xb = np.append(x_tilde,1) if bias else x_tilde
            
            y = (wb*xb).sum()
            delta = z-y
            MSE_list.append((y-z)**2)

            # meta update
            meta_beta_increment = delta * (xb*h).sum() 
            
            trace_meta_post_norm = trace_meta_post_norm + (1-meta_post_norm) * (meta_beta_increment**2 - trace_meta_post_norm)
            current_meta_beta_increment = meta_beta_increment / np.clip(np.sqrt(trace_meta_post_norm/(1-meta_post_norm**t)), a_min=1e-8, a_max=None)
            
            beta = beta + np.clip(meta_stepsize * current_meta_beta_increment, a_min=beta_inc_clip_min, a_max=beta_inc_clip_max) 
            beta = np.clip(beta, a_min=beta_clip_min, a_max=beta_clip_max)
            alpha = act_sigmoid(beta)

            norm_x = (xb*xb).sum() # for  Overshoot bounding
            h_projection =  (xb*h).sum() * alpha*xb/norm_x
            h_overshoor_bound = min([1, np.sqrt(np.sum(h**2) / ((h_projection**2).sum()+1e-8))])
            h = h - h_overshoor_bound*h_projection + delta*xb/norm_x
            
            # base update
            wb = wb + (alpha/ norm_x) *  delta * xb 

            if np.isnan(wb).any(): break
        else:
            break


    ###-----------------------
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    