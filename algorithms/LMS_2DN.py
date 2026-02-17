import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.data_loader import data_loader
from utils.sanitize_and_save import sanitize_and_save
from datetime import datetime



alg_name =  'LMS-2DN'  # combinaiton of MDN and GMDN 
alpha = 1e-4
eta0 = .001
theta_MDN = 1e-4
transform = 'pre:none,post:both'  # 'None' 'signW'  'postnorm_both'
weight_of_MDN = 0.2  # weight of MDN in the combination

bias = True
start_of_steady_state = 900_001

dataset = 'SALTdata-RSS1-20-1100000.txt' 
save_to='csvs/test2.txt' + f'++{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


def param_sweeps():
    sweeps=[]
    for bias in ['True']:
            for eta0 in [.001]:
                for theta_MDN in [1e-3]:#, 1e-2]:
                    for weight_of_MDN in [.1,.5]:
                        for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                            for transform in [
                                            #   'pre:none,post:none', 
                                            #   'pre:signW,post:none', 
                                            #   'pre:none,post:v@0.01', 
                                            #   'pre:signW,post:v@0.01', 
                                            #   'pre:none,post:v@0.001', 
                                            #   'pre:signW,post:v@0.001', 
                                            #   'pre:none,post:both@0.01@0.01', 
                                            'pre:signW,post:both@0.01@0.01',
                                            #   'pre:signW,post:both@0.001@0.001',
                                            ]:
                                sweeps.append({
                                                'file_path':os.path.abspath(__file__),
                                                'alpha':alpha,
                                                'eta0':eta0,
                                                'theta_MDN':theta_MDN,
                                                'bias':bias,
                                                'transform':transform,
                                                'weight_of_MDN':weight_of_MDN
                                            })
    return sweeps

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("utils/configurator.py").read())  # overrides from command line or config file

bias = False if bias in [False,'False',0,'0'] else True
alg_config={
            'dataset':dataset,
            'alg_name':alg_name,
            'params':{
                    'alpha':alpha,
                    'theta':theta_MDN,
                    'eta0':eta0,
                    'transform':transform,
                    'weight_of_MDN':weight_of_MDN,
                    },
            'bias':bias,
            'model':('affine' if bias else 'linear'),
            'save_to':save_to,
            }
# -----------------------------------------------------------------------------


class _2DN():
    def __init__(self, theta, eta0, transform='None', weight_of_MDN=0.1, activation='sigmoid'):
        if activation=='sigmoid':
            self.act = lambda b: 1/(1+np.exp(-np.clip(b, -500, 500)))
            act_inv = lambda a: -np.log(1/a -1)
            self.d_dbeta_given_eta = lambda eta: eta*(1.0-eta)
        self.theta = theta
        self.weight_of_MDN = weight_of_MDN
        self.var = 0.
        # extract prenorm_type  from transform string if applicable
        if transform.split(',')[0].split(':')[1].lower() in ['none']:
            self.prenorm_type = 'None'
        else:
            self.prenorm_type = transform.split(',')[0].split(':')[1]
        if transform.split(',')[1].split(':')[1].lower() in ['none']:
            self.post_norm_type = 'None'
        else:
            self.post_norm_type = 'postnorm_'+transform.split(',')[1].split(':')[1]

        
        self.t = 0
        self.epsilon = 1e-8
        self.beta_mu = act_inv(eta0)
        self.beta_v = act_inv(eta0)
        self.h_mu = 0.0
        self.h_v = 0.0
        self.eta_mu_prod = 1.0
        self.eta_v_prod = 1.0
        self.postnorm_v_MDN = 1.0
        self.postnorm_mu_MDN = 1.0
        self.postnorm_v_GMDN = 1.0
        self.postnorm_mu_GMDN = 1.0
        self.update_trace = 1.0

        if self.post_norm_type=='None':
            self.postnorm_coeff_mu = 1.0
            self.postnorm_coeff_v = 1.0
        elif self.post_norm_type.split('@')[0]=='postnorm_both':
            self.postnorm_coeff_mu = float(self.post_norm_type.split('@')[1]) if '@' in self.post_norm_type else 0.01
            self.postnorm_coeff_v = float(self.post_norm_type.split('@')[2]) if len(self.post_norm_type.split('@'))>2 else 0.01
        else:
            raise NotImplementedError(f'Normalization type {self.post_norm_type} not implemented yet.')
        

    def step(self, x, z, w, b):
        self.t+=1
        eta_mu = self.act(self.beta_mu)
        eta_v = self.act(self.beta_v)

        if self.t==1:
            self.mu = x
            self.eta_mu_prod *= 1.0-eta_mu 
            x_tilde = np.zeros_like(x)
            sigma = self.epsilon * np.ones_like(x)
            return x_tilde, self.mu, sigma
        
        self.eta_mu_prod *= 1.0-eta_mu  
        self.eta_v_prod *= 1.0-eta_v

        self.var = self.var + eta_v*((x-self.mu)**2 - self.var) / (1-self.eta_v_prod) # (1-self.eta_v_prod) is not present in the original MDN
        sigma = np.clip(np.sqrt(self.var), a_min=self.epsilon, a_max=None)
        x_tilde = (x-self.mu) /  sigma
        self.mu = self.mu + eta_mu*(x-self.mu) / (1-self.eta_mu_prod) # (1-self.eta_v_prod) is not present in the original MDN

        delta = z  - ((w*x_tilde).sum() + b)


        # GMDN update
        if self.prenorm_type=='None':
            pre_norm_mu_GMDN,pre_norm_v_GMDN = 1,1
        elif self.prenorm_type=='signW':
            pre_norm_mu_GMDN = 1/(np.abs(w)+1e-8)
            pre_norm_v_GMDN = 1/(np.abs(w)+1e-8)

        meta_update_mu_GMDN = pre_norm_mu_GMDN  * delta * w * self.h_mu
        meta_update_v_GMDN  = pre_norm_v_GMDN   * delta * w * x_tilde * self.h_v

        if self.post_norm_type=='None':
            final_beta_mu_update_GMDN, final_beta_v_update_GMDN = meta_update_mu_GMDN,meta_update_v_GMDN
        elif self.post_norm_type.split('@')[0]=='postnorm_both':
            self.postnorm_mu_GMDN += self.postnorm_coeff_mu * (meta_update_mu_GMDN**2 - self.postnorm_mu_GMDN)
            self.postnorm_v_GMDN += self.postnorm_coeff_v * (meta_update_v_GMDN**2 - self.postnorm_v_GMDN)
            final_beta_mu_update_GMDN = meta_update_mu_GMDN / np.sqrt(self.postnorm_mu_GMDN + 1e-8)
            final_beta_v_update_GMDN = meta_update_v_GMDN / np.sqrt(self.postnorm_v_GMDN + 1e-8)
            
        

        # MDN update
        meta_update_mu_MDN = x_tilde *self.h_mu 
        meta_update_v_MDN  = (x_tilde**2-1) *self.h_v  

        if self.post_norm_type=='None':
            final_beta_mu_update_MDN, final_beta_v_update_MDN = meta_update_mu_MDN,meta_update_v_MDN
        elif self.post_norm_type.split('@')[0]=='postnorm_both':
            self.postnorm_mu_MDN += self.postnorm_coeff_mu * (meta_update_mu_MDN**2 - self.postnorm_mu_MDN)
            self.postnorm_v_MDN += self.postnorm_coeff_v * (meta_update_v_MDN**2 - self.postnorm_v_MDN)
            final_beta_mu_update_MDN = meta_update_mu_MDN / np.sqrt(self.postnorm_mu_MDN + 1e-8)
            final_beta_v_update_MDN = meta_update_v_MDN / np.sqrt(self.postnorm_v_MDN + 1e-8)
        
        
        # h and beta updates
        self.h_mu = (1-eta_mu)*self.h_mu + x_tilde 
        self.h_v = (1-eta_v)*self.h_v + (x_tilde**2-1) 

        self.beta_mu = self.beta_mu - self.theta * (self.weight_of_MDN*final_beta_mu_update_MDN + (1-self.weight_of_MDN)*final_beta_mu_update_GMDN)
        self.beta_v  = self.beta_v  - self.theta * (self.weight_of_MDN*final_beta_v_update_MDN + (1-self.weight_of_MDN)*final_beta_v_update_GMDN)


        return x_tilde, self.mu, sigma


def npfmt5(a):
    return np.array2string(a[:5], precision=4, floatmode='fixed', separator=' ')


def main_loop(dataset=dataset, alpha=alpha, eta0=eta0, theta_MDN=theta_MDN, bias=bias):
    env, num_steps, d = data_loader(dataset)
    w = np.zeros([d]) #+ (1 if bias else 0)])
    b = 0.0
    dd = d+1 if bias else d
    input_normalizer = _2DN(theta=theta_MDN, eta0=eta0, weight_of_MDN=weight_of_MDN, transform=transform)
    MSE_list = []

    for iteration in range(num_steps):
        x,z = env.next()
        x_tilde, mu, sigma = input_normalizer.step(x, z, w, b)

        y = (w*x_tilde).sum() + b
        try:
            MSE_list.append((y-z)**2)
        except:
            break

        w = w + (alpha/dd) * (z-y)*x_tilde
        if bias:
            b = b + (alpha/dd)*(z-y)
        
    MSE_list = np.append(MSE_list, np.nan*np.ones(num_steps-len(MSE_list)))  # padding with np.nan in case of divergence
    steady_state_RMSE = np.sqrt(MSE_list[start_of_steady_state:].mean())
    
    return MSE_list, steady_state_RMSE




if __name__ == "__main__":
    MSE_list, steady_state_RMSE = main_loop()

    ###-----------------------
    print(f'RMSE after {start_of_steady_state}:  {np.round(steady_state_RMSE,3)},     {[alg_config[key] for key in alg_config]}')
    sanitize_and_save(save_to=save_to, MSE_list=MSE_list, alg_config=alg_config)
    