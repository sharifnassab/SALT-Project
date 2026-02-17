import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
import ast
import itertools
import numpy as np
import csv
import json
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from tqdm import tqdm

csv_file = 'csvs/results_.csv'
#x_lim = None
x_lim = [1e-6, 1]

dataset_info = [
    {'name':'RSS1', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 22.5},
    {'name':'RSS2', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 28.0},
    {'name':'RSS3', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 28.0},
    {'name':'RSS4', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 28.0},
    {'name':'RSS5', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 27.0},
    {'name':'RSS6', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 28.0},
    {'name':'RSS7', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 23.0},
    {'name':'ASH1', "dim":10, 'From':'900k', "To":"1.1m", "ylim_max": 15.0},
    {'name':'ASH2', "dim":1, 'From':'900k', "To":"1.1m", "ylim_max": 15.0},
    {'name':'ASH3', "dim":3, 'From':'900k', "To":"1.1m", "ylim_max": 25.0},
    {'name':'ASH4', "dim":3, 'From':'900k', "To":"1.1m", "ylim_max": 35.0},
    {'name':'ASH5', "dim":16, 'From':'900k', "To":"1.1m", "ylim_max": 25.0},
    {'name':'ASH6', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 40.0},
    {'name':'ASH7', "dim":30, 'From':'900k', "To":"1.1m", "ylim_max": 50.0},
    {'name':'ASH8', "dim":10, 'From':'900k', "To":"1.1m", "ylim_max": 100.0},
    {'name':'ASH9', "dim":1, 'From':'900k', "To":"1.1m", "ylim_max": 24.0},
    {'name':'ASH10', "dim":5, 'From':'900k', "To":"1.1m", "ylim_max": 20.0},
    {'name':'ASH11', "dim":4, 'From':'900k', "To":"1.1m", "ylim_max": 20.0},
    {'name':'ASH12', "dim":5, 'From':'900k', "To":"1.1m", "ylim_max": 60.0},
    {'name':'ASH13', "dim":2, 'From':'900k', "To":"1.1m", "ylim_max": 80.0},
]

experiments = [
    #{"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"LMS-EANcenter", "Model":"affine", "Params":{'eta':'0.001'}, "x_axis":"alpha", "label_override":r"LMS_EANcenter ($\eta=0.001$)"},
    #{"Alg":"LMS-EAN", "Model":"affine", "Params":{"tau":100}, "x_axis":"alpha", "label_override":r"LMS EAN $\tau=100$"}, # best in class (among all EAN and EAN2)
    #{"Alg":"LMS-EAN", "Model":"affine", "Params":{"tau":1000}, "x_axis":"alpha", "label_override":r"LMS EAN $\tau=1000$"},
    #{"Alg":"LMS_EAN2", "Model":"affine", "Params":{"eta":0.001, "eta_v":0.01}, "x_axis":"alpha", "label_override":r"LMS EAN2 $\eta_v=0.01, \eta_{cent}=0.001$"}, # best in class
    ##{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha", "label_override":r"LMS MDN \theta^{norm}=0.01"},
    #{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha", "label_override":r"LMS MDN $\theta^{norm}=0.001$"},
    # {"Alg":"LMS_MDNEAN", "Model":"affine", "Params":{"theta":0.001, "eta_v":0.001}, "x_axis":"alpha", "label_override":"\n"+r"LMS MDNEAN"+"\n   "+r"$\theta^{norm}=0.001, \eta_v=0.001$"},
    # {"Alg":"LMS_MDNEAN", "Model":"affine", "Params":{"theta":0.001, "eta_v":0.01}, "x_axis":"alpha", "label_override":"\n"+r"LMS MDNEAN"+"\n   "+r"$\theta^{norm}=0.001, \eta_v=0.01$"},
    ##{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.01, "transform": "pre:none,post:none"}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.01$"+"\n   "+r"pre:none,post:none"},
    # {"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": "pre:none,post:none"}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r"pre:none,post:none"},
    # {"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": "pre:signW,post:none"}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r"pre:signW,post:none"},
    # {"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:none,post:v@0.01'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:none,post:v@0.01'},
    #{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:signW,post:v@0.01'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:signW,post:v@0.01'},
    #{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:none,post:v@0.001'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:none,post:v@0.001'},
    #{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:signW,post:v@0.001'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:signW,post:v@0.001'},
    #{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:none,post:both@0.01@0.01'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:none,post:both@0.01@0.01'},
    #######{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:signW,post:both@0.01@0.01'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:signW,post:both@0.01@0.01'},
    #{"Alg":"LMS-GMDN", "Model":"affine", "Params":{"theta":0.001, "transform": 'pre:signW,post:both@0.001@0.001'}, "x_axis":"alpha", "label_override":"\n"+r"LMS GMDN $\theta^{meta}=0.001$"+"\n   "+r'pre:signW,post:both@0.001@0.001'},
    #{"Alg":"LMS-MDN2", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha", "label_override":r"LMS MDN2 $\theta^{meta}=0.01$"},
    #{"Alg":"LMS-MDN2", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha", "label_override":r"LMS MDN2 $\theta^{meta}=0.001$"},
    #{"Alg":"CGD_EAN", "Model":"affine", "Params":{"normalizer_params.eta":0.001}, "x_axis":"alpha", "label_override":r"CGD_EAN $\eta=0.001$"},
    #{"Alg":"CGD_EAN", "Model":"affine", "Params":{"normalizer_params.eta":0.0001}, "x_axis":"alpha", "label_override":r"CGD_EAN $\eta=0.0001$"},   # good
    #{"Alg":"CGD_MDN", "Model":"affine", "Params":{"normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":r"CGD_MDN $\theta^{norm}=0.001$"},
    #{"Alg":"CGD_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"x_type_in_meta_grad":"x_center", "h_decay": 0.9, "base.theta_MDN":"theta_meta/10"}, "x_axis":"metastep", "label_override":"\n\n"+r"CGD^MGEN (x in $grad^{meta}$=$x_{center}$)"+"\n  "+f"      (h_decay=0.9)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"CGD_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"x_type_in_meta_grad":"x", "h_decay": 0.9, "base.theta_MDN":"theta_meta/10"}, "x_axis":"metastep", "label_override":"\n\n"+r"CGD^MGEN (x in $grad^{meta}$=$x$)"+"\n  "+f"      (h_decay=0.9)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"CGD_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"x_type_in_meta_grad":"x_center", "h_decay": 0.9, "base.eta":"0.0001"}, "x_axis":"metastep", "label_override":"\n\n"+r"CGD^MGEN (x in $grad^{meta}$=$x_{center}$)"+"\n  "+f"      (h_decay=0.9)"+"\n  +"+r"EAN ($\eta=0.0001$)"},
    #{"Alg":"CGD_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"x_type_in_meta_grad":"x", "h_decay": 0.9, "base.eta":"0.0001"}, "x_axis":"metastep", "label_override":"\n\n"+r"CGD^MGEN (x in $grad^{meta}$=x)"+"\n  "+f"      (h_decay=0.9)"+"\n  +"+r"EAN ($\eta=0.0001$)"},
    #{"Alg":"Obn_normalized_to_x_not_xb", "Model":"affine", "Params":{"eta_norm_x":0.0}, "x_axis":"alpha", "label_override":"\n"+r"Obn_normalized_to_x_not_xb" + "\n  "+ r"($\eta_{||x||}=0$)"},
    #{"Alg":"Obn_none", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"none"}, "x_axis":"alpha", "label_override":r"Obn ($\eta_{||x||}=1$)"},   # champion for ASH
    ##{"Alg":"Obn_none", "Model":"affine", "Params":{"eta_norm_x":0.01, "normalizer":"none"}, "x_axis":"alpha", "label_override":r"Obn ($\eta_{||x||}=0.01$)"},
    #{"Alg":"Obn_EAN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN", "normalizer_params.eta":0.01}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"EAN($\eta=0.01$)"},  # best in class overall (better in RSS and worse in ASH compared to 0.001)
    #{"Alg":"Obn_EAN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN", "normalizer_params.eta":0.001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"EAN($\eta=0.001$)"},
    #{"Alg":"Obn_EAN2", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN2", "normalizer_params.eta":0.1, "normalizer_params.eta_v":0.01}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"EAN2($\eta=0.1, \eta_v=0.01$)"}, # bad in ASH
    ## {"Alg":"Obn_EAN2", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN2", "normalizer_params.eta":0.01, "normalizer_params.eta_v":0.001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"EAN2($\eta=0.01, \eta_v=0.001$)"},
    ## {"Alg":"Obn_EAN2", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN2", "normalizer_params.eta":0.001, "normalizer_params.eta_v":0.0001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"EAN2($\eta=0.001, \eta_v=0.0001$)"},
    #{"Alg":"Obn_MDNEAN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"MDNEAN", "normalizer_params.theta_MDN":0.001, "normalizer_params.eta_v":0.001}, "x_axis":"alpha", "label_override":"\n\n"+r"Obn ($\eta_{||x||}=1$)" +"\n + MDNEAN"+"\n   "+r"$\theta^{norm}=0.001, \eta_v=0.001$"},
    #{"Alg":"Obn_MDNEAN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"MDNEAN", "normalizer_params.theta_MDN":0.001, "normalizer_params.eta_v":0.01}, "x_axis":"alpha", "label_override":"\n\n"+r"Obn ($\eta_{||x||}=1$)" +"\n + MDNEAN"+"\n   "+r"$\theta^{norm}=0.001, \eta_v=0.01$"}, # best in subclass of Obn_MDNEAN
    ##{"Alg":"Obn_EAN", "Model":"affine", "Params":{"eta_norm_x":0.01, "normalizer":"EAN", "normalizer_params.eta":0.001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=0.01$)" +"\n  +"+r"EAN($\eta=0.001$)"},
    #{"Alg":"Obn_MDN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"MDN", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"MDN($\theta^{norm}=0.001$)"},
    ##{"Alg":"Obn_MDN", "Model":"affine", "Params":{"eta_norm_x":0.01, "normalizer":"MDN", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=0.01$)" +"\n  +"+r"MDN($\theta^{norm}=0.001$)"},
    #{"Alg":"Obn_GMDN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"GMDN", 'normalizer_params.theta_MDN':0.001, "normalizer_params.transform":'pre:signW,post:both@0.01@0.01'}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1.0$)" +"\n  +"+r"GMDN($\theta^{norm}=0.001$)" + "\n"+r"pre:signW,post:both@0.01@0.01"}, 
    #### {"Alg":"Obn_GMDN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"GMDN", 'normalizer_params.theta_MDN':0.0001, "normalizer_params.transform":'pre:signW,post:both@0.01@0.01'}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1.0$)" +"\n  +"+r"GMDN($\theta^{norm}=0.0001$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},  # best in subclass of normalized Obn
    #{"Alg":"Obn_GMDN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"GMDN", 'normalizer_params.theta_MDN':0.001, "normalizer_params.transform":'pre:signW,post:both@0.001@0.001'}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1.0$)" +"\n  +"+r"GMDN($\theta^{norm}=0.001$)" + "\n"+r"pre:signW,post:both@0.001@0.001"}, 
    ##{"Alg":"ObnCenter_EAN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN", "normalizer_params.eta":0.001}, "x_axis":"alpha", "label_override":"\n"+r"ObnCenter ($\eta_{||x||}=1$)" +"\n  +"+r"EAN($\eta=0.001$)"},
    ##{"Alg":"ObnCenter_EAN", "Model":"affine", "Params":{"eta_norm_x":0.01, "normalizer":"EAN", "normalizer_params.eta":0.001}, "x_axis":"alpha", "label_override":"\n"+r"ObnCenter ($\eta_{||x||}=0.01$)" +"\n  +"+r"EAN($\eta=0.001$)"},
    ##{"Alg":"ObnCenter_MDN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"MDN", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"ObnCenter ($\eta_{||x||}=1$)" +"\n  +"+r"MDN($\theta^{norm}=0.001$)"},
    ##{"Alg":"ObnCenter_MDN", "Model":"affine", "Params":{"eta_norm_x":0.01, "normalizer":"MDN", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"ObnCenter ($\eta_{||x||}=0.01$)" +"\n  +"+r"MDN($\theta^{norm}=0.001$)"},
    #{"Alg":"TwoLLMs", "Model":"affine", "Params":{"alpha1": "alpha/2", "RMSPw": 0.999, "RMSPc": 0.999}, "x_axis":"alpha", "label_override":"\n"+r"TwoLLMs ($\alpha_1=0.5\alpha_2$)" +"\n    +"+r"($RMSP=0.999$)"},
    #{"Alg":"TwoLLMs", "Model":"affine", "Params":{"alpha1": "alpha/2", "RMSPw": 0.99, "RMSPc": 0.99}, "x_axis":"alpha", "label_override":"\n"+r"TwoLLMs ($\alpha_1=0.5\alpha_2$)" +"\n    +"+r"($RMSP=0.99$)"}, # best TwoLLMs (surpassed by TwoLLMs2 only in RSS)   Base Champion
    #{"Alg":"TwoLLMs2", "Model":"affine", "Params":{"alpha1": "alpha/2", "RMSPw": 0.999, "RMSPc": 0.999}, "x_axis":"alpha", "label_override":"\n"+r"TwoLLMs2 ($\alpha_1=0.5\alpha_2$)" +"\n    +"+r"($RMSP=0.999$)"},
    #{"Alg":"TwoLLMs2", "Model":"affine", "Params":{"alpha1": "alpha/2", "RMSPw": 0.99, "RMSPc": 0.99}, "x_axis":"alpha", "label_override":"\n"+r"TwoLLMs2 ($\alpha_1=0.5\alpha_2$)" +"\n    +"+r"($RMSP=0.99$)"},
    #{"Alg":"TwoLLMsOb", "Model":"affine", "Params":{"a1_to_a2_ratio": "0.5", "RMSPw": 0.99, "RMSPc": 0.99}, "x_axis":"alpha", "label_override":"\n"+r"TwoLLMsOb ($\alpha_1/\alpha_2=0.5$)" +"\n    +"+r"($RMSP=0.99$)"}, # best TwoLLMs (surpassed by TwoLLMs2 only in RSS)
    #{"Alg":"RMSProp", "Model":"affine", "Params":{"beta_nu":.999}, "x_axis":"alpha", "label_override":r"RMSProp ($\beta_\nu=.999$)"},
    #{"Alg":"RMSProp", "Model":"affine", "Params":{"beta_nu":.99}, "x_axis":"alpha", "label_override":r"RMSProp ($\beta_\nu=.99$)"}, # best RMSP (only better at ASH9, similar elsewhere)
    #{"Alg":"Adam", "Model":"affine", "Params":{"beta_nu":.999}, "x_axis":"alpha"},
    #{"Alg":"RMSPropCenter", "Model":"affine", "Params":{"beta_nu":.99, "beta_centering": 0.99}, "x_axis":"alpha", "label_override":r"RMSPropC ($\beta_\nu=0.99, \beta_{center}=0.99$)"},
    #{"Alg":"RMSPropCenter", "Model":"affine", "Params":{"beta_nu":.99, "beta_centering": 0.999}, "x_axis":"alpha", "label_override":r"RMSPropC ($\beta_\nu=0.99, \beta_{cent}=0.999$)"},  # best RMSPropCenter except for RSS1.
    #{"Alg":"AdamCenter", "Model":"affine", "Params":{"beta_nu":.99, "beta_centering": 0.999}, "x_axis":"alpha", "label_override":r"AdamC ($\beta_\nu=0.99, \beta_{center}=0.999)"},
    #{"Alg":"AdamCenter", "Model":"affine", "Params":{"beta_centering": 0.999}, "x_axis":"alpha", "label_override":r"AdamC (centering=0.999)"},
    #{"Alg":"AdamCenter", "Model":"affine", "Params":{"beta_centering": 0.9999}, "x_axis":"alpha", "label_override":r"AdamC (centering=0.9999)"},
    #{"Alg":"AdamCenterSplit", "Model":"affine", "Params":{"beta_centering": 0.999}, "x_axis":"alpha", "label_override":r"AdamCA (centering=0.999)"},
    #{"Alg":"AdamCenterSplit", "Model":"affine", "Params":{"beta_centering": 0.9999}, "x_axis":"alpha", "label_override":r"AdamCA (centering=0.9999)"},
    #{"Alg":"Newton", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': .001, 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGEN"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta'}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}$)"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta/10', "meta_eta": 0.001}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN (\eta^{meta}=0.001)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},   # best in class
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta/10', "meta_eta": 0.01}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN (\eta^{meta}=0.01)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:none"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:none,post:none"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:v@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:none,post:v@0.01"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:v@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:signW,post:v@0.01"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:v@0.001"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:none,post:v@0.001"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:v@0.001"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:signW,post:v@0.001"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:none,post:both@0.01@0.01"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_GMDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.001@0.001"}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.001@0.001"},
    #{"Alg":"LMS_MDN^IDBD_MGENC", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENC"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENC_alpha_only", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.0", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(0)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.9", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.9)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.99", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.99)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.999", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.999)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha*x2", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.0", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(0)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.9", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.9)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.99", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.99)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.999", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.999)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha*x2", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center_and_augment'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN CA ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta':'0.001', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN ($\eta=10^{-3}$)"},
    # {"Alg":"LMS_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta':'0.001', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN C ($\eta=10^{-3}$)"},
    # {"Alg":"LMS_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta':'0.001', 'base.cente_only':'center_and_augment'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN CA ($\eta=10^{-3}$)"},
    #{"Alg":"LMS_EAN_eta_eq_alpha^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta/alpha':1}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN ($\eta=\alpha$)"},
    #{"Alg":"LMS_EAN_eta_eq_alpha^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta/alpha':.1}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN ($\eta=\alpha/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENC_delta_delta", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_MGENChd(0)"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"+"\n [delta delta]"},
    #{"Alg":"LMS_MDN^IDBD_MGENChdOb", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.01, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdOb"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.01$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    # {"Alg":"LMS_MDNEAN^IDBD_MGENChdOb", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.01, 'base.theta_MDN':'0.001', 'base.eta_v':'0.01', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdOb"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.01$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDNEAN"+"\n     "+r"$\theta^{norm}=0.001, \eta_v=0.01$"}, 
    # {"Alg":"LMS_MDNEAN^IDBD_MGENChdOb", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.01, 'base.theta_MDN':'0.001', 'base.eta_v':'0.001', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdOb"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.01$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDNEAN"+"\n     "+r"$\theta^{norm}=0.001, \eta_v=0.001$"}, 
    # #{"Alg":"LMS_MDNEAN^IDBD_MGENChdOb", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.01, 'base.theta_MDN':'theta_meta/10', 'base.eta_v':'0.01', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdOb"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.01$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDNEAN"+"\n     "+r"$\theta^{norm}=\theta^{meta}/10, \eta_v=0.01$"}, 
    # {"Alg":"LMS_MDNEAN^IDBD_MGENChdOb", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.001, 'base.theta_MDN':'0.001', 'base.eta_v':'0.001', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdOb"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.001$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDNEAN"+"\n     "+r"$\theta^{norm}=0.001, \eta_v=0.001$"}, 
    #{"Alg":"LMS_MDN^IDBD_MGENChdObCorrected", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.01, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdObCorrected"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.01$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    #{"Alg":"LMS_MDN^IDBD_MGENChdObCorrected", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.1, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdObCorrected"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.1$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    #{"Alg":"LMS_MDNEAN^IDBD_MGENChdObCorrected", "Model":"affine", "Params":{"h_decay":"1-alpha", "meta_eta": 0.1, 'base.theta_MDN':'0.001', 'base.eta_v':'0.01', 'base.cente_only':'N/A', "alpha_min":0.001, "os_decay": 0.99}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENChdObCorrected"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.1$"+"\n  "+r"$\alpha_{min}=0.001/d, decay_{os}=0.99$"+"\n +"+r"MDNEAN ($\theta^{norm}=0.001, \eta_v=0.01$)"},  # good in the class
    #{"Alg":"LMS_MDN^IDBD_MGENedge", "Model":"affine", "Params":{"edge_param":0.5, "h_decay":"1-alpha", "meta_eta": 0.1, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENedge"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.1$"+"\n  "+r"$\alpha_{min}=0.001/d, edge=0.5$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    #{"Alg":"LMS_MDN^IDBD_MGENedge", "Model":"affine", "Params":{"edge_param":0.25, "h_decay":"1-alpha", "meta_eta": 0.1, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENedge"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.1$"+"\n  "+r"$\alpha_{min}=0.001/d, edge=0.25$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    #{"Alg":"LMS_MDN^IDBD_MGENedge", "Model":"affine", "Params":{"edge_param":0.75, "h_decay":"1-alpha", "meta_eta": 0.1, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENedge"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.1$"+"\n  "+r"$\alpha_{min}=0.001/d, edge=0.75$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    #{"Alg":"LMS_MDN^IDBD_MGENedge", "Model":"affine", "Params":{"edge_param":0.1, "h_decay":"1-alpha", "meta_eta": 0.1, 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A', "alpha_min":0.001}, "x_axis":"metastep", "label_override":"\n\n\n\n"+r"MGENedge"+"\n  "+r"$hd=1-\alpha, \eta^{meta}=0.1$"+"\n  "+r"$\alpha_{min}=0.001/d, edge=0.1$"+"\n +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"}, 
    #{"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    # {"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}$)"},
    #{"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:none"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:none,post:none"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:v@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:none,post:v@0.01"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:v@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:signW,post:v@0.01"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:none,post:v@0.001"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:none,post:v@0.001"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:v@0.001"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n   "+r"pre:signW,post:v@0.001"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_GMDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.001@0.001"}, "x_axis":"metastep", "label_override": "\n"+r"NIDBD3"+"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.001@0.001"},
    #{"Alg":"LMS_MDN^NIDBD3hd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3hd(1-\alpha)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^IDBD_comb", "Model":"affine", "Params":{"meta_eta":"0.01", "delta_norm":"none", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_comb"+"\n  " +r"($\delta^{norm}$:none, $\eta^{meta}$:.01)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^IDBD_comb", "Model":"affine", "Params":{"meta_eta":"0.01", "delta_norm":"eta", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_comb"+"\n  " +r"($\delta^{norm}:\eta$, $\eta^{meta}$:.01)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^IDBD_comb", "Model":"affine", "Params":{"meta_eta":"0.01", "delta_norm":".1eta", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_comb"+"\n  " +r"($\delta^{norm}:.1\eta$, $\eta^{meta}$:.01)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^IDBD_comb", "Model":"affine", "Params":{"meta_eta":"theta_meta/10", "delta_norm":"none", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_comb"+"\n  " +r"($\delta^{norm}$:none, $\eta^{meta}:\theta^{meta}/10$)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^IDBD_comb", "Model":"affine", "Params":{"meta_eta":"theta_meta/10", "delta_norm":"eta", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_comb"+"\n  " +r"($\delta^{norm}:\eta$, $\eta^{meta}:\theta^{meta}/10$)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^IDBD_comb", "Model":"affine", "Params":{"meta_eta":"theta_meta/10", "delta_norm":".1eta", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_comb"+"\n  " +r"($\delta^{norm}:.1\eta$, $\eta^{meta}:\theta^{meta}/10$)"+"\n  +"+r"MDN ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_withMetaPostNorm", "Model":"affine", "Params":{"meta_post_norm":"0.01", "beta_inc_clip":"none", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01"+ "\n  "+r"   $clip^{\beta_{inc}}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_withMetaPostNorm", "Model":"affine", "Params":{"meta_post_norm":"0.01", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01"+ "\n  "+r"   $clip^{\beta_{inc}}$=0.1)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_withMetaPostNorm", "Model":"affine", "Params":{"meta_post_norm":"0.001", "beta_inc_clip":"none", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.001"+ "\n  "+r"   $clip^{\beta_{inc}}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_withMetaPostNorm", "Model":"affine", "Params":{"meta_post_norm":"0.001", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.001"+ "\n  "+r"   $clip^{\beta_{inc}}$=0.1)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"steady", "h_pre_norm":"none", "meta_post_norm":"0.01", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=steady, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"none", "h_pre_norm":"steady", "meta_post_norm":"0.01", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=steady)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.01", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":True, "act": "softmax", "beta_pre_norm":"none", "h_pre_norm":"steady", "meta_post_norm":"0.01", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=yes, $pre_{\beta}$=none, $pre_{h}$=steady)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":True, "act": "softmax", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.01", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.01, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=yes, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "act": "softmax", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},    # Champion
    # {"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "act": "sigmoid", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n\n"+r"NIDBD3 (sigmoid)"+ "\n  " +r"PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "act": "exp", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n\n"+r"NIDBD3 (exp)"+ "\n  " +r"PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre_K1", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3_K1 (PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_MetaPost_vs_Pre_K1", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"none", "h_pre_norm":"alpha", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3_K1 (PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=alpha)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_GMDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "act": "softmax", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"NIDBD3 (PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},   
    # {"Alg":"LMS_GMDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "act": "softmax", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.001@0.001"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"NIDBD3 (PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.001@0.001"}, 
    # {"Alg":"LMS_GMDN^NIDBD3_MetaPost_vs_Pre", "Model":"affine", "Params":{"delta_norm":False, "act": "softmax", "beta_pre_norm":"none", "h_pre_norm":"none", "meta_post_norm":"0.99", "beta_inc_clip":"0.1", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"NIDBD3 (PostN=0.99, $clip^{\beta_{inc}}$=0.1"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}$=none)" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},    # best in class
    #{"Alg":"LMS_MDN^NIDBD3_Oct2025_champion", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3_OctChamp (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},    # Champion
    # {"Alg":"LMS^K1", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"none", "h_pre_norm":"__1-alpha*x2__alpha", "meta_post_norm":"none", "beta_inc_clip":"none", "h_decay": "1-alpha*x2"}, "x_axis":"metastep", "label_override":"\n"+r"K1 (PostN=0.99, $clip^{\beta_{inc}}$=none"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}=(1-\alpha x^2)\alpha$)"},
    # {"Alg":"LMS^K1", "Model":"affine", "Params":{"delta_norm":False, "beta_pre_norm":"none", "h_pre_norm":"alpha", "meta_post_norm":"none", "beta_inc_clip":"none", "h_decay": "1-alpha*x2"}, "x_axis":"metastep", "label_override":"\n"+r"K1 (PostN=0.99, $clip^{\beta_{inc}}$=none"+ "\n  "+r"$N_{\delta}$=no, $pre_{\beta}$=none, $pre_{h}=\alpha$)"},
    # {"Alg":"LMS_MDN^NIDBD3_noSoftmax", "Model":"affine", "Params":{"act":"softmax", "N_delta_decay":"0.99", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (act=softmax"+ "\n  "+r"     $N_{\delta}$=0.99)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_noSoftmax", "Model":"affine", "Params":{"act":"sigmoid", "N_delta_decay":"0.99", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (act=sigmoid"+ "\n  "+r"     $N_{\delta}$=0.99)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^NIDBD3_noSoftmax", "Model":"affine", "Params":{"act":"exp", "N_delta_decay":"0.99", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 (act=exp"+ "\n  "+r"     $N_{\delta}$=0.99)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"none", "beta_inc_clip":"none", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=none"+ "\n  "+r"   $clip^{\beta_{inc}}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"maxmin_10", "beta_inc_clip":"none", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=maxmin_10"+ "\n  "+r"   $clip^{\beta_{inc}}$=none)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"maxmin_10", "beta_inc_clip":"max_0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=maxmin_10"+ "\n  "+r"   $clip^{\beta_{inc}}$=max_0.1)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"none", "beta_inc_clip":"maxmin_0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=none"+ "\n  "+r"   $clip^{\beta_{inc}}$=maxmin_0.1)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"maxmin_10", "beta_inc_clip":"maxmin_0.1", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=maxmin_10"+ "\n  "+r"   $clip^{\beta_{inc}}$=maxmin_0.1)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    ##{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"none", "beta_inc_clip":"maxmin_0.5", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=none"+ "\n  "+r"   $clip^{\beta_{inc}}$=maxmin_0.5)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3_ClipTest", "Model":"affine", "Params":{"beta_clip":"maxmin_10", "beta_inc_clip":"maxmin_0.5", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3 ($clip^{\beta}$=maxmin_10"+ "\n  "+r"   $clip^{\beta_{inc}}$=maxmin_0.5)" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"TwoLLMs_MGEN", "Model":"affine", "Params":{"h_decay":0.9, "RMSPw": 0.99, "RMSPc": 0.99, "RMSPw_meta": 0.99, "RMSPc_meta": 0.99, "alpha_min": 0.0001}, "x_axis":"metastep", "label_override":"\n"+r"TwoLLMs_MGEN (hd=0.9)"+"\n  "+r"$RMSP=0.99, \alpha_{min}=0.0001/d$"},  
    #{"Alg":"TwoLLMs_MGEN", "Model":"affine", "Params":{"h_decay":"1-alpha", "RMSPw": 0.999, "RMSPc": 0.999, "RMSPw_meta": 0.99, "RMSPc_meta": 0.99, "alpha_min": 0.0001}, "x_axis":"metastep", "label_override":"\n"+r"TwoLLMs_MGEN (hd=1-$\alpha$)"+"\n  "+r"$RMSP_{base}=0.999, RMSP_{meta}=0.99, \alpha_{min}=0.0001/d$"}, 
    #{"Alg":"TwoLLMsL1C_MGEN", "Model":"affine", "Params":{"h_decay":0.9, "RMSPw": 0.99, "RMSPw_meta": 0.99, "RMSPc_meta": 0.99, "alpha_min": 0.0001}, "x_axis":"metastep", "label_override":"\n"+r"TwoLLMsL1C_MGEN (hd=0.9)"+"\n  "+r"$RMSP=0.99, \alpha_{min}=0.0001/d$"},  # best in the class
    #{"Alg":"TwoLLMsL1C_MGEN", "Model":"affine", "Params":{"h_decay":"1-alpha", "RMSPw": 0.999, "RMSPw_meta": 0.99, "RMSPc_meta": 0.99, "alpha_min": 0.0001}, "x_axis":"metastep", "label_override":"\n"+r"TwoLLMsL1C_MGEN (hd=1-$\alpha$)"+"\n  "+r"$RMSP_{base}=0.999$" +"\n  "+r"$RMSP_{meta}=0.99, \alpha_{min}=0.0001/d$"},  
    # {"Alg":"LMS_MDN^SIDBD", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS^SIDBD", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n"+r"SIDBD (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10"},
    #{"Alg":"LMS^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10"},   # champion for ASH
    #{"Alg":"LMS_MDN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # good
    #{"Alg":"LMS_GMDN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_GMDN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"}, # good (pareto)
    #{"Alg":"LMS_GMDNcenter^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDNcenter($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:0.01"}, # 
    #{"Alg":"LMS_GMDNcenter^SIDBD_ObnSig", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_ObnSig (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDNcenter($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:0.01"}, # 
    # {"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "1x_meta_stepsize", "meta_reg":"N/A", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=\theta, PostN=0.99$"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # best in class of HIDBD
    # {"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "10x_meta_stepsize", "meta_reg":"N/A", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=10\theta, PostN=0.99$"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # {"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "100x_meta_stepsize", "meta_reg":"N/A", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=100\theta, PostN=0.99$"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    # #
    #{"Alg":"LMS_GMDN^SIDBD_ObnDe", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_ObnDe (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"}, # good
    # {"Alg":"LMS_GMDNcenter^SIDBD_ObnDe", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_ObnDe (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDNcenter($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:0.01"}, # 
    #{"Alg":"ObnDe_EAN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"EAN", "normalizer_params.eta":0.01}, "x_axis":"alpha", "label_override":"\n"+r"ObnDe ($\eta_{||x||}=1$)" +"\n  +"+r"EAN($\eta=0.01$)"},  # best in class overall (better in RSS and worse in ASH compared to 0.001)
    #{"Alg":"ObnDe_MDN", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"MDN", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"ObnDe ($\eta_{||x||}=1$)" +"\n  +"+r"MDN($\theta^{norm}=0.001$)"},
    # {"Alg":"Obn_GMDNcenter", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"GMDNcenter", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"Obn ($\eta_{||x||}=1$)" +"\n  +"+r"GMDNcenter($\theta^{norm}=0.001$)"},
    # {"Alg":"ObnDe_GMDNcenter", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"GMDNcenter", "normalizer_params.theta_MDN":0.001}, "x_axis":"alpha", "label_override":"\n"+r"ObnDe ($\eta_{||x||}=1$)" +"\n  +"+r"GMDNcenter($\theta^{norm}=0.001$)"},
    #{"Alg":"LMS_MDNcenter^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"post:0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDNcenter($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"post:0.01"}, # 
    #{"Alg":"LMS_MDCGN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDCGN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"}, # good
    #
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.5", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=\theta^{meta}, w_{MDN}=.5$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=\theta^{meta}, w_{MDN}=.2$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.1", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=\theta^{meta}, w_{MDN}=.1$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.5", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=.5$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=.2$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.1", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=.1$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-0.1", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=\theta^{meta}, w_{MDN}=-.1$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-0.2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=\theta^{meta}, w_{MDN}=-.2$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-0.5", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=\theta^{meta}, w_{MDN}=-.5$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-0.1", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=-.1$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    # {"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-0.2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=-.2$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-0.5", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=-.5$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_2DN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"-1.0", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN($\theta^{norm}=.1\theta^{meta}, w_{MDN}=-1$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
    #{"Alg":"LMS_2DN_^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN_($\theta^{norm}=\theta^{meta}, w_{MDN}=.2$)" + "\n"+r"pre:signW,post:both@0.01@0.01"}, # maybe best in class (others are also equally good)
    #
    # ##{"Alg":"LMS_GC", "Model":"affine", "Params":{'alpha_mu':'alpha/2'}, "x_axis":"alpha", "label_override":r"LMS_GC ($\alpha_\mu=\alpha/2$)"},
    # {"Alg":"LMS_GC", "Model":"affine", "Params":{'alpha_mu':'alpha/10'}, "x_axis":"alpha", "label_override":r"LMS_GC ($\alpha_\mu=\alpha/10$)"},
    # ##{"Alg":"LMS_GCobn", "Model":"affine", "Params":{'alpha_mu':'0.01'}, "x_axis":"alpha", "label_override":r"LMS_GCobn ($\alpha_\mu=0.01$)"},
    # {"Alg":"LMS_GCobn", "Model":"affine", "Params":{'alpha_mu':'0.001'}, "x_axis":"alpha", "label_override":r"LMS_GCobn ($\alpha_\mu=0.001$)"},
    # ##{"Alg":"LMS_GCsignObn", "Model":"affine", "Params":{'alpha_mu':'0.01'}, "x_axis":"alpha", "label_override":r"LMS_GCsignObn ($\alpha_\mu=0.01$)"},
    # {"Alg":"LMS_GCsignObn", "Model":"affine", "Params":{'alpha_mu':'0.001'}, "x_axis":"alpha", "label_override":r"LMS_GCsignObn ($\alpha_\mu=0.001$)"},
    #
    # {"Alg":"LMS_GCsidbd", "Model":"affine", "Params":{'meta_stepsize_mu':'0.001', 'meta_post_norm_mu':'0.99'}, "x_axis":"alpha", "label_override":r"LMS_GCsidbd ($\theta_\mu=0.001, PostN_\mu=0.99$)"},
    # {"Alg":"LMS_GCsidbdObn", "Model":"affine", "Params":{'meta_stepsize_mu':'0.001', 'meta_post_norm_mu':'0.99'}, "x_axis":"alpha", "label_override":r"LMS_GCsidbdObn ($\theta_\mu=0.001, PostN_\mu=0.99$)"},
    # {"Alg":"LMS_GCsidbdSignObn", "Model":"affine", "Params":{'meta_stepsize_mu':'0.001', 'meta_post_norm_mu':'0.99'}, "x_axis":"alpha", "label_override":r"LMS_GCsidbdSignObn ($\theta_\mu=0.001, PostN_\mu=0.99$)"},
    # #
    # {"Alg":"LMS_GC^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"1e-05", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GC($\alpha_{\mu}=10^{-5}$)"}, 
    # {"Alg":"LMS_GC^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"1e-06", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GC($\alpha_{\mu}=10^{-6}$)"}, # best in class
    ##{"Alg":"LMS_GCobn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"0.01", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCobn($\alpha_{\mu}=0.01$)"}, 
    ##{"Alg":"LMS_GCobn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCobn($\alpha_{\mu}=0.001$)"},
    ##{"Alg":"LMS_GCobn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"alpha/2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCobn($\alpha_{\mu}=\alpha/2$)"}, 
    # {"Alg":"LMS_GCobn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"alpha/10", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCobn($\alpha_{\mu}=\alpha/10$)"},
    ##{"Alg":"LMS_GCsignObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"0.01", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsignObn($\alpha_{\mu}=0.01$)"}, 
    ##{"Alg":"LMS_GCsignObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsignObn($\alpha_{\mu}=0.001$)"},
    ##{"Alg":"LMS_GCsignObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"alpha/2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsignObn($\alpha_{\mu}=\alpha/2$)"}, 
    ##{"Alg":"LMS_GCsignObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.alpha_mu":"alpha/10", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsignObn($\alpha_{\mu}=\alpha/10$)"},
    # #
    # {"Alg":"LMS_GCsidbd^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.meta_stepsize_mu":"0.001", "base.meta_post_norm_mu":"0.99", "base.alpha0_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsidbd($\theta_\mu=0.001,PostN=0.99$)"},  # best in classs
    ##{"Alg":"LMS_GCsidbd^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.meta_stepsize_mu":"theta_meta", "base.meta_post_norm_mu":"0.99", "base.alpha0_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsidbd($\theta_\mu=\theta,PostN=0.99$)"}, 
    # {"Alg":"LMS_GCsidbdObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.meta_stepsize_mu":"0.001", "base.meta_post_norm_mu":"0.99", "base.alpha0_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsidbdObn($\theta_\mu=0.001,PostN=0.99$)"}, 
    ##{"Alg":"LMS_GCsidbdObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.meta_stepsize_mu":"theta_meta", "base.meta_post_norm_mu":"0.99", "base.alpha0_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsidbdObn($\theta_\mu=\theta,PostN=0.99$)"}, 
    ##{"Alg":"LMS_GCsidbdSignObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.meta_stepsize_mu":"0.001", "base.meta_post_norm_mu":"0.99", "base.alpha0_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsidbdSignObn($\theta_\mu=0.001,PostN=0.99$)"}, 
    # {"Alg":"LMS_GCsidbdSignObn^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99",  "base.meta_stepsize_mu":"theta_meta", "base.meta_post_norm_mu":"0.99", "base.alpha0_mu":"0.001", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"GCsidbdSignObn($\theta_\mu=\theta,PostN=0.99$)"}, 
    # #
    # {"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "1x_meta_stepsize", "meta_reg":"N/A", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=\theta, PostN=0.99$"+ "\n  "+r"$reg=0, clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # best in class of HIDBD
    # {"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "1x_meta_stepsize", "meta_reg":"0.01", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=\theta, PostN=0.99$"+ "\n  "+r"$reg=0.01, clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # best in class of HIDBD
    ##{"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "1x_meta_stepsize", "meta_reg":"0.1", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=\theta, PostN=0.99$"+ "\n  "+r"$reg=0.1, clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # best in class of HIDBD
    ##{"Alg":"LMS_MDN^HIDBD", "Model":"affine", "Params":{"meta_stepsize_scalar": "1x_meta_stepsize", "meta_reg":"1.0", "meta_post_norm":"0.99", "meta_post_norm_scalar": "0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"HIDBD ($\theta_s=\theta, PostN=0.99$"+ "\n  "+r"reg=1, $clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # best in class of HIDBD
    ]


# experiments = [
#     #{"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
#     #{"Alg":"Obn_none", "Model":"affine", "Params":{"eta_norm_x":1.0, "normalizer":"none"}, "x_axis":"alpha", "label_override":r"Obn ($\eta_{||x||}=1$)"},   # champion for ASH
#     #{"Alg":"LMS^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10"}, "x_axis":"metastep", "label_override":"\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10"},   # champion for ASH
#     # {"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
#     {"Alg":"LMS_MDN^NIDBD3_Oct2025_champion", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"NIDBD3_OctChamp (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},    # Champion
#     {"Alg":"LMS_MDN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"}, # good
#     # {"Alg":"LMS_GMDN^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN': 'theta_meta/10', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n  +"+r"GMDN($\theta^{norm}=\theta^{meta}/10$)" + "\n"+r"pre:signW,post:both@0.01@0.01"}, # good (pareto)
#     #{"Alg":"LMS_2DN_^SIDBD_Obn", "Model":"affine", "Params":{"meta_post_norm":"0.99", "base.weight_of_MDN":"0.2", "beta_inc_clip":"maxmin_0.5", "beta_clip":"maxmin_10", 'base.theta_MDN':'theta_meta', "base.transform":"pre:signW,post:both@0.01@0.01"}, "x_axis":"metastep", "label_override":"\n\n\n"+r"SIDBD_Obn (PostN=0.99"+ "\n  "+r"$clip^{\beta_{inc}}$=0.5, $clip^{\beta}$=10" +"\n+"+r"2DN_($\theta^{norm}=\theta^{meta}, w_{MDN}=.2$)" + "\n"+r"pre:signW,post:both@0.01@0.01"},
# ]

if 1:

    experiments.append(
        {"Alg":"global best",
        "among_algorithms":[""],           # [''] means include all algs
        "excluding_algorithms":["Newton"],         # substrings to exclude
        "x_axis":"", # "", "alpha" , # "metastep"
        "label_override":"\nGlobal Best\n(all algorithms no Newton)"}
 )
if 0:
    experiments.append(
        {"Alg":"global best",
        "among_algorithms":[""],           # [''] means include all algs
        "excluding_algorithms":[],         # substrings to exclude
        "x_axis":"", # "", "alpha" , # "metastep"
        "label_override":"\nGlobal Best\n(all algorithms)"}
 )


'''
#{"Alg":"LMS_MDNPN_KC^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
##{"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
#{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha"},
##{"Alg":"IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
#{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
#{"Alg":"LMS_MDN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
'''





# --- helpers: border "X" and non-overlapping up-arrows ---
def _draw_left_border_X(ax, color, size_axes=0.03, lw=1.6):
    """
    Draw an 'X' near the left border, centered vertically, fully inside the axes
    so bbox='tight' won't expand or error.
    """
    trans = ax.transAxes
    x0, y0 = 0.015, 0.5   # was 0.0 -> move slightly inside the axes
    dx, dy = size_axes, size_axes
    ax.plot([x0, x0+dx], [y0-dy, y0+dy], transform=trans, color=color, lw=lw, clip_on=True)
    ax.plot([x0, x0+dx], [y0+dy, y0-dy], transform=trans, color=color, lw=lw, clip_on=True)


def _arrow_slot_generator(n_slots: int = 24):
    """
    Evenly spaced x-positions across the axes width (in axes fraction).
    Keeps arrows separated no matter the x-limits or log scale.
    """
    # Leave small margins left/right so heads aren't clipped.
    slots = np.linspace(0.07, 0.93, n_slots)
    for s in slots:
        yield float(s)
    # If more arrows than slots, recycle with a tiny jitter so they still don't overlap
    k = 0
    while True:
        base = slots[k % len(slots)]
        jitter = 0.002 * ((k // len(slots)) + 1)  # 0.2% per wrap
        yield float(np.clip(base + jitter, 0.05, 0.95))
        k += 1


# ---------------- core helpers (safe edits) ----------------

csv_column_numbers = {'Alg': 0, 'Model': 1, 'Params': 2, 'Dataset': 4, "Dim":5, "From":8, "To":9, 'Result': 16}

def get_label(exp):
    if 'label_override' in exp:
        return exp['label_override']
    return exp['Alg']

def _flatten_dict(d, parent_key='', sep='.'):
    """Utility used by both filtering and curve building to ensure consistent key access."""
    if not isinstance(d, dict): 
        return {}
    items = {}
    for k, v in d.items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, nk, sep))
        else:
            items[nk] = v
    return items

def filter_experiments_with_data(csv_file, dataset, experiments):
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        all_rows = list(csv.reader(file))

    def exists(exp):
        # --- special case: virtual experiment "global best" ---
        if exp.get('Alg') == 'global best':
            among = [s for s in exp.get('among_algorithms', ['']) if s is not None]
            excludes = exp.get('excluding_algorithms', [])

            def alg_ok(name: str) -> bool:
                if any(excl in name for excl in excludes):
                    return False
                if (not among) or any(s == '' for s in among):
                    return True
                return any(sub in name for sub in among)

            for row in all_rows:
                if (row[csv_column_numbers['Dataset']] != str(dataset['name']) or
                    row[csv_column_numbers['Dim']]     != str(dataset['dim']) or
                    row[csv_column_numbers['From']]    != str(dataset['From']) or
                    row[csv_column_numbers['To']]      != str(dataset['To'])):
                    continue
                if not alg_ok(row[csv_column_numbers['Alg']]):
                    continue
                # Parse params & check x-axis existence
                try:
                    param_str = row[csv_column_numbers['Params']].replace('""', '"')
                    params_raw = json.loads(param_str)
                except Exception:
                    try:
                        params_raw = ast.literal_eval(row[csv_column_numbers['Params']])
                    except Exception:
                        continue
                params = _flatten_dict(params_raw)
                if exp.get('x_axis', '') == '' or (exp['x_axis'] in params):
                    return True
            return False

        for row in all_rows:
            if (row[csv_column_numbers['Dataset']] != str(dataset['name']) or
                row[csv_column_numbers['Dim']]     != str(dataset['dim']) or
                row[csv_column_numbers['From']]    != str(dataset['From']) or
                row[csv_column_numbers['To']]      != str(dataset['To']) or
                row[csv_column_numbers['Alg']]     != str(exp['Alg']) or
                row[csv_column_numbers['Model']]   != str(exp['Model'])):
                continue
            # Parse params safely
            try:
                param_str = row[csv_column_numbers['Params']].replace('""', '"')
                params_raw = json.loads(param_str)
            except Exception:
                try:
                    params_raw = ast.literal_eval(row[csv_column_numbers['Params']])
                except Exception:
                    continue
            params = _flatten_dict(params_raw)

            # Must contain the x_axis key after flattening (e.g., "metastep")
            if exp['x_axis'] not in params:
                continue

            # Ensure any specified Params match (after flattening)
            if all(((k not in params) if str(v) == "N/A" else (k in params and str(params[k]) == str(v))) for k, v in exp.get('Params', {}).items()):
                return True
        return False

    return [e for e in experiments if exists(e)]

def build_curves(csv_file, experiments, csv_column_numbers, dataset):
    def flatten_dict(d, parent_key='', sep='.'):
        if not isinstance(d, dict): return {}
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict): items.update(flatten_dict(v, new_key, sep=sep))
            else: items[new_key] = v
        return items
    
    def _safe_parse_params(s):
        if pd.isna(s):
            return {}
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return json.loads(s.replace('""', '"'))
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}
    
    df = pd.read_csv(csv_file, header=None, low_memory=False)
    rename = csv_column_numbers
    df.rename(columns={rename['Alg']:'Alg', rename['Model']:'Model', rename['Params']:'Params',
                       rename['Dataset']:'Dataset', rename['Result']:'Result',
                       rename['Dim']:'Dim', rename['From']:'From', rename['To']:'To'}, inplace=True)

    #df['Params_dict'] = df['Params'].apply(ast.literal_eval).apply(flatten_dict)
    df['Params_dict'] = df['Params'].apply(_safe_parse_params).apply(flatten_dict)


    base = df[(df['Dataset'] == dataset['name']) &
              (df['Dim'] == dataset['dim']) &
              (df['From'] == dataset['From']) &
              (df['To'] == dataset['To'])]

    curves = []
    for exp in experiments:
        # Read only what's guaranteed, branch early for the virtual experiment
        alg = exp.get("Alg")
        x_axis = exp["x_axis"]  # required

        # --- special case: "global best" -> per-dataset winner line + tiny label ---
        if alg == 'global best':
            among = [s for s in exp.get('among_algorithms', ['']) if s is not None]
            excludes = exp.get('excluding_algorithms', [])

            def alg_ok(name: str) -> bool:
                if any(excl in name for excl in excludes):
                    return False
                if (not among) or any(s == '' for s in among):
                    return True
                return any(sub in name for sub in among)

            temp = base[ base['Alg'].apply(alg_ok) ].copy()
            # If x_axis == '', consider ALL rows regardless of params
            if x_axis:
                temp = temp[temp['Params_dict'].apply(lambda d: x_axis in d)]
            if temp.empty:
                continue

            best_idx = temp['Result'].astype(float).idxmin()
            best_y   = float(temp.loc[best_idx, 'Result'])
            best_alg = str(temp.loc[best_idx, 'Alg'])

            # Only compute a data-driven x-range if we actually have an x-axis
            xr = None
            if x_axis:
                temp['__x'] = temp['Params_dict'].apply(lambda d: d[x_axis]).astype(float)
                xr = (float(temp['__x'].min()), float(temp['__x'].max()))

            curves.append({
                'alg': 'global best',
                'x': np.array([0.0, 1.0]),  # placeholder; real span is the axis width at draw time
                'y': np.array([best_y, best_y]),
                'label': f"{get_label(exp)}",
                'linestyle': '--',
                'is_horizontal': True,
                'best_alg_name': best_alg,
                'xrange': xr,
            })
            continue


        # Normal experiments: only now read optional keys safely
        model = exp.get("Model", None)
        params_filter = exp.get("Params", {})
        if model is None:
            # Malformed normal experiment; skip gracefully
            continue

        filtered = base[(base['Alg'] == alg) & (base['Model'] == model)]

        if filtered.empty:
            continue

        all_params = filtered['Params_dict'].apply(lambda d: set(d.keys())).explode().unique().tolist()
        other_params = [p for p in all_params if p not in params_filter and p != x_axis and p is not None]

        param_values = {}
        for param in other_params:
            param_values[param] = filtered['Params_dict'].apply(lambda d: d.get(param)).dropna().unique().tolist()

        combinations = list(itertools.product(*param_values.values())) if param_values else [()]

        for comb in combinations:
            temp = filtered.copy()
            comb_dict = dict(zip(other_params, comb))
            for k, v in {**params_filter, **comb_dict}.items():
                temp = temp[temp['Params_dict'].apply(lambda p, kk=k, vv=v: ((kk not in p) if str(vv) == "N/A" else (str(p.get(kk)) == str(vv))))]
                #temp = temp[temp['Params_dict'].apply(lambda p, kk=k, vv=v: (kk not in p) if vv == "N/A" else (p.get(kk) == vv))]
                #temp = temp[temp['Params_dict'].apply(lambda p, kk=k, vv=v: p.get(kk) == vv)]

            temp = temp[temp['Params_dict'].apply(lambda p: x_axis in p)]
            if temp.empty:
                continue

            temp['x_value'] = temp['Params_dict'].apply(lambda p: p[x_axis])
            temp = temp.sort_values(by='x_value')

            if temp['Result'].empty:
                continue

            curves.append({
                'alg': alg,
                'x': temp['x_value'].to_numpy(),
                'y': temp['Result'].to_numpy(),
                'label': f"{get_label(exp)}",
                'linestyle': '-'                 # enforce solid line
            })
    return curves

def plot_curves(curves, dataset, ax, alg_color_map):
    if not curves:
        ax.axis('off')
        return set()

    ylim_max = dataset.get('ylim_max', None)
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])

    # Plot — solid for normal curves; defer horizontal overlays
    seen_algs = set()
    horizontals, normal_curves = [], []
    # Split incoming curves
    for curve in curves:
        if curve.get('is_horizontal', False):
            horizontals.append(curve)
        else:
            normal_curves.append(curve)

    # Build quick lookups
    by_label = {}
    for c in normal_curves:
        by_label.setdefault(c['label'], []).append(c)

    # Arrow placement manager for this axes
    next_arrow_x = _arrow_slot_generator()

    # We'll need ylim_max to decide arrows
    ylim_max = dataset.get('ylim_max', None)

    # Exclude horizontal-overlay label(s) from "missing => X" handling
    horizontal_labels = set(h['label'] for h in horizontals)

    # Iterate over every expected label for this subplot (from alg_color_map order)
    for label in alg_color_map.keys():
        if label in horizontal_labels:
            continue  # handled later in the horizontals section

        color = alg_color_map.get(label, None)
        curve_list = by_label.get(label, [])

        if not curve_list:
            # No datapoints at all for this label in this subplot -> draw X
            _draw_left_border_X(ax, color)
            # (Optional) count it as seen if you want the legend entry no matter what:
            # seen_algs.add(label)  # only if you change legend code to use labels
            continue

        # There are one or more curves for this label (e.g., from param combinations)
        for c in curve_list:
            x = np.asarray(c['x'])
            y = np.asarray(c['y'])
            ls = c.get('linestyle', '-')

            # If truly empty arrays -> treat as missing -> left X
            if x.size == 0 or y.size == 0:
                _draw_left_border_X(ax, color)
                continue

            finite_mask = np.isfinite(y)
            has_finite = bool(finite_mask.any())

            # Decide arrow condition:
            # (1) all points are NaN  -> ↑
            # (2) OR (all finite points are strictly above ylim_max) -> ↑
            needs_arrow = False
            if not has_finite:
                needs_arrow = True
            elif ylim_max is not None:
                finite_y = y[finite_mask]
                if finite_y.size > 0 and np.all(finite_y > ylim_max):
                    needs_arrow = True

            if needs_arrow:
                # Get a unique x-slot across the width (axes fraction)
                x_frac = next(next_arrow_x)

                # Arrow fully inside axes; spread along x, fixed y near top
                ax.annotate(
                    '',
                    xy=(x_frac, 0.995),            # arrow head just under top edge
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(x_frac, 0.96),         # short tail
                    textcoords=('axes fraction', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-|>', lw=1.6, color=color),
                    clip_on=True
                )

                # Count as seen
                seen_algs.add(c['alg'])
            else:
                # Normal in-range plotting
                ax.plot(x, y, linestyle=ls, linewidth=1.6, color=color, label=label)
                seen_algs.add(c['alg'])


    # If there are no normal curves but there is a horizontal overlay, set x-limits from it
    if not normal_curves and horizontals:
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ranges = [h['xrange'] for h in horizontals if h.get('xrange') is not None]
            if ranges:
                ax.set_xlim(min(r[0] for r in ranges), max(r[1] for r in ranges))
            else:
                # No x info at all (e.g., x_axis == ""); choose a safe positive span for log scale
                ax.set_xlim(1e-4, 1.0)


    # Y limits for normal curves (top only; bottom handled after drawing horizontals)
    if ylim_max is not None and normal_curves:
        y_min_norm = np.nanmin([np.nanmin(c['y']) for c in normal_curves])
        ax.set_ylim(y_min_norm - 1, ylim_max)

    ax.set_xscale("log")
    ax.grid(True, linewidth=0.6, alpha=0.6)

    # Draw horizontal "global best" overlays across full axis span and annotate winner
    for h in horizontals:
        color = alg_color_map.get(h['label'], None)
        x0, x1 = ax.get_xlim()
        ax.hlines(h['y'][0], x0, x1, linestyles='--', linewidth=1.6, colors=color, label=h['label'])
        # annotate tiny winner label just below the line
        xmid = np.sqrt(x0 * x1) if ax.get_xscale() == 'log' else (x0 + x1) / 2.0
        cur_ymin, cur_ymax = ax.get_ylim()
        y_top = dataset.get('ylim_max', ax.get_ylim()[1])
        # pad = 0.015 * (cur_ymax - cur_ymin) if np.isfinite(cur_ymax - cur_ymin) else 0.5
        # text_y = h['y'][0] - 1.3 * pad
        # ax.set_ylim(text_y - 0.6 * pad, y_top)
        pad = max(0.06 * (cur_ymax - cur_ymin), 0.5)
        text_y = h['y'][0] - 0.3 * pad
        ax.set_ylim(text_y - pad, y_top)
        ax.text(xmid, text_y, h.get('best_alg_name', ''), ha='center', va='top', fontsize=6)
        seen_algs.add(h['alg'])

    # Short title only
    ax.set_title(f"{dataset['name']}-{dataset['dim']}", fontsize=12)

    # Keep axes labels on outer edges only to reduce clutter
    return seen_algs

# ---------------- figure assembly ----------------

rows, cols = 4, 5
fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
axes = axes.flatten()

# Build a consistent color map for algorithms (cycled if more algs than default colors)
all_algs = [e['Alg'] for e in experiments]

palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
alg_color_map = {get_label(exp): palette[i % len(palette)] for i, exp in enumerate(experiments)}

glob_seen_algs = set()

for idx in tqdm(range(rows * cols)):
    ax = axes[idx]
    if idx >= len(dataset_info) or not dataset_info[idx]:
        ax.axis('off')
        continue

    dataset = dataset_info[idx]
    # Skip if any mandatory field is missing
    if not all(k in dataset for k in ['name','dim','From','To']):
        ax.axis('off')
        continue

    # Only keep experiments that exist for this dataset (now using flattened params)
    verified = filter_experiments_with_data(csv_file, dataset, experiments)
    if not verified:
        ax.axis('off')
        continue

    curves = build_curves(csv_file, verified, csv_column_numbers, dataset=dataset)
    seen = plot_curves(curves, dataset, ax=ax, alg_color_map=alg_color_map)
    glob_seen_algs |= seen

# Reduce label clutter: only outer axes get axis labels
for i, ax in enumerate(axes):
    r, c = divmod(i, cols)
    if r < rows - 1:  # not bottom row
        ax.set_xlabel('')
    else:
        ax.set_xlabel('alpha / metastep', fontsize=13)
    if c != 0:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('RMSE', fontsize=13)

# Global legend (one entry per Alg actually plotted) — boxed at right
legend_handles = []
legend_labels  = []
for exp in experiments:
    if exp['Alg'] in glob_seen_algs:
        ls = '--' if exp['Alg'] == 'global best' else '-'
        legend_handles.append(Line2D([0], [0], color=alg_color_map[get_label(exp)], lw=2, linestyle=ls))
        legend_labels.append(get_label(exp))

# Reserve space on the right for the legend box
# (you can tweak the 0.80/0.82 numbers to grow/shrink the legend column)
plt.tight_layout(rect=[0.03, 0.05, .80, .98])

if legend_handles:
    # Place the legend to the right, centered vertically, inside a box
    legend = fig.legend(legend_handles, legend_labels,
                        loc='center left',
                        bbox_to_anchor=(0.792, 0.5),  # x just to the right of plotting area
                        ncol=1, frameon=True, fancybox=True, framealpha=0.95,
                        edgecolor='0.5', borderpad=0.8, fontsize=12, title="Algorithms")
    legend.get_title().set_fontsize(12)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/all_subplots.png', dpi=300, bbox_inches='tight')
plt.show()


