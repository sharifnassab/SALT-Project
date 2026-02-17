import time
from datetime import datetime
import os
import numpy as np
from utils.data_loader import get_dir
import multiprocessing
import json
from algorithms.IDBD import param_sweeps as IDBD_params
from algorithms.IDBD2 import param_sweeps as IDBD2_params
from algorithms.IDBD_MGEN import param_sweeps as IDBD_MGEN_params
from algorithms.IDBD_comb import param_sweeps as IDBD_comb_params
from algorithms.IDBD_MGENC import param_sweeps as IDBD_MGENC_params
from algorithms.IDBD_MGENC_delta_delta import param_sweeps as IDBD_MGENC_delta_delta_params
from algorithms.IDBD_MGENChd import param_sweeps as IDBD_MGENChd_params
from algorithms.IDBD_MGENChdOb import param_sweeps as IDBD_MGENChdOb_params
from algorithms.IDBD_MGENChdObCorrected import param_sweeps as IDBD_MGENChdObCorrected_params
from algorithms.IDBD_MGENedge import param_sweeps as IDBD_MGENedge_params
from algorithms.IDBD_MGENChd_CGD import param_sweeps as IDBD_MGENChd_CGD_params
from algorithms.NIDBD3 import param_sweeps as NIDBD3_params
from algorithms.NIDBD3_Oct2025_champion import param_sweeps as NIDBD3_Oct2025_champion_params
from algorithms.NIDBD3hd import param_sweeps as NIDBD3hd_params
from algorithms.NIDBD3_ClipTest import param_sweeps as NIDBD3_ClipTest_params
from algorithms.NIDBD3_noSoftmax import param_sweeps as NIDBD3_noSoftmax_params
from algorithms.NIDBD3_withMetaPostNorm import param_sweeps as NIDBD3_withMetaPostNorm_params
from algorithms.NIDBD3_MetaPost_vs_Pre import param_sweeps as NIDBD3_MetaPost_vs_Pre_params
from algorithms.NIDBD3_MetaPost_vs_Pre_K1 import param_sweeps as  NIDBD3_MetaPost_vs_Pre_K1_params
from algorithms.HIDBD import param_sweeps as  HIDBD_params # hierarchical IDBD
from algorithms.SIDBD import param_sweeps as  SIDBD_params # scalar IDBD
from algorithms.SIDBD_Obn import param_sweeps as  SIDBD_Obn_params  # scalar IDBD on Obn
from algorithms.SIDBD_ObnSig import param_sweeps as  SIDBD_ObnSig_params  # scalar IDBD on Obn
from algorithms.SIDBD_ObnDe import param_sweeps as  SIDBD_ObnDe_params  # scalar IDBD on ObnDe
from algorithms.K1 import param_sweeps as K1_params
from algorithms.TwoLLMs_MGEN import param_sweeps as TwoLLMs_MGEN_params
from algorithms.TwoLLMsL1C_MGEN import param_sweeps as TwoLLMsL1C_MGEN_params



num_parallel_processes = 2
# num_parallel_processes = multiprocessing.cpu_count()-1

ASH_dir = get_dir('ASH')
RSS_dir = get_dir('RSS')

def run_experiment(run_index, total_runs, command):
    if run_index> num_parallel_processes:
        time_per_run = (time.time()-start_time)/(run_index)
        time_remaining = time_per_run*(total_runs-run_index+1)
        print(f"{run_index}/{total_runs},  \t\t\t time per run: {time_per_run:.1f} sec, remaining time {time.strftime('%H:%M:%S', time.gmtime(time_remaining))}")
    else:
        print(f"{run_index}/{total_runs}")
    
    os.system(command)

def params_to_commands(alg_config):
    return (
        f"python3 {alg_config['file_path']} "
        + ' '.join(
            f"--{param}="
            + (
                '"' + f"{alg_config[param]}" + '"'
                if isinstance(alg_config[param], dict)
                else f"{alg_config[param]}"
            )
            for param in alg_config
            if param != 'file_path'
        )
    )

def replace_sweep_of_a_variable(all_configs, new_param_name, list_new_param):
    unique_keys = {
        json.dumps({k: v for k, v in cfg.items() if k != new_param_name}, sort_keys=True)
        for cfg in all_configs
    }

    return [
        {**json.loads(cfg_json), new_param_name: val}
        for cfg_json in unique_keys
        for val in list_new_param
    ]

start_time = time.time()
if __name__=='__main__':
    UID = 'results_'
    list_meta_stepsize = [x.item() for x in list(10**(np.linspace(start=-6, stop=.5, num=int(6.5/.5) +1)))]
    #list_meta_stepsize = [x.item() for x in list(10**(np.linspace(start=-4, stop=0, num=int(4/.5) +1)))]
    #list_meta_stepsize = [1e-3]
    
    
    dataset_list = [
                    'SALTdata-RSS1-20-1100000.txt',
                    'SALTdata-RSS2-20-1100000.txt',
                    'SALTdata-RSS3-20-1100000.txt',
                    'SALTdata-RSS4-20-1100000.txt',
                    'SALTdata-RSS5-20-1100000.txt',
                    'SALTdata-RSS6-20-1100000.txt',
                    'SALTdata-RSS7-20-1100000.txt',
                    'SALTdata-ASH1-10-1100000.txt',
                    'SALTdata-ASH2-1-1100000.txt',
                    'SALTdata-ASH3-3-1100000.txt',
                    'SALTdata-ASH4-3-1100000.txt',
                    'SALTdata-ASH5-16-1100000.txt',
                    'SALTdata-ASH6-20-1100000.txt',
                    'SALTdata-ASH7-30-1100000.txt',
                    'SALTdata-ASH8-10-1100000.txt',
                    'SALTdata-ASH9-1-1100000.txt',
                    'SALTdata-ASH10-5-1100000.txt',
                    'SALTdata-ASH11-4-1100000.txt',
                    'SALTdata-ASH12-5-1100000.txt',
                    'SALTdata-ASH13-2-1100000.txt',
                    ]

    command_list = []
    for alg_params in [
                        #IDBD_params(),
                        #IDBD2_params(),
                        #NIDBD3_params(),
                        #NIDBD3_Oct2025_champion_params(),
                        #NIDBD3hd_params(),
                        #NIDBD3_ClipTest_params(),
                        #NIDBD3_noSoftmax_params(),
                        #NIDBD3_withMetaPostNorm_params(),
                        #NIDBD3_MetaPost_vs_Pre_params(),
                        #NIDBD3_MetaPost_vs_Pre_K1_params(),
                        HIDBD_params(),
                        #SIDBD_params(),
                        #SIDBD_Obn_params(),
                        #SIDBD_ObnSig_params(),
                        #SIDBD_ObnDe_params(),
                        #K1_params(),
                        #IDBD_MGEN_params(),
                        #IDBD_comb_params(),
                        #IDBD_MGENC_params(),
                        #IDBD_MGENC_delta_delta_params(),
                        #IDBD_MGENChd_params(),
                        #IDBD_MGENedge_params(),
                        #IDBD_MGENChdOb_params(),
                        #IDBD_MGENChdObCorrected_params(),
                        #IDBD_MGENChd_CGD_params(),
                        #TwoLLMs_MGEN_params(),
                        #TwoLLMsL1C_MGEN_params(),
                        ]:
        alg_params = replace_sweep_of_a_variable(alg_params, new_param_name='meta_stepsize', list_new_param=list_meta_stepsize)
        for alg_config in alg_params:
            alg_command = params_to_commands(alg_config)
            for dataset in dataset_list:
                command_list.append(alg_command + ' ' +
                                    ' '.join([f'--save_to=csvs/{UID}.csv', 
                                            f'--dataset={dataset}'])
                                    )
    if False:
        print(len(command_list))
        assert 0
    
    if False: # remove existing runs
        from utils.remove_existing_runs import remove_existing_runs
        keys_for_config = ['dataset', 'alg_name', ['params',['alpha']], 'bias', 'input_normalization', 'input_norm_gamma', 'save_to']
        args_list = remove_existing_runs([x+['--alg_name=SGD'] for x in args_list], keys_for_config, csv_file_path)

    indexed_commands = [(i, len(command_list), cmd) for i, cmd in enumerate(command_list, start=1)]
    with multiprocessing.Pool(processes=num_parallel_processes) as pool:
        pool.starmap(run_experiment, indexed_commands, chunksize=1)

