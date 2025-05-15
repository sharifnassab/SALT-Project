import pandas as pd
import matplotlib.pyplot as plt
import ast
import itertools
import numpy as np
from adjustText import adjust_text

csv_file = 'csvs/results_.csv'
x_scale = "log"

dataset_info = [
    {'name':'RSS1', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 24.0},
    {'name':'RSS5', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 27.0},
    {'name':'RSS6', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 28.0},
    {'name':'RSS7', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 23.0},
    {'name':'ASH5', "dim":16, 'From':'900k', "To":"1.1m", "ylim_max": 43.0},
    {'name':'ASH6', "dim":20, 'From':'900k', "To":"1.1m", "ylim_max": 40.0},
    {'name':'ASH7', "dim":30, 'From':'900k', "To":"1.1m", "ylim_max": 50.0},
    {'name':'ASH9', "dim":1, 'From':'900k', "To":"1.1m", "ylim_max": 24.0},
    #{'name':'ASH10', "dim":5, 'From':'900k', "To":"1.1m"},
    #{'name':'ASH11', "dim":4, 'From':'900k', "To":"1.1m", "ylim_max": 100.0},
    {'name':'ASH12', "dim":5, 'From':'900k', "To":"1.1m", "ylim_max": 60.0},
]



algorithms = [
    {"Alg":"Adam", "Model":"affine", "Params":{"beta_nu":0.999, "beta_m":0.9}, "x_axis":"alpha"},
    {"Alg":"Newton", "Model":"affine", "Params":{"gamma":0.9999}, "x_axis":"alpha"},
    #{"Alg":"NAD-SAN", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"NAD-EAN", "Model":"affine", "Params":{"tau":100}, "x_axis":"alpha"},
    {"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"LMS-SAN", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    {"Alg":"LMS-EAN", "Model":"affine", "Params":{"tau":100}, "x_axis":"alpha"},
    #{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha"},
    #{"Alg":"LMS-MDNa", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha"},
    #{"Alg":"LMS-MDNPN", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha"},
    {"Alg":"LMS-MDNPN_KC", "Model":"affine", "Params":{"theta":0.01, "eta0":0.001}, "x_axis":"alpha"},
    
    {"Alg":"IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"IDBD2", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    
    #{"Alg":"LMS^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_EAN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNa^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNPN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    {"Alg":"LMS_MDNPN_KC^IDBD_MGEN", "Model":"affine", "Params":{"meta_eta":0.001, "alpha0":1e-05, "base.eta0":0.001, "base.theta_MDN":0.01}, "x_axis":"metastep"},
    
    #{"Alg":"LMS^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_EAN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNa^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    #{"Alg":"LMS_MDNPN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
    {"Alg":"LMS_MDNPN_KC^Normalized_IDBD", "Model":"affine", "Params":{"alpha0":1e-05, "base.eta0":0.001, "base.theta_MDN":0.01}, "x_axis":"metastep"},
]


csv_column_numbers = {'Alg':0, 'Model':1, 'Params':2, 'Dataset':4, "Dim":5, "From":8, "To":9, 'Result':16}

# --- Reuse the original helper functions ---
# (Assuming distributed_legend, manage_outlier_segments and flatten_dict exist as provided.)

def distributed_legend(curves, lines, y_max, legend_fontsize=13):
    x_all = np.concatenate([c['x'] for c in curves])
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    x_positions = np.linspace(x_min, x_max, len(curves) + 2)[1:-1]

    texts, out_of_bounds = [], []
    for line, xpos in zip(lines, x_positions):
        xdata, ydata = line.get_data()
        label = line.get_label().split(',')[0]
        valid = np.isfinite(ydata) & (ydata <= y_max if y_max else True)

        if np.any(valid):
            idx = np.abs(xdata[valid] - xpos).argmin()
            texts.append(plt.text(xdata[valid][idx], ydata[valid][idx], label,
                                  color=line.get_color(), fontsize=legend_fontsize, fontweight='bold'))
        else:
            out_of_bounds.append((label, line.get_color()))

    adjust_text(texts, avoid_self=True, avoid_points=True, expand_text=(1.2, 1.4))

    # Handle out-of-bounds labels
    if out_of_bounds:
        y_bottom, y_top = plt.ylim()
        x_left, _ = plt.xlim()
        vertical_spacing = (y_top - y_bottom) * 0.04

        for i, (label, color) in enumerate(out_of_bounds):
            plt.text(x_left, y_bottom + vertical_spacing * i, f"{label}: out of bounds",
                     fontsize=legend_fontsize, fontweight='bold', color=color, va='bottom')



def manage_outlier_segments(x, y, threshold, color, epsilon=.5):
    for i in range(len(x) - 1):
        if y[i+1] > threshold:
            dy,dx = (y[i+1] - y[i]) , np.abs(x[i+1] - x[i])
            y_segment = [y[i], y[i] + epsilon*dy/(np.abs(dy)+dx+1e-10)]
            x_segment = [x[i], x[i] + epsilon*dx/(np.abs(dy)+dx+1e-10)]
            plt.plot(x_segment, y_segment, linestyle=':', color=color)
    for i in range(1,len(x)):
        if y[i-1] > threshold:
            dy,dx = np.abs(y[i-1] - y[i]) , np.abs(x[i] - x[i-1])
            y_segment = [y[i] + epsilon*dy/(np.abs(dy)+dx+1e-10), y[i]]
            x_segment = [x[i] - epsilon*dx/(np.abs(dy)+dx+1e-10), x[i]]
            plt.plot(x_segment, y_segment, linestyle=':', color=color)





def plot_curves(csv_file, algorithm, csv_column_numbers, datasets, x_scale='linear'):
    def flatten_dict(d, parent_key='', sep='.'):
        items = {}
        if not isinstance(d, dict):
            return {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    df = pd.read_csv(csv_file, header=None)

    # Rename columns based on provided indices
    col_indices = csv_column_numbers
    df.rename(columns={
        col_indices['Alg']: 'Alg',
        col_indices['Model']: 'Model',
        col_indices['Params']: 'Params',
        col_indices['Dataset']: 'Dataset',
        col_indices['Result']: 'Result',
        col_indices['Dim']: 'Dim',
        col_indices['From']: 'From',
        col_indices['To']: 'To'
    }, inplace=True)

    df['Params_dict'] = df['Params'].apply(ast.literal_eval).apply(flatten_dict)
    min_result = float('inf')

    curves = []

    for dataset in datasets:
        base_filtered_df = df[(df['Dataset'] == dataset['name']) &
                              (df['Dim'] == dataset['dim']) &
                              (df['From'] == dataset['From']) &
                              (df['To'] == dataset['To']) &
                              (df['Alg'] == algorithm["Alg"]) &
                              (df['Model'] == algorithm["Model"])]

        all_params = base_filtered_df['Params_dict'].apply(lambda d: set(d.keys())).explode().unique().tolist()
        params_filter, x_axis = algorithm["Params"], algorithm["x_axis"]
        other_params = [p for p in all_params if p not in params_filter and p != x_axis]

        param_values_combinations = {}
        for param in other_params:
            param_values_combinations[param] = base_filtered_df['Params_dict'].apply(lambda d: d.get(param)).dropna().unique().tolist()

        combinations = list(itertools.product(*param_values_combinations.values())) if param_values_combinations else [()]

        for comb in combinations:
            temp_df = base_filtered_df.copy()
            comb_dict = dict(zip(other_params, comb))
            for key, value in {**params_filter, **comb_dict}.items():
                temp_df = temp_df[temp_df['Params_dict'].apply(lambda p: p.get(key) == value)]

            temp_df = temp_df[temp_df['Params_dict'].apply(lambda p: x_axis in p)]
            temp_df['x_value'] = temp_df['Params_dict'].apply(lambda p: p[x_axis])

            temp_df = temp_df.sort_values(by='x_value')

            if not temp_df['Result'].empty:
                min_result = min(min_result, temp_df['Result'].min())

            #params_str = ', '.join(f'{k}={v}' for k, v in {**params_filter, **comb_dict}.items())
            label = f'{dataset["name"]}-{dataset["dim"]}'
            curves.append({'x': temp_df['x_value'],
                           'y': temp_df['Result'],
                           'label': label,
                           'linestyle': ('--' if algorithm['x_axis'] == 'metastep' else '-' if algorithm['x_axis'] == 'alpha' else '-')})

    for curve in curves:
        old_x = np.nan
        print('\n\n', curve['label'])
        for x,y in zip(list(curve['x']),list(curve['y'])):
            if not x==old_x:
                print(f'alpha=10^({np.log10(x)})\t  RMSE={y}')
            old_x = x
            

    ## ------ plotting ---------
    remove_outlier_segments_larger_than = 1.5*dataset['ylim_max'] if 'ylim_max' in dataset and dataset['ylim_max'] is not None else None
    fontsize = {
                "in_figure_legend":15,
                "detailed_legend":9,
                "axes":15,
                "axes_numbers":14,
                "title":16
                }
    ylim_min_epsilon = 1

    plt.figure(figsize=(14, 10))
    lines = []
    for curve in curves:
        if remove_outlier_segments_larger_than is not None:
            y_with_naned_outliers = np.array(curve['y']) 
            y_with_naned_outliers[np.array(curve['y'])>remove_outlier_segments_larger_than] =np.nan
            line, = plt.plot(np.array(curve['x']), y_with_naned_outliers, marker='o', label=curve['label'], linestyle=curve['linestyle'])
            y_with_nan_to_num = np.nan_to_num(np.array(curve['y']) , nan=1e16)
            manage_outlier_segments(list(curve['x']), y_with_nan_to_num , epsilon=2, threshold=remove_outlier_segments_larger_than, color=line.get_color())
        else:
            line, = plt.plot(curve['x'], curve['y'], marker='o', label=curve['label'], linestyle=curve['linestyle'])
        lines.append(line)
    if dataset['ylim_max'] is not None:
        plt.ylim(min_result - ylim_min_epsilon, dataset['ylim_max'])
    plt.tight_layout(rect=[0.03, 0.3, .97, .97])
    plt.xscale(x_scale)
    plt.grid(True)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=fontsize['detailed_legend'])
    distributed_legend(curves=curves, lines=lines, y_max=dataset['ylim_max']-1, legend_fontsize=fontsize['in_figure_legend'])

    plt.xlabel(algorithm['x_axis'], fontsize=fontsize['axes'])
    plt.ylabel('RMSE', fontsize=fontsize['axes'])
    title = ( f"{algorithm['Alg']}   {algorithm['Params']}"
             f" ({dataset['From']}--{dataset['To']})")
    plt.xticks(fontsize = fontsize['axes_numbers'])
    plt.yticks(fontsize = fontsize['axes_numbers'])
    plt.title(title, fontsize=fontsize['title'])
    
    plt.savefig(f'plots/{algorithm['Alg']}.png', dpi=300, bbox_inches='tight')
    plt.show()


# Call the new plotting function
for algorithm in algorithms:
    plot_curves(csv_file, algorithm, csv_column_numbers, datasets=dataset_info, x_scale=x_scale)
