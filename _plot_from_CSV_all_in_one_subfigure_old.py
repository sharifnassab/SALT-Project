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
from tqdm import tqdm

csv_file = 'csvs/results_.csv'
x_lim = None
# x_lim = [1e-4, 1]

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
    {'name':'ASH8', "dim":30, 'From':'900k', "To":"1.1m", "ylim_max": 100.0},
    {'name':'ASH9', "dim":1, 'From':'900k', "To":"1.1m", "ylim_max": 24.0},
    {'name':'ASH10', "dim":5, 'From':'900k', "To":"1.1m", "ylim_max": 20.0},
    {'name':'ASH11', "dim":4, 'From':'900k', "To":"1.1m", "ylim_max": 20.0},
    {'name':'ASH12', "dim":5, 'From':'900k', "To":"1.1m", "ylim_max": 60.0},
    {'name':'ASH13', "dim":2, 'From':'900k', "To":"1.1m", "ylim_max": 80.0},
]

experiments = [
    #{"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    ##{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.01}, "x_axis":"alpha", "label_override":r"\theta^{meta}=0.01"},
    #{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha", "label_override":r"\theta^{meta}=0.001"},
    #{"Alg":"Adam", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"AdamCenter", "Model":"affine", "Params":{"beta_centering": 0.999}, "x_axis":"alpha", "label_override":r"AdamC (centering=0.999)"},
    #{"Alg":"AdamCenter", "Model":"affine", "Params":{"beta_centering": 0.9999}, "x_axis":"alpha", "label_override":r"AdamC (centering=0.9999)"},
    #{"Alg":"AdamCenterSplit", "Model":"affine", "Params":{"beta_centering": 0.999}, "x_axis":"alpha", "label_override":r"AdamCA (centering=0.999)"},
    #{"Alg":"AdamCenterSplit", "Model":"affine", "Params":{"beta_centering": 0.9999}, "x_axis":"alpha", "label_override":r"AdamCA (centering=0.9999)"},
    #{"Alg":"Newton", "Model":"affine", "Params":{}, "x_axis":"alpha"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': .001, 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGEN"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta'}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}$)"},
    #{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override": "\n"+r"IDBD_MGEN"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENC", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENC"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENC_alpha_only", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    {"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.0", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(0)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    {"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.9", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.9)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    {"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.99", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.99)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    {"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.999", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.999)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    {"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha*x2", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.0", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(0)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.9", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.9)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.99", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.99)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"0.999", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(.999)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha*x2", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    {"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN C ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN':'theta_meta/10', 'base.cente_only':'center_and_augment'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"MDN CA ($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta':'0.001', 'base.cente_only':'N/A'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN ($\eta=10^{-3}$)"},
    #{"Alg":"LMS_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta':'0.001', 'base.cente_only':'center'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN C ($\eta=10^{-3}$)"},
    #{"Alg":"LMS_EAN^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta':'0.001', 'base.cente_only':'center_and_augment'}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN CA ($\eta=10^{-3}$)"},
    #{"Alg":"LMS_EAN_eta_eq_alpha^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta/alpha':1}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN ($\eta=\alpha$)"},
    #{"Alg":"LMS_EAN_eta_eq_alpha^IDBD_MGENChd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.eta/alpha':.1}, "x_axis":"metastep", "label_override":"\n"+r"IDBD_MGENChd(1-\alpha)"+"\n  +"+r"EAN ($\eta=\alpha/10$)"},
    #{"Alg":"LMS_MDN^IDBD_MGENC_delta_delta", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n\n"+r"IDBD_MGENChd(0)"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"+"\n [delta delta]"},
    #{"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': .001}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=1e-3$)"},
    #{"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}$)"},
    #{"Alg":"LMS_MDN^NIDBD3", "Model":"affine", "Params":{'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
    #{"Alg":"LMS_MDN^NIDBD3hd", "Model":"affine", "Params":{"h_decay":"1-alpha", 'base.theta_MDN': 'theta_meta/10'}, "x_axis":"metastep", "label_override":"\n"+r"NIDBD3hd(1-\alpha)"+"\n  +"+r"MDN($\theta^{norm}=\theta^{meta}/10$)"},
]



'''
#{"Alg":"LMS_MDNPN_KC^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
##{"Alg":"LMS", "Model":"affine", "Params":{}, "x_axis":"alpha"},
#{"Alg":"LMS-MDN", "Model":"affine", "Params":{"theta":0.001}, "x_axis":"alpha"},
##{"Alg":"IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
#{"Alg":"LMS_MDN^IDBD_MGEN", "Model":"affine", "Params":{}, "x_axis":"metastep"},
#{"Alg":"LMS_MDN^Normalized_IDBD", "Model":"affine", "Params":{}, "x_axis":"metastep"},
'''

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
            #if all(((k not in params) if str(v) == "N/A" else (k in params and str(params[k]) == str(v))) for k, v in exp.get('Params', {}).items()):
            #if all(k in params and str(params[k]) == str(v) for k, v in exp.get('Params', {}).items()):
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
    
    df = pd.read_csv(csv_file, header=None)
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
        alg, model, params_filter, x_axis = exp["Alg"], exp["Model"], exp["Params"], exp["x_axis"]

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

    # Plot — solid lines, consistent colors, NO markers, NO per-axes legend
    seen_algs = set()
    for curve in curves:
        color = alg_color_map.get(curve['label'], None)
        line, = ax.plot(curve['x'], curve['y'],
                        linestyle='-', linewidth=1.6,
                        color=color, label=curve['label'])
        seen_algs.add(curve['alg'])

    # Y limits if requested
    if ylim_max is not None:
        # try to keep a tiny margin
        y_min = np.nanmin([np.nanmin(c['y']) for c in curves])
        ax.set_ylim(y_min - 1, ylim_max)

    ax.set_xscale("log")
    ax.grid(True, linewidth=0.6, alpha=0.6)

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
legend_handles = [Line2D([0], [0], color=alg_color_map[get_label(exp)], lw=2, linestyle='-')
                  for exp in experiments if exp['Alg'] in glob_seen_algs]
legend_labels  = [get_label(exp) for exp in experiments if exp['Alg'] in glob_seen_algs]

# Reserve space on the right for the legend box
# (you can tweak the 0.80/0.82 numbers to grow/shrink the legend column)
plt.tight_layout(rect=[0.03, 0.05, .80, .98])

if legend_handles:
    # Place the legend to the right, centered vertically, inside a box
    legend = fig.legend(legend_handles, legend_labels,
                        loc='center left',
                        bbox_to_anchor=(0.82, 0.5),  # x just to the right of plotting area
                        ncol=1, frameon=True, fancybox=True, framealpha=0.95,
                        edgecolor='0.5', borderpad=0.8, fontsize=12, title="Algorithms")
    legend.get_title().set_fontsize(12)

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/all_subplots.png', dpi=300, bbox_inches='tight')
plt.show()
