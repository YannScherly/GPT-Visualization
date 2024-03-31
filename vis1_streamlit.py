import pandas as pd
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots


bigfive_human_data = pd.read_csv('data/bigfive_data.csv', delimiter='\t')
bigfive_human_data['hue'] = 'Human'
bigfive_human_data.head()

### independent 30 instances
records_gpt4 = json.load(open('records/bigfive_gpt4_2023_06_26-01_37_11_PM.json', 'r'))
records_turbo = json.load(open('records/bigfive_turbo_2023_06_26-02_06_26_AM.json', 'r'))
bigfive_model_data = {}
bigfive_model_data['gpt4'] = pd.DataFrame(records_gpt4['choices'])
bigfive_model_data['turbo'] = pd.DataFrame(records_turbo['choices'])
bigfive_model_data['gpt4']['hue'] = 'ChatGPT-4'
bigfive_model_data['turbo']['hue'] = 'ChatGPT-3'
questions = {}
with open('data/bigfive.tsv', 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        questions[row[0]] = row[1]

keyed = {}
indices = defaultdict(int)
dimensions = 'EACNO'
with open('data/bigfive_IPIP.tsv', 'r') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        d = dimensions[int(row[-1][1])-1]
        v = row[-1][2]
        indices[d] += 1
        k = '%s%i' % (d, indices[d])
        keyed[k] = v
models = ['gpt4', 'turbo']
beg_pos = list(range(7, 100, 10))[:5]
# d_scores_model = defaultdict(dict)

dimensions = 'ENACO'
for i, d in enumerate(dimensions):
    
    ### human scores
    d_score = 0
    for j in range(10):
        k = '%s%i' % (d, j+1)
        v = keyed[k]
        score = bigfive_human_data.iloc[:, beg_pos[i]+j]
        if v == '-': score = 6 - score
        d_score += score
    bigfive_human_data[d] = d_score

    ### model scores
    for model in models:
        d_score = 0
        records = eval('records_%s' % model)
        for j in range(10):
            k = '%s%i' % (d, j+1)
            v = keyed[k]
            score = bigfive_model_data[model].iloc[:, i*10+j]
            # score = np.mean(records['choices'][k])
            if v == '-': score = 6 - score
            d_score += score
        # d_scores_model[model][d] = d_score
        bigfive_model_data[model][d] = d_score
data = pd.concat([
    bigfive_human_data, 
    bigfive_model_data['gpt4'], 
    bigfive_model_data['turbo']
], ignore_index=True)
data[['E', 'hue']]
data['N'] = 60 - data['N']
data[[*dimensions, 'hue']]



categories = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
N = len(categories)
categories += categories[:1] # add the first category again to close the last line in polar figure

hues = ['Human', 'ChatGPT-4', 'ChatGPT-3']
hue_colors = {
    'Human': 'rgba(31, 119, 180, 1)',
    'ChatGPT-4': 'rgba(255, 127, 14, 1)',
    'ChatGPT-3': 'rgba(214, 39, 40, 1)',
}
value_type = 'median'

layout = go.Layout(
    width=800,
    height=800, 
    polar = dict(
        radialaxis = dict(
            visible = True,
            range = [0, 50],
        )
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend = True,
)

fig_radar = go.Figure(layout=layout)

fig = make_subplots(
    rows=2, cols=1,
    column_widths=[1],
    row_heights=[1, 1],
    specs=[
        [{"type": "scatterpolar"}],
        [{"type": "bar"}] 
    ],
    figure=fig_radar
)


data_trace_polar = []
data_trace_bar = []
for hue in hues:
    d_scores = data[data['hue'] == hue][[*dimensions]].agg(['median', 'mean', 'std'])
    values = d_scores.loc[value_type].values.tolist()
    std_dev = d_scores.loc['std'].values.tolist()
    values += values[:1]
    # print(f'values after: {values}')
    std_dev += std_dev[:1]
    
    upper_bound = np.array(values) + np.array(std_dev)
    lower_bound = np.array(values) - np.array(std_dev)
    
    upper_bound = np.array(values) + np.array(std_dev)
    lower_bound = np.array(values) - np.array(std_dev)

    
    trace = go.Scatterpolar(
        r = values,
        theta = categories,
        name = hue,
        hoverinfo = 'all',
        line=dict(color=hue_colors[hue]),
        legendgroup=hue
    )

    trace_upper = go.Scatterpolar(
        r=upper_bound.tolist(),
        theta=categories,
        marker=dict(color=hue_colors[hue].replace('1)', '0.2)')),
        line=dict(width=0.1),
        hoverinfo='none',
        legendgroup=hue,
        showlegend=False,
        name = hue + ' error band'
    )

    trace_lower = go.Scatterpolar(
        r=lower_bound.tolist(),
        theta=categories,
        marker= dict(color=hue_colors[hue].replace('1)', '0.2)')),
        fill='tonext',
        line=dict(width=0.1),
        fillcolor=hue_colors[hue].replace('1)', '0.2)'),
        legendgroup=hue,
        showlegend=False,
        name = hue + ' error band'
    )

    bar_trace = go.Bar(
        x=categories[:(len(categories)-1)],#adjusted for extra category
        y=values,
        name=hue,
        marker_color=hue_colors[hue],
        legendgroup=hue
    )
    
    # data_trace_bar.append(bar_trace)
    # data_trace_polar.extend([trace_upper, trace_lower, trace])
    fig.add_trace(trace, row=1, col=1)
    fig.add_trace(trace_upper, row=1, col=1)
    fig.add_trace(trace_lower, row=1, col=1)
    fig.add_trace(bar_trace,row=2, col=1)

fig.show()