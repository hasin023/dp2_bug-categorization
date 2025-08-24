import pandas as pd
import plotly.graph_objects as go

# Use provided JSON data
bug_data = {
    'concurrency': [20.20, 18.25, 1.4, 87.7],
    'memory': [41.47, 161.31, 2.8, 1778.2],
    'other': [42.28, 187.38, 3.5, 2056.6],
    'semantic': [39.34, 35.87, 3.4, 243.4],
}

# Prepare dataframe
rows = []
for cat, times in bug_data.items():
    cat_abr = cat[:15]
    for t in times:
        rows.append({'bug_cat': cat_abr, 'time_spent': t})
df = pd.DataFrame(rows)

colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']
cat_order = list(bug_data.keys())
color_map = {cat: colors[i%len(colors)] for i,cat in enumerate(cat_order)}

fig = go.Figure()
for i, cat in enumerate(cat_order):
    data = df[df['bug_cat'] == cat]
    # box trace
    fig.add_trace(go.Box(
        x=[cat]*len(data),
        y=data['time_spent'],
        name=cat,
        marker_color=colors[i],
        boxpoints=False,
        showlegend=False
    ))
    # scatter trace
    fig.add_trace(go.Scatter(
        x=[cat]*len(data),
        y=data['time_spent'],
        mode='markers',
        name=cat,
        marker=dict(color=colors[i], size=8),
        showlegend=False,
        cliponaxis=False,
        hovertemplate=f'Bug Cat: {cat[:15]}<br>Time: %{{y:.2f}}'
    ))

fig.update_yaxes(title_text='Time (s)', title_font=dict(size=14), tickfont=dict(size=13))
fig.update_xaxes(title_text='Bug Cat', title_font=dict(size=14), tickfont=dict(size=13))
fig.update_layout(title='Annotation Time Complexity by Bug Category')
fig.write_image('annotation_time_bug_cat.png')