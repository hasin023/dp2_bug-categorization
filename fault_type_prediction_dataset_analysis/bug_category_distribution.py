import plotly.graph_objects as go

categories = ["semantic", "memory", "other", "concurrency"]
counts = [164, 125, 119, 102]
percentages = [32.16, 24.51, 23.33, 20.00]
colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F"]

fig = go.Figure(
    go.Bar(
        y=categories,
        x=counts,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.2f}%" for p in percentages],
        textposition='auto',
        cliponaxis=False
    )
)
fig.update_layout(
    title_text='Bug Category Distribution in Dataset',
    xaxis_title='Cnt',
    yaxis_title='Cat',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig.update_xaxes(title_text='Cnt')
fig.update_yaxes(title_text='Cat')

fig.write_image('bug_category_distribution.png')
