import plotly.graph_objects as go
import pandas as pd

data = {
    'repos': [
        "netty/netty", "elastic/elasticsearch", "spring-projects/spring-boot",
        "redisson/redisson", "bazelbuild/bazel", "spring-projects/spring-session",
        "checkstyle/checkstyle", "hazelcast/hazelcast", "spring-projects/spring-security",
        "AsyncHttpClient/async-http-client"
    ],
    'semantic': [22, 16, 13, 5, 4, 2, 7, 3, 4, 4],
    'memory': [33, 10, 4, 9, 5, 2, 1, 6, 2, 3],
    'concurrency': [17, 11, 7, 4, 6, 1, 3, 4, 2, 2],
    'other': [9, 10, 8, 4, 6, 12, 6, 3, 8, 6]
}
df = pd.DataFrame(data)

colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

df_sorted = df.copy()
df_sorted['total'] = df_sorted[['semantic', 'memory', 'concurrency', 'other']].sum(axis=1)
df_sorted = df_sorted.sort_values('total', ascending=False).head(10)

fig = go.Figure()
fig.add_bar(
    name='Semantic',
    x=df_sorted['repos'], y=df_sorted['semantic'],
    marker_color=colors[0],
    cliponaxis=False
)
fig.add_bar(
    name='Memory',
    x=df_sorted['repos'], y=df_sorted['memory'],
    marker_color=colors[1],
    cliponaxis=False
)
fig.add_bar(
    name='Concurr.',
    x=df_sorted['repos'], y=df_sorted['concurrency'],
    marker_color=colors[2],
    cliponaxis=False
)
fig.add_bar(
    name='Other',
    x=df_sorted['repos'], y=df_sorted['other'],
    marker_color=colors[3],
    cliponaxis=False
)
fig.update_layout(
    barmode='stack',
    title_text='Bug Category Distribution by Repository',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig.update_xaxes(title_text='Repos', tickfont=dict(size=12))
fig.update_yaxes(title_text='Issues', tickfont=dict(size=12))
fig.write_image('bug_category_by_repo.png')
