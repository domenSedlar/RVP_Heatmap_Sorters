from Visualizations.vis import read, tm_vs_score, size_vs_col
from matplotlib import pyplot as plt

df = read()

df_rand = df[df['dataset']=='Random']
df_gds = df[df['dataset']=='GDS_rand']

tm_vs_score(df_rand, title='time vs score on random dataset', s=211)
tm_vs_score(df_gds, title='time vs score on GDS dataset', s=212)
plt.show()