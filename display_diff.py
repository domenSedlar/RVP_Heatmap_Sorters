from matplotlib import pyplot as plt
import pandas as pd

from OptimalSorters.tsp_gurobi import TSP_gurobi
from Heuristics.bae import BAE

from eval import get_matrix

def read(path='./results.parquet')->pd.DataFrame:
    df = pd.read_parquet(path)
    df['n'] = df[['row_size', 'col_size']].values.min(1)
    df = df[df['algo'] != 'BAE->Hclust']
    #df['algo'] = df['algo'].astype("category")

    return df

df = read()
df['id'] = df['dir'] +'/'+ df['file_name']
# 1. Get the total number of unique algorithms in your dataset
num_algos = df['algo'].nunique()

# 2. Filter groups that contain all unique algorithms
df = df.groupby('id').filter(
    lambda group: group['algo'].nunique() == num_algos
)
df_BAE = df[df['algo'] == 'BAE'][['algo', 'id', 'NS4', 'file_name']]
df_BAE['NS4_BAE'] = df_BAE['NS4']
df_BAE = df_BAE.drop(columns=['NS4', 'file_name'])
df_gur = df[df['algo'] == 'TSP_gurobi'][['algo', 'id', 'NS4', 'file_name']]

data = pd.merge(df_BAE, df_gur, left_on='id', right_on='id', how='right')

data['diff'] = data['NS4_BAE'] - data['NS4']

data = data[data['diff'] <= 500]
m = data['diff'].argmax()

print(data.iloc[m])
p = data.iloc[m]['id'].replace('\\', '/')

H = get_matrix(p)
bae = BAE()

bae_h = bae(H)

plt.subplot(121)
plt.imshow(bae_h, cmap="viridis", interpolation="nearest")
plt.colorbar()  # Adds the color rating scale on the side
plt.title("BAE " + data.iloc[m]['file_name'])


gui = TSP_gurobi()
tsp_h = gui(H)
plt.subplot(122)
plt.imshow(tsp_h, cmap="viridis", interpolation="nearest")
plt.colorbar()  # Adds the color rating scale on the side
plt.title("TSP_gurobi " + data.iloc[m]['file_name'])
plt.show()
