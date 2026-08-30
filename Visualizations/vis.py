from matplotlib import pyplot as plt
import pandas as pd
from adjustText import adjust_text
import seaborn as sns

def read(path='./results.parquet')->pd.DataFrame:
    df = pd.read_parquet(path)
    df['n'] = df[['row_size', 'col_size']].values.min(1)
    df = df[df['algo'] != 'BAE->Hclust']
    #df['algo'] = df['algo'].astype("category")

    return df

def tm_vs_score(df, title='', only_small=True):
    df['algo'] = df['algo'].astype("category")
    if only_small:
        size_lim = df[df['algo'] == 'TSP_gurobi']['size'].max()
        df = df[df['size']<=size_lim]
    points_x = []
    points_y = []
    labels = []
    fig, ax = plt.subplots(figsize=(9,9))
    for algo in df['algo'].cat.categories:
        if algo == 'BAE->Hclust':
            continue
        labels.append(algo)
        a = df[df['algo'] == algo]
        x = a['time'].mean()
        y = a['NS4'].mean()
        points_x.append(x)
        points_y.append(y)
    
    texts = []
    for x, y, s in zip(points_x, points_y, labels):
        texts.append(plt.text(x, y, s))
    ax.scatter(points_x, points_y)
    plt.xlabel("Mean Time")
    plt.ylabel("Mean NS4 Score")
    plt.title('Time vs Score' + title)
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
    plt.savefig(f"./Results/{('Time vs Score' + title).replace(' ', '_')}.png", bbox_inches='tight')

    plt.show()

df = read()
tm_vs_score(df)
tm_vs_score(df[df['dataset']=='Random'], title=' on Random Subset')
tm_vs_score(df[df['dataset']=='GDS_rand'], title=' on GDS_Rand Subset')
#tm_vs_score(df[df['dataset']=='SparseMatrixSuite'], title='SparseMatrixSuite')

def size_vs_col(df, col='time', only_small=False, title_addon=''):
    df['algo'] = df['algo'].astype("category")
    if only_small:
        size_lim = df[df['algo'] == 'TSP_gurobi']['size'].max()
        df = df[df['size']<=size_lim]

    if col == 'time':
        title = "Execution Time vs. Input Size" + title_addon
        ylabel = "Execution Time"
    else:
        title = "Score vs. Input size" + title_addon
        ylabel = col + " Score"
    
    size_map = {algo: 6.0 if algo == 'TSP_gurobi' or algo == 'TSP_LIN_TimeLim=30' else 1.2 for algo in df['algo'].cat.categories}
    print(size_map)
    plt.figure(figsize=(12, 12))
    sns.lineplot(data=df, x="n", y=col, hue="algo", marker="o", size='algo', errorbar=None, sizes=size_map)

    plt.xlabel("Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"./Results/{title.replace(' ', '_')}.png", bbox_inches='tight')
    plt.show()

"""data = df

size_vs_col(
    df[df['algo'] != 'BAE->Hclust'][df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'],
    col='NS4',
    )

size_vs_col(
    df[df['algo'] != 'BAE->Hclust'][df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'],
    col='time'
    )

df = data[data['dataset']=='Random']

size_vs_col(
    df[df['algo'] != 'BAE->Hclust'][df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'],
    col='NS4',
    title_addon=' on Random subset'
    )

size_vs_col(
    df[df['algo'] != 'BAE->Hclust'][df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'],
    col='time',
    title_addon=' on Random subset'
    )

df = data[data['dataset']=='GDS_rand']

size_vs_col(
    df[df['algo'] != 'BAE->Hclust'][df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'],
    col='NS4',
    title_addon=' on GDS_rand subset'
    )

size_vs_col(
    df[df['algo'] != 'BAE->Hclust'][df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'][df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000'],
    col='time',
    title_addon=' on GDS_rand subset'
    )"""