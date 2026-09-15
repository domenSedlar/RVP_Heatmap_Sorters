from matplotlib import pyplot as plt
import pandas as pd
from adjustText import adjust_text
import seaborn as sns

def read(path='./results.parquet')->pd.DataFrame:
    df = pd.read_parquet(path)
    df['n'] = df[['row_size', 'col_size']].values.min(1)
    df = df[df['algo'] != 'BAE->Hclust']
    #df['algo'] = df['algo'].astype("category")
    df = df[df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000']
    df = df[df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000']
    df = df[df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000']

    return df

def tm_vs_score(df, title='', only_small=False, only_common=True):
    df['algo'] = df['algo'].astype("category")
    df = df.copy()
    if only_common:
        df['id'] = df['dir'] + df['file_name']
        # 1. Get the total number of unique algorithms in your dataset
        num_algos = df['algo'].nunique()

        # 2. Filter groups that contain all unique algorithms
        df = df.groupby('id').filter(
            lambda group: group['algo'].nunique() == num_algos
        )
        print(max(df['row_size']))

        fdf = df.groupby('id').filter(
            lambda group: round(min(group[group['algo'] == 'TSP_gurobi']['NS4']), 2) != round(min(group[group['algo'] == 'BAE->TSP_gurobi']['NS4']), 2)
        )

        print(fdf[['id', 'algo', 'NS4']])
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
    #plt.savefig(f"./Results/{('Time vs Score' + title).replace(' ', '_')}.png", bbox_inches='tight')

    plt.show()

df = read()
tm_vs_score(df, title=', only common')
tm_vs_score(df[df['dataset']=='Random'], title=' on Random Subset, only common')
tm_vs_score(df[df['dataset']=='GDS_rand'], title=' on GDS_Rand Subset, only common')
#tm_vs_score(df[df['dataset']=='SparseMatrixSuite'], title='SparseMatrixSuite')

def size_vs_col(df, col='time', only_small=False, title_addon=''):
    df = df.copy()
    
    if only_small:
        size_lim = df[df['algo'] == 'TSP_gurobi']['size'].max()
        df = df[df['size'] <= size_lim]

    df = df.groupby(['algo', 'n'], as_index=False)[col].mean()
    df = df.sort_values(by=['algo', 'n'])

    if col == 'time':
        title = "Execution Time vs. Input Size" + title_addon
        ylabel = "Execution Time"
    else:
        title = "Score vs. Input size" + title_addon
        ylabel = col + " Score"

    plt.figure(figsize=(10, 8))
    
    # Track algorithm styles manually using Matplotlib directly
    # This guarantees Seaborn will never alter hue assignments or data grouping
    palette = sns.color_palette(None, n_colors=df['algo'].nunique())
    
    for i, (algo, group) in enumerate(df.groupby('algo', observed=True)):
        linewidth = 4.0 if algo in ['TSP_gurobi', 'TSP_LIN_TimeLim=30'] else 1.5
        plt.plot(
            group['n'], 
            group[col], 
            label=algo, 
            linewidth=linewidth, 
            color=palette[i]
        )

    plt.xlabel("Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Algorithm", loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.6)
    #plt.savefig(f"./Results/{title.replace(' ', '_')}.png", bbox_inches='tight')
    plt.show()
data = df

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
    )