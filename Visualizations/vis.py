from matplotlib import pyplot as plt
import pandas as pd
from adjustText import adjust_text
import seaborn as sns
import re

def read(path='./Results/res.parquet')->pd.DataFrame:
    df = pd.read_parquet(path)
    df['n'] = df[['row_size', 'col_size']].values.min(1)
    df['n'] = df['size']
    df = df[df['algo'] != 'BAE->Hclust']
    #df['algo'] = df['algo'].astype("category")
    df = df[df['algo'] != 'Random_swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000']
    df = df[df['algo'] != 'Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000']
    df = df[df['algo'] != 'Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=50000']
    #df = df[~((df['algo'] == 'TSP_gurobi') & (df['opt'] == False))]
    #df = df[~((df['algo'] == 'TSP_gurobi') & (df['n'] > 60))]
    #df = df[df['NS4'] < 1e30] # This removes mistakes

    return df

def tm_vs_score(df, title='', only_small=False, only_common=False, out_dir='./Results/imgs', s=111):
    s = 111
    df = df.copy()
    #ax = plt.subplot(s)
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

        title = title + ' only common'

    if only_small:
        size_lim = df[df['algo'] == 'TSP_gurobi']['size'].max()
        df = df[df['size']<=size_lim]

        title = title + f"small matrices (< {size_lim})"

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
        s = s[:25]
        texts.append(plt.text(x, y, s))
    ax.scatter(points_x, points_y)
    plt.xlabel("Mean Time")
    plt.ylabel("Mean NS4 Score")
    plt.title('Time vs Score' + title)
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='gray', lw=1.5), force_text=(0.1,0.5))
    #plt.savefig(f"{out_dir}{('Time vs Score' + title).replace(' ', '_')}.png", bbox_inches='tight')

    plt.show()

df = read()
# tm_vs_score(df, title='', s=321)
#tm_vs_score(df, title='', only_common=True)
#tm_vs_score(df[df['dataset']=='Random'], title=' on Random Subset', s=322)
#tm_vs_score(df[df['dataset']=='GDS_rand'], title=' on GDS_Rand Subset', s=323)

#tm_vs_score(df, title='', only_common=True, s=324)
#tm_vs_score(df[df['dataset']=='Random'], title=' on Random Subset', only_common=True, s=325)
#tm_vs_score(df[df['dataset']=='GDS_rand'], title=' on GDS_Rand Subset', only_common=True, s=326)
#tm_vs_score(df[df['dataset']=='SparseMatrixSuite'], title='SparseMatrixSuite')
#plt.show()

def size_vs_col(df, col='time', only_small=False, title_addon='', out_dir='./Results/imgs', s=111):
    #s=111
    #plt.subplot(s)
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

#    plt.figure(figsize=(10, 8))
    
    # Track algorithm styles manually using Matplotlib directly
    # This guarantees Seaborn will never alter hue assignments or data grouping
    palette = sns.color_palette('tab20', n_colors=df['algo'].nunique())
    linestyle_map = {
    'TSP_gurobi': '--',             # Dashed
    'TSP_LIN_TimeLim=30': '--',     # Dashed
    'Hclust': ':',                     # Dotted
    'BAE': '-.'                  # Dash-dot
}
    for i, (algo, group) in enumerate(df.groupby('algo', observed=True)):
        linewidth = 3.0 if algo in ['TSP_gurobi', 'TSP_LIN_TimeLim=30'] else 2
        linestyle = linestyle_map.get(algo, '-')
        plt.plot(
            group['n'], 
            group[col], 
            label=algo, 
            linewidth=linewidth,
            linestyle=linestyle,
            color=palette[i]
        )

    plt.xlabel("Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Algorithm", loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.6)
    #plt.savefig(f"./Results/{title.replace(' ', '_')}.png", bbox_inches='tight')
    plt.show()

def size_vs_col2(
    df,
    col="NS4",
    only_small=False,
    title_addon="",
    out_dir="./Results/imgs",
    s=111,
):
    df = df.copy()

    # Aggregate by algo and n
    df = df.groupby(["algo", "n"], as_index=False)[col].mean()

    # --- Pivot to subtract the baseline ('TSP_gurobi') ---
    df_pivot = df.pivot(index="n", columns="algo", values=col)

    if "TSP_gurobi" in df_pivot.columns:
        # Subtract the baseline from all algorithm columns
        df_pivot = df_pivot.sub(df_pivot["TSP_gurobi"], axis=0)

    # Convert back to long format for plotting
    df = df_pivot.reset_index().melt(
        id_vars="n", var_name="algo", value_name=col
    )
    df = df.dropna(subset=[col]).sort_values(by=["algo", "n"])
    # ----------------------------------------------------

    # Update axis labels and titles to reflect the baseline difference
    baseline_str = " (Diff from TSP_gurobi)"

    title = "Score Difference vs. Input Size" + title_addon
    ylabel = col + " Score Difference"

    palette = sns.color_palette("tab10", n_colors=df["algo"].nunique())
    linestyle_map = {
        "TSP_gurobi": "--",  # Baseline (will sit flat at y=0)
        "TSP_LIN_TimeLim=30": "--",
        "Hclust": ":",
        "BAE": "-.",
    }

    for i, (algo, group) in enumerate(df.groupby("algo", observed=True)):
        linewidth = 3.0 if algo in ["TSP_gurobi", "TSP_LIN_TimeLim=30"] else 2
        if "TSP_gurobi" in algo or "TSP_LIN_TimeLim=30" in algo:
            continue
        linestyle = linestyle_map.get(algo, "-")
        plt.plot(
            group["n"],
            group[col],
            label=algo,
            linewidth=linewidth,
            linestyle=linestyle,
            color=palette[i],
        )

    plt.xlabel("Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Algorithm", loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


"""size_vs_col(
    df,
    col='time'
    )

size_vs_col(
    df,
    col='NS4',
)

size_vs_col(
    df[df['dataset']=='Random'],
    col='NS4',
    title_addon=' on Random dataset'
)"""

good_algos = [
    "gurobi",
    "Hclust",
    "BAE",
    'TSP_LK',
    "TSP_LIN"
]
pattern = "|".join(map(re.escape, good_algos))
df = df[df["algo"].str.contains(pattern, na=False)]

data = df

size_vs_col2(
    df[df['dataset']=='Random'],
    col='NS4'
)
"""size_vs_col(
    data[data['dataset']=='GDS_rand'],
    col='NS4',
    s=321,
    title_addon=' on GDS dataset'
    )

size_vs_col(
    df,
    col='time',
    s=322
    )

df = data[data['dataset']=='Random']

size_vs_col(
    df,
    col='NS4',
    title_addon=' on Random subset',
    s=323,
    )

size_vs_col(
    df,
    col='time',
    title_addon=' on Random subset',
    s=324,
    )

df = data[data['dataset']=='GDS_rand']

size_vs_col(
    df,
    col='NS4',
    title_addon=' on GDS_rand subset',
    s=325,
    )

size_vs_col(
    df,
    col='time',
    title_addon=' on GDS_rand subset',
    s=326
    )
"""
#plt.show()