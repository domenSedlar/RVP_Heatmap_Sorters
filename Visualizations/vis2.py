from Visualizations.vis import read, tm_vs_score, size_vs_col
from matplotlib import pyplot as plt
import re

df = read()

long_categories = [
       'GTSP->Rand_Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'GTSP->Rand_Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'GTSP->Rand_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'GTSP->TSP_LIN_TimeLim=30',
       'Hclust->Rand_Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'Hclust->Rand_Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'Hclust->Rand_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'Rand_Block_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'Rand_Mirror_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'Rand_Swaps_Tries=500_Temp=0.0_Cooling=0.0_NumIter=1',
       'TSP_LIN_TimeLim=30', 'TSP_LK_runs=10']
new_categories=[
       'GTSP->Rand_Block_Swaps',
       'GTSP->Rand_Mirror',
       'GTSP->Rand_Swaps',
       'GTSP->TSP_LIN',
       'Hclust->Rand_Block_Swaps',
       'Hclust->Rand_Mirror',
       'Hclust->Rand_Swaps',
       'Rand_Block_Swaps',
       'Rand_Mirror',
       'Rand_Swaps',
       'TSP_LIN', 'TSP_LK'
]

def mk_tabl1(df):
    df['algo'] = df['algo'].astype('category').cat.rename_categories(dict(zip(long_categories,new_categories)))

    df_rand = df[df['dataset']=='Random']
    df_gds = df[df['dataset']=='GDS_rand']

    tm_vs_score(df_rand, title=' on random dataset', s=121)
    tm_vs_score(df_gds, title=' on GDS dataset', s=122)
    plt.show()

def mk_fig2(df):
    df['algo'] = df['algo'].astype('category').cat.rename_categories(dict(zip(long_categories,new_categories)))

    size_vs_col(df, 'time', s=121)
    size_vs_col(df, 'NS4', s=122)

    plt.show()


good_algos = [
    "gurobi",
    "Hclust",
    "BEA",
    'TSP_LK',
    "TSP_LIN",
]


def mk_fig3(df):
    pattern = "|".join(map(re.escape, good_algos))
    df = df[~df['algo'].str.contains('GTSP')]
    df = df[df["algo"].str.contains(pattern, na=False)]

    df['algo'] = df['algo'].astype('category').cat.rename_categories(dict(zip(long_categories,new_categories)))

    size_vs_col(df, 'time', s=121)
    size_vs_col(df, 'NS4', s=122)
    plt.show()

def mk_fig4(df):
    pattern = "|".join(map(re.escape, good_algos))
    df = df[~df['algo'].str.contains('GTSP')]
    df = df[df["algo"].str.contains(pattern, na=False)]

    df['algo'] = df['algo'].astype('category').cat.rename_categories(dict(zip(long_categories,new_categories)))

    df_rand = df[df['dataset']=='Random']
    df_gds = df[df['dataset']=='GDS_rand']

    size_vs_col(df_rand, 'time', s=221, title_addon=' on random dataset')
    size_vs_col(df_rand, 'NS4', s=222, title_addon=' on random dataset')

    size_vs_col(df_gds, 'time', s=223, title_addon=' on gds dataset')
    size_vs_col(df_gds, 'NS4', s=224, title_addon=' on gds dataset')
    plt.show()

def mk_fig5(df):
    chain_algos = [
        'Hclust->Rand_Block_Swaps',
        'Hclust->Rand_Mirror',
        'Hclust->Rand_Swaps',
        'Hclust->TSP_gurobi',
    ]

    non_chains = [
        'Rand_Block_Swaps',
        'Rand_Mirror',
        'Rand_Swaps',
        'TSP_gurobi',
    ]
    df = df[~df['algo'].str.contains('GTSP')]
    pattern = "|".join(map(re.escape, chain_algos + non_chains + ['Hclust']))
    df = df[df["algo"].str.contains(pattern, na=False)]

    df['algo'] = df['algo'].astype('category').cat.rename_categories(dict(zip(long_categories,new_categories)))

    linestyle = [':' for _ in non_chains] + ['--']
    linestyle_map = dict(zip(non_chains + ['Hclust'], linestyle))
    size_vs_col(df, 'time', s=121, linestyle_map=linestyle_map)
    size_vs_col(df, 'NS4', s=122, linestyle_map=linestyle_map)

    plt.show()
#mk_tabl1(df)
#mk_fig2(df)
#mk_fig3(df)
mk_fig4(df)
#mk_fig5(df)