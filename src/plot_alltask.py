import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    accuracy_tables_alltask = {
        'listops': load_table_zero_shot_cot('listops'),
        'arithmetic': load_table_zero_shot_cot('arithmetic'),
        'algebra': load_table_zero_shot_cot('algebra'),
    }
    plot_accuracy_tables_alltask(accuracy_tables_alltask)


def load_table_zero_shot_cot(task_name):
    df = pd.read_csv(f'../out/ours_accuracy_tables/alltask/window/{task_name}.csv', index_col=0)
    df = df.T.drop('_wandb').sort_index()
    df.index = [v.replace('seq_acc/', '').replace(f'__{task_name}', '') for v in df.index]
    df = df.rename(columns={'polar-wave-99': 'Seq Acc'})
    
    if task_name == 'listops':
        new_df = pd.DataFrame({1: [], 2: [], 3: [], 4: []})
        new_df.loc[:, 1] = df.iloc[:3, 0].values
        new_df.loc[:, 2] = df.iloc[3:6, 0].values
        new_df.loc[:, 3] = df.iloc[6:9, 0].values
        new_df.loc[:, 4] = df.iloc[9:12, 0].values
        df = new_df
        df = df.rename(index={'N1_O2': 2, 'N1_O3': 3, 'N1_O4': 4})
    return revert_rows_order(reformat_floats(df.dropna(axis=1)))


def reformat_floats(df):
    return df.astype(str).map(lambda x: x.replace(',', '.')).astype(float)


def revert_rows_order(df):
    return df.iloc[::-1]


def format_task_name(task_name):
    if task_name == 'Listops':
        return 'ListOps'
    else:
        return task_name


def plot_accuracy_tables_alltask(accuracy_tables_by_task):
    base_fontsize = 8
    
    fig = plt.figure(layout="constrained", figsize=(4, 2.2))
    gs = GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    axes = [ax1, ax2, ax3]

    for task_name, ax in zip(['listops', 'arithmetic', 'algebra'], axes):
        table = accuracy_tables_by_task[task_name]
        if task_name == 'listops':
            ax = sns.heatmap(table, ax=ax, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': base_fontsize-2}, cbar=False, square=True)
        else:
            ax = sns.heatmap(table.iloc[::-1].T, ax=ax, vmin=0, vmax=1, annot=True, annot_kws={'fontsize': base_fontsize-2}, cbar=False, square=True)
        
        ax.set_title(format_task_name(task_name.capitalize()), fontsize=base_fontsize)
        ax.tick_params(axis="x", labelsize=base_fontsize-4)
        ax.tick_params(axis="y", labelsize=base_fontsize-4)
        ax.set_ylabel('arguments', fontsize=base_fontsize-3)

    axes[0].set_xlabel('nesting', fontsize=base_fontsize-3)
    axes[-1].set_xlabel('nesting', fontsize=base_fontsize-3)

    plt.savefig(f'../out/plots/accuracy_tables_alltask.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()