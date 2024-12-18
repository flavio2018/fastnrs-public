import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines


ANNOT_FONTSIZE = 25
AXES_RIGHTLAB_FONTSIZE = 20
AXES_LAB_FONTSIZE = 18
AXES_TICK_FONTSIZE = 15
TITLE_FONTSIZE = 25


def load_table_o1(task_name):
    df = pd.read_csv(
        f"../gpt/output/accuracy_tables/o1-preview_{task_name}.csv",
        index_col=0,
    )
    df = df.replace(-1, np.nan)
    df.rename(columns={"0": "2"}, inplace=True)
    return revert_rows_order(reformat_floats(df))


def main():
    accuracy_tables_by_task = {
        "listops": {
            "ours_w": load_table_ours("listops", window=True),
            "ours_textseg": load_table_textseg("listops"),
            # "gpt": load_table_gpt("listops"),
            # "o1": load_table_o1("listops"),
            # "ndr": load_table_ndr("listops"),
            "ndr_ext": load_table_ndr("listops", extended=True),
            # "ours": load_table_ours('listops'),
        },
        "arithmetic": {
            "ours_w": load_table_ours("arithmetic", window=True),
            "ours_textseg": load_table_textseg("arithmetic"),
            # "gpt": load_table_gpt("arithmetic"),
            # "o1": load_table_o1("arithmetic"),
            # "ndr": load_table_ndr("arithmetic"),
            "ndr_ext": load_table_ndr("arithmetic", extended=True),
            # "ours": load_table_ours('arithmetic'),
        },
        "algebra": {
            "ours_w": load_table_ours("algebra", window=True),
            "ours_textseg": load_table_textseg("algebra"),
            # "gpt": load_table_gpt("algebra"),
            # "o1": load_table_o1("algebra"),
            # "ndr": load_table_ndr("algebra"),
            "ndr_ext": load_table_ndr("algebra", extended=True),
            # "ours": load_table_ours('algebra'),
        },
        "logic": {
            "ours_w": load_table_ours("logic", window=True),
            "ours_textseg": load_table_textseg("logic"),
            # "gpt": load_table_gpt("logic"),
            # "o1": load_table_o1("logic"),
            # "ndr": load_table_ndr("logic"),
            "ndr_ext": load_table_ndr("logic", extended=True),
        },
    }

    # accuracy_tables_alltask = {
    #     "arithmetic": load_table_alltask("arithmetic"),
    #     "algebra": load_table_alltask("algebra"),
    #     "logic": load_table_alltask("logic"),
    #     "listops": load_table_alltask("listops"),
    # }

    # accuracy_tables_textseg_alltask = {
    #     "arithmetic": load_table_alltask("arithmetic", textseg=True),
    #     "algebra": load_table_alltask("algebra", textseg=True),
    #     "logic": load_table_alltask("logic", textseg=True),
    #     "listops": load_table_alltask("listops", textseg=True),
    # }

    accuracy_tables_o1_alltask = {
        "arithmetic": {
            "o1": load_table_o1("arithmetic"),
            # "nrs_alltask": load_table_alltask("arithmetic"),
            # "fastnrs_alltask": load_table_alltask("arithmetic", textseg=True),
            "gpt4": load_table_gpt("arithmetic"),
        },
        "algebra": {
            "o1": load_table_o1("algebra"),
            # "nrs_alltask": load_table_alltask("algebra"),
            # "fastnrs_alltask": load_table_alltask("algebra", textseg=True),
            "gpt4": load_table_gpt("algebra"),
        },
        "logic": {
            "o1": load_table_o1("logic"),
            # "nrs_alltask": load_table_alltask("logic"),
            # "fastnrs_alltask": load_table_alltask("logic", textseg=True),
            "gpt4": load_table_gpt("logic"),
        },
        "listops": {
            "o1": load_table_o1("listops"),
            # "nrs_alltask": load_table_alltask("listops"),
            # "fastnrs_alltask": load_table_alltask("listops", textseg=True),
            "gpt4": load_table_gpt("listops"),
        },
    }

    # plot_accuracy_tables_listops(accuracy_tables_by_task['listops'])
    # plot_accuracy_tables_arit_alg(accuracy_tables_by_task['arithmetic'], 'arithmetic')
    # plot_accuracy_tables_arit_alg(accuracy_tables_by_task['algebra'], 'algebra')

    # plot_accuracy_tables_all(accuracy_tables_by_task)

    # plot_accuracy_table_lineplot(accuracy_tables_by_task)
    # plot_accuracy_table_alltask(accuracy_tables_alltask, False)
    # plot_accuracy_table_alltask(accuracy_tables_textseg_alltask, True)
    plot_accuracy_table_o1_alltask(accuracy_tables_o1_alltask)

    # accuracy_tables_zscot = {
    #     'listops': load_table_zero_shot_cot('listops'),
    #     'arithmetic': load_table_zero_shot_cot('arithmetic'),
    #     'algebra': load_table_zero_shot_cot('algebra'),
    # }
    #
    # plot_accuracy_tables_zero_shot_cot(accuracy_tables_zscot)

    # accuracy_tables_ndr_expanded = {
    #     "listops": load_table_ndr_expanded("listops"),
    #     "arithmetic": load_table_ndr_expanded("arithmetic"),
    #     "algebra": load_table_ndr_expanded("algebra"),
    # }
    # plot_accuracy_tables_zero_shot_cot(accuracy_tables_ndr_expanded, "ndrexp")


def load_table_ndr(task_name, extended=False):
    if extended:
        df = pd.read_csv(
            f"../out/ndr_accuracy_tables/{task_name}_extended.csv", index_col=0
        )
    else:
        df = pd.read_csv(f"../out/ndr_accuracy_tables/{task_name}.csv", index_col=0)
    return revert_rows_order(reformat_floats(df))


def load_table_gpt(task_name):
    df = pd.read_csv(
        f"../gpt/output/accuracy_tables/gpt4_{task_name}_self_consistency.csv",
        index_col=0,
    )
    return revert_rows_order(reformat_floats(df.dropna(axis=1)))


def load_table_zero_shot_cot(task_name):
    df = pd.read_csv(
        f"../gpt/output/accuracy_tables/gpt4_{task_name}_zero_shot_cot.csv", index_col=0
    )
    return revert_rows_order(reformat_floats(df.dropna(axis=1)))


def load_table_ours(task_name, window=False):
    if window:
        df = pd.read_csv(
            f"../out/ours_accuracy_tables/window/{task_name}.csv", index_col=0
        )
        return revert_rows_order(reformat_floats(df))
    else:
        df = pd.read_csv(
            f"../out/ours_accuracy_tables/no_window/{task_name}.csv", index_col=0
        )
        return revert_rows_order(reformat_floats(df))


def load_table_alltask(task_name, textseg=False):
    if textseg:
        df = pd.read_csv(
            f"../out/ours_accuracy_tables/alltask/textseg/{task_name}.csv", index_col=0
        )
    else:
        df = pd.read_csv(
            f"../out/ours_accuracy_tables/alltask/window/{task_name}.csv", index_col=0
        )
    return revert_rows_order(reformat_floats(df))


def load_table_textseg(task_name):
    df = pd.read_csv(
        f"../out/ours_accuracy_tables/textseg/{task_name}.csv", index_col=0
    )
    return revert_rows_order(reformat_floats(df))


def reformat_floats(df):
    return df.astype(str).map(lambda x: x.replace(",", ".")).astype(float)


def revert_rows_order(df):
    return df.iloc[::-1]


def plot_accuracy_tables_listops(accuracy_tables):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True, sharex=True)

    for (model_name, table), ax in zip(accuracy_tables.items(), axes.flat):
        ax = sns.heatmap(
            table.iloc[::-1, ::-1].T,
            ax=ax,
            vmin=0,
            vmax=1,
            annot=True,
            annot_kws={"fontsize": ANNOT_FONTSIZE},
            cbar=False,
            square=True,
        )
        ax.set_title(format_model_name(model_name), fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

    # axes[1, 0].set_xlabel('nesting', fontsize=AXES_LAB_FONTSIZE)
    # axes[1, 1].set_xlabel('nesting', fontsize=AXES_LAB_FONTSIZE)
    # axes[0, 0].set_ylabel('arguments', fontsize=AXES_LAB_FONTSIZE)
    # axes[1, 0].set_ylabel('arguments', fontsize=AXES_LAB_FONTSIZE)
    # axes[1, 1].set_title('NRS\n(- Dynamic Windowing)', fontsize=TITLE_FONTSIZE)

    plt.savefig("../out/plots/accuracy_tables_listops.pdf", bbox_inches="tight")


def plot_accuracy_tables_arit_alg(accuracy_tables, task_name):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for (model_name, table), ax in zip(accuracy_tables.items(), axes.flat):
        ax = sns.heatmap(
            table.iloc[::-1].T,
            ax=ax,
            vmin=0,
            vmax=1,
            annot=True,
            annot_kws={"fontsize": ANNOT_FONTSIZE},
            cbar=False,
            square=True,
        )
        ax.set_title(format_model_name(model_name), fontsize=TITLE_FONTSIZE)
        ax.set_ylabel("arguments", fontsize=AXES_LAB_FONTSIZE)
        ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

    axes[-1].set_xlabel("nesting", fontsize=AXES_LAB_FONTSIZE)

    plt.savefig(f"../out/plots/accuracy_tables_{task_name}.pdf", bbox_inches="tight")


def plot_accuracy_tables_all(accuracy_tables):
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    spec = fig.add_gridspec(nrows=5, ncols=3)
    ax01 = fig.add_subplot(spec[:3, 0])
    ax02 = fig.add_subplot(spec[:3, 1])
    ax03 = fig.add_subplot(spec[:3, 2])
    ax11 = fig.add_subplot(spec[3, 0])
    ax12 = fig.add_subplot(spec[3, 1])
    ax13 = fig.add_subplot(spec[3, 2])
    ax21 = fig.add_subplot(spec[4, 0])
    ax22 = fig.add_subplot(spec[4, 1])
    ax23 = fig.add_subplot(spec[4, 2])
    axes = [[ax01, ax02, ax03], [ax11, ax12, ax13], [ax21, ax22, ax23]]
    ANNOT_FONTSIZE = 14

    for task_idx, task_name in enumerate(accuracy_tables.keys()):
        if task_name == "listops":
            for (model_name, table), ax in zip(
                accuracy_tables[task_name].items(), axes[task_idx]
            ):
                ax = sns.heatmap(
                    table.iloc[::-1, ::-1].T,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    annot=True,
                    annot_kws={"fontsize": ANNOT_FONTSIZE},
                    cbar=False,
                )
                if model_name == "ndr":
                    ax.set_ylabel("args", fontsize=AXES_LAB_FONTSIZE)
                elif model_name == "ours_w":
                    ax.text(
                        4.2,
                        2.0,
                        format_task_name(task_name.capitalize()),
                        rotation=270,
                        fontsize=AXES_LAB_FONTSIZE,
                    )
                    ax.set_yticklabels([])
                else:
                    ax.set_yticklabels([])
                if task_name == "algebra":
                    ax.set_xlabel("nesting", fontsize=AXES_LAB_FONTSIZE)
                ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
                ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)
                ax.set_title(format_model_name(model_name), fontsize=TITLE_FONTSIZE)
        else:
            for (model_name, table), ax in zip(
                accuracy_tables[task_name].items(), axes[task_idx]
            ):
                ax = sns.heatmap(
                    table.iloc[::-1].T,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    annot=True,
                    annot_kws={"fontsize": ANNOT_FONTSIZE},
                    cbar=False,
                )
                if model_name == "ndr":
                    ax.set_ylabel("args", fontsize=AXES_LAB_FONTSIZE)
                elif model_name == "ours_w":
                    if task_name == "arithmetic":
                        ax.text(
                            6.3,
                            1.0,
                            format_task_name("Arithm."),
                            rotation=270,
                            fontsize=AXES_LAB_FONTSIZE,
                        )
                    else:
                        ax.text(
                            6.3,
                            1.0,
                            format_task_name(task_name.capitalize()),
                            rotation=270,
                            fontsize=AXES_LAB_FONTSIZE,
                        )
                    ax.set_yticklabels([" "])
                else:
                    ax.set_yticklabels([" "])
                if task_name == "algebra":
                    ax.set_xlabel("nesting", fontsize=AXES_LAB_FONTSIZE)
                ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
                ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)
    # fig.suptitle(' NDR                   GPT-4                NRS', fontsize=TITLE_FONTSIZE)
    plt.savefig(f"../out/plots/accuracy_tables_all.pdf", bbox_inches="tight")


def plot_accuracy_tables_zero_shot_cot(accuracy_tables_by_task, name="zscot"):
    base_fontsize = 8

    fig = plt.figure(layout="constrained", figsize=(4, 2.2))
    gs = GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    axes = [ax1, ax2, ax3]

    for task_name, ax in zip(["listops", "arithmetic", "algebra"], axes):
        table = accuracy_tables_by_task[task_name]
        if task_name == "listops":
            ax = sns.heatmap(
                table.iloc[::-1, ::-1].T,
                ax=ax,
                vmin=0,
                vmax=1,
                annot=True,
                annot_kws={"fontsize": base_fontsize - 2},
                cbar=False,
                square=True,
            )
        else:
            ax = sns.heatmap(
                table.iloc[::-1].T,
                ax=ax,
                vmin=0,
                vmax=1,
                annot=True,
                annot_kws={"fontsize": base_fontsize - 2},
                cbar=False,
                square=True,
            )

        ax.set_title(format_task_name(task_name.capitalize()), fontsize=base_fontsize)
        ax.tick_params(axis="x", labelsize=base_fontsize - 4)
        ax.tick_params(axis="y", labelsize=base_fontsize - 4)
        ax.set_ylabel("arguments", fontsize=base_fontsize - 3)

    axes[0].set_xlabel("nesting", fontsize=base_fontsize - 3)
    axes[-1].set_xlabel("nesting", fontsize=base_fontsize - 3)

    plt.savefig(f"../out/plots/accuracy_tables_{name}.pdf", bbox_inches="tight")


def plot_accuracy_table_lineplot(tables):
    TITLE_FONTSIZE = 25
    AXIS_LABEL_FONTSIZE = 29
    AXIS_TICKS_FONTSIZE = 24
    LEGEND_FONTSIZE = 24
    TEXT_FONTSIZE = 24
    colors = {
        "ours_w": "tab:blue",
        "ours_textseg": "tab:red",
        "ndr_ext": "tab:pink",
    }
    # linestyles = {
    #     "ours_w": "solid",
    #     "ours_textseg": "dashdot",
    #     "ndr_ext": "dotted",
    #     # "gpt": "dashed",
    #     # "o1": (5, (10, 3)),
    # }
    for task_name, tables_task in tables.items():
        for model_name, table in tables_task.items():
            if model_name == "ndr":
                continue
            if task_name == "listops" and model_name != "o1":
                ax = sns.lineplot(
                    table.mean(axis=1).iloc[:],
                    label=format_model_name(model_name),
                    marker="o",
                    markeredgecolor=colors[model_name],
                    markeredgewidth=0.3,
                    # linestyle=linestyles[model_name],
                    color=colors[model_name],
                )
            else:
                ax = sns.lineplot(
                    table["2"],
                    label=format_model_name(model_name),
                    marker="o",
                    markeredgecolor=colors[model_name],
                    markeredgewidth=0.3,
                    #                     linestyle=linestyles[model_name],
                    color=colors[model_name],
                )
            if task_name != "algebra":
                ax.legend().set_visible(False)
            else:
                custom_markers = [
                    mlines.Line2D(
                        [],
                        [],
                        color=colors[model_name],
                        marker="o",
                        markeredgecolor=colors[model_name],
                        markeredgewidth=0.3,
                        # linestyle=linestyles[model_name],
                        label=format_model_name(model_name),
                    )
                    for model_name in colors
                ]

                plt.legend(
                    handles=custom_markers,
                    fontsize=LEGEND_FONTSIZE,
                    ncol=2,
                    loc="lower left",
                    columnspacing=0.5,
                    handlelength=1.0,
                )
            if task_name == "logic" or task_name == "arithmetic":
                ax.set_ylabel("Sequence Accuracy")
                ax.yaxis.set_label_coords(-0.2, 0.5)
            else:
                ax.set_ylabel("")
        # ax.set_title(format_task_name(task_name))
        ax.set_xlabel("Nesting depth", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(fontsize=AXIS_TICKS_FONTSIZE)
        plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)

        ax.axvline(3, linestyle="--", color="lightgrey", label="")
        ax.text(
            0.42 if task_name != "logic" else 0.22,
            0.5,
            "OOD ↓",
            rotation=90,
            fontsize=TEXT_FONTSIZE,
            transform=ax.transAxes,
        )
        ax.text(
            0.34 if task_name != "logic" else 0.14,
            0.4,
            "IID ↑",
            rotation=90,
            fontsize=TEXT_FONTSIZE,
            transform=ax.transAxes,
        )
        # plt.show()
        plt.savefig(
            f"../out/plots/accuracy_tables_lineplot_{task_name}.pdf",
            bbox_inches="tight",
        )
        plt.clf()


def plot_accuracy_table_alltask(tables, textseg=False):
    AXIS_LABEL_FONTSIZE = 20
    AXIS_TICKS_FONTSIZE = 15
    LEGEND_FONTSIZE = 15
    TEXT_FONTSIZE = 15

    colors = {
        "logic": "tab:blue",
        "listops": "tab:red",
        "arithmetic": "tab:green",
        "algebra": "tab:orange",
    }
    # if textseg:
    #     title = "FastNRS (multi)"
    # else:
    #     title = "NRS (multi)"
    for task_name, table_logic in tables.items():
        if task_name == "listops":
            ax = sns.lineplot(
                table_logic.mean(axis=1).iloc[:],
                label=format_task_name(task_name),
                marker="o",
                markeredgecolor=colors[task_name],
                markeredgewidth=0.5,
                color=colors[task_name],
            )
        else:
            table_logic = table_logic[-7:]
            ax = sns.lineplot(
                table_logic["2"],
                label=(
                    "Arithm."
                    if "Arithmetic" == format_task_name(task_name)
                    else format_task_name(task_name)
                ),
                marker="o",
                markeredgecolor=colors[task_name],
                markeredgewidth=0.5,
                color=colors[task_name],
            )
    # ax.set_title(title)

    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    #
    # inset_ax = inset_axes(
    #     ax,
    #     width="30%",  # width = 30% of parent_bbox
    #     height=1.0,  # height : 1 inch
    #     loc="right",
    # )
    # table_logic = tables["logic"][:-5]
    # inset_ax = sns.lineplot(
    #     table_logic["2"],
    #     marker="o",
    #     markeredgecolor=colors["logic"],
    #     markeredgewidth=0.5,
    #     color=colors["logic"],
    #     ax=inset_ax,
    # )
    # inset_ax.set_ylim((-0.1, 1.1))
    # inset_ax.set_xlim((6.5, 12.5))
    # inset_ax.set_ylabel("")
    # plt.xticks(fontsize=AXIS_TICKS_FONTSIZE)
    # plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)

    # ax.lines((4.5, 0.6), (5.5, 1.6))

    ax.set_ylim((-0.1, 1.1))
    ax.set_xlim((0.8, 6.2))
    ax.text(
        0.42,
        0.5,
        "OOD ↓",
        rotation=90,
        fontsize=TEXT_FONTSIZE,
        transform=ax.transAxes,
    )
    ax.text(
        0.36,
        0.4,
        "IID ↑",
        rotation=90,
        fontsize=TEXT_FONTSIZE,
        transform=ax.transAxes,
    )
    if not textseg:
        ax.legend().set_visible(False)
        ax.set_ylabel("Sequence Accuracy", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2, columnspacing=0.5)
    ax.set_xlabel("Nesting depth", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(fontsize=AXIS_TICKS_FONTSIZE)
    plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)
    ax.axvline(3, linestyle="--", color="lightgrey", label="")
    plt.savefig(
        f"../out/plots/accuracy_tables_lineplot_{'textseg_' if textseg else ''}alltask.pdf",
        bbox_inches="tight",
    )
    plt.clf()


def plot_accuracy_table_o1_alltask(tables):
    TITLE_FONTSIZE = 25
    AXIS_LABEL_FONTSIZE = 29
    AXIS_TICKS_FONTSIZE = 24
    LEGEND_FONTSIZE = 24
    TEXT_FONTSIZE = 24
    colors = {
        # "nrs_alltask": "tab:blue",
        # "fastnrs_alltask": "tab:red",
        "gpt4": "tab:green",
        "o1": "tab:orange",
    }
    # linestyles = {
    #     "nrs_alltask": "solid",
    #     "fastnrs_alltask": "dashed",
    #     "gpt4": "",
    #     "o1": "dotted",
    # }
    for task_name, tables_task in tables.items():
        for model_name, table in tables_task.items():
            marker = "o"
            if task_name == "listops" and model_name != "o1":
                ax = sns.lineplot(
                    table.mean(axis=1).iloc[:],
                    label=format_model_name(model_name),
                    marker=marker,
                    markeredgecolor=colors[model_name],
                    markeredgewidth=0.3,
                    # linestyle=linestyles[model_name],
                    color=colors[model_name],
                )
            else:
                ax = sns.lineplot(
                    table["2"],
                    label=format_model_name(model_name),
                    marker=marker,
                    markeredgecolor=colors[model_name],
                    markeredgewidth=0.3,
                    #                     linestyle=linestyles[model_name],
                    color=colors[model_name],
                )
            if task_name != "algebra":
                ax.legend().set_visible(False)
            else:
                custom_markers = [
                    mlines.Line2D(
                        [],
                        [],
                        color=colors[model_name],
                        # marker=marker,
                        markeredgecolor=colors[model_name],
                        markeredgewidth=0.3,
                        #                         linestyle=linestyles[model_name],
                        label=format_model_name(model_name),
                    )
                    for model_name in colors
                ]

                plt.legend(
                    handles=custom_markers,
                    fontsize=LEGEND_FONTSIZE,
                    ncol=2,
                    loc="lower left",
                    columnspacing=0.5,
                    handlelength=1.0,
                )
            if task_name == "logic" or task_name == "arithmetic":
                ax.set_ylabel("Sequence Accuracy")
                ax.yaxis.set_label_coords(-0.2, 0.5)
            else:
                ax.set_ylabel("")
            # ax.set_title(format_task_name(task_name))
        ax.set_xlabel("Nesting depth", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(0.00, 1.05)
        plt.xticks(fontsize=AXIS_TICKS_FONTSIZE)
        plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)

        ax.axvline(3, linestyle="--", color="lightgrey", label="")
        ax.text(
            0.42 if task_name != "logic" else 0.22,
            0.5,
            "OOD ↓",
            rotation=90,
            fontsize=TEXT_FONTSIZE,
            transform=ax.transAxes,
        )
        ax.text(
            0.34 if task_name != "logic" else 0.14,
            0.4,
            "IID ↑",
            rotation=90,
            fontsize=TEXT_FONTSIZE,
            transform=ax.transAxes,
        )
        # plt.show()
        plt.savefig(
            f"../out/plots/accuracy_tables_lineplot_o1_alltask_{task_name}_presentation.pdf",
            bbox_inches="tight",
        )
        plt.clf()


def format_model_name(model_name):
    return {
        "ndr": "NDR",
        "ndr_ext": "NDR",
        "gpt": "GPT-4",
        "gpt4": "GPT-4",
        "o1": "o1-prev",
        "ours": "NRS (- Dynamic Windowing)",
        "ours_w": "NRS",
        "ours_textseg": "FastNRS",
        "nrs_alltask": "NRS$^{MD}$",
        "fastnrs_alltask": "FastNRS$^{MD}$",
    }[model_name]


def format_task_name(task_name):
    if task_name == "Listops":
        return "ListOps"
    else:
        return task_name.capitalize()


if __name__ == "__main__":
    main()
