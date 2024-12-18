import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data.vocabulary import Vocabulary

TITLE_FONTSIZE = 20
AXES_LAB_FONTSIZE = 16
AXES_TICK_FONTSIZE = 14


def main():
    confidence_scores_by_task = {
        "logic": load_conf_score_df("logic"),
        "listops": load_conf_score_df("listops"),
        "arithmetic": load_conf_score_df("arithmetic"),
        "algebra": load_conf_score_df("algebra"),
        "alltask_logic": load_conf_score_df("logic", alltask=True),
        "alltask_listops": load_conf_score_df("listops", alltask=True),
        "alltask_arithmetic": load_conf_score_df("arithmetic", alltask=True),
        "alltask_algebra": load_conf_score_df("algebra", alltask=True),
    }
    print("Plotting...")
    plot_conf_scores(
        {k: v for k, v in confidence_scores_by_task.items() if "alltask" not in k},
        alltask=False,
    )
    plot_conf_scores(
        {
            k.split("_")[1]: v
            for k, v in confidence_scores_by_task.items()
            if "alltask" in k
        },
        alltask=True,
    )


def load_conf_score_df(task_name, alltask=False):
    # f'../out/conf_scores/{task_name}_solve_1000_input_len_Vs_conf_scores.csv'  # old files
    if not alltask:
        return pd.read_csv(
            f"../out/conf_scores/new_confidence_scores_df_{task_name}.csv", index_col=0
        )
    else:
        task_name = "alltask_" + task_name
        return pd.read_csv(
            f"../out/conf_scores/new_confidence_scores_df_{task_name}.csv", index_col=0
        )
    # if alltask:
    #     task_name = "alltask_test_" + task_name
    # if task_name == "logic" or (alltask and "algebra" not in task_name):
    #     num_outputs = "100"
    # else:
    #     num_outputs = "1000"
    # return pd.read_csv(
    #     f"../out/conf_scores/{task_name}_{num_outputs}_input_len_Vs_conf_scores.csv"
    # )


def plot_conf_scores(dfs, alltask):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))  # , sharex=True, sharey=True)
    conf_threshold_per_task = {
        "logic": None,
        "listops": 150,
        "arithmetic": 125,
        "algebra": 150,
    }

    for ax, (task_name, df) in zip(axes.flat, dfs.items()):
        df = (
            df.groupby("input_length")
            .mean()
            .rename(columns={"confidence_score": "avg_conf_score"})
        )
        df["input_len"] = df.index

        ax = sns.scatterplot(
            data=df,
            x="input_len",
            y="avg_conf_score",
            ax=ax,
            linewidth=0,
            edgecolors="face",
        )
        ax.axvline(max_len_input_per_task[task_name], color="gray", linestyle="--")
        # if conf_threshold_per_task[task_name]:
        #     ax.axvline(conf_threshold_per_task[task_name], color="red", linestyle="--")
        ax.set_title(format_task_name(task_name.capitalize()), fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

    for ax in axes[:, 0]:
        ax.set_ylabel("Avg Confidence Score", fontsize=AXES_LAB_FONTSIZE)
    for ax in axes[1, :]:
        ax.set_xlabel("Input Lenght", fontsize=AXES_LAB_FONTSIZE)
    plt.tight_layout()
    plt.savefig(
        f"../out/plots/conf_score_analysis{'_alltask' if alltask else ''}.pdf",
        bbox_inches="tight",
    )
    plt.clf()


def get_max_len_input_per_task():
    print("Computing max len input per task...")
    return {
        "logic": (
            pd.read_csv("../datasets/logic_controlled_select/train.csv")["X"]
            .apply(len)
            .max()
        ),
        "listops": (
            pd.read_csv("../datasets/listops_controlled_select/train.csv")["X"]
            .apply(Vocabulary._tokenize_listops)
            .apply(len)
            .max()
        ),
        "arithmetic": (
            pd.read_csv("../datasets/arithmetic_controlled_select/train.csv")["X"]
            .apply(Vocabulary._tokenize_arithmetic)
            .apply(len)
            .max()
        ),
        "algebra": (
            pd.read_csv("../datasets/algebra_controlled_select/train.csv")["X"]
            .apply(Vocabulary._tokenize_algebra)
            .apply(len)
            .max()
        ),
    }


def format_task_name(task_name):
    if task_name == "Listops":
        return "ListOps"
    else:
        return task_name


max_len_input_per_task = get_max_len_input_per_task()

if __name__ == "__main__":
    main()
