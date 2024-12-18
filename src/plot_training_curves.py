import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    for task in [
        # ("algebra", "clean"),
        ("algebra", "rural"),
        ("arithmetic", "stellar"),
        ("listops", "stoic"),
        ("logic", "colorful"),
    ]:
        plot_for_task(task)


def plot_for_task(task):
    TITLE_FONTSIZE = 25
    AXIS_LABEL_FONTSIZE = 34
    AXIS_TICKS_FONTSIZE = 29
    LEGEND_FONTSIZE = 29
    task_name, sweep_name = task
    metric_name = "seqacc"  # loss
    df = pd.read_csv(
        f"../out/textseg_train_curves/{metric_name}_{task_name}_{sweep_name}_sweep.csv"
    )
    for col_name in df.columns:
        if "MIN" in col_name or "MAX" in col_name:
            df.drop(col_name, inplace=True, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for col_name in df.columns:
        if col_name != "Step":
            set_name = col_name.split("/")[1]
            sns.lineplot(
                data=df,
                x="Step",
                y=col_name,
                ax=ax,
                label=format_split_name(set_name),
            )
    # ax.set_ylim((0, 0.3))
    if task_name == "logic" or task_name == "arithmetic":
        ax.set_ylabel(format_metric_name(metric_name), fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    if task_name == "algebra":
        plt.legend(fontsize=LEGEND_FONTSIZE)
    else:
        ax.legend().set_visible(False)
    ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(fontsize=AXIS_TICKS_FONTSIZE)
    plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)
    plt.savefig(
        f"../out/plots/textseg_{task_name}_{metric_name}_curves.pdf",
        bbox_inches="tight",
    )


def format_split_name(split_name):
    if split_name == "train":
        return "Training set"
    elif split_name == "valid_iid":
        return "Validation set"
    elif split_name == "valid_ood":
        return "Validation set (OOD)"


def format_task_name(task_name):
    if task_name == "Listops":
        return "ListOps"
    else:
        return task_name.capitalize()


def format_metric_name(metric_name):
    if metric_name == "seqacc":
        return "Sequence Accuracy"
    elif metric_name == "loss":
        return "Loss"


if __name__ == "__main__":
    main()
