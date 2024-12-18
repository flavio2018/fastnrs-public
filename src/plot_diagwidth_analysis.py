import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    for metric_name in ["seqacc", "loss"]:
        for task in [
            "algebra",
            "arithmetic",
            "listops",
            "logic",
        ]:
            plot_for_task(task, metric_name, "maskwidth")
            plot_for_task(task, metric_name, "depth")


def plot_for_task(task, metric_name, analysis_name):
    TITLE_FONTSIZE = 25
    AXIS_LABEL_FONTSIZE = 34
    AXIS_TICKS_FONTSIZE = 29
    LEGEND_FONTSIZE = 29

    df = pd.read_csv(f"../out/{analysis_name}_analysis/{task}/{metric_name}.csv")
    for col_name in df.columns:
        if "MIN" in col_name or "MAX" in col_name or "step" in col_name:
            df.drop(col_name, inplace=True, axis=1)

    df.set_index("iteration", drop=True, inplace=True)

    # Group the columns in sets of three
    groups = [df.iloc[:, i : i + 3] for i in range(0, df.shape[1], 3)]
    if task != "listops" or analysis_name == "depth":
        groups = groups[::-1]
    # Prepare to plot
    plt.figure(figsize=(10, 6))

    for idx, group in enumerate(groups):
        # Calculate mean and standard deviation
        group_mean = group.mean(axis=1)
        group_std = group.std(axis=1)

        # Plot the mean line
        ax = sns.lineplot(x=group.index, y=group_mean, label=f"{idx+1}")

        # Fill the area between (mean - std) and (mean + std)
        plt.fill_between(
            group.index, group_mean - group_std, group_mean + group_std, alpha=0.3
        )

    # Customize the plot
    # plt.title("Grouped Series with Mean and Standard Deviation")
    plt.xlabel(
        "Iterations",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    if task == "algebra":
        plt.legend(
            fontsize=LEGEND_FONTSIZE,
            title=(
                "Attn window width"
                if analysis_name == "maskwidht"
                else "Num Selector layers"
            ),
            title_fontsize=LEGEND_FONTSIZE,
            ncols=2,
        )

    else:
        plt.legend().set_visible(False)

    if task == "logic":
        plt.ylabel(
            metric_name.capitalize() if metric_name == "loss" else "Sequence Accuracy",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
    else:
        ax.set_ylabel("")
    plt.xticks(fontsize=AXIS_TICKS_FONTSIZE)
    plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)
    # plt.show()
    plt.savefig(
        f"../out/plots/{analysis_name}_analysis_{task}_{metric_name}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
