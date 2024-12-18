import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def format_task_name(task_name):
    if task_name == "Listops":
        return "ListOps"
    else:
        return task_name


def plot_stacked_barplot(model_name, pure_task_name, ax):
    TITLE_FONTSIZE = 36
    AXES_LAB_FONTSIZE = 32
    AXES_TICK_FONTSIZE = 25

    # Read the CSV file into a Pandas DataFrame
    # The CSV file should have the format: split_name, selector_value, solver_value
    filename = f"../out/new_error_analysis/{model_name}_{pure_task_name}.csv"
    filename_alltask = (
        f"../out/new_error_analysis/{model_name}_alltask_{pure_task_name}.csv"
    )
    df = pd.read_csv(filename)
    df_alltask = pd.read_csv(filename_alltask)

    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns={"selector": "malformed"}, inplace=True)
    df.rename(columns={"no_valid_leaf": "no_valid_leafs"}, inplace=True)
    df_alltask.rename(columns=lambda x: x.strip(), inplace=True)
    df_alltask.rename(columns={"selector": "malformed"}, inplace=True)
    df_alltask.rename(columns={"no_valid_leaf": "no_valid_leafs"}, inplace=True)

    if model_name == "nrs":
        df = df[["split", "solver", "malformed", "missing"]]
        df_alltask = df_alltask[["split", "solver", "malformed", "missing"]]
    else:
        df = df[["split", "solver", "malformed", "missing"]]
        df_alltask = df_alltask[["split", "solver", "malformed", "missing"]]

    # Set the index to the 'split' column
    df.set_index("split", inplace=True)
    df["Nesting"] = df.index.map(lambda x: int(x.split("_")[0][1:]))
    df.set_index("Nesting", inplace=True)

    df_alltask.set_index("split", inplace=True)
    df_alltask["Nesting"] = df_alltask.index.map(lambda x: int(x.split("_")[0][1:]))
    df_alltask.set_index("Nesting", inplace=True)

    if pure_task_name in ["listops", "alltask_listops"]:
        df = df.groupby("Nesting").mean()
        df_alltask = df_alltask.groupby("Nesting").mean()

    bar_width = 0.4
    # Plot the first stacked bar plot
    # bottoms1 = np.zeros(len(df))  # Initialize bottoms for the first stack
    # single-task
    for nes_idx in range(len(df)):
        ax.bar(
            df.index[nes_idx],
            df["solver"].iloc[nes_idx],
            width=bar_width,
            edgecolor="tab:blue" if df["solver"].iloc[nes_idx] > 0 else "none",
            linewidth=2,
            label="solver",
            facecolor="none",
            hatch="xx",
        )
        ax.bar(
            df.index[nes_idx],
            df["malformed"].iloc[nes_idx],
            width=bar_width,
            edgecolor="tab:orange" if df["malformed"].iloc[nes_idx] > 0 else "none",
            linewidth=2,
            bottom=df["solver"].iloc[nes_idx],
            label="malformed",
            facecolor="none",
            hatch="xx",
        )
        ax.bar(
            df.index[nes_idx],
            df["missing"].iloc[nes_idx],
            width=bar_width,
            edgecolor="tab:green" if df["missing"].iloc[nes_idx] > 0 else "none",
            linewidth=2,
            bottom=df["solver"].iloc[nes_idx] + df["malformed"].iloc[nes_idx],
            label="missing",
            facecolor="none",
            hatch="xx",
        )
        # multi-task
        ax.bar(
            df_alltask.index[nes_idx] + bar_width,
            df_alltask["solver"].iloc[nes_idx],
            width=0.4,
            edgecolor="tab:blue" if df_alltask["solver"].iloc[nes_idx] > 0 else "none",
            linewidth=2,
            label="solver",
            facecolor="none",
            hatch="..",
        )
        ax.bar(
            df_alltask.index[nes_idx] + bar_width,
            df_alltask["malformed"].iloc[nes_idx],
            width=0.4,
            edgecolor=(
                "tab:orange" if df_alltask["malformed"].iloc[nes_idx] > 0 else "none"
            ),
            linewidth=2,
            bottom=df_alltask["solver"].iloc[nes_idx],
            label="malformed",
            facecolor="none",
            hatch="..",
        )
        ax.bar(
            df_alltask.index[nes_idx] + bar_width,
            df_alltask["missing"].iloc[nes_idx],
            width=0.4,
            edgecolor=(
                "tab:green" if df_alltask["missing"].iloc[nes_idx] > 0 else "none"
            ),
            linewidth=2,
            bottom=df_alltask["solver"].iloc[nes_idx]
            + df_alltask["malformed"].iloc[nes_idx],
            label="missing",
            facecolor="none",
            hatch="..",
        )

    # Create a stacked bar plot
    # ax = df.plot(kind="bar", stacked=True, ax=ax, width=0.4)
    # for bar in ax.patches:
    #     bar.set_hatch("/")
    # ax = df_alltask.plot(kind="bar", stacked=True, ax=ax, width=0.4)
    # for bar in ax.patches[len(df) :]:
    #     bar.set_hatch("-")
    # # Adjusting the position of the bars to avoid overlap
    # for i, bar in enumerate(ax.patches[: len(df)]):  # First 4 bars belong to data1
    #     bar.set_x(bar.get_x() - 0.2)  # Shift left by 0.2
    #
    # for i, bar in enumerate(ax.patches[len(df) :]):  # Last 4 bars belong to data2
    #     bar.set_x(bar.get_x() + 0.2)  # Shift right by 0.2

    ax.set_ylim(0, 0.4)
    ax.tick_params(axis="x", labelsize=AXES_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=AXES_TICK_FONTSIZE)

    # Add labels and title
    ax.set_xlabel("Nesting", fontsize=AXES_LAB_FONTSIZE)

    if pure_task_name in ["logic"]:
        ax.set_ylabel("Error %", fontsize=AXES_LAB_FONTSIZE)
    else:
        ax.set_yticklabels([])
    ax.set_title(
        f"{format_task_name(pure_task_name.capitalize())}",
        fontsize=TITLE_FONTSIZE,
    )

    # Color legend marker (filled color)
    solver_legend = Patch(
        facecolor="none", edgecolor="tab:blue", label="solver", linewidth=5
    )
    missing_legend = Patch(
        facecolor="none", edgecolor="tab:green", label="missing", linewidth=5
    )
    malformed_legend = Patch(
        facecolor="none", edgecolor="tab:orange", label="malformed", linewidth=5
    )

    # Hatch legend marker (hatch pattern with no fill)
    md_legend = Patch(
        facecolor="none", edgecolor="gray", hatch="..", label="Multi-domain"
    )
    sd_legend = Patch(
        facecolor="none", edgecolor="gray", hatch="xx", label="Single-domain"
    )

    # Add custom legend
    ax.legend(
        handles=[solver_legend, missing_legend, malformed_legend, sd_legend, md_legend]
    )

    if pure_task_name not in ["logic"]:
        ax.legend().set_visible(False)
    else:
        ax.legend(fontsize=AXES_TICK_FONTSIZE - 2, handlelength=1)
        ax.legend().set_visible(False)
        ax.legend(
            handles=[
                solver_legend,
                missing_legend,
                malformed_legend,
                sd_legend,
                md_legend,
            ],
            fontsize=AXES_TICK_FONTSIZE,
        )

    # Show the plot
    if pure_task_name == "logic":
        ax.set_xticks(
            [x + 0.2 for x in range(2, 14, 2)],
            range(2, 14, 2),
            rotation=45,
            fontsize=AXES_TICK_FONTSIZE,
        )
    else:
        ax.set_xticks(
            # np.linspace(-1, len(df) + 3.4, len(df)),
            [x + 0.2 for x in range(1, 7)],
            df.index,
            rotation=45,
            fontsize=AXES_TICK_FONTSIZE,
        )
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    for model_name in ["fastnrs", "nrs"]:
        ax_idx = 0
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

        for pure_task_name in [
            "logic",
            "listops",
            "arithmetic",
            "algebra",
        ]:
            plot_stacked_barplot(model_name, pure_task_name, axes.flat[ax_idx])
            ax_idx += 1
        plt.savefig(
            f"../out/plots/new_error_analysis/{model_name}.pdf",
            bbox_inches="tight",
        )
