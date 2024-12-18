import logging
import pandas as pd

import utils


def main():
    setup_logging()

    for task_name in ["controlled_select", "solve_atomic"]:
        for split_name in ["train", "valid_iid", "valid_ood"]:
            if task_name == "solve_atomic" and split_name == "valid_ood":
                continue
            resample_split(task_name, split_name)


def resample_split(task_name, split_name):
    logging.info(f"Loading {split_name} dataframes...")
    algebra_df = pd.read_csv(f"../datasets/algebra_{task_name}/{split_name}.csv")
    arithmetic_df = pd.read_csv(f"../datasets/arithmetic_{task_name}/{split_name}.csv")
    listops_df = pd.read_csv(f"../datasets/listops_{task_name}/{split_name}.csv")
    logic_df = pd.read_csv(f"../datasets/logic_{task_name}/{split_name}.csv")
    logging.info(f"Dataframes loaded.")

    if split_name == "valid_ood":
        groupby_fields = ["depth", "subexpr_ops"]
    else:
        groupby_fields = ["nesting", "num_operands", "extra"]

    print(algebra_df.groupby(groupby_fields, dropna=False).count())
    print(arithmetic_df.groupby(groupby_fields, dropna=False).count())
    print(listops_df.groupby(groupby_fields, dropna=False).count())
    print(logic_df.groupby(groupby_fields, dropna=False).count())

    algebra_df = resample_set(algebra_df, split_name)
    arithmetic_df = resample_set(arithmetic_df, split_name)
    listops_df = resample_set(listops_df, split_name)
    logic_df = resample_set(logic_df, split_name)

    pd.concat([algebra_df, arithmetic_df, listops_df, logic_df]).to_csv(
        f"../datasets/alltask_{task_name}/{split_name}_new.csv", index=False
    )


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s\n%(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename=f"../datasets/alltask_resampling_run.log",
        filemode="w",
    )
    utils.mirror_logging_to_console()


def resample_set(df, split_name):
    new_df_parts = []
    logging.info(f"Resampling {split_name} dataset...")

    if split_name == "valid_ood":
        groupby_fields = ["depth", "subexpr_ops"]
    else:
        groupby_fields = ["nesting", "num_operands", "extra"]

    splits_counts = df.groupby(groupby_fields, dropna=False).count()
    if split_name == "train":
        max_count = 40000
    else:
        max_count = 1000

    for split in splits_counts.index:
        split_count = splits_counts.loc[split, "X"]

        if split_name == "valid_ood":
            depth, subexpr_ops = split
            split_samples = df[
                (df["depth"] == depth) & (df["subexpr_ops"] == subexpr_ops)
            ]
        else:
            nesting, num_operands, extra = split
            split_samples = df[
                (df["nesting"] == nesting)
                & (df["num_operands"] == num_operands)
                & (df["extra"] == extra)
            ]

        if split_count < max_count:
            diff = max_count - split_count
            upsampled_split_samples = split_samples.sample(n=diff, replace=True)
            new_df_parts.append(split_samples)
            new_df_parts.append(upsampled_split_samples)

        elif split_count > max_count:
            downsampled_split_samples = split_samples.sample(n=max_count)
            new_df_parts.append(downsampled_split_samples)

        elif split_count == max_count:
            new_df_parts.append(split_samples)

    new_df = pd.concat(new_df_parts)
    logging.info(f"{len(new_df)} samples.")
    logging.info(new_df.groupby(groupby_fields, dropna=False).count())

    return new_df


if __name__ == "__main__":
    main()
