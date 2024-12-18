import os.path
import re

import pandas as pd

from data.vocabulary import Vocabulary


def main():
    def get_select_dataset_path(task_name, split_name):
        return f"../datasets/{task_name}_controlled_select/{split_name}.csv"

    task_names = ["logic", "listops", "arithmetic", "algebra"]
    split_names = ["train", "valid_iid", "valid_ood"]

    for task_name in task_names:
        print(f"Processing {task_name}...")

        for split_name in split_names:
            print(f"Processing split {split_name}...")
            df = pd.read_csv(get_select_dataset_path(task_name, split_name))

            df_textseg = make_text_segmentation_dataset(df, split_name, task_name)

            if not os.path.exists(f"../datasets/{task_name}_text_segmentation/"):
                os.mkdir(f"../datasets/{task_name}_text_segmentation")

            df_textseg.to_csv(
                f"../datasets/{task_name}_text_segmentation/{split_name}.csv",
                index=False,
            )

    for split_name in split_names:
        print(f"Processing alltask {split_name}...")
        df = pd.read_csv(get_select_dataset_path("alltask", split_name))
        df["task_name"] = df["X"].apply(Vocabulary.task_name_from_sample)

        dfs = []
        for task_name in task_names:
            df_task = df[df["task_name"] == task_name].copy()
            df_textseg = make_text_segmentation_dataset(df_task, split_name, task_name)
            dfs.append(df_textseg)

        if not os.path.exists(f"../datasets/alltask_text_segmentation/"):
            os.mkdir(f"../datasets/alltask_text_segmentation")

        pd.concat(dfs).to_csv(
            f"../datasets/alltask_text_segmentation/{split_name}.csv",
            index=False,
        )


def make_text_segmentation_dataset(df, split_name, task_name):
    def make_char_level_mask(row):
        if "nesting" in row:
            if (row["nesting"] == 1 and row["num_operands"] == 1) or (
                row["nesting"] == 0
            ):
                return "1" * len(row["X"])
        x = row["X"]
        leafs = leaf_expr_re.findall(x)
        for leaf in leafs:
            x = x.replace(leaf, "£" * len(leaf))
        new_x = ""
        for char in x:
            if char == "£":
                new_x += "1"
            else:
                new_x += "0"
        return new_x

    def make_token_level_mask(row):
        tokenized_x = row["tokenized_X"]
        char_to_token_map = []
        for token_id, token in enumerate(tokenized_x):
            for char in token:
                char_to_token_map += [token_id]
        char_level_mask = row["char_level_mask"]
        token_level_mask = []
        curr_token = None
        for char_id, char_mask in enumerate(char_level_mask):
            if curr_token != char_to_token_map[char_id]:
                token_level_mask.append(char_mask)
                curr_token = char_to_token_map[char_id]
        assert len(token_level_mask) == len(tokenized_x)
        return "".join(token_level_mask)

    if task_name == "logic":
        leaf_expr_re = re.compile("\([a-zTF][|&][a-zTF]\)|\(![a-zTF]\)")
        df["tokenized_X"] = df["X"].apply(lambda x: Vocabulary._tokenize_char(x))
    elif task_name == "listops":
        leaf_expr_re = re.compile(r"\[[A-Z]+[\d{1}]+\]")
        df["tokenized_X"] = df["X"].apply(lambda x: Vocabulary._tokenize_listops(x))
    elif task_name == "arithmetic":
        leaf_expr_re = re.compile(r"\([+\-*]*[0-9]+[+\-*]*[0-9]+\)")
        df["tokenized_X"] = df["X"].apply(lambda x: Vocabulary._tokenize_arithmetic(x))
    elif task_name == "algebra":
        leaf_expr_re = re.compile(r"\([+\-]*[0-9]*[abxy*]+[+\-]*[+\-0-9]*[abxy*]+\)")
        df["tokenized_X"] = df["X"].apply(lambda x: Vocabulary._tokenize_algebra(x))

    df["char_level_mask"] = df.apply(
        lambda row: make_char_level_mask(row), axis=1
    ).astype(str)
    df["token_level_mask"] = df.apply(
        lambda row: make_token_level_mask(row), axis=1
    ).astype(str)
    df["Y"] = df["token_level_mask"]
    df.drop(columns=["token_level_mask"], inplace=True)
    df.drop(columns=["tokenized_X"], inplace=True)
    return df


if __name__ == "__main__":
    main()
