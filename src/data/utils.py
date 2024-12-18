import logging

from data.dataset import ItersolvDataset, RegressionDataset


def build_split_dataset(
    data_cfg, split, device, tokenizer=None, difficulty_split=None, task_name=None
):
    if "regr" in data_cfg.name:
        return RegressionDataset(
            dataset_name=data_cfg.name,
            split=split,
            train_batch_size=data_cfg.train_batch_size,
            eval_batch_size=data_cfg.eval_batch_size,
            device=device,
            task_name=task_name,
        )
    else:
        return ItersolvDataset(
            dataset_name=data_cfg.name,
            split=split,
            train_batch_size=data_cfg.train_batch_size,
            eval_batch_size=data_cfg.eval_batch_size,
            device=device,
            sos=data_cfg.sos,
            eos=data_cfg.eos,
            tokenizer=tokenizer,
            difficulty_split=difficulty_split,
            specials_in_x=data_cfg.specials_in_x,
            specials_in_y=data_cfg.specials_in_y,
            task_name=task_name,
        )


def build_datasets(cfg):
    dataset_splits = dict()

    if "train" in cfg.name and "solve_atomic" in cfg.name:
        tokenizer = "char"
    elif "plot_solver" in cfg.name:
        tokenizer = "char"
    else:
        tokenizer = None
    # here we normally build a dataset for each train/valid/test split
    for split in cfg.data.splits:
        dataset_splits[split] = build_split_dataset(
            cfg.data, split, cfg.device, tokenizer=tokenizer
        )

    # here we want to build splits for each type of ood test formula
    if "difficulty_splits" in cfg.data:
        # in the case of alltask
        if "alltask" in cfg.data.name:

            for task_name in ["algebra", "arithmetic", "listops", "logic"]:
                for nesting, num_operands in cfg.data.difficulty_splits:
                    logging.info(
                        f"Building {task_name} ({nesting}, {num_operands}) dataset"
                    )
                    dataset_splits[f"test_{task_name}_{nesting}_{num_operands}"] = (
                        build_split_dataset(
                            cfg.data,
                            "test",
                            cfg.device,
                            difficulty_split=(nesting, num_operands),
                            task_name=task_name,
                        )
                    )
        # and in the other cases
        else:
            for nesting, num_operands in cfg.data.difficulty_splits:
                dataset_splits[f"test_{nesting}_{num_operands}"] = build_split_dataset(
                    cfg.data,
                    "test",
                    cfg.device,
                    difficulty_split=(nesting, num_operands),
                )
    set_vocabularies(dataset_splits, cfg)
    return dataset_splits


def set_vocabularies(datasets, cfg):
    # here we ensure that each of those datasets has the right vocab
    if "valid_iid" in datasets:
        datasets["valid_iid"].set_vocabulary(datasets["train"].vocabulary)
    if "valid_ood" in datasets:
        datasets["valid_ood"].set_vocabulary(datasets["train"].vocabulary)

    if "difficulty_splits" in cfg.data:
        # here we take into account that for the text segmentation task, the Y training
        # vocabulary for the selector is just 0, 1, and this should not be the same for the
        # Y test vocab, which should be the normal vocab
        train_vocab = build_split_dataset(
            cfg.selector_data, "train", cfg.device
        ).vocabulary
        if "textseg" in cfg.name:  # in this case the y train vocab is {0, 1}
            datasets["test"].set_vocabulary(train_vocab, only_x=True)
        else:
            datasets["test"].set_vocabulary(train_vocab)
        if "alltask" in cfg.data.name:
            for task_name in ["algebra", "arithmetic", "listops", "logic"]:
                for nesting, num_operands in cfg.data.difficulty_splits:
                    if (
                        "textseg" in cfg.name
                    ):  # in this case the y train vocab is {0, 1}
                        datasets[
                            f"test_{task_name}_{nesting}_{num_operands}"
                        ].set_vocabulary(train_vocab, only_x=True)
                    else:
                        datasets[
                            f"test_{task_name}_{nesting}_{num_operands}"
                        ].set_vocabulary(train_vocab)
        else:
            for nesting, num_operands in cfg.data.difficulty_splits:
                if "textseg" in cfg.name:  # in this case the y train vocab is {0, 1}
                    datasets[f"test_{nesting}_{num_operands}"].set_vocabulary(
                        train_vocab, only_x=True
                    )
                else:
                    datasets[f"test_{nesting}_{num_operands}"].set_vocabulary(
                        train_vocab
                    )


def get_vocab(cfg, datasets):
    if (
        ("test_selsolcom" in cfg.name)
        or ("test_encselsolcom" in cfg.name)
        or ("test_textseg" in cfg.name)
    ):
        vocab = dict()
        if not "test_textseg" in cfg.name:
            vocab["selector"] = datasets["test"].vocabulary
        else:
            vocab["selector"] = build_split_dataset(
                cfg.selector_data, "train", cfg.device, tokenizer=None
            ).vocabulary
        tokenizer_solver = "char"  # if not 'listops' in cfg.data.name else 'listops'
        vocab["solver"] = build_split_dataset(
            cfg.solver_data, "train", cfg.device, tokenizer=tokenizer_solver
        ).vocabulary
        vocab["selsolcom"] = datasets["test"].vocabulary

    elif "plot_solver_confidence_score_distribution" in cfg.name:
        vocab = dict()
        test_dataset = build_split_dataset(
            cfg.selsolcom_data, "test", cfg.device, tokenizer=None
        )
        vocab["selector"] = build_split_dataset(
            cfg.selector_data, "train", cfg.device, tokenizer=None
        ).vocabulary
        tokenizer_solver = "char"  # if not 'listops' in cfg.data.name else 'listops'
        vocab["solver"] = build_split_dataset(
            cfg.solver_data, "train", cfg.device, tokenizer=tokenizer_solver
        ).vocabulary
        vocab["selsolcom"] = test_dataset.vocabulary

    else:
        if ("train" in cfg.name) or ("valid" in cfg.name) or ("collect" in cfg.name):
            vocab_split = "train"
        else:
            vocab_split = "test"
        print(f"Building vocab with {vocab_split} dataset split.")
        vocab = datasets[vocab_split].vocabulary
    return vocab
