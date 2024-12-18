import logging
import torch
import pandas as pd
from data.vocabulary import Vocabulary


class ItersolvDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        dataset_name,
        split,
        train_batch_size,
        eval_batch_size,
        device,
        sos,
        eos,
        tokenizer=None,
        difficulty_split=None,
        specials_in_x=False,
        specials_in_y=True,
        task_name=None,
    ):
        self.dataset_name = dataset_name
        self.set_tokenizer(tokenizer)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.split = split
        self.device = device
        self.specials_in_x = specials_in_x
        self.specials_in_y = specials_in_y
        self.sos = sos
        self.eos = eos
        self.difficulty_split = difficulty_split
        self.task_name = task_name
        self.vocab_tokens_x = None
        self.vocab_tokens_y = None
        self._build_dataset_df(dataset_name, split)
        self._build_vocabulary()
        self._slice_by_task_name()
        self._slice_difficulty_split()

    def __iter__(self):
        return self._generate_dict()

    def __len__(self):
        return len(self.df)

    def set_tokenizer(self, tokenizer):
        if tokenizer is None:
            if "listops" in self.dataset_name:
                self.tokenizer = "listops"
            elif (
                "arithmetic" in self.dataset_name
                and "text_segmentation" in self.dataset_name
            ):
                self.tokenizer = "arithmetic_textseg"
            elif "arithmetic" in self.dataset_name:
                self.tokenizer = "arithmetic"
            elif (
                "algebra" in self.dataset_name
                and "text_segmentation" in self.dataset_name
            ):
                self.tokenizer = "algebra_textseg"
            elif "algebra" in self.dataset_name:
                self.tokenizer = "algebra"
            elif (
                "alltask" in self.dataset_name
                and "text_segmentation" in self.dataset_name
            ):
                self.tokenizer = "alltask_textseg"
            elif "alltask" in self.dataset_name:
                self.tokenizer = "alltask"
            else:
                self.tokenizer = "char"
        else:
            self.tokenizer = tokenizer

    def _build_dataset_df(self, dataset_name, split):
        logging.info(f"Loading dataset {dataset_name}, split {split}...")
        self.df = pd.read_csv(
            f"../datasets/{dataset_name}/{split}.csv", dtype={"X": str, "Y": str}
        )

    def _slice_by_task_name(self):
        if self.task_name is not None:
            logging.info(f"Slicing task {self.task_name}")
            self.df = self.df.loc[self.df["task"] == self.task_name]

    def _slice_difficulty_split(self):
        if self.difficulty_split is not None:
            logging.info(f"Slicing difficulty split: {self.difficulty_split}")
            nesting, num_operands = self.difficulty_split
            self.df = self.df.loc[
                (self.df["nesting"] == nesting)
                & (self.df["num_operands"] == num_operands)
            ]
        logging.info(f"{len(self.df)} total samples in {self.split} split.")

    def _build_vocabulary(self):
        x_vocab_tokens, y_vocab_tokens = self.get_vocab_tokens()
        if "text_segmentation" in self.dataset_name:
            y_vocab_tokens = ["0", "1"]
        self.vocabulary = Vocabulary(
            x_vocab_tokens,
            y_vocab_tokens,
            self.device,
            self.sos,
            self.eos,
            self.specials_in_y,
            self.specials_in_x,
            tokenizer=self.tokenizer,
        )

    def set_vocabulary(self, vocabulary, only_x=False):
        if only_x:
            old_vocab_y_tokens = self.vocabulary.y_vocab.get_itos()
            new_vocab_x_tokens = vocabulary.x_vocab.get_itos()
            self.vocabulary = Vocabulary(
                new_vocab_x_tokens,
                old_vocab_y_tokens,
                self.device,
                vocabulary.sos,
                vocabulary.eos,
                vocabulary.specials_in_y,
                vocabulary.specials_in_x,
                tokenizer=vocabulary.tokenizer,
            )
        else:
            self.vocabulary = vocabulary

    def get_vocab_tokens(self):
        if self.vocab_tokens_x is not None and self.vocab_tokens_y is not None:
            return self.vocab_tokens_x, self.vocab_tokens_y

        if self.tokenizer == "listops":
            return self._get_vocab_tokens_listops()
        elif self.tokenizer == "arithmetic":
            return self._get_vocab_tokens_arithmetic()
        elif self.tokenizer == "arithmetic_textseg":
            return self._get_vocab_tokens_arithmetic()
        elif self.tokenizer == "algebra":
            return self._get_vocab_tokens_algebra()
        elif self.tokenizer == "algebra_textseg":
            return self._get_vocab_tokens_algebra()
        elif self.tokenizer == "alltask_textseg":
            return self._get_vocab_tokens_alltask()
        elif self.tokenizer == "alltask":
            return self._get_vocab_tokens_alltask()
        elif self.tokenizer == "char":
            return self._get_vocabs_chars()

    def _get_vocab_tokens_listops(self):
        x_tokens_sets = (
            self.df["X"].apply(Vocabulary._tokenize_listops).apply(lambda s: set(s))
        )
        y_tokens_sets = (
            self.df["Y"].apply(Vocabulary._tokenize_listops).apply(lambda s: set(s))
        )
        self.vocab_tokens_x, self.vocab_tokens_y = self._build_vocab_tokens_lists(
            x_tokens_sets, y_tokens_sets
        )
        return self.vocab_tokens_x, self.vocab_tokens_y

    def _get_vocab_tokens_arithmetic(self):
        x_tokenized = self.df["X"].apply(Vocabulary._tokenize_arithmetic)
        x_tokens_sets = x_tokenized.apply(lambda s: set(s))
        del x_tokenized
        y_tokenized = self.df["Y"].apply(Vocabulary._tokenize_arithmetic)
        y_tokens_sets = y_tokenized.apply(lambda s: set(s))
        del y_tokenized
        self.vocab_tokens_x, self.vocab_tokens_y = self._build_vocab_tokens_lists(
            x_tokens_sets, y_tokens_sets
        )
        return self.vocab_tokens_x, self.vocab_tokens_y

    def _get_vocab_tokens_algebra(self):
        x_tokens_sets = (
            self.df["X"].apply(Vocabulary._tokenize_algebra).apply(lambda s: set(s))
        )
        y_tokens_sets = (
            self.df["Y"].apply(Vocabulary._tokenize_algebra).apply(lambda s: set(s))
        )
        self.vocab_tokens_x, self.vocab_tokens_y = self._build_vocab_tokens_lists(
            x_tokens_sets, y_tokens_sets
        )
        return self.vocab_tokens_x, self.vocab_tokens_y

    def _get_vocab_tokens_alltask(self):
        x_tokenized = self.df["X"].apply(Vocabulary._tokenize_alltask)
        x_tokens_sets = x_tokenized.apply(lambda s: set(s))
        y_tokenized = self.df["Y"].apply(Vocabulary._tokenize_alltask)
        y_tokens_sets = y_tokenized.apply(lambda s: set(s))
        self.vocab_tokens_x, self.vocab_tokens_y = self._build_vocab_tokens_lists(
            x_tokens_sets, y_tokens_sets
        )
        return self.vocab_tokens_x, self.vocab_tokens_y

    def _get_vocabs_chars(self):
        x_chars_sets = self.df["X"].apply(lambda s: set(s))
        y_chars_sets = self.df["Y"].apply(lambda s: set(s))
        self.vocab_tokens_x, self.vocab_tokens_y = self._build_vocab_tokens_lists(
            x_chars_sets, y_chars_sets
        )
        return self.vocab_tokens_x, self.vocab_tokens_y

    @staticmethod
    def _build_vocab_tokens_lists(x_tokens_sets, y_tokens_sets):
        x_vocab_tokens = set()
        for token_set in x_tokens_sets:
            x_vocab_tokens |= token_set

        y_vocab_tokens = set()
        for token_set in y_tokens_sets:
            y_vocab_tokens |= token_set

        x_vocab_tokens_list = sorted(list(x_vocab_tokens))
        y_vocab_tokens_list = sorted(list(y_vocab_tokens))

        return x_vocab_tokens_list, y_vocab_tokens_list

    @property
    def batch_size(self):
        if self.split == "train":
            return self.train_batch_size
        else:
            return self.eval_batch_size

    def _generate_dict(self):
        def _continue():
            if self.split == "train":
                return True
            else:
                max_iter = len(self.df) // self.eval_batch_size
                if len(self.df) % self.eval_batch_size != 0:
                    max_iter += 1
                self.curr_iter += 1
                return self.curr_iter <= max_iter

        self.curr_iter = 0

        while _continue():
            batch_df = self.get_batch(self.batch_size)
            X, Y = (
                batch_df["X"].astype(str).tolist(),
                batch_df["Y"].astype(str).tolist(),
            )
            token_X, token_Y = self.vocabulary.str_to_batch(
                X
            ), self.vocabulary.str_to_batch(Y, x=False)
            yield token_X, token_Y

    def get_batch(self, batch_size):
        if self.split == "train":
            return self.df.sample(n=batch_size)
        else:
            return self.df[
                (self.curr_iter - 1) * batch_size : self.curr_iter * batch_size
            ]


class RegressionDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset_name,
        split,
        train_batch_size,
        eval_batch_size,
        device,
        task_name=None,
    ):
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.split = split
        self.device = device
        self.task_name = task_name
        self._build_dataset_df(dataset_name, split)
        self._build_vocabulary()

    def __iter__(self):
        return self._generate_dict()

    def __len__(self):
        return len(self.df)

    def _build_dataset_df(self, dataset_name, split):
        logging.info(f"Loading dataset {dataset_name}, split {split}...")
        self.df = pd.read_csv(f"../datasets/{dataset_name}/{split}.csv")
        self.df["X"] = self.df["X"].astype("str")
        self.df["y_regr"] = self.df["y_regr"].astype("int")

    def _build_vocabulary(self):
        x_vocab_tokens = self.get_vocab_tokens()
        tokenizer = "listops" if "listops" in self.dataset_name else "char"
        self.vocabulary = Vocabulary(
            x_vocab_tokens, [], self.device, False, False, False, tokenizer=tokenizer
        )

    def get_vocab_tokens(self):
        if "listops" in self.dataset_name:
            return self._get_vocab_tokens_listops()
        else:
            return self._get_vocabs_chars()

    def _get_vocab_tokens_listops(self):
        x_tokens_sets = (
            self.df["X"].apply(Vocabulary._tokenize_listops).apply(lambda s: set(s))
        )
        return self._build_vocab_tokens_lists(x_tokens_sets)

    def _get_vocabs_chars(self):
        x_chars_sets = self.df["X"].apply(lambda s: set(s))
        return self._build_vocab_tokens_lists(x_chars_sets)

    @staticmethod
    def _build_vocab_tokens_lists(x_tokens_sets):
        x_vocab_tokens = set()
        for token_set in x_tokens_sets:
            x_vocab_tokens |= token_set

        x_vocab_tokens_list = sorted(list(x_vocab_tokens))

        return x_vocab_tokens_list

    @property
    def batch_size(self):
        if self.split == "train":
            return self.train_batch_size
        else:
            return self.eval_batch_size

    def _generate_dict(self):
        def _continue():
            if self.split == "train":
                return True
            else:
                self.curr_iter += 1
                return self.curr_iter <= len(self.df) // self.eval_batch_size

        self.curr_iter = 0

        while _continue():
            batch_df = self.get_batch(self.batch_size)
            X, Y = (
                batch_df["X"].astype(str).tolist(),
                batch_df["y_regr"].astype(float).tolist(),
            )
            token_X, token_Y = self.vocabulary.str_to_batch(X), torch.tensor(
                Y, device=self.device
            )
            yield token_X, token_Y

    def get_batch(self, batch_size):
        if self.split == "train":
            return self.df.sample(n=batch_size)
        else:
            return self.df[
                (self.curr_iter - 1) * batch_size : self.curr_iter * batch_size
            ]
