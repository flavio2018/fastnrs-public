import copy

import pandas as pd
import torch

from tasks.base import BaseTask
from models.selsolcom import SelSolCom
from data.vocabulary import Vocabulary
from tqdm import tqdm


class CollectSelectorConfidenceScores(BaseTask):

    def __init__(self, model, datasets, cfg):
        datasets["valid_iid"].df = datasets["valid_iid"].df.drop_duplicates(
            subset="X", keep="first"
        )
        datasets["valid_ood"].df = datasets["valid_ood"].df.drop_duplicates(
            subset="X", keep="first"
        )
        super().__init__(model, datasets, cfg)

    def _load_ckpt(self, opt=True):
        assert self.model is not None

        if self.cfg.model.ckpt:
            self.model.load_model_weights(self.cfg.model.ckpt)

    def run(self):
        super().run()

        task_name = self.cfg.name.split("_")[-1]

        if "alltask" in self.cfg.data.name:
            task_names = [
                "alltask_algebra",
                "alltask_arithmetic",
                "alltask_listops",
                "alltask_logic",
            ]
        else:
            task_names = [task_name]

        for task_name in task_names:
            global_confidence_scores = []
            global_expression_lengths = []
            for split_name in ["valid_iid", "valid_ood"]:
                if "alltask" in self.cfg.data.name:
                    ds = copy.deepcopy(self.dataloaders[split_name].dataset)
                    df = ds.df
                    df["task_name"] = df["X"].apply(
                        lambda x: Vocabulary.task_name_from_sample(x)
                    )
                    df = df[df["task_name"] == task_name.split("_")[1]]
                    df.reset_index(drop=True, inplace=True)
                    ds.df = df
                    dl = torch.utils.data.DataLoader(
                        dataset=ds, collate_fn=lambda l: l[0]
                    )
                else:
                    dl = self.dataloaders[split_name]
                total_it = len(dl) // self.cfg.data.eval_batch_size
                for X, Y in tqdm(iter(dl), total=total_it):
                    expression_lengths = (
                        X != self.model.vocabulary.get_special_idx("pad")
                    ).sum(1)
                    transformer_test_output = self.model(X)
                    proba_confidence_scores = SelSolCom._get_confidence_score(
                        transformer_test_output, use_proba=True
                    )
                    global_confidence_scores += proba_confidence_scores.tolist()
                    global_expression_lengths += expression_lengths.tolist()
            df = pd.DataFrame(
                {
                    "input_length": global_expression_lengths,
                    "confidence_score": global_confidence_scores,
                }
            )
            df.to_csv(f"../out/new_confidence_scores_df_{task_name}.csv")
