import pandas as pd
from matplotlib import pyplot as plt

from tasks.base import BaseTask
from tqdm import tqdm


class PlotSolverConfidenceScoreDistribution(BaseTask):

    def __init__(self, model, datasets, cfg):
        datasets["train"].df = datasets["train"].df.drop_duplicates(
            subset="X", keep=False
        )
        super().__init__(model, datasets, cfg)

    def _load_ckpt(self, opt=True):
        assert self.model is not None

        if self.cfg.model.solver.ckpt:
            self.model.solver.load_model_weights(self.cfg.model.solver.ckpt)

    def run(self):
        TITLE_FONTSIZE = 25
        AXIS_LABEL_FONTSIZE = 32
        AXIS_TICKS_FONTSIZE = 27
        LEGEND_FONTSIZE = 29

        super().run()
        task_name = self.cfg.name.split("_")[-1]
        confidence_scores = []
        total_it = len(self.dataloaders["train"]) // self.cfg.data.eval_batch_size
        self.dataloaders["train"].dataset.split = "test"  # to trick the dataset

        for X, Y in tqdm(iter(self.dataloaders["train"]), total=total_it):
            solver_outputs = self.model._solver_multi_run_hq(X)
            confidence_scores += [o.confidence_score for o in solver_outputs]
        confidence_scores_df = pd.DataFrame(confidence_scores)
        print(confidence_scores_df.describe())
        # df_no_outliers = confidence_scores_df[
        #     confidence_scores_df[0] > confidence_scores_df[0].quantile(0.01)
        # ]
        ax = confidence_scores_df.plot(
            kind="hist",
            # title=f"Confidence Score Distribution {task_name.capitalize()}",
            bins=100,
        )
        if task_name == "logic":
            ax.set_ylabel("Confidence Score", fontsize=AXIS_LABEL_FONTSIZE)
            plt.xticks(fontsize=AXIS_TICKS_FONTSIZE, rotation=45)
            # ax.set_xticks(
            #     plt.gca().get_xticks(),
            #     [tick.get_text() for tick in plt.gca().get_xticklabels()],
            #     rotation=45,
            #     fontsize=AXIS_TICKS_FONTSIZE,
            # )
        else:
            plt.xticks(fontsize=AXIS_TICKS_FONTSIZE, rotation=45)
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.legend().set_visible(False)

        plt.yticks(fontsize=AXIS_TICKS_FONTSIZE)

        plt.yscale("log")
        plt.savefig(
            f"../out/plots/conf_scores_hist_{task_name}.pdf", bbox_inches="tight"
        )
