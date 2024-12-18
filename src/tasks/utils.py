from models.utils import build_model
from data.utils import build_datasets, get_vocab
from tasks.train import TrainTask
from tasks.train_regr import TrainRegrTask
from tasks.test import TestSSCTask, TestOnValid
from tasks.train_encsel import TrainEncoderSelectorTask

from tasks.plot_solver_confidence_score_distribution import (
    PlotSolverConfidenceScoreDistribution,
)
from tasks.collect_nrs_selector_conf_scores import CollectSelectorConfidenceScores


def build_task(cfg):
    datasets = build_datasets(cfg)
    vocab = get_vocab(cfg, datasets)
    model = build_model(cfg, vocab)

    if "regr" in cfg.name:
        task = TrainRegrTask(model, datasets, cfg)

    elif cfg.name.startswith("collect_nrs_selector_conf_scores"):
        task = CollectSelectorConfidenceScores(model, datasets, cfg)

    elif cfg.name.startswith("plot_solver_confidence_score_distribution"):
        task = PlotSolverConfidenceScoreDistribution(model, datasets, cfg)

    elif cfg.task.name == "train_encsel":
        task = TrainEncoderSelectorTask(model, datasets, cfg)

    elif "train" in cfg.name:
        task = TrainTask(model, datasets, cfg)

    elif cfg.task.name == "test_valid":
        task = TestOnValid(model, datasets, cfg)

    elif "test" in cfg.name:
        task = TestSSCTask(model, datasets, cfg)

    return task
