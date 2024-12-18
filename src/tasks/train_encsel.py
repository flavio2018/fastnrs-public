import datetime as dt
from tasks.train import TrainTask


class TrainEncoderSelectorTask(TrainTask):

    def __init__(self, model, datasets, cfg):
        super(TrainEncoderSelectorTask, self).__init__(model, datasets, cfg)

    def train(self):
        for it in range(self.cfg.task.max_iter):
            self.train_step()

            if it % self.FREQ_WANDB_LOG == 0:
                self.valid_step(it)
                self.serialize_state(
                    self.valid_step_metrics[
                        "metrics/" + self.cfg.task.early_stop_metric
                    ],
                    it,
                )
                if self.use_early_stopping and self.early_stopping.early_stop:
                    return
                self.reset_metrics_dict()

    def train_step(self):
        self.start_train_step_timestamp = dt.datetime.now()
        self.model.train()
        self.opt.zero_grad()
        X, Y = next(iter(self.dataloaders["train"]))
        outputs = self.model(X, Y=None, tf=self.tf)
        acc, std = self.batch_acc(outputs, Y)
        seq_acc, std = self.batch_seq_acc(outputs, Y)
        loss = self.compute_loss(outputs, Y)
        loss.backward()
        self.opt.step()
        if self.cfg.task.lr_scheduler is not None:
            self.scheduler.step()
        self.valid_step_metrics["metrics/train/loss"] = loss.item()
        self.valid_step_metrics["metrics/train/char_acc"] = acc.item()
        self.valid_step_metrics["metrics/train/seq_acc"] = seq_acc.item()
        self.end_train_step_timestamp = dt.datetime.now()
