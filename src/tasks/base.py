import hydra
from hydra.core.hydra_config import HydraConfig
import datetime as dt
import logging
import numpy as np
import omegaconf
import os
import torch
import wandb
from utils import mirror_logging_to_console


class BaseTask:

    def __init__(self, model, datasets, cfg):
        self.cfg = cfg
        self.model = model
        self.datasets = datasets
        if (
            ("train" in cfg.name)
            or ("valid" in cfg.name)
            or ("plot" in cfg.name)
            or ("collect" in cfg.name)
        ):
            vocab_split = "train"
        else:
            vocab_split = "test"
        self.vocabulary = datasets[vocab_split].vocabulary
        self.dataloaders = {
            name: torch.utils.data.DataLoader(
                self.datasets[name], collate_fn=lambda l: l[0]
            )
            for name in self.datasets.keys()
        }
        self.FREQ_WANDB_LOG = 0
        self.start_timestamp = dt.datetime.now()
        torch.set_printoptions(linewidth=200, sci_mode=False)
        mirror_logging_to_console()

    def eta(self, it):
        if not "max_iter" in self.cfg.task:
            logging.info("Cannot estimate ETA without max_iter.")
            return
        if it < 500:
            return None
        elapsed_time = (dt.datetime.now() - self.start_timestamp).total_seconds()
        estimated_time_per_iter = elapsed_time / (it + 1)
        remaining_iters = self.cfg.task.max_iter - it + 1
        return (remaining_iters * estimated_time_per_iter) / 3600

    def run(self):
        self._setup_folders()
        self._resolve_model_cfg()
        self._setup_wandb()
        self._load_ckpt()

    def _setup_folders(self):
        if not os.path.exists("../out/"):
            logging.info("out/ folder not found, creating...")
            os.mkdir("../out/")
        if not os.path.exists("../checkpoints/"):
            logging.info("checkpoints/ folder not found, creating...")
            os.mkdir("../checkpoints/")

    def _setup_wandb(self):
        logging.info("Setting up wandb...")

        if self.cfg.task.name == "train":
            self.FREQ_WANDB_LOG = np.ceil(self.cfg.task.max_iter / 500)
        else:
            self.FREQ_WANDB_LOG = 1
        logging.info(f"Logging to W&B every {self.FREQ_WANDB_LOG} iterations.")

        if self.cfg.tags:
            run_tags = self.cfg.tags.split(",")
        else:
            run_tags = None

        if self.cfg.notes:
            run_notes = self.cfg.notes
        else:
            run_notes = None

        if self.cfg.wandb_disabled:
            mode = "disabled"
        else:
            mode = "online"

        wandb.init(
            project=self.cfg.wandb_proj,
            entity="flapetr",
            mode=mode,
            notes=run_notes,
            tags=run_tags,
            settings=wandb.Settings(start_method="fork"),
        )
        if self.cfg.wandb_name:
            wandb.run.name = self.cfg.wandb_name
        self.register_config_to_wandb()
        wandb.watch(self.model, log_freq=self.FREQ_WANDB_LOG)
        logging.info("Done.")

    def register_config_to_wandb(self):
        wandb.config.update(
            omegaconf.OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True
            )
        )

    def _resolve_model_cfg(self):
        if ("ckpt" in self.cfg.model) and (self.cfg.model.ckpt is not None):
            overridden_params = [
                overridden_param.split("=")
                for overridden_param in HydraConfig.get().overrides.task
            ]
            for name, value in overridden_params:
                if name == "model.ckpt":
                    torch_ckpt = torch.load(
                        os.path.join(
                            hydra.utils.get_original_cwd(), f"../checkpoints/{value}"
                        )
                    )
                    if "model_cfg" in torch_ckpt:
                        self.cfg.model = torch_ckpt["model_cfg"]
                        self.cfg.model.ckpt = value
                    else:
                        print("model_cfg was not saved in checkpoint.")
                elif "model" in name:
                    self.cfg.model[name.replace("model.", "")] = value

    def _load_ckpt(self, opt=True):
        if self.cfg.model.ckpt:
            assert self.model is not None
            self.model.load_model_weights(self.cfg.model.ckpt)
            # logging.info('Loading model from checkpoint...')
            # ckpt = torch.load(
            # 	os.path.join(hydra.utils.get_original_cwd(),
            # 		f'../checkpoints/{self.cfg.model.ckpt}'), map_location=self.cfg.device)
            # self.model.load_state_dict(ckpt['model'])
            # logging.info('Done.')
