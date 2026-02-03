import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch # added for torch.set_float32_matmul_precision

from utils import dataset

from config.MyMVWSLConfig import MyMVWSLConfig
from config.MyMVWSLConfig import LogConfig
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.ModelConfig import SplitModelConfig
from config.ModelConfig import JointPriorModelConfig
from config.DatasetConfig import PMtranslatedData75Config
from config.DatasetConfig import CelebADataConfig
from config.DatasetConfig import scMNCDataConfig
from config.MyMVWSLConfig import EvalConfig

from mv_vaes.mv_joint_vae import MVJointVAE as MVJointVAE
from mv_vaes.mv_split_vae import MVSplitVAE as MVSplitVAE
from mv_vaes.mv_unimodal_vae import MVunimodalVAE as MVunimodalVAE
from mv_vaes.mv_mixedprior_vae import MVMixedPriorVAE as MVMixedPriorVAE
from mv_vaes.mv_jointprior_vae import MVJointPriorVAE as MVJointPriorVAE

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(group="log", name="log", node=LogConfig)
cs.store(group="model", name="joint", node=JointModelConfig)
cs.store(group="model", name="mixedprior", node=MixedPriorModelConfig)
cs.store(group="model", name="jointprior", node=JointPriorModelConfig)
cs.store(group="model", name="unimodal", node=UnimodalModelConfig)
cs.store(group="model", name="split", node=SplitModelConfig)
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="PMtranslated75", node=PMtranslatedData75Config)
cs.store(group="dataset", name="CelebA", node=CelebADataConfig)
cs.store(group="dataset", name="scMNC", node=scMNCDataConfig)
cs.store(name="base_config", node=MyMVWSLConfig)

torch.set_float32_matmul_precision('high')

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_experiment(cfg: MyMVWSLConfig):
    print(cfg)
    
    # WANDB_API_KEY=$YOUR_API_KEY
    if cfg.log.wandb_local_instance:
        wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
    elif not cfg.log.wandb_offline:
        # wandb.login(host="https://api.wandb.ai")
        wandb.login()
        
    pl.seed_everything(cfg.model.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, _ = dataset.get_dataset(cfg)
    label_names = train_dst.label_names

    # init model
    model = None
    if cfg.model.name == "joint":
        model = MVJointVAE(cfg)
    elif cfg.model.name == "mixedprior":
        model = MVMixedPriorVAE(cfg)
    elif cfg.model.name == "unimodal":
        model = MVunimodalVAE(cfg)
    elif cfg.model.name == "split":
        model = MVSplitVAE(cfg)
    elif cfg.model.name == "jointprior":
        model = MVJointPriorVAE(cfg)
    assert model is not None
    model.assign_label_names(label_names)
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    early_stopping = EarlyStopping(monitor="val/loss/loss_rec", mode="min")

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    wandb_logger = WandbLogger(
        name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu" if cfg.model.device == "cuda" else cfg.model.device,
        logger=wandb_logger,
        check_val_every_n_epoch=cfg.log.val_freq,
        deterministic=True,
        callbacks=[early_stopping] if cfg.model.early_stop else [],
    )

    if cfg.log.debug:
        trainer.logger.watch(model, log="all")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model.logger.log_metrics({"final_scores/rec_loss": model.final_scores_rec_loss})
    model.logger.log_metrics(
        {"final_scores/cond_rec_loss": model.final_scores_cond_rec_loss}
    )
    model.logger.log_metrics(
      {"final_scores/cond_rec_loss_cov": model.final_scores_cond_rec_loss_cov }
    )
      
    
    for m, key in enumerate(model.modality_names):
        model.logger.log_metrics(
            {
                f"final_scores/downstream_lr/aggregated/{key}": model.final_scores_lr_aggregated[
                    m
                ]
            }
        )
        model.logger.log_metrics(
            {
                f"final_scores/downstream_lr/unimodal/{key}": model.final_scores_lr_unimodal[
                    m
                ]
            }
        )
        if cfg.dataset.name == "celeba":
            for k, l_name in enumerate(label_names):
                model.logger.log_metrics(
                    {
                        f"final_scores/downstream_lr/aggregated/{key}/{l_name}": model.final_scores_lr_aggregated_alllabels[
                            m, k
                        ]
                    }
                )
                model.logger.log_metrics(
                    {
                        f"final_scores/downstream_lr/unimodal/{key}/{l_name}": model.final_scores_lr_unimodal_alllabels[
                            m, k
                        ]
                    }
                )

    for m, key in enumerate(model.modality_names):
        for m_tilde, key_tilde in enumerate(model.modality_names):
            model.logger.log_metrics(
                {
                    f"final_scores/coherence/{key}_to_{key_tilde}": model.final_scores_coh[
                        m, m_tilde, :
                    ].mean()
                }
            )
            model.logger.log_metrics(
                {
                    f"final_scores/coherence/{key}_to_{key_tilde}_cov": model.final_scores_coh_cov[
                        m, m_tilde, :
                    ].mean()
                }
            )


if __name__ == "__main__":
    run_experiment()
