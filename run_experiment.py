import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch # added for torch.set_float32_matmul_precision
from utils import dataset
from config.UserVariables import folder_path
torch.set_float32_matmul_precision('high')
os.environ["WANDB__SERVICE_WAIT"] = "300"

# experiment configs 
from config.ExperimentConfig import ExperimentConfig
from config.ExperimentConfig import LogConfig
from config.ExperimentConfig import EvalConfig
# model configs
from config.ModelConfig import JointModelConfig
from config.ModelConfig import MixedPriorModelConfig
from config.ModelConfig import UnimodalModelConfig
from config.ModelConfig import JointPriorModelConfig
# dataset configs
from config.DatasetConfig import PMtranslatedData75Config
from config.DatasetConfig import CelebADataConfig
from config.DatasetConfig import scMNCDataConfig
# models
from mv_vaes.mv_joint_vae import MVJointVAE as MVJointVAE
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
cs.store(group="eval", name="eval", node=EvalConfig)
cs.store(group="dataset", name="PMtranslated75", node=PMtranslatedData75Config)
cs.store(group="dataset", name="CelebA", node=CelebADataConfig)
cs.store(group="dataset", name="scMNC", node=scMNCDataConfig)
cs.store(name="base_config", node=ExperimentConfig)

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_experiment(cfg: ExperimentConfig):
    print(cfg)
    
    if cfg.log.wandb_local_instance:
        wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
    elif not cfg.log.wandb_offline:
        wandb.login()
        
    pl.seed_everything(cfg.model.seed, workers=True)

    # get data loaders
    train_loader, train_dst, val_loader, _ = dataset.get_dataset(cfg)
    label_names = train_dst.label_names

    # if using the cpu, use cpu specific classifier directoryß
    if cfg.model.device != "cuda":
        cfg.dataset.dir_clf = cfg.dataset.dir_clf + "_cpu"
    # initialise model
    model = None
    if cfg.model.name == "joint":
        model = MVJointVAE(cfg)
    elif cfg.model.name == "mixedprior":
        model = MVMixedPriorVAE(cfg)
    elif cfg.model.name == "unimodal":
        model = MVunimodalVAE(cfg)
    elif cfg.model.name == "jointprior":
        model = MVJointPriorVAE(cfg)
    assert model is not None
    model.assign_label_names(label_names)
    summary = ModelSummary(model, max_depth=2)
    print(summary)
    filename = f'{cfg.dataset.name}_{cfg.model.name}_agg{cfg.model.aggregation}_alpha{cfg.model.alpha_scalar}_batch_size_{cfg.model.batch_size}_{cfg.model.epochs}_seed{cfg.model.seed}_estop{cfg.model.early_stop}_beta_anneal{cfg.model.beta_annealing}_beta_M{cfg.model.beta_M}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=folder_path + "/runs",
        monitor=cfg.checkpoint_metric,
        mode="max",
        save_last=True,
        filename=filename
    )
    # early stopping specification
    early_stopping = EarlyStopping(monitor="val/loss/loss", mode="min")
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
        callbacks=[checkpoint_callback, early_stopping] if cfg.model.early_stop else [checkpoint_callback],
    )

    if cfg.log.debug:
        trainer.logger.watch(model, log="all")
    # train and evaluate model
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # log metrics
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
