def get_dataset_cub(cfg):
    dir_data = os.path.join(cfg.dataset.dir_data)

    train_dst = CUB(dir_data, train=True)
    val_dst = CUB(dir_data, train=False)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dst,
        batch_size=cfg.model.batch_size_eval,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    return train_loader, train_dst, val_loader, val_dst


if cfg.dataset.name.startswith("CUB"):
            self.train_clf_lr = train_clf_lr_cub
            self.eval_clf_lr = eval_clf_lr_cub
            self.eval_downstream_task = self.eval_downstream_task_cub
            self.calc_coherence = calc_coherence_ap
            self.from_preds_to_clf_metric = from_preds_to_ap
            self.modality_names = ["text", "img"]
            self.betas = {"img": cfg.dataset.beta_img, "text": cfg.dataset.beta_text}
            self.ref_mod_d_size = cfg.dataset.img_size * cfg.dataset.img_size
            self.modalities_size = {
                "img": cfg.dataset.img_size * cfg.dataset.img_size,
                "text": cfg.dataset.len_sequence,
            }

@dataclass
class CUBDataConfig(DataConfig):
    name: str = "CUB"
    num_views: int = 2
    dir_data: str = "/rds/general/user/eso18/home/mmvmvae/data/cub"
    num_labels: int = 6
    dir_clf: str = (
        "/usr/scratch/projects/multimodality/mvvae/experiments/trained_clfs/cub"
    )
    beta_img: float = 1.0
    beta_text: float = 1.0
    len_sequence: int = 32
    img_size: int = 64
    n_clfs_outputs: int = 6
    label_names: List[str] = field(
        default_factory=lambda: [
            "blue2red",
            "brown",
            "grey",
            "yellow",
            "black",
            "white",
        ]
    )
    
    def eval_downstream_task_cub(self, str_ds, clfs, enc_mu_val, labels_val):
        n_labels = labels_val.shape[1]
        scores = torch.zeros((self.cfg.dataset.num_views, n_labels))
        for m, key in enumerate(self.modality_names):
            clf_m = clfs[m]
            enc_mu_m_val = enc_mu_val[key]
            scores_m = self.eval_clf_lr(
                clf_m,
                enc_mu_m_val,
                labels_val,
            )
            scores[m, :] = scores_m
            self.log("val/downstream/" + str_ds + "/" + key, scores_m.mean())
            for k, l_name in enumerate(self.label_names):
                self.log(f"val/downstream/{str_ds}/{key}/{l_name}", scores_m[k])
        return scores