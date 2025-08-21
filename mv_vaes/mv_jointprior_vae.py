import torch
import math
from torch import nn
from mv_vaes.mv_vae import MVVAE


class MVJointPriorVAE(MVVAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters()

    def log_additional_values(self, out):
        pass

    def log_additional_values_val(self):
        pass

    def forward(self, batch):
        data = batch[0]

        dists_enc_out = {}
        dists_out = {}
        mods_rec = {}
        for m, key in enumerate(data.keys()):
            # encode views: img_m -> z_m
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            dists_enc_out[key] = [mu_m, lv_m]
            z_m = self.reparametrize(mu_m, lv_m)

            # decode views: z_m -> img_hat_m
            mod_hat_m = self.decoders[m](z_m)
            mods_rec[key] = mod_hat_m

            dist_out_m = [mu_m, lv_m]
            dists_out[key] = dist_out_m
        return (mods_rec, dists_out, dists_enc_out)
      
    def get_latent_representations(self, batch):
        data = batch
        # z_ms = {}
        z_ms = []
        for m, key in enumerate(data.keys()):
            # encode views: img_m -> z_m
            mod_m = data[key]
            mu_m, lv_m = self.encoders[m](mod_m)
            # dists_enc_out[key] = [mu_m, lv_m]
            z_m = self.reparametrize(mu_m, lv_m)
            # z_ms[key] = z_m
            z_ms.append(z_m)
        z = torch.cat(z_ms, dim=1) # [z_m1, z_m2, ...]
        return (z)

    def get_reconstructions(self, mods_out, key, n_samples):
        mod_rec = mods_out[key][0][:n_samples]
        return mod_rec

    def cond_generate_samples(self, m, z):
        mod_c_gen_m_tilde = self.decoders[m](z)
        return mod_c_gen_m_tilde
      
    

    def compute_loss(self, str_set, batch, forward_out):
        imgs, labels = batch
        imgs_rec = forward_out[0]
        dists_out = forward_out[1]

        if self.cfg.model.alpha_annealing:
            init_temp = self.cfg.model.init_alpha_value
            final_temp = self.cfg.model.final_alpha_value
            annealing_steps = self.cfg.model.alpha_annealing_steps
            alpha_weight = self.compute_current_temperature(
                init_temp, final_temp, annealing_steps
            )
        else:
            alpha_weight = self.cfg.model.final_alpha_value
        self.log("alpha annealing", alpha_weight)
        klds = []
        # m = 0
        scalar = 1 + (self.cfg.dataset.num_views - 1) * (1 - self.cfg.model.cov_scalar) / self.cfg.model.cov_scalar
        key_m = self.modality_names[0]
        kld_m = self.kl_div_orthog(dists_out[key_m], 1 / scalar)
        klds.append(kld_m.unsqueeze(1))
        # remaining modalities
        sum_product = torch.zeros_like(dists_out[key_m][0].transpose(0, 1))
        for m, key in enumerate(self.modality_names[1:]):
            dist_m = dists_out[key]
            kld_m = self.kl_div_orthog(dist_m, self.cfg.model.cov_scalar)
            # add N(0,1) as a component
            klds.append(kld_m.unsqueeze(1))
            # can do this putting dist_m instead of identity
            sum_product += self.cov_mat[m](dist_m[0].transpose(0, 1))

        cross_term = torch.squeeze(torch.matmul(dists_out[key_m][0].unsqueeze(1), sum_product.transpose(0, 1).unsqueeze(2))) / self.cfg.model.cov_scalar
        # cov_inv = 
        # add last term
        # SUM_{m}:( alpha * KL(q_m|N(0,1)) + (1-alpha)/M * SUM_{m_tilde}:KL(q_m|q_m_tilde) )
        # when alpha = 0: mixedprior
        # when alpha = 1: unimodal
        # # when alpha = 1/(M+1): mixedpriorstdnorm
        # klds_sum = torch.cat(klds, dim=1).sum(dim=1) / self.cfg.dataset.num_views
        klds_sum = torch.cat(klds, dim=1).sum(dim=1) 
        # klds_sum = klds_sum / self.cfg.dataset.num_views
        klds_term = klds_sum - cross_term - 0.5 * self.cfg.model.latent_dim * math.log(self.cfg.model.cov_scalar)
  

        ## compute reconstruction loss/ conditional log-likelihood out data
        ## given latents
        loss_rec, loss_rec_mods, loss_rec_mods_weighted = self.compute_rec_loss(
            imgs, imgs_rec
        )
        for m, key in enumerate(self.modality_names):
            self.log(
                f"{str_set}/loss/weighted_rec_loss_{key}",
                loss_rec_mods_weighted[key],
            )
            self.log(
                f"{str_set}/loss/rec_loss_{key}",
                loss_rec_mods[key],
            )

        beta = self.cfg.model.beta
        loss_mv_vae = (loss_rec + beta * klds_term).mean(dim=0)
        total_loss = loss_mv_vae
        # logging
        self.log(str_set + "/loss/klds_avg", klds_term.mean(dim=0))
        self.log(str_set + "/loss/loss_rec", loss_rec.mean(dim=0))
        self.log(str_set + "/loss/mv_vae", loss_mv_vae)
        self.log(str_set + "/loss/loss", total_loss)
        return total_loss, loss_rec
