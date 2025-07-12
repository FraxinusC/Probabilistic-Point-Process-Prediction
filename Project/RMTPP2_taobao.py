# %%
import torch
from torch import nn
import math
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.utils import Registrable


class RMTPP2(TorchBaseModel, Registrable):
    @Registrable.register("RMTPP2") 
    def __init__(self, model_config):
        super(RMTPP2, self).__init__(model_config)
        
        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)
        self.layer_rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                               num_layers=1, batch_first=True)
        
        self.hidden_to_intensity_mean = nn.Linear(self.hidden_size, self.num_event_types)
        self.hidden_to_intensity_logvar = nn.Linear(self.hidden_size, self.num_event_types)
        
        self.prior_mean = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.prior_logvar = nn.Parameter(torch.zeros(1, self.num_event_types))
        
        self.b_t_mu = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.w_t_mu = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.b_t_sigma = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.w_t_sigma = nn.Parameter(torch.zeros(1, self.num_event_types))
        nn.init.xavier_normal_(self.b_t_mu)
        nn.init.xavier_normal_(self.w_t_mu)
        nn.init.xavier_normal_(self.b_t_sigma)
        nn.init.xavier_normal_(self.w_t_sigma)

    def evolve_and_get_intentsity(self, right_hiddens_BNH, dts_BNG):
        """Get mean and logvar of intensity."""
        hidden_proj = right_hiddens_BNH[..., None, :]
        mean = self.hidden_to_intensity_mean(hidden_proj)
        logvar = self.hidden_to_intensity_logvar(hidden_proj)

        mean = mean + self.w_t_mu[None, None, :] * dts_BNG[..., None] + self.b_t_mu[None, None, :]
        logvar = logvar + self.w_t_sigma[None, None, :] * dts_BNG[..., None] + self.b_t_sigma[None, None, :]
        logvar = logvar.clamp(min=-10, max=10)
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=10)
        eps = torch.randn_like(std)
        intensity_BNGM = torch.exp(mean + eps * std).clamp(max=1e5)
        
        return intensity_BNGM

    def forward(self, batch):
        t_BN, dt_BN, marks_BN, _, _ = batch
        mark_emb_BNH = self.layer_type_emb(marks_BN)
        time_emb_BNH = self.layer_temporal_emb(t_BN[..., None])
        right_hiddens_BNH, _ = self.layer_rnn(mark_emb_BNH + time_emb_BNH)
        
        left_intensity_B_Nm1_M = self.evolve_and_get_intentsity(right_hiddens_BNH[:, :-1, :], dt_BN[:, 1:][...,None]).squeeze(-2)
        
        return left_intensity_B_Nm1_M, right_hiddens_BNH


    def loglike_loss(self, batch):
        """Compute the log-likelihood loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch
        left_intensity_B_Nm1_M, right_hiddens_BNH = self.forward((ts_BN, dts_BN, marks_BN, None, None))
        right_hiddens_B_Nm1_H = right_hiddens_BNH[..., :-1, :]

        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])
        intensity_dts_B_Nm1_G_M = self.evolve_and_get_intentsity(right_hiddens_B_Nm1_H, dts_sample_B_Nm1_G)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=left_intensity_B_Nm1_M,
            lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
            time_delta_seq=dts_BN[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:]
        )
        
        loss = -(event_ll - non_event_ll).sum()
        
        return loss, num_events

    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        _input = time_seqs, time_delta_seqs, type_seqs, None, None
        _, right_hiddens_BNH = self.forward(_input)

        if compute_last_step_only:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH[:, -1:, :], sample_dtimes[:, -1:, :])
        else:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH, sample_dtimes)  # shape: [B, N, G, M]
        return sampled_intensities







# %%
from easy_tpp.utils import Registrable

available_models = Registrable.list_available()
print("Registered models:", available_models)



# %%
import argparse
from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_dir', type=str, required=False, default='configs/experiment_config.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')

    parser.add_argument('--experiment_id', type=str, required=False, default='RMTPP2_gen_taobao',
                        help='Experiment id in the config file.')

    args = parser.parse_args()

  
    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)


    model_runner = Runner.build_from_config(config)

    model_runner.run()

if __name__ == '__main__':
    main()

