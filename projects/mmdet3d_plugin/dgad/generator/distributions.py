import torch
import torch.nn as nn

from mmdet.models import LOSSES


class DistributionModule(nn.Module):
    """Encode per-agent/per-mode state tokens into codebook tokens."""

    def __init__(
        self, in_channels, in_channels_2, latent_dim, codebook_dim=None,
        codebook_num_vectors=None, hidden_channels=256,
        token_hidden_channels=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_dim = codebook_dim or latent_dim
        self.codebook_num_vectors = codebook_num_vectors or int(latent_dim * 2)
        self.in_channels = in_channels
        self.num_input_tokens = in_channels_2
        self.hidden_channels = hidden_channels
        self.token_hidden_channels = token_hidden_channels

        self.token_encoder = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.token_compressor = nn.Sequential(
            nn.Linear(self.num_input_tokens, token_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(token_hidden_channels, self.codebook_num_vectors),
        )
        self.to_code = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, self.codebook_dim),
        )

    def forward(self, s_t):
        if s_t.shape[-1] != self.in_channels:
            raise ValueError(
                f'Expected input feature dim {self.in_channels}, '
                f'but got {s_t.shape[-1]}.')
        if s_t.shape[1] != self.num_input_tokens:
            raise ValueError(
                f'Expected {self.num_input_tokens} input tokens, '
                f'but got {s_t.shape[1]}.')

        encoding = self.token_encoder(s_t)
        encoding = self.token_compressor(
            encoding.transpose(1, 2)).transpose(1, 2)
        return self.to_code(encoding)


class SampleDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=256,
        codebook_dim=32,
        token_hidden_channels=512,
    ):
        super().__init__()
        self.num_code_tokens = in_channels
        self.num_output_tokens = out_channels
        self.hidden_channels = hidden_channels
        self.codebook_dim = codebook_dim
        self.token_hidden_channels = token_hidden_channels

        self.decoder = nn.Sequential(
            nn.LayerNorm(codebook_dim),
            nn.Linear(codebook_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, codebook_dim),
        )
        self.token_expander = nn.Sequential(
            nn.Linear(self.num_code_tokens, token_hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(token_hidden_channels, self.num_output_tokens),
        )

    def forward(self, z):
        if z.shape[1] != self.num_code_tokens:
            raise ValueError(
                f'Expected {self.num_code_tokens} code tokens, '
                f'but got {z.shape[1]}.')
        if z.shape[-1] != self.codebook_dim:
            raise ValueError(
                f'Expected codebook dim {self.codebook_dim}, '
                f'but got {z.shape[-1]}.')

        z = self.decoder(z)
        z = self.token_expander(z.transpose(1, 2)).transpose(1, 2)
        return z


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding_1 = nn.Embedding(self.n_e, self.e_dim)
        self.embedding_1.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.embedding_2 = nn.Embedding(self.n_e, self.e_dim)
        self.embedding_2.weight.data.copy_(self.embedding_1.weight.data)

    def forward(self, present_z, future_z):
        def quantize_and_loss(z, embedding):
            if z.shape[-1] != self.e_dim:
                raise ValueError(
                    f'Expected codebook input channel dim {self.e_dim}, '
                    f'but got {z.shape[-1]}.')
            z_flattened = z.reshape(-1, self.e_dim)

            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding.weight ** 2, dim=1) - 2 * \
                torch.matmul(z_flattened, embedding.weight.t())

            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            min_encodings = z.new_zeros(
                min_encoding_indices.shape[0], self.n_e)
            min_encodings.scatter_(1, min_encoding_indices, 1)

            z_q = torch.matmul(min_encodings, embedding.weight).view(z.shape)
            loss = torch.mean((z.detach() - z_q) ** 2)
            loss += self.beta * torch.mean((z - z_q.detach()) ** 2)
            z_q = z + (z_q - z).detach()

            return z_q, loss

        z_q_present, loss_present = quantize_and_loss(
            present_z, self.embedding_1)
        if future_z is not None:
            z_q_future, loss_future = quantize_and_loss(
                future_z, self.embedding_2)
        else:
            z_q_future, loss_future = None, 0

        loss = loss_present + loss_future

        return (
            z_q_present,
            z_q_future,
            loss,
            self.embedding_1.weight,
            self.embedding_2.weight,
        )


class PredictModel(nn.Module):
    """Predict future states with a GRU."""

    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
        )
        self.linear1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.linear2 = nn.Linear(hidden_channels * 2, hidden_channels * 4)
        self.linear3 = nn.Linear(hidden_channels * 4, out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x, h):
        x, h = self.gru(x, h)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


@LOSSES.register_module()
class ProbabilisticLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output):
        return output['vq_loss'] * self.loss_weight
