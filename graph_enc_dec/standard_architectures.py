from torch import nn
from torch import zeros


class ConvAutoencoder(nn.Module):
    def __init__(self, features_enc, kernel_enc, features_dec, kernel_dec,
                 batch_norm=False, act_fn=nn.Tanh(), last_act_fn=nn.Tanh()):
        if features_enc[-1] != features_dec[0]:
            raise RuntimeError('Different dimensions for the latent space')

        super(ConvAutoencoder, self).__init__()
        self.autoenc = nn.Sequential()
        self.fts_enc = features_enc
        self.kernel_enc = kernel_enc
        self.fts_dec = features_dec
        self.kernel_dec = kernel_dec
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.build_network()

    def build_network(self):
        # Encoder section
        for l in range(len(self.fts_enc)-1):
            self.add_layer(nn.Conv1d(self.fts_enc[l], self.fts_enc[l+1],
                           self.kernel_enc, bias=False))
            if self.act_fn is not None:
                self.add_layer(self.act_fn)
            if self.batch_norm:
                self.add_layer(nn.BatchNorm1d(self.fts_enc[l+1]))

        # Decoder Section
        for l in range(len(self.fts_dec)-1):
            self.add_layer(nn.ConvTranspose1d(self.fts_dec[l],
                           self.fts_dec[l+1], self.kernel_dec, bias=False))
            if self.act_fn is not None:
                self.add_layer(self.act_fn)
            if self.batch_norm:
                self.add_layer(nn.BatchNorm1d(self.fts_dec[l+1]))

        if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)

    def add_layer(self, module):
        self.autoenc.add_module(str(len(self.autoenc) + 1), module)

    def forward(self, x):
        return self.autoenc(x)


class FCAutoencoder(nn.Module):
    def __init__(self, nodes_enc, nodes_dec, bias=True,
                 batch_norm=False, act_fn=nn.Tanh(), last_act_fn=nn.Tanh()):
        if nodes_enc[-1] != nodes_dec[0]:
            raise RuntimeError('Different dimensions for the latent space')

        super(FCAutoencoder, self).__init__()
        self.autoenc = nn.Sequential()
        self.nodes_enc = nodes_enc
        self.nodes_dec = nodes_dec
        self.batch_norm = batch_norm
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.build_network()

    def build_network(self):
        # Encoder section
        for l in range(len(self.nodes_enc)-1):
            self.add_layer(nn.Linear(self.nodes_enc[l], self.nodes_enc[l+1],
                           bias=False))
            if self.act_fn is not None:
                self.add_layer(self.act_fn)
            if self.batch_norm:
                self.add_layer(nn.BatchNorm1d(1))

        # Decoder Section
        for l in range(len(self.nodes_dec)-1):
            self.add_layer(nn.Linear(self.nodes_dec[l], self.nodes_dec[l+1],
                           bias=False))
            if self.act_fn is not None:
                self.add_layer(self.act_fn)
            if self.batch_norm:
                self.add_layer(nn.BatchNorm1d(1))

    def add_layer(self, module):
        self.autoenc.add_module(str(len(self.autoenc) + 1), module)

    def forward(self, x):
        return self.autoenc(x)
