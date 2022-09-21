'''
Modified based on https://github.com/wengong-jin/icml18-jtnn.git.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnutils import create_var, flatten_tensor, avg_pool
from .jtnn_enc import JTNNEncoder
from .jtnn_dec import JTNNDecoder
from .params import VOCAB_SIZE, CONTACT_SIZE
import copy
from .jtnn_pred import JTNNPredictor

vocab_size = VOCAB_SIZE
class JTNNVAE(nn.Module):

    def __init__(self, hidden_size, latent_size, depthT, encoding, pred_prop):
        super(JTNNVAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size #Tree and Mol has two vectors
        self.pred_prop = pred_prop

        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab_size, hidden_size), encoding)
        self.decoder = JTNNDecoder(hidden_size, latent_size, nn.Embedding(vocab_size, hidden_size))
        if pred_prop:
            self.predictor = JTNNPredictor(latent_size, CONTACT_SIZE, n_hidden_layers=2, hidden_layer_size=128)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)


    def encode(self, jtenc_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        return tree_vecs, tree_mess # , mol_vecs

    def encode_latent(self, x_batch):
        jtenc_holder = x_batch[1]
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        tree_mean = self.T_mean(tree_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        return tree_mean, tree_var

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, prob_decode)

    def contact_pred(self, z_vecs, W_mean, loc_batch):
        z_mean = W_mean(z_vecs)
        pred_loss = self.predictor(z_mean, loc_batch)
        return pred_loss

    # alpha controls the word_loss proportion
    def forward(self, x_batch, loc_batch, beta, alpha, gamma):
        x_batch, x_jtenc_holder = x_batch
        x_tree_vecs, x_tree_mess = self.encode(x_jtenc_holder)
        z_tree_vecs, kl_div = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        if self.pred_prop:
            pred_loss = self.contact_pred(x_tree_vecs, self.T_mean, loc_batch)
        else:
            pred_loss = torch.tensor(0.0)
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        return alpha * word_loss + topo_loss + beta * kl_div + gamma * pred_loss, kl_div.item(), \
               word_acc, topo_acc, pred_loss.item()

    def decode(self, x_tree_vecs, prob_decode):
        assert x_tree_vecs.size(0) == 1
        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        return pred_root, pred_nodes


