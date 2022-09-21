'''
For training the generative model that encodes designs
example:
python vae_train.py --save_dir sum_ls28_pred20 --data_dir new_train_data_loc_merge --gamma 20
python vae_train.py --save_dir random_ls28 --data_dir random_data --gamma 0 --pred
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os

import math, sys
import numpy as np
import argparse
from fast_jtnn import *

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=28)
parser.add_argument('--depthT', type=int, default=20)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=400000)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--encode', type=str, default="sum")
parser.add_argument('--pred', default=True, action="store_false")


args = parser.parse_args()
print(args)


model = JTNNVAE(args.hidden_size, args.latent_size, args.depthT, args.encode, args.pred).cuda()
print(model)


for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model_path = os.path.join(args.save_dir, "model.iter-" + str(args.load_epoch))
    model.load_state_dict(torch.load(model_path))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = args.beta
alpha = args.alpha
gamma = args.gamma
meters = np.zeros(4)

adj_data_name = os.path.join("data", args.data_dir, "adj.npy")
feat_data_name = os.path.join("data", args.data_dir, "feat.npy")
if args.pred:
    loc_data_name = os.path.join("data", args.data_dir, "loc.npy")

print(f"loading data from {adj_data_name}")

attr_init = np.load(feat_data_name, allow_pickle=True)
conn_init = np.load(adj_data_name, allow_pickle=True)
if args.pred:
    loc_init = np.load(loc_data_name, allow_pickle=True)
assert (attr_init.shape[0] == conn_init.shape[0])
data_length = attr_init.shape[0]

for epoch in range(args.epoch):
    perm = np.random.permutation(data_length)
    attr = attr_init[perm]
    conn = conn_init[perm]
    if args.pred:
        loc = loc_init[perm]
    else:
        loc = np.zeros([data_length, 16])  # dummy padding

    cur_loader_idx = 0
    batch_size = args.batch_size

    # loop through the dataset by batch
    while cur_loader_idx < data_length:
        cur_attr = attr[cur_loader_idx: cur_loader_idx + batch_size]
        cur_conn = conn[cur_loader_idx: cur_loader_idx + batch_size]
        cur_loc = loc[cur_loader_idx: cur_loader_idx + batch_size]
        cur_loader_idx += batch_size

        batch = tensorize(cur_attr, cur_conn)
        loc_batch = torch.tensor(cur_loc, dtype=torch.float32).cuda().reshape(cur_attr.shape[0], 16)
        total_step += 1

        model.zero_grad()
        loss, kl_div, wacc, tacc, pred_loss = model(batch, loc_batch, beta, alpha, gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, pred_loss * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Pred_Loss: %.2f, PNorm: %.2f, GNorm: %.2f" %
                  (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
            sys.stdout.flush()
            meters *= 0

        if total_step % args.save_iter == 0:
            if not os.path.exists(args.save_dir):
                # Create a new directory because it does not exist
                os.makedirs(args.save_dir)
                print(f"creating model filepath {args.save_dir}...")
            model_path = os.path.join(args.save_dir, "model.iter-" + str(total_step))
            torch.save(model.state_dict(), model_path)

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)
