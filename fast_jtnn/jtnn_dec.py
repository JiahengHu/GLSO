'''
Modified based on https://github.com/wengong-jin/icml18-jtnn.git.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nnutils import create_var, GRU
from .mod_tree import ModTree, TreeNode
from .params import MAX_NB, MAX_DECODE_LEN, VOCAB_SIZE



class JTNNDecoder(nn.Module):

    def __init__(self, hidden_size, latent_size, embedding):
        super(JTNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = VOCAB_SIZE
        # self.vocab = vocab
        self.embedding = embedding

        #GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        #Word Prediction Weights 
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        #Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        #Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        tree_contexts = x_tree_vecs.index_select(0, contexts)
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        output_vec = F.relu( V(input_vec) )
        return V_o(output_vec)

    def forward(self, mol_batch, x_tree_vecs):
        pred_hiddens,pred_contexts,pred_targets = [],[],[]
        stop_hiddens,stop_contexts,stop_targets = [],[],[]
        traces = []
        for mol_tree in mol_batch:
            s = []
            dfs(s, mol_tree.nodes[0], -1)
            traces.append(s)
            for node in mol_tree.nodes:
                node.neighbors = []

        #Predict Root
        batch_size = len(mol_batch)
        pred_hiddens.append(create_var(torch.zeros(len(mol_batch),self.hidden_size)))
        pred_targets.extend([mol_tree.nodes[0].wid for mol_tree in mol_batch])
        pred_contexts.append( create_var( torch.LongTensor(range(batch_size)) ) )

        max_iter = max([len(tr) for tr in traces])
        padding = create_var(torch.zeros(self.hidden_size), False)
        h = {}

        # Create a batch of input
        # All the neighbors are set to [] initially
        # Max_nb: max neighbor
        for t in range(max_iter):
            prop_list = []
            batch_list = []
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i)

            cur_x = []
            cur_h_nei,cur_o_nei = [],[]

            for node_x, real_y, _ in prop_list:
                #Neighbors for message passing (target not included)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
                pad_len = MAX_NB - len(cur_nei)
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
                pad_len = MAX_NB - len(cur_nei)
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                #Current clique embedding
                cur_x.append(node_x.wid)

            #Clique embedding
            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x) 
            
            #Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

            #Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
            cur_o = cur_o_nei.sum(dim=1)

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(prop_list):
                node_x,node_y,direction = m
                x,y = node_x.idx,node_y.idx
                h[(x,y)] = new_h[i]
                node_y.neighbors.append(node_x)
                if direction == 1:
                    pred_target.append(node_y.wid)
                    pred_list.append(i) 
                stop_target.append(direction)

            #Hidden states for stop prediction
            cur_batch = create_var(torch.LongTensor(batch_list))
            stop_hidden = torch.cat([cur_x,cur_o], dim=1)
            stop_hiddens.append( stop_hidden )
            stop_contexts.append( cur_batch )
            stop_targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                batch_list = [batch_list[i] for i in pred_list]
                cur_batch = create_var(torch.LongTensor(batch_list))
                pred_contexts.append( cur_batch )

                cur_pred = create_var(torch.LongTensor(pred_list))
                pred_hiddens.append( new_h.index_select(0, cur_pred) )
                pred_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in mol_batch:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.wid)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var(torch.LongTensor(cur_x))
        cur_x = self.embedding(cur_x) 
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x,cur_o], dim=1)
        stop_hiddens.append( stop_hidden )
        stop_contexts.append( create_var( torch.LongTensor(range(batch_size)) ) )
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'word')
        pred_targets = create_var(torch.LongTensor(pred_targets))

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(mol_batch)
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, pred_targets).float()
        pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

        #Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu( self.U_i(stop_hiddens) )
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, x_tree_vecs, 'stop')
        stop_scores = stop_scores.squeeze(-1)
        stop_targets = create_var(torch.Tensor(stop_targets))
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(mol_batch)
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_targets).float()
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

    def decode(self, x_tree_vecs, prob_decode):
        assert x_tree_vecs.size(0) == 1

        stack = []
        init_hiddens = create_var( torch.zeros(1, self.hidden_size) )
        zero_pad = create_var(torch.zeros(1,1,self.hidden_size))
        contexts = create_var( torch.LongTensor(1).zero_() )

        #Root Prediction
        root_score = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'word')
        _,root_wid = torch.max(root_score, dim=1)
        root_wid = root_wid.item()

        root = TreeNode(root_wid)
        root.wid = root_wid
        root.idx = 0
        stack.append( (root, None) )

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, _ = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
            else:
                cur_h_nei = zero_pad

            cur_x = create_var(torch.LongTensor([node_x.wid]))
            cur_x = self.embedding(cur_x)

            #Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
            stop_hiddens = F.relu( self.U_i(stop_hiddens) )
            stop_score = self.aggregate(stop_hiddens, contexts, x_tree_vecs, 'stop')

            if prob_decode:
                backtrack = (torch.bernoulli( torch.sigmoid(stop_score) ).item() == 0)
            else:
                backtrack = (stop_score.item() < 0)
                # print(f'step = {step}, backtrack = {backtrack}, stopscore = {stop_score}')

            if not backtrack: #Forward: Predict next clique
                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                pred_score = self.aggregate(new_h, contexts, x_tree_vecs, 'word')

                type_range = VOCAB_SIZE #originally 5
                if prob_decode:
                    sort_wid = torch.multinomial(F.softmax(pred_score, dim=1).squeeze(), type_range)
                else:
                    _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:type_range]:
                    next_wid = wid
                    break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = TreeNode(next_wid)
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y, None) )
                    all_nodes.append(node_y)

            if backtrack: #Backtrack, use if instead of else
                if len(stack) == 1:
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()

        return root, all_nodes

"""
Helper Functions:
"""
def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        dfs(stack, y, x.idx)
        stack.append( (y,x,0) )
