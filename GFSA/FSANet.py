import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

import time
from pathlib import Path
from typing import List, Dict

from GFSA.FSARegular import FSARegular
from GFSA.FSARound import FSARound, FSANetRound

class FSANet(torch.nn.Module):
    def __init__(self, num_state, num_word, batch_size, learn_rate, formulaes_info, train_args, reinit=False):
        super(FSANet, self).__init__()
        # sigmoid([-2.2, -1.4, -0.85, 0, 0.4, 0.85, 1.4, 2.2]) = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self._num_models = train_args.num_models
        self._num_state = num_state
        self._num_word = num_word
        self._batch_size = batch_size
        self._learn_rate = learn_rate
        self._init_weight_center = 0
        self._init_weight_radius = 3
        self.INIT_POS_MASK = self._init_weight_center + self._init_weight_radius
        self.INIT_NEG_MASK = self._init_weight_center - self._init_weight_radius
        self.INIT_POS_MASK_RAND = -0.5
        self.INIT_NEG_MASK_RAND = -1.5
        self.INIT_POS_MASK_RAND_ACC = -0.5
        self.INIT_NEG_MASK_RAND_ACC = -1.5
        self.FIX_POS_MASK = 100
        self.FIX_NEG_MASK = -100
        self._train_args = train_args
        self.device = train_args.device

        self._neighbor = torch.nn.Parameter(torch.randn((self._num_models, self._num_state, self._num_word * self._num_state), device=self.device),requires_grad=False)
        self._accept = torch.nn.Parameter(torch.randn((self._num_models, self._num_state, 1), device=self.device),requires_grad=False)
        self._neighbor_grad_mask = torch.ones((self._num_models, self._num_state, self._num_word * self._num_state), device=self.device, requires_grad=False)
        
        self.init_info = formulaes_info['init_info'] if formulaes_info is not None else None
        self._init_random()

        self._accept.requires_grad_(True)
        self._neighbor.requires_grad_(True)
        
        self.loss_weight = self._train_args.w_loss_pos
        self._regular = FSARegular(*self.read_parameter(), self._neighbor_grad_mask, train_args,num_word,num_state)
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self._learn_rate)
        self.best_score = float('inf')
        self.best_score_allpos = float('inf')
        self.best_fsa = None
        self.best_fsa_allpos = None
        self.best_neighbor = None
        self.best_accept = None
        self.best_step = 0
        self.best_fsa_idx = 0
        self.best_step_allpos = 0
        self.best_fsa_idx_allpos = 0
        self.fsa_hashtable = set()
        self.restart_steps = train_args.restart_steps
        self.cnt = 0
        self.conflicts = 0

    def _init_random(self, num_state=5, reinit=False):
        with torch.no_grad():
            self._init_weight()
            self._neighbor.data = self._neighbor.data.reshape(self._num_models, self._num_state, self._num_word, self._num_state)
            self._neighbor_grad_mask = self._neighbor_grad_mask.reshape(self._num_models, self._num_state, self._num_word, self._num_state)
            if self._train_args.neighbor_init:
                if self.init_info is not None:
                    if len(self.init_info['not_init']) > 0:
                        not_init_tensor = torch.tensor(self.init_info['not_init'], dtype=torch.long, device=self.device)
                        self._neighbor[:, 0, :, 1:].index_fill_(1, not_init_tensor, self.FIX_NEG_MASK)
                        self._neighbor_grad_mask[:, 0, :, 1:].index_fill_(1, not_init_tensor, 0)
                    if len(self.init_info['init']) > 0:
                        init_tensor = torch.tensor(self.init_info['init'], dtype=torch.long, device=self.device)
                        self._neighbor[:, 1:, :, :].index_fill_(2, init_tensor, self.FIX_NEG_MASK)
                        self._neighbor_grad_mask[:, 1:, :, :].index_fill_(2, init_tensor, 0)
                    if len(self.init_info['no_loop']) == 1:
                        self._neighbor[:, :, :, 0] = self.FIX_NEG_MASK
                        self._neighbor_grad_mask[:, :, :, 0] = 0
            self._neighbor.data = self._neighbor.data.reshape(self._num_models, self._num_state, self._num_word * self._num_state)
            self._neighbor_grad_mask = self._neighbor_grad_mask.reshape(self._num_models, self._num_state, self._num_word * self._num_state)
        self._rand = True
        self._accept.requires_grad_(True)
        self._neighbor.requires_grad_(True)
        return

    def _init_from_ltl(self, formulaes_info: Dict):
        n = self._num_word
        formulaes = formulaes_info['formulaes']
        states = formulaes_info['states']
        accept_list = formulaes_info['accept']
        neighbor_list = formulaes_info['neighbor']

        accept = torch.ones(size=(self._num_state, 1)) * self.INIT_NEG_MASK
        neighbor = torch.ones(size=(self._num_state, self._num_state * self._num_word)) * self.INIT_NEG_MASK
        for i in accept_list:
            if i < self._num_state:
                accept[i] = self.INIT_POS_MASK
        for _from, _to, label in neighbor_list:
            if _from >= self._num_state or _to >= self._num_state:
                continue
            neighbor[_from][_to + self._num_state * label] = self.INIT_POS_MASK

        self._rand = False
        return accept, neighbor, states

    def _init_weight(self):
        with torch.no_grad():
            self._neighbor.data.uniform_(self.INIT_NEG_MASK_RAND, self.INIT_POS_MASK_RAND)
            self._accept.data.uniform_(self.INIT_POS_MASK_RAND_ACC, self.INIT_POS_MASK_RAND_ACC)

    def _active(self, x):
        if self._train_args.active == 1: # a variant of leaky ReLU: max⁡(min⁡(x,1),0)
            x = torch.where(x < 0, x * 0.001 + 0.0, x)
            x = torch.where(x > 1, x * 0.001 + 0.999, x)
        else: # min(max(0,x),1)
            x = torch.min(torch.max(torch.zeros(x.shape, device=self.device), x), torch.ones(x.shape, device=self.device))
        return x
    
    def _criterion(self, predicts, labels, flags):
        if self._train_args.criterion == 1: # MSE
            loss = torch.mean(((predicts - labels) ** 2) * flags, dim=[1,2])
        else: # BCELoss
            loss = F.binary_cross_entropy(predicts, labels, reduction='none') * flags
            loss = torch.mean(loss, dim=[1,2])
        return loss

    def read_parameter(self):
        return self._accept, self._neighbor

    # # batch version
    def forward(self, traces):
        # traces: Tensor(batch_size * max_len * (num_state num_word) * num_state)
        # results: Tensor(num_models * batch_size * num_state * max_len)
        # x_next: Tensor(num_models * 1 * num_state * 1)
        results = torch.zeros((self._num_models,len(traces),len(traces[0][0][0]),len(traces[0])), device=self.device)
        x_next = torch.sigmoid(self._accept).unsqueeze(1)
        neighbor = torch.sigmoid(self._neighbor)
        neighbor = torch.mul(neighbor, self._neighbor_grad_mask)

        for state_index in range(len(traces[0])-1,-1,-1):
            # self._neighbor: Tensor(num_models * num_state * (num_state num_word))
            # neighbor.unsqueeze(1): Tensor(num_models * 1 * num_state * (num_state num_word))
            # trace: Tensor(batch_size * (num_state num_word) * num_state)
            trace = traces[:,state_index]
            z = torch.matmul(neighbor.unsqueeze(1), trace)
            # z: Tensor(num_models * batch_size * num_state * num_state)
            x = self._active(torch.matmul(z,x_next))
            # x: Tensor(num_models * batch_size * num_state * 1)

            results[:,:,:,state_index] = x[:,:,:,0]
            x_next = x
            # x_next: Tensor(num_models * batch_size * num_state * 1)
        return results

    def net_train_batch(self, train_dataloaders, epoch_idx, writer, debug_args=None):
        train_dataloader = train_dataloaders['train_dataloader']
        valid_dataloader = train_dataloaders['valid_dataloader']
        
        step_per_epoch = len(train_dataloader)
        total_loss = 0
        regular_loss = 0
        unlabel_loss = 0
        train_time = 0
        valid_time = 0
        total_score = []

        for step_idx, (traces, labels, masks, flags) in enumerate(train_dataloader):
            idx = (epoch_idx - 1) * step_per_epoch + step_idx
            start_time = time.time()

            # traces: [batch_size, length, num_word * num_state, num_state]
            # labels: [batch_size, 1]
            # masks: [batch_size, length, 1]
            # flags: [batch_size, 1]
            traces, labels, masks = traces.to(self.device), labels.to(self.device), masks.to(self.device)
            flags = flags.to(self.device)
            # predicts: [num_models, batch_size, 1]
            predicts = torch.matmul(self(traces), masks)[:, :, 0]
            # labels: [num_models, batch_size, 1]
            labels = labels.unsqueeze(0).repeat(self._num_models,1,1)
            loss = self._criterion(predicts, labels, flags) * self.loss_weight
            loss = torch.mean(loss)
            total_loss += loss.detach()
            loss.backward()

            loss_2 = self._regular()
            reg_loss = loss_2
            regular_loss += reg_loss.detach()
            loss_2.backward()
            loss_total = loss + loss_2
            # print(f'({idx}) loss_2: {loss_2}')

            writer.add_scalar('train label loss', loss, idx)
            writer.add_scalar('train regular loss', loss_2, idx)
            writer.add_scalar('train loss', loss_total, idx)

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_time = time.time() - start_time
            train_time += end_time

        # 评估更新后的公式
        if not self._train_args.no_valid:
            start_time = time.time()
            total_score = self.net_valid(valid_dataloader, idx, writer, debug_args)
            end_time = time.time() - start_time
            valid_time += end_time
        
        valid_loss = 0
        if len(total_score) > 0:
            valid_loss = torch.mean(total_score).item()

        total_loss = total_loss / step_per_epoch
        regular_loss = regular_loss / step_per_epoch
        return total_loss, regular_loss, valid_loss, train_time, valid_time

    def net_valid_hash(self, valid_dataloader, idx, writer, debug_args=None):
        with torch.no_grad():
            total_score: list = []
            fsas = self.to_fsa_round(theta=0.5)
            for fsa_idx, fsa in enumerate(fsas):
                if fsa in self.fsa_hashtable:
                    self.conflicts += 1
                    self.cnt += 1
                else:
                    self.fsa_hashtable.add(fsa)
                
                    score = 0
                    for i, (valid_traces, valid_labels, valid_masks, valid_flags) in enumerate(valid_dataloader):
                        valid_traces = valid_traces.to(self.device)
                        valid_labels = valid_labels.to(self.device)
                        valid_masks = valid_masks.to(self.device)
                        # valid_flags = valid_flags.to(self.device)
                        valid_predicts = fsa.check_batch(valid_traces, valid_masks).to(torch.float32)
                        score += torch.sum(((valid_predicts - valid_labels) ** 2)) / len(valid_dataloader.dataset)
                    score = score.item()
                    total_score.append(score)
    
                    if score < self.best_score:
                        self.cnt = 0
                        self.best_score = score
                        self.best_fsa = fsa
                        self.best_step = idx
                        self.best_fsa_idx = fsa_idx
                    else:
                        self.cnt += 1
        return total_score

    def net_valid(self, valid_dataloader, idx, writer, debug_args=None):
        with torch.no_grad():
            net_round = self.to_fsa_net_round(theta=0.5)
            scores = torch.zeros(self._num_models, device=self.device)
            tps = torch.zeros(self._num_models, device=self.device)
            fns = torch.zeros(self._num_models, device=self.device)
            for i, (valid_traces, valid_labels, valid_masks, valid_flags) in enumerate(valid_dataloader):
                valid_traces = valid_traces.to(self.device)
                valid_labels = valid_labels.to(self.device)
                valid_masks = valid_masks.to(self.device)
                # valid_flags = valid_flags.to(self.device)
                valid_predicts = net_round.check_batch(valid_traces, valid_masks).to(torch.float32)
                valid_labels = valid_labels.unsqueeze(0).repeat(self._num_models,1,1) # [num_models, batch_size, 1]
                scores += torch.sum(((valid_predicts - valid_labels) ** 2), dim=[1,2]) / len(valid_dataloader.dataset)
                if self._train_args.save_best_allpos:
                    tps += torch.sum(((valid_labels == 1) & (valid_predicts == 1)).squeeze(), dim=1)
                    fns += torch.sum(((valid_labels == 1) & (valid_predicts == 0)).squeeze(), dim=1)
            recs = tps / (tps + fns)
            sat_allpos = (recs == 1)
            score = torch.min(scores).item()
            fsa_idx = torch.argmin(scores).item()

            if self._train_args.save_best_allpos and torch.any(sat_allpos):
                allpos_idx = torch.nonzero(sat_allpos).squeeze(dim=1)
                scores_allpos = scores.index_select(0, allpos_idx)
                score_allpos, score_allpos_idx = torch.min(scores_allpos, dim=0)
                fsa_idx_allpos = allpos_idx[score_allpos_idx]
                if score_allpos < self.best_score_allpos:
                    self.cnt = 0
                    self.best_score_allpos = score_allpos
                    self.best_fsa_allpos = net_round[fsa_idx_allpos]
                    self.best_step_allpos = idx
                    self.best_fsa_idx_allpos = fsa_idx_allpos
    
            if score < self.best_score:
                self.cnt = 0
                self.best_score = score
                self.best_fsa = net_round[fsa_idx]
                self.best_step = idx
                self.best_fsa_idx = fsa_idx
                if self._train_args.save_best_net:
                    if self.best_neighbor is not None:
                        self.best_neighbor.copy_(self._neighbor[fsa_idx].data)
                    else:
                        self.best_neighbor = self._neighbor[fsa_idx].data.clone()
                    if self.best_accept is not None:
                        self.best_accept.copy_(self._accept[fsa_idx].data)
                    else:
                        self.best_accept = self._accept[fsa_idx].data.clone()
            else:
                self.cnt += 1
        return scores

    def _restart(self, reason=''):
        print(f'{self.cnt} steps Not Improved: restart, reason: {reason}')
        self.cnt = 0
        with torch.no_grad():
            self._neighbor.data[0].copy_(torch.where(self.best_fsa.neighbor.to(torch.bool), self.INIT_POS_MASK, self.INIT_NEG_MASK))
            self._accept.data[0].copy_(torch.where(self.best_fsa.accept.to(torch.bool), self.INIT_POS_MASK, self.INIT_NEG_MASK))
        self._accept.requires_grad_(True)
        self._neighbor.requires_grad_(True)

    def net_test(self, dataset):
        TP = torch.zeros(self._num_models)
        FN = torch.zeros(self._num_models)
        FP = torch.zeros(self._num_models)
        TN = torch.zeros(self._num_models)
        test_results=[]
        for trace, label in dataset:
            trace = trace.to(self.device)
            predicts = self.infer(trace).cpu()

            label = bool(label > 0.5)
            TP += label & predicts
            FN += label & ~predicts
            FP += (not label) & predicts
            TN += (not label) & ~predicts
            test_results.append([label,predicts.tolist()])

        acc = (TP+TN)/(TP+FN+FP+TN)
        pre = (TP)/(TP+FP)
        rec = (TP)/(TP+FN)
        F1 = (2*pre*rec)/(pre+rec)
        return TP, FN, FP, TN, acc, pre, rec, F1, test_results
    
    def infer(self, trace):
        self.eval()
        with torch.no_grad():
            x_next=torch.sigmoid(self._accept) # [num_models, num_state, 1]
            neighbor=torch.sigmoid(self._neighbor) # [num_models, num_state, num_state * num_word]
            for s in torch.flip(trace, [0]):
                z = torch.matmul(neighbor, s) # [num_models, num_state, num_state]
                x = self._active(torch.matmul(z, x_next)) # [num_models, num_state, 1]
                x_next = x
            is_accept = x_next[:, 0, 0] > 0.5
        self.train()
        return is_accept

    def net_test_best(self, dataset):
        assert(self.best_neighbor is not None and self.best_accept is not None)
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        test_results=[]
        for x in dataset:
            trace = x[0]
            label = x[1]
            trace = trace.to(self.device)
            predicts = bool(self.infer_best(trace).item())

            label = bool(label > 0.5)
            TP += label & predicts
            FN += label & ~predicts
            FP += (not label) & predicts
            TN += (not label) & ~predicts
            test_results.append([label,predicts])

        acc = (TP+TN)/(TP+FN+FP+TN)
        pre = (TP)/(TP+FP)
        rec = (TP)/(TP+FN)
        F1 = (2*pre*rec)/(pre+rec)
        return TP, FN, FP, TN, acc, pre, rec, F1, test_results
    
    def infer_best(self, trace):
        assert(self.best_neighbor is not None and self.best_accept is not None)
        self.eval()
        with torch.no_grad():
            x_next=torch.sigmoid(self.best_accept) # [num_state, 1]
            neighbor=torch.sigmoid(self.best_neighbor) # [num_state, num_state * num_word]
            for s in torch.flip(trace, [0]):
                z = torch.matmul(neighbor, s) # [num_state, num_state]
                x = self._active(torch.matmul(z, x_next)) # [num_state, 1]
                x_next = x
            is_accept = x_next[0, 0] > 0.5
        self.train()
        return is_accept

    def to_fsa_round(self, theta=0.5) -> List[FSARound]:
        accept = torch.sigmoid(self._accept).to(self.device)
        neighbor = torch.sigmoid(self._neighbor).to(self.device)
        accept = (accept > theta).float()
        neighbor = (neighbor > theta).float()
        return [FSARound(neighbor[i], accept[i]) for i in range(self._num_models)]

    def to_fsa_round_index(self, index, theta=0.5) -> FSARound:
        accept = torch.sigmoid(self._accept[index]).to(self.device)
        neighbor = torch.sigmoid(self._neighbor[index]).to(self.device)
        accept = (accept > theta).float()
        neighbor = (neighbor > theta).float()
        return FSARound(neighbor, accept)

    def to_fsa_net_round(self, theta=0.5) -> FSANetRound:
        accept = torch.sigmoid(self._accept).to(self.device)
        neighbor = torch.sigmoid(self._neighbor).to(self.device)
        accept = (accept > theta).float()
        neighbor = (neighbor > theta).float()
        return FSANetRound(neighbor, accept)
