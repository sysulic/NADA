import torch
import torch.nn.functional as F

def compute_range_loss(v, l_b, r_b):
    return torch.sum(torch.relu(l_b - v) + torch.relu(v - r_b))

def compute_close_loss(v1, v2):
    return torch.exp(-torch.abs(v1 - v2))

def compute_away_loss(v1, v2):
    return torch.sum(-torch.relu(v1-v2)-torch.relu(v2-v1))
    # return torch.mean(-torch.relu(v1-v2)-torch.relu(v2-v1))

def log_modify(v):
    return torch.where(v <= 0, 0, torch.log(v))

def compute_entropy_reg_loss(v):
    return torch.mean(- v * log_modify(v) - (1 - v) * log_modify(1 - v))

class FSARegular(object):
    def __init__(self, para_accept, para_neighbor, para_neighbor_grad_mask, train_args, _num_word,_num_state):
        self._para_accept = para_accept
        self._para_neighbor = para_neighbor
        self._para_neighbor_grad_mask = para_neighbor_grad_mask
        self._args = train_args
        self._num_word = _num_word
        self._num_state=_num_state
        self._num_models = para_accept.size(0)

    def __call__(self):
        neighbor = torch.sigmoid(self._para_neighbor)
        neighbor = torch.mul(neighbor, self._para_neighbor_grad_mask)
        accept = torch.sigmoid(self._para_accept)

        faithful_loss1 = -torch.sigmoid(torch.max(self._para_accept)) + 1

        if self._args.faithfula2_old:
            faithful_loss2 = (compute_away_loss(accept, 0.5)+compute_away_loss(neighbor, 0.5)) / self._num_models
        else:
            faithful_loss2 = (compute_entropy_reg_loss(accept)+compute_entropy_reg_loss(neighbor)) / 2

        a= faithful_loss1 * self._args.faithfula1 + faithful_loss2 * self._args.faithfula2
        return a
