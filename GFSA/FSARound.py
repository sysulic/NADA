from typing import Dict, List, Set, Tuple
import torch

def write_fsa(fsa,path):
  with open(path,'w') as f:
    print(len(fsa['start']),file=f)
    for i in fsa['start']:
      print(i,file=f)
    print(len(fsa['accept']),file=f)
    for i in fsa['accept']:
      print(i,file=f)
    print(len(fsa['neighbor']),file=f)
    for i in fsa['neighbor']:
      print(i[0],'\t',i[1],'\t',i[2],file=f)

def get_adj_list(states, neighbor):
  adj_list = [[] for i in range(states)]
  for i in range(states):
    for _from, _to, label in neighbor:
      if i == _from:
        adj_list[i].append([_from, _to, label])
  return adj_list

def remove_unreachable(adj_list: List[List[List[int]]], accept: List[int]) -> Tuple[List[int], List[List[int]], int, List[List[List[int]]]]:
  # dfs
  found = set()
  stack = [0]
  while len(stack) > 0:
    _from = stack.pop()
    if _from not in found:
      found.add(_from)
      for _, _to, _ in adj_list[_from][::-1]:
        if _to not in found:
          stack.append(_to)
  adj_list = [adj_list[_from] for _from in found]
  states = len(adj_list)
  acc = accept
  accept = []
  for i in acc:
    if i in found:
      accept.append(i)

  total = set(range(len(found)))
  not_found = sorted(list(total.difference(found)))
  over = sorted(list(found.difference(total)))
  for idx, i in enumerate(accept):
    new_i = i
    if i in over:
      new_i = not_found[over.index(i)]
    accept[idx] = new_i
  accept.sort()
  for i in range(len(adj_list)):
    for j, (_from, _to, label) in enumerate(adj_list[i]):
      new_from = _from
      new_to = _to
      if _from in over:
        new_from = not_found[over.index(_from)]
      if _to in over:
        new_to = not_found[over.index(_to)]
      adj_list[i][j] = [new_from, new_to, label]
  neighbor = []
  for i in range(len(adj_list)):
    neighbor += adj_list[i]
  neighbor.sort(key=lambda x: x[0])
  adj_list = get_adj_list(states, neighbor)
  return accept, neighbor, states, adj_list

def remove_not_lead_accept(adj_list, accept):
  def dfs(adj_list, start=0):
    found = set()
    stack = [start]
    while len(stack) > 0:
      _from = stack.pop()
      if _from not in found:
        found.add(_from)
        for _, _to, _ in adj_list[_from][::-1]:
          if _to not in found:
            stack.append(_to)
    has_loop = False
    for _, _to, _ in adj_list[start]:
      if _to == start:
        has_loop = True
        break
    if not has_loop:
      found.remove(start)
    return found
  
  states = len(adj_list)
  not_lead_accept = set()
  for i in range(1, states):
    found = dfs(adj_list, i)
    if (len(found.intersection(accept)) == 0) and (i not in accept):
      not_lead_accept.add(i)
  found = set(list(range(states))).difference(not_lead_accept)
  
  adj_list_ori = adj_list
  adj_list = []
  for _from in found:
    adj = []
    for _f, _to, label in adj_list_ori[_from]:
      assert(_f == _from)
      if _to in found:
        adj.append([_f, _to, label])
    adj_list.append(adj)
  states = len(adj_list)
  acc = accept
  accept = []
  for i in acc:
    if i in found:
      accept.append(i)

  total = set(range(len(found)))
  not_found = sorted(list(total.difference(found)))
  over = sorted(list(found.difference(total)))
  for idx, i in enumerate(accept):
    new_i = i
    if i in over:
      new_i = not_found[over.index(i)]
    accept[idx] = new_i
  accept.sort()
  for i in range(len(adj_list)):
    for j, (_from, _to, label) in enumerate(adj_list[i]):
      new_from = _from
      new_to = _to
      if _from in over:
        new_from = not_found[over.index(_from)]
      if _to in over:
        new_to = not_found[over.index(_to)]
      adj_list[i][j] = [new_from, new_to, label]
  neighbor = []
  for i in range(len(adj_list)):
    neighbor += adj_list[i]
  neighbor.sort(key=lambda x: x[0])
  adj_list = get_adj_list(states, neighbor)
  return accept, neighbor, states, adj_list

class FSARound:
  def __init__(self, neighbor: torch.Tensor, accept: torch.Tensor) -> None:
    # neighbor: Tensor(num_state * (num_state num_word))
    # accept: Tensor(num_state * 1)
    self.num_state = neighbor.size(0)
    self.num_word = neighbor.size(1) // self.num_state
    self.neighbor = neighbor
    self.accept = accept
    self.neighbor_list = None
    self.accept_list = None
    self.adj_list = None
    return

  def _init_from_accept_neighbor_list(self, neighbor_list, accept_list, states=None):
    n = self.num_word
    if states is not None:
      self.num_state = states

    self.accept = torch.zeros(size=(self.num_state, 1))
    self.neighbor = torch.zeros(size=(self.num_state, self.num_state * self.num_word))
    for i in accept_list:
      if i < self.num_state:
        self.accept[i] = 1
    for _from, _to, label in neighbor_list:
      if _from >= self.num_state or _to >= self.num_state:
        continue
      self.neighbor[_from][_to + self.num_state * label] = 1
    return

  def get_accept_neighbor_list(self, vocab=None) -> Tuple[List]:
    accept_list = torch.nonzero(self.accept.squeeze()).squeeze(1).tolist()
    neighbor_list = []
    neighbor_index = torch.nonzero(self.neighbor)
    for transition in neighbor_index:
      _from = int(transition[0])
      _to = int(transition[1]) % self.num_state
      token = int(transition[1]) // self.num_state
      if vocab is None:
        neighbor_list.append([_from, _to, token])
      else:
        assert(len(vocab) == self.num_word)
        neighbor_list.append([_from, _to, vocab[token]])
    return accept_list, neighbor_list

  def check_trace(self, trace: torch.Tensor) -> int:
    # trace: [length, num_word * num_state, num_state]
    # print(f'trace: {trace.size()}, {trace}')
    with torch.no_grad():
      # trace = trace.to(torch.int32)
      x = self.accept
      for i, s in enumerate(torch.flip(trace, [0])): # trace: [num_word * num_state, num_state]
        # print(f's: {s.size()}, {s}')
        z = torch.mm(self.neighbor, s) # [num_state, num_state]
        # print(f'z: {z.size()}, {z}')
        x = torch.mm(z, x) # [num_state, 1]
        # print(f'x: {x.size()}, {x}')
      x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)) # min(max(0, x), 1)
      # print(f'x: {x.size()}, {x}') # [num_state, 1]
      is_accept = x[0][0]
    return is_accept

  def check(self, trace: torch.Tensor, masks: torch.Tensor) -> int:
    # trace: [length, num_word * num_state, num_state]
    # masks: [length, 1]
    # print(f'trace: {trace.size()}, {trace}')
    # print(f'masks: {masks.size()}, {masks}')
    device = trace.device
    length = len(trace)
    results = torch.zeros((len(trace[0][0]), len(trace)), device=device) # [num_state, length]
    with torch.no_grad():
      # trace = trace.to(torch.int32)
      x = self.accept
      for i, s in enumerate(torch.flip(trace, [0])): # trace: [num_word * num_state, num_state]
        # print(f's: {s.size()}, {s}')
        z = torch.mm(self.neighbor, s) # [num_state, num_state]
        # print(f'z: {z.size()}, {z}')
        x = torch.mm(z, x) # [num_state, 1]
        # print(f'x: {x.size()}, {x}')
        x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)) # min(max(0, x), 1)
        results[:, length-1-i] = x[:, 0]
      # print(f'results: {results.size()}, {results}')
      ret = torch.matmul(results, masks) # [num_state, 1]
      # print(f'ret: {ret.size()}, {ret}')
      is_accept = ret[0][0]
    return is_accept
  
  def check_batch(self, traces: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # traces: [batch_size, length, num_word * num_state, num_state]
    # masks: [batch_size, length, 1]
    device = traces.device
    length = len(traces[0])
    results = torch.zeros((len(traces), len(traces[0][0][0]), len(traces[0])), device=device) # [batch_size, num_state, length]
    with torch.no_grad():
      # traces = traces.to(torch.int32)
      traces = torch.transpose(traces, 0, 1) # [length, batch_size, num_word * num_state, num_state]
      x = self.accept
      for i, s in enumerate(torch.flip(traces, [0])):
        # s: [batch_size, num_word * num_state, num_state]
        z = torch.matmul(self.neighbor, s) # [batch_size, num_state, num_state]
        x = torch.matmul(z, x) # [batch_size, num_state, 1]
        x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)) # min(max(0, x), 1)
        results[:, :, length-1-i] = x[:, :, 0]
      is_accept = torch.matmul(results, masks)[:, 0] # matmul: [batch_size, num_state, 1]
    return is_accept # [batch_size, 1]
  
  def __hash__(self) -> int:
    neighbor = self.neighbor.tolist()
    neighbor = tuple([tuple(i) for i in neighbor])
    accept = self.accept.tolist()
    accept = tuple([tuple(i) for i in accept])
    return hash((neighbor, accept))
  
  def __eq__(self, obj: object) -> bool:
    return torch.equal(self.neighbor, obj.neighbor) and torch.equal(self.accept, obj.accept)

  def write_fsa(self, vocab, fsa_path, theta=0.5):
    ret_fsa = self.to_dict(vocab, theta=theta)
    write_fsa(ret_fsa, fsa_path)
    return

  def to_dict(self, vocab, theta=0.5):
    cnt=0
    ret_fsa={'accept':[],'neighbor':[],'states':['C%d'%i for i in range(self.num_state)],'start':['C0'],'neighbormap':{}}
    accept_list = torch.nonzero(self.accept.squeeze(dim=1) >= theta).tolist()
    # print(accept_list)
    for acc in accept_list:
      ret_fsa['accept'].append(f'C{acc[0]}')
    neighbor_list = torch.nonzero(self.neighbor.view(self.num_state, self.num_word, self.num_state) >= theta).tolist()
    for _from, label, _to in neighbor_list:
      word = vocab[label]
      if (f'C{_from}', word) not in ret_fsa['neighbormap'].keys():
        ret_fsa['neighbormap'][(f'C{_from}', word)] = []
      ret_fsa['neighbormap'][(f'C{_from}', word)].append(f'C{_to}')
      ret_fsa['neighbor'].append([f'C{_from}', f'C{_to}', word])
    return ret_fsa

  def neighbor_postprocess(self):
    # self.accept_list, self.neighbor_list = self.get_accept_neighbor_list()
    # print(f'before, accept_list: {self.accept_list}\nneighbor_list: {self.neighbor_list}')
    self.neighbor = self.neighbor.reshape(self.num_state, self.num_word, self.num_state)
    self.neighbor[0, 1:, :] = 0
    self.neighbor[:, :, 0] = 0
    self.neighbor[1:, 0, :] = 0
    self.neighbor = self.neighbor.reshape(self.num_state, self.num_word * self.num_state)
    # self.accept_list, self.neighbor_list = self.get_accept_neighbor_list()
    # print(f'after, accept_list: {self.accept_list}\nneighbor_list: {self.neighbor_list}')
    return self

  def remove_unreachable(self, init=False):
    if (self.accept_list is None) or (self.neighbor_list is None):
      self.accept_list, self.neighbor_list = self.get_accept_neighbor_list()
    # print(f'before, accept_list: {self.accept_list}\nneighbor_list: {self.neighbor_list}')
    if self.adj_list is None:
      self.adj_list = get_adj_list(self.num_state, self.neighbor_list)
    self.accept_list, self.neighbor_list, self.num_state, self.adj_list = remove_unreachable(self.adj_list, self.accept_list)
    # print(f'after, accept_list: {self.accept_list}\nneighbor_list: {self.neighbor_list}')
    if init:
      self._init_from_accept_neighbor_list(self.neighbor_list, self.accept_list, self.num_state)
    return self

  def remove_not_lead_accept(self, init=False):
    if (self.accept_list is None) or (self.neighbor_list is None):
      self.accept_list, self.neighbor_list = self.get_accept_neighbor_list()
    # print(f'before, accept_list: {self.accept_list}\nneighbor_list: {self.neighbor_list}')
    if self.adj_list is None:
      self.adj_list = get_adj_list(self.num_state, self.neighbor_list)
    self.accept_list, self.neighbor_list, self.num_state, self.adj_list = remove_not_lead_accept(self.adj_list, self.accept_list)
    # print(f'after, accept_list: {self.accept_list}\nneighbor_list: {self.neighbor_list}')
    if init:
      self._init_from_accept_neighbor_list(self.neighbor_list, self.accept_list, self.num_state)
    return self

class FSANetRound:
  def __init__(self, neighbor: torch.Tensor, accept: torch.Tensor) -> None:
    # neighbor: Tensor(num_models * num_state * (num_state num_word))
    # accept: Tensor(num_models * num_state * 1)
    self.num_models = neighbor.size(0)
    self.num_state = neighbor.size(1)
    self.num_word = neighbor.size(2) // self.num_state
    self.neighbor = neighbor
    self.accept = accept
    return
  
  def check_batch(self, traces: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # traces: [batch_size, length, num_word * num_state, num_state]
    # masks: [batch_size, length, 1]
    device = traces.device
    length = len(traces[0])
    results = torch.zeros((self.num_models, len(traces), len(traces[0][0][0]), len(traces[0])), device=device) # [num_models, batch_size, num_state, length]
    with torch.no_grad():
      traces = torch.transpose(traces, 0, 1) # [length, batch_size, num_word * num_state, num_state]
      x = self.accept.unsqueeze(1) # [num_models, 1, num_state, 1]
      for i, s in enumerate(torch.flip(traces, [0])):
        # s: [batch_size, num_word * num_state, num_state]
        # neighbor.unsqueeze(1): [num_models, 1, num_state, num_state * num_word]
        z = torch.matmul(self.neighbor.unsqueeze(1), s) # [num_models, batch_size, num_state, num_state]
        x = torch.matmul(z, x) # [num_models, batch_size, num_state, 1]
        x = torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x)) # min(max(0, x), 1)
        results[:, :, :, length-1-i] = x[:, :, :, 0]
      is_accept = torch.matmul(results, masks)[:, :, 0] # matmul: [num_models, batch_size, num_state, 1]
    return is_accept # [num_models, batch_size, 1]

  def __getitem__(self, index) -> FSARound:
    return FSARound(self.neighbor[index], self.accept[index])

if __name__ == '__main__':
  states = 5
  vocab = ["ZipOutputStream", "closeEntry", "putNextEntry", "write", "close"]
  neighbor_list = [[0, 2, 'ZipOutputStream'], [0, 4, 'ZipOutputStream'], [0, 3, 'close'], [1, 1, 'write'], [1, 2, 'write'], [1, 0, 'close'], [2, 1, 'closeEntry'], [2, 2, 'closeEntry'], [2, 1, 'putNextEntry'], [2, 2, 'putNextEntry'], [2, 0, 'close'], [2, 3, 'close'], [3, 0, 'close'], [4, 0, 'close']]
  neighbor_list = [[_from, _to, vocab.index(label)] for _from, _to, label in neighbor_list]
  accept_list = [0]
  adj_list = get_adj_list(states, neighbor_list)
  print(f'states: {states}')
  print(f'accept_list: {accept_list}, {len(accept_list)}')
  print(f'neighbor_list: {neighbor_list}, {len(neighbor_list)}')
  print(f'adj_list: {adj_list}')
  fsa = FSARound(torch.zeros(states,states*len(vocab)), torch.zeros(states,1))
  fsa._init_from_accept_neighbor_list(neighbor_list, accept_list, states)
  fsa.neighbor_postprocess()
  accept_list, neighbor_list = fsa.get_accept_neighbor_list()
  adj_list = get_adj_list(states, neighbor_list)
  print(f'states: {states}')
  print(f'accept_list: {accept_list}, {len(accept_list)}')
  print(f'neighbor_list: {neighbor_list}, {len(neighbor_list)}')
  print(f'adj_list: {adj_list}')
  accept_list, neighbor_list, states, adj_list = remove_unreachable(adj_list, accept_list)
  print(f'states: {states}')
  print(f'accept_list: {accept_list}, {len(accept_list)}')
  print(f'neighbor_list: {neighbor_list}, {len(neighbor_list)}')
  print(f'adj_list: {adj_list}')
  accept_list, neighbor_list, states, adj_list = remove_not_lead_accept(adj_list, accept_list)
  print(f'states: {states}')
  print(f'accept_list: {accept_list}, {len(accept_list)}')
  print(f'neighbor_list: {neighbor_list}, {len(neighbor_list)}')
  print(f'adj_list: {adj_list}')
