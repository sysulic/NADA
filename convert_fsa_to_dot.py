import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set
import argparse
from FSM_for_cmp import FSM
from sympy import symbols, parse_expr

def fa2dot(state_set, transitions, initial_states, final_states) -> str:
  state2index = {i: int(re.search('\d+', str(i)).group()) for i in state_set}
  dot = 'digraph FSA {\n'
  dot += ' rankdir = LR;\n'
  dot += ' center = true;\n'
  dot += ' size = "7.5,10.5";\n'
  dot += ' edge [fontname = Courier];\n'
  dot += ' node [height = .5, width = .5];\n'
  if final_states is not None and len(final_states) > 0:
    acc_states = sorted([str(state2index[i]) for i in final_states])
    for s in acc_states:
      dot += f' {s} [shape = "doublecircle"];\n'
  for state in state_set:
    if state not in final_states:
      dot += f' {state2index[state]} [shape = "circle"];\n'
  dot += f' __start0 [label = "" shape = none width="0" height="0"];\n'
  if isinstance(initial_states, str) or isinstance(initial_states, int):
    dot += f' __start0 -> {state2index[initial_states]};\n'
  else:
    for i in sorted(initial_states):
      dot += f' __start0 -> {state2index[i]};\n'
  for _from in transitions.keys():
    for label in transitions[_from].keys():
      if isinstance(transitions[_from][label], str) or isinstance(transitions[_from][label], int):
        _to = transitions[_from][label]
        dot += f' {state2index[_from]} -> {state2index[_to]} [label="{label}"];\n'
      else:
        for _to in sorted(transitions[_from][label]):
          dot += f' {state2index[_from]} -> {state2index[_to]} [label="{label}"];\n'
  dot += '}'
  return dot

def fsm2dot(fsa: FSM) -> str:
  state2index = {i: int(re.search('\d+', i).group()) for i in fsa.state_set}
  dot = 'digraph FSA {\n'
  dot += ' rankdir = LR;\n'
  dot += ' center = true;\n'
  dot += ' size = "7.5,10.5";\n'
  dot += ' edge [fontname = Courier];\n'
  dot += ' node [height = .5, width = .5];\n'
  if fsa.final_states is not None and len(fsa.final_states) > 0:
    acc_states = sorted([str(state2index[i]) for i in fsa.final_states])
    for s in acc_states:
      dot += f' {s} [shape = "doublecircle"];\n'
  for state in fsa.state_set:
    if state not in fsa.final_states:
      dot += f' {state2index[state]} [shape = "circle"];\n'
  dot += f' __start0 [label = "" shape = none width="0" height="0"];\n'
  for i in sorted(fsa.init_states):
    dot += f' __start0 -> {state2index[i]};\n'
  for _from in fsa.G.keys():
    for label in fsa.G[_from].keys():
      for _to in sorted(fsa.G[_from][label]):
        dot += f' {state2index[_from]} -> {state2index[_to]} [label="{label}"];\n'
  dot += '}'
  return dot

def neighbor2dot(accept, neighbor, vocab):
  n = len(vocab)
  dot = """digraph FSA {\n rankdir = LR;\n center = true;\n size = "7.5,10.5";\n edge [fontname = Courier];\n node [height = .5, width = .5];\n"""
  if len(accept) > 0:
    dot += " node [shape = doublecircle]; {};\n".format("; ".join([str(i) for i in accept]))
  dot += f""" node [shape = circle]; 1;\n init [shape = plaintext, label = ""];\n"""
  dot += f' init -> 0;\n'
  for _from, _to, label in neighbor:
    dot += f' {_from} -> {_to} [label="{vocab[label]}"];\n'
  dot += '}'
  return dot

def write_dot_img_with_neighbor(accept, neighbor, vocab, dot_file_path, fsa_img_path, silent=False):
  dot = neighbor2dot(accept, neighbor, vocab)
  dot_file_path = Path(dot_file_path)
  if not os.path.exists(dot_file_path.parent):
    os.makedirs(dot_file_path.parent)
  with open(dot_file_path, 'w') as f:
    f.write(dot)
  if not silent: print(f'successfully write {dot_file_path}')
  fsa_img_path = Path(fsa_img_path)
  if not os.path.exists(fsa_img_path.parent):
    os.makedirs(fsa_img_path.parent)
  os.popen(f"dot -Tpng {dot_file_path} -o {fsa_img_path}").close()
  if not silent: print(f'successfully write {fsa_img_path}')
  return

def write_dot_img_with_dot(dot, dot_file_path, fsa_img_path, silent=False):
  dot_file_path = Path(dot_file_path)
  if not os.path.exists(dot_file_path.parent):
    os.makedirs(dot_file_path.parent)
  with open(dot_file_path, 'w') as f:
    f.write(dot)
  if not silent: print(f'successfully write {dot_file_path}')
  fsa_img_path = Path(fsa_img_path)
  if not os.path.exists(fsa_img_path.parent):
    os.makedirs(fsa_img_path.parent)
  os.popen(f"dot -Tpng {dot_file_path} -o {fsa_img_path}").close()
  if not silent: print(f'successfully write {fsa_img_path}')
  return

def write_dot_img(fsa_path, dot_file_path, fsa_img_path, silent=False, only_dot=False):
  fsa_path = Path(fsa_path)
  fsa = FSM(fsa_path) 
  dot = fsm2dot(fsa)
  dot_file_path = Path(dot_file_path)
  if not os.path.exists(dot_file_path.parent):
    os.makedirs(dot_file_path.parent)
  with open(dot_file_path, 'w') as f:
    f.write(dot)
  if not silent: print(f'successfully write {dot_file_path}')
  if only_dot:
    return
  fsa_img_path = Path(fsa_img_path)
  if not os.path.exists(fsa_img_path.parent):
    os.makedirs(fsa_img_path.parent)
  os.popen(f"dot -Tpng {dot_file_path} -o {fsa_img_path}").close()
  if not silent: print(f'successfully write {fsa_img_path}')
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='fsa to dot')
  parser.add_argument('--tag', type=int, required=False, help='tag', default=-1)
  parser.add_argument('--gt', action='store_true', help='write gt fsa')
  parser.add_argument('--init', action='store_true', help='write init fsa')
  parser.add_argument('--init_postprocess', action='store_true', help='write init_postprocess fsa')
  parser.add_argument('--val_k', type=int, required=False, default=0)
  parser.add_argument('--data_synthesis', type=int, required=False, default=0)
  args = parser.parse_args()

  dot_path = Path(f'./graph/dot')
  img_path = Path(f'./graph/img')
  if not dot_path.exists():
    os.makedirs(dot_path)
  if not img_path.exists():
    os.makedirs(img_path)

  if args.data_synthesis:
    datasets = [f'state{i}_vocab{j}' for j in [20,22,24,26,28] for i in [7,9,11,13,15]]
  else:
    datasets = ['ArrayList', 'HashMap', 'HashSet', 'Hashtable', 'LinkedList', 'NumberFormatStringTokenizer',
                'Signature', 'Socket', 'StringTokenizer', 'ZipOutputStream','StackAr']
  for dataset in datasets:
    print(f'----{dataset}----')
    # gt_fsm
    if args.gt:
      if args.data_synthesis:
        gt_fsa = FSM(Path(f'data_synthesis/{dataset}/gt_fsm.txt'))
      else:
        gt_fsa = FSM(Path(f'data_ori/{dataset}/gt_fsm.txt'))
      gt_dot = fsm2dot(gt_fsa)
      gt_dir = dot_path / 'gt'
      gt_path = gt_dir / f'{dataset}_gt_fsm.dot'
      if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
      with open(gt_path, 'w') as f:
        f.write(gt_dot)
      print(f'successfully write {gt_path}')
      gt_img_dir = img_path / 'gt'
      if not os.path.exists(gt_img_dir):
        os.makedirs(gt_img_dir)
      gt_img_path = gt_img_dir / f'{dataset}_gt_fsm.png'
      os.popen(f"dot -Tpng {gt_path} -o {gt_img_path}").close()
      print(f'successfully write {gt_img_path}')
      
    if args.init:
      init_path = Path(f'model/init/{args.tag}/valk{args.val_k}/{dataset}/init_fsa.txt')
      fsa = FSM(init_path)
      dot = fsm2dot(fsa)
      fsa_dir = dot_path / f'init/{args.tag}/valk{args.val_k}'
      fsa_path = fsa_dir / f'{dataset}_init_fsa.dot'
      if not os.path.exists(fsa_dir):
        os.makedirs(fsa_dir)
      with open(fsa_path, 'w') as f:
        f.write(dot)
      print(f'successfully write {fsa_path}')
      fsa_img_dir = img_path / f'init/{args.tag}/valk{args.val_k}'
      if not os.path.exists(fsa_img_dir):
        os.makedirs(fsa_img_dir)
      fsa_img_path = fsa_img_dir / f'{dataset}_init_fsa.png'
      os.popen(f"dot -Tpng {fsa_path} -o {fsa_img_path}").close()
      print(f'successfully write {fsa_img_path}')

    # final_fsa
    if args.tag >= 0:
      fsa = FSM(Path(f'model/{args.tag}/{dataset}/final_fsa.txt')) 
      dot = fsm2dot(fsa)
      fsa_dir = dot_path / f'{args.tag}'
      fsa_path = fsa_dir / f'{dataset}_final_fsa.dot'
      if not os.path.exists(fsa_dir):
        os.makedirs(fsa_dir)
      with open(fsa_path, 'w') as f:
        f.write(dot)
      print(f'successfully write {fsa_path}')
      fsa_img_dir = img_path / f'{args.tag}'
      if not os.path.exists(fsa_img_dir):
        os.makedirs(fsa_img_dir)
      fsa_img_path = fsa_img_dir / f'{dataset}_final_fsa.png'
      os.popen(f"dot -Tpng {fsa_path} -o {fsa_img_path}").close()
      print(f'successfully write {fsa_img_path}')