import json
import random
import time
import torch
import os
import numpy
from torch.utils.data import DataLoader, Dataset
from GFSA.LTLfMiner import LTLf
import copy
import spot
from typing import List, Dict
from sympy import symbols, parse_expr
from GFSA.Net2FSA import write_fsa
from FSM_for_cmp import FSM
from pathlib import Path
import psutil

TIMEOUT_COUNT = 10000

process = psutil.Process()

def dfa2fsa(dfa: str) -> List[int]:
  accept = []
  neighbor = []
  states = 0
  lines = dfa.split(sep='\n')
  for line in lines:
    line = line.strip()
    if line.find('node [shape = doublecircle];') != -1:
      accept = line[len('node [shape = doublecircle]; '):-1]
      accept = accept.split(sep=';')
      accept = [int(x)-1 for x in accept]
      break
  begin = lines.index(' init -> 1;') + 1
  for i in range(begin, len(lines)-1):
    line = lines[i].strip()
    from_to = line[:line.find('[')]
    from_to = from_to.split(sep='->')
    for x in from_to:
        x = int(x)
        if x > states:
            states = x
    from_to = [int(x)-1 for x in from_to]
    label = line[line.find('"')+1 : -3]
    from_to.append(label)
    neighbor.append(from_to)
  return accept, neighbor, states

def formulae2fsa(ltlfs: List, traces_pos: List[str], vocab: List[str], dataset: str, args, save_path=None) -> List[str]:
    ltlf_strs = [lambda i, j, k: f'(G(v{i} -> X(F(v{j}))))',
                 lambda i, j, k: f'(G(v{i} -> X(G(!v{j}))))',
                 lambda i, j, k: f'((!v{i} U v{j}) | G(!v{i}))',
                 lambda i, j, k: f'(G(v{i} -> X(v{j})))',
                 lambda i, j, k: f'(G(v{i} -> X(!v{j})))',
                 lambda i, j, k: f'(F(v{i}) -> (!v{i} U (v{j} & X(v{i}))))',
                 lambda i, j, k: f'(G(v{i} -> ((!v{j} U v{k}) | G(!v{j}))))']
    total = len(traces_pos)

    formulaes = []
    fml_str = ''
    for num_sat, var, idx, _ in ltlfs:
        assert(num_sat / total >= args.threshold1)
        v1 = var[0]
        v2 = var[1]
        i = vocab.index(v1)
        j = vocab.index(v2)
        k = 0
        if len(var) == 3:
            v3 = var[2]
            k = vocab.index(v3)
        ltlf = ltlf_strs[idx](i, j, k)
        formulaes.append(ltlf)
        if fml_str == '':
            fml_str = ltlf
        else:
            fml_str += '&' + ltlf
    if fml_str == '':
        fml_str = 'true'
    if not args.silent: print(f'formulaes: {formulaes}')
    if not args.silent: print(f'formulaes len: {len(formulaes)}')
    
    fml_str += '&G('
    for i in range(len(vocab)):
      if i != 0:
        fml_str += ' | '
      fml_str += '('
      fml_str += ' & '.join([f'v{j}' if j == i else f'!v{j}' for j in range(len(vocab))])
      fml_str += ')'
    fml_str += ')'

    if not args.silent: print('parsing ltlf to fsa ...')
    f = spot.formula(fml_str)
    fsa = spot.translate(f, 'Buchi')
    hoa = fsa.to_str('hoa')
    states: int = fsa.num_states()
    init_states: int = fsa.get_init_state_number()
    neighbor_list = []
    accept_list = set()
    convert_index = {i: i for i in range(states)}
    convert_index[0] = init_states
    convert_index[init_states] = 0
    bdict = fsa.get_dict()
    for s in range(0, states):
        for t in fsa.out(s):
            src = convert_index[t.src]
            dst = convert_index[t.dst]
            label = spot.bdd_format_formula(bdict, t.cond)
            neighbor_list.append([src, dst, label])
            if t.acc.count() != 0:
                assert(t.src == s)
                accept_list.add(src)
    accept_list = sorted(list(accept_list))

    neighbor_ori = neighbor_list
    neighbor_list = []
    n = len(vocab)
    for _from, _to, label in neighbor_ori:
        if label == 'true':
            for i, token in enumerate(vocab):
                neighbor_list.append([_from, _to, i])
        else:
            variables = []
            for j in range(n):
                variables.append(symbols(f'v{j}'))
            label = label.replace('!', '~')
            label_formula = parse_expr(label)
            dic = {}
            for j in range(n):
                dic[variables[j]] = False
            for j in range(n):
                if j > 0:
                    dic[variables[j-1]] = False
                dic[variables[j]] = True
                if bool(label_formula.subs(dic)):
                    neighbor_list.append([_from, _to, j])
    accept_list.sort()
    neighbor_list.sort(key=lambda x: x[0])
    if not args.silent: print('successfully parse ltlf to fsa')

    fsa = {'accept':[],'neighbor':[],'states':[f'C{i}' for i in range(states)],'start':['C0'],'neighbormap':{}}
    for i in accept_list:
        fsa['accept'].append(f'C{i}')
    for _from, _to, label in neighbor_list:
        word = vocab[label]
        fsa['neighbor'].append([f'C{_from}', f'C{_to}', word])
    init_dir = f'model/init/{args.tag}/valk{args.val_k}/{dataset}'
    if not os.path.exists(init_dir):
        os.makedirs(init_dir)
    fsa_path = f'{init_dir}/init_fsa.txt'
    write_fsa(fsa, fsa_path)

    return formulaes, accept_list, neighbor_list, states

def fun(v1, v2, cnt, vocab, traces_pos, args, temp_save_path=None):
    def get_sat_array(traces_pos, check_pos_mark):
        sat_array = []
        for idx, trace in enumerate(traces_pos):
            mark = 0
            if check_pos_mark[idx]:
                for state in trace:
                    if v1 in state:
                        mark = 1
                        break
            sat_array.append(mark)
        return sat_array
    
    if not args.silent:
        print('\rltl miner processing:%d/%d' % (cnt[0], cnt[1]),end='      ')
        print(f'current formulaes count: {cnt[2]}', end='      ')
        print(f"memory_usage: {process.memory_info().rss / 1024**2} MB", end='      ')
    ret_ltlfs = []
    ltlf_trees = [('U', ('->', v1, ('X', ('F', v2))),('!',('X','true'))),
                  ('U', ('->', v1, ('X', ('U', ('!', v2),('!',('X','true'))))),('!',('X','true'))),
                  ('W', ('!', v1), v2),
                  ('U', ('->', v1, ('X', v2)),('!',('X','true'))),
                  ('U', ('->', v1, ('X', ('!', v2))),('!',('X','true'))),
                  ('->', ('F', v1), ('U', ('!', v1), ('&', v2, ('X', v1))))]
    ltlf_idx_pair = [[3, 0], [1, 4], [5, 2]]
    if not args.fml_opt1:
        sat_count = 0
        for ltlf_tmp_idx, ltlf_tree in enumerate(ltlf_trees):
            if v1 == v2 and ltlf_tmp_idx == 2:
                continue
            ltlf = LTLf(vocab, ltlf_tree)
            check_pos_mark, _ = ltlf.evaluate(traces_pos, [])
            num_sat = sum(check_pos_mark)
            sat_array = []
            for idx, trace in enumerate(traces_pos):
                mark = 0
                if check_pos_mark[idx]:
                    for state in trace:
                        if v1 in state:
                            mark = 1
                            break
                sat_array.append(mark)
            if num_sat / len(traces_pos) >= args.threshold1:
                ret_ltlfs.append((num_sat, (v1, v2), ltlf_tmp_idx, sat_array))
                sat_count += 1
                if sat_count + cnt[2] == args.fml_topk:
                    break
        return ret_ltlfs
    else:
        sat_count = 0
        for ltlf_spe_idx, ltlf_uni_idx in ltlf_idx_pair:
            flag = True
            ltlf_tree = ltlf_trees[ltlf_spe_idx]
            ltlf = LTLf(vocab, ltlf_tree)
            check_pos_mark, _ = ltlf.evaluate(traces_pos, [])
            num_sat = sum(check_pos_mark)
            if num_sat == len(traces_pos):
                sat_array = get_sat_array(traces_pos, check_pos_mark)
                ret_ltlfs.append((num_sat, (v1, v2), ltlf_spe_idx, sat_array))
            elif num_sat / len(traces_pos) >= args.threshold1:
                if v1 == v2 and ltlf_uni_idx == 2:
                    continue
                ltlf_tree = ltlf_trees[ltlf_uni_idx]
                ltlf = LTLf(vocab, ltlf_tree)
                check_pos_mark_uni, _ = ltlf.evaluate(traces_pos, [])
                num_sat_uni = sum(check_pos_mark_uni)
                if num_sat_uni == len(traces_pos):
                    sat_array = get_sat_array(traces_pos, check_pos_mark_uni)
                    ret_ltlfs.append((num_sat_uni, (v1, v2), ltlf_uni_idx, sat_array))
                else:
                    sat_array = get_sat_array(traces_pos, check_pos_mark)
                    ret_ltlfs.append((num_sat, (v1, v2), ltlf_spe_idx, sat_array))
            else:
                if v1 == v2 and ltlf_uni_idx == 2:
                    continue
                ltlf_tree = ltlf_trees[ltlf_uni_idx]
                ltlf = LTLf(vocab, ltlf_tree)
                check_pos_mark_uni, _ = ltlf.evaluate(traces_pos, [])
                num_sat_uni = sum(check_pos_mark_uni)
                if num_sat_uni / len(traces_pos) >= args.threshold1: # 如果大于置信度阈值，加入通用公式
                    sat_array = get_sat_array(traces_pos, check_pos_mark_uni)
                    ret_ltlfs.append((num_sat_uni, (v1, v2), ltlf_uni_idx, sat_array))
                else:
                    flag = False
            
            if flag:
                sat_count += 1
                if sat_count + cnt[2] == args.fml_topk:
                    break
        return ret_ltlfs

def get_formulae(traces_pos:list,vocab:list,args, save_path=None):
    ltlfs = []

    cnt = 0
    total = len(vocab)**2
    sat_cnt = 0

    result=[]

    if args.fml_opt2:
        premise = torch.zeros((len(vocab), len(traces_pos)), dtype=torch.float64)
        for i, v in enumerate(vocab):
            for j, trace in enumerate(traces_pos):
                for state in trace:
                    if v in state:
                        premise[i,j] = 1.0
                        break
        premise = torch.mean(premise, dim=1)
        premise, order = torch.sort(premise, descending=True)
        order = order.tolist()

        end_flag = False
        for idx, idx_v1 in enumerate(order):
            v1 = vocab[idx_v1]
            if args.fml_opt2 and premise[idx].item() < args.threshold2:
                break
            for v2 in vocab:
                cnt+=1
                ret_ltlfs = fun(v1, v2, (cnt,total,sat_cnt), vocab, traces_pos, args=args)
                sat_cnt += len(ret_ltlfs)
                assert(args.fml_topk == -1 or sat_cnt <= args.fml_topk)
                result.append(ret_ltlfs)
                if sat_cnt == args.fml_topk:
                    end_flag = True
                    break
            if end_flag:
                break
    else:
        for v1 in vocab:
            for v2 in vocab:
                cnt+=1
                ret_ltlfs = fun(v1, v2, (cnt,total,sat_cnt), vocab, traces_pos, args=args)
                sat_cnt += len(ret_ltlfs)
                result.append(ret_ltlfs)
    
    if not args.silent: print('')                         
    for ret in result:
        ltlfs.extend(ret)
    ltlfs.sort(key=lambda x: x[0], reverse=True)
    return ltlfs

def generate_neg_trace(trace,v1,v2,ltlf_tmp_idx):
    a_idxs = []
    b_idxs = []
    for state_idx, state in enumerate(trace):
        if v1 in state:
            a_idxs.append(state_idx)
        if v2 in state:
            b_idxs.append(state_idx)

    if len(a_idxs) == 0:
        return trace
    a_idx = random.choice(a_idxs)

    neg_trace = []
    if ltlf_tmp_idx == 0:
        for state_idx, state in enumerate(trace):
            if (state_idx > a_idx and state_idx in b_idxs):
                continue
            neg_trace.append(state)
    elif ltlf_tmp_idx == 1:
        b_idx = random.randint(a_idx, len(trace) - 1)
        for state_idx, state in enumerate(trace):
            neg_trace.append(state)
            if state_idx == b_idx:
                neg_trace.append([v2])
    elif ltlf_tmp_idx == 2:
        if len(b_idxs)==0:
            return trace
        b_idx = b_idxs[0]
        if b_idx > a_idx:
            return trace
        for state_idx, state in enumerate(trace):
            if state_idx == b_idx:
                neg_trace.append([v1])
            elif state_idx == a_idx:
                neg_trace.append([v2])
            else:
                neg_trace.append(state)
    elif ltlf_tmp_idx == 3:
        remove_mode = True
        for state_idx, state in enumerate(trace):
            if state_idx > a_idx and state_idx not in b_idxs:
                remove_mode = False
            if state_idx > a_idx and remove_mode and state_idx in b_idxs:
                continue
            neg_trace.append(state)
    elif ltlf_tmp_idx == 4:
        for state_idx, state in enumerate(trace):
            neg_trace.append(state)
            if state_idx == a_idx:
                neg_trace.append([v2])
    elif ltlf_tmp_idx == 5:
        if len(b_idxs)==0 or a_idxs[0]-1 not in b_idxs:
            return trace
        for state_idx, state in enumerate(trace):
            if state_idx == a_idxs[0]:
                neg_trace.append([v1])
                neg_trace.append([v2])
                continue
            if state_idx == a_idxs[0] - 1:
                continue
            neg_trace.append(state)
    return neg_trace

def gen_formulae_and_fsa(traces_pos, vocab, dataset, train, data_time, args):
    traces_pos = copy.deepcopy(traces_pos)
    stime = time.time()
    for idx,trace in enumerate(traces_pos):
        traces_pos[idx]=traces_pos[idx]+[[]]
    if not args.silent: print(f'getting formulaes, total: {len(vocab)**2} ...\n')
    ltlfs=get_formulae(traces_pos,vocab,args)
    if not args.silent: print(f'\nget formulae done, len: {len(ltlfs)}')
    etime = time.time() - stime
    if train:
        data_time['gen_formulae_time'] = etime
    else:
        data_time['test_gen_formulae_time'] = etime
    print(f'gen formulae time: {etime:.03f}')

    for idx,trace in enumerate(traces_pos):
        traces_pos[idx]=traces_pos[idx][:-1]
    if len(ltlfs) == 0:
        return [], None, None, None, None

    stime = time.time()
    formulaes = accept_list = neighbor_list = states = None
    if (args.sample_unlabel):
        formulaes, accept_list, neighbor_list, states = formulae2fsa(ltlfs, traces_pos, vocab, dataset, args)
    etime = time.time() - stime
    if train:
        data_time['ltl2fsa_time'] = etime
    else:
        data_time['test_ltl2fsa_time'] = etime
    print(f'ltl2fsa time: {etime:.03f}')
    return ltlfs, formulaes, accept_list, neighbor_list, states

def gen_potential_pos_traces(traces_pos, vocab, dataset, train, valid, data_time, args, n=-1):
    stime = time.time()
    traces_unlabel = []
    max_length = -1
    for trace in traces_pos:
        if len(trace) > max_length:
            max_length = len(trace)

    total_count = 0
    conflict_count1 = 0
    conflict_count2 = 0
    not_accept_count = 0
    timeout_count = 0
    if (train or valid) and args.sample_unlabel:
        if n == -1:
            n = len(traces_pos) if len(vocab) > 5 else 0
        fsa_path = f'model/init/{args.tag}/valk{args.val_k}/{dataset}/init_fsa.txt'
        fsm = FSM(Path(fsa_path))
        traces_unlabel = []
        traces_set = set()
        while len(traces_unlabel) < n:
            try:
                flag = True
                count = 0
                s = ''
                while flag:
                    flag = False
                    count += 1
                    total_count += 1
                    if count > TIMEOUT_COUNT:
                        raise TimeoutError(f'Timeout')
                    pos_length = random.randint(1, max_length)
                    trace = fsm.sample_pos_trace(pos_length)
                    t = [[state] for state in trace]
                    if not fsm.check(trace):
                        flag = True
                        not_accept_count += 1
                    else:
                        s = ','.join(trace)
                        if s in traces_set:
                            flag = True
                            conflict_count1 += 1
                        if t in traces_pos:
                            flag = True
                            conflict_count2 += 1
            except TimeoutError as e:
                timeout_count += 1
            finally:
                traces_set.add(s)
                traces_unlabel.append(t)
    etime = time.time() - stime
    if train:
        data_time['train_gen_unlabel_time'] = etime
        print(f'gen unlabel time: {etime:.03f}')
    elif valid:
        data_time['valid_gen_unlabel_time'] = etime
        print(f'gen unlabel time: {etime:.03f}')
    print(f'total: {total_count}, duplicate: {conflict_count1}, true_pos: {conflict_count2}, not_accept: {not_accept_count}, timeout: {timeout_count}')
    if total_count != 0:
        print(f'ratio, duplicate: {conflict_count1 / total_count:.3f}, true_pos: {conflict_count2 / total_count}, sum: {(conflict_count1 + conflict_count2) / total_count}')
    return traces_unlabel

def generate_neg_traces(traces_pos, vocab, dataset, ltlfs, train, valid, data_time: Dict, args, n=-1):
    if len(ltlfs) == 0:
        return []
    if args.generate_neg_simple:
        return generate_neg_traces_simple(traces_pos, vocab, dataset, ltlfs, train, valid, data_time, args, n=n)
    stime = time.time()
    traces_neg=[]
    eq_count = 0
    if n == -1:
        n = len(traces_pos)

    for t_idx,trace in enumerate(traces_pos):
        change_list=[]
        for num_sat, (v1, v2), ltlf_tmp_idx, sat_array in ltlfs:
            assert(num_sat / len(traces_pos) >= args.threshold1)
            if sat_array[t_idx]==1:
                change_list.append((v1,v2,ltlf_tmp_idx))
        if len(change_list)>0:
            for v1,v2,ltlf_tmp_idx in change_list:
                pass
            if len(change_list)==0:
                continue
            v1,v2,ltlf_tmp_idx=random.choice(change_list)
            new_trace = generate_neg_trace(trace,v1,v2,ltlf_tmp_idx)
            if new_trace == trace:
                eq_count += 1
            traces_neg.append(new_trace)
            ltlf_trees = [('U', ('->', v1, ('X', ('F', v2))), ('!', ('X', 'true'))),
                            ('U', ('->', v1, ('X', ('U', ('!', v2), ('!', ('X', 'true'))))), ('!', ('X', 'true'))),
                            ('W', ('!', v1), v2),
                            ('U', ('->', v1, ('X', v2)), ('!', ('X', 'true'))),
                            ('U', ('->', v1, ('X', ('!', v2))), ('!', ('X', 'true'))),
                            ('->', ('F', v1), ('U', ('!', v1), ('&', v2, ('X', v1))))]
    etime = time.time() - stime
    if train:
        data_time['train_gen_neg_time'] = etime
        print(f'gen neg time: {etime:.03f}')
    elif valid:
        data_time['valid_gen_neg_time'] = etime
        print(f'gen neg time: {etime:.03f}')
    else:
        data_time['test_gen_neg_time'] = etime
        print(f'gen neg time: {etime:.03f}')

    return traces_neg

def mutation(trace: List, vocab: List):
    if len(trace) == 0:
        return trace
    n = len(trace)
    new_trace = copy.deepcopy(trace)
    if n == 1:
        mode = random.choice([0,1])
    elif trace[:-1] == trace[1:]:
        mode = random.choice([0,1,3])
    else:
        mode = random.randint(0,3)
    if mode == 0: # Replacement
        idx = random.randint(0, n-1)
        token_range = vocab.copy()
        token_range.remove(new_trace[idx][0])
        token = random.choice(token_range)
        new_trace[idx] = [token]
    elif mode == 1: # Insertion
        idx = random.randint(0, n)
        token = random.choice(vocab)
        new_trace.insert(idx, [token])
    elif mode == 2: # Exchange
        a, b = random.sample(range(n), k=2)
        while new_trace[a] == new_trace[b]:
            a, b = random.sample(range(n), k=2)
        temp = new_trace[a]
        new_trace[a] = new_trace[b]
        new_trace[b] = temp
    else: # Deletion
        idx = random.randint(0, n-1)
        new_trace.pop(idx)
    return new_trace

def inject_noise(traces, eta, vocab, silent=False):
    n = len(traces)
    if eta <= 0 or eta > 1 or n == 0:
        return traces
    noise_num = int(eta * n)
    if not silent:
        print(f'inject noise num: {noise_num}')
    traces_noise = copy.deepcopy(traces)
    indices = random.choices(range(n), k=noise_num)
    for idx in indices:
        traces_noise[idx] = mutation(traces_noise[idx], vocab)
    return traces_noise

def noise_statistics(traces_pos, traces_neg, traces_unlabel, dataset, args):
    gt_path = f'{args.data_dir}/{dataset}/gt_fsm.txt'
    gt = FSM(Path(gt_path))
    TP = FP = TN = FN = TUN = FUN = 0
    for trace in traces_pos:
        t = [state[0] for state in trace]
        if gt.check(t):
            TP += 1
        else:
            FP += 1
    for trace in traces_neg:
        t = [state[0] for state in trace]
        if gt.check(t):
            FN += 1
        else:
            TN += 1
    for trace in traces_unlabel:
        t = [state[0] for state in trace]
        if gt.check(t):
            TUN += 1
        else:
            FUN += 1
    print(f'(actual traces statistics) TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}, TUN: {TUN}, FUN: {FUN}, noise rate pos (FP/P): {FP/(TP+FP)}, noise rate neg (FN/N): {FN/(TN+FN)}, noise rate (TUN/UN): {TUN/(TUN+FUN) if (TUN+FUN)!=0 else -1}, noise rate (FP+FN)/(P+N): {(FP+FN)/(TP+FN+FP+TN)}')
    return TP, FN, FP, TN, TUN, FUN

def generate_neg_traces_test(traces_pos, vocab, dataset, data_time: Dict, args, n=-1):
    stime = time.time()
    traces_neg = []
    eq_count = 0
    if n == -1:
        n = len(traces_pos)
    gt_path = f'{args.data_dir}/{dataset}/gt_fsm.txt'
    gt = FSM(Path(gt_path))
    traces_set = set()
    max_length = -1
    for trace in traces_pos:
        if len(trace) > max_length:
            max_length = len(trace)

    total_count = 0
    conflict_count = 0
    accept_count = 0
    timeout_count = 0
    while len(traces_neg) < n:
        try:
            flag = True
            count = 0
            s = ''
            while flag:
                flag = False
                count += 1
                total_count += 1
                if count > TIMEOUT_COUNT:
                    raise TimeoutError(f'Timeout')
                neg_length = random.randint(1, max_length)
                trace = gt.sample_neg_trace(neg_length)
                t = [[state] for state in trace]
                if gt.check(trace):
                    flag = True
                    accept_count += 1
                else:
                    s = ','.join(trace)
                    if s in traces_set:
                        flag = True
                        conflict_count += 1
        except TimeoutError as e:
            timeout_count += 1
        finally:
            traces_set.add(s)
            traces_neg.append(t)
            
    etime = time.time() - stime
    data_time['test_gen_neg_time'] = etime
    print(f'gen neg time: {etime:.03f}')
    print(f'total: {total_count}, duplicate: {conflict_count}, accept: {accept_count}, timeout: {timeout_count}')
    if total_count != 0:
        print(f'ratio, duplicate: {conflict_count / total_count:.3f}')
    return traces_neg

def generate_neg_traces_simple(traces_pos, vocab, dataset, ltlfs, train, valid, data_time: Dict, args, n=-1):
    stime = time.time()
    traces_neg=[]
    if n == -1:
        n = len(traces_pos)

    for t_idx, trace in enumerate(traces_pos):
        op = random.randint(0,5)
        trace_set = set(t[0] for t in trace)
        if len(trace_set) < 2:
            traces_neg.append(trace)
            continue
        v1, v2 = random.sample(list(trace_set), k=2)
        new_trace = generate_neg_trace(trace,v1,v2,op)
        traces_neg.append(new_trace)

    etime = time.time() - stime
    if train:
        data_time['train_gen_neg_time'] = etime
        print(f'gen neg time: {etime:.03f}')
    elif valid:
        data_time['valid_gen_neg_time'] = etime
        print(f'gen neg time: {etime:.03f}')
    else:
        data_time['test_gen_neg_time'] = etime
        print(f'gen neg time: {etime:.03f}')
    return traces_neg

def neighbor_init(ltlfs, vocab, word2index):
    init_info = {'not_init': [],
                 'init': [],
                 'no_loop': []}
    if len(ltlfs) == 0:
        return None
    init = [set() for i in range(len(vocab))]
    no_loop1 = [set() for i in range(len(vocab))]
    no_loop2 = set()
    for num_sat, (v1, v2), idx, _ in ltlfs:
        a = word2index[v1]
        b = word2index[v2]
        if idx == 2 or idx == 5: # !a W b, F(a) -> (!a U (b & Xa))
            init_info['not_init'].append(a)
            no_loop1[b].add(a)
        elif idx == 1: # G(a -> X(G(!b)))
            init[b].add(a)
            if a == b:
                no_loop2.add(b)
    for b in range(len(vocab)):
        if len(init[b]) == len(vocab):
            init_info['init'].append(b)
        if (len(no_loop1[b]) == len(vocab) - 1) and (b not in no_loop1) and (b in no_loop2):
            init_info['no_loop'].append(b)
    init_info['not_init'] = sorted(list(set(init_info['not_init'])))
    init_info['init'] = sorted(list(set(init_info['init'])))
    init_info['no_loop'] = sorted(list(set(init_info['no_loop'])))
    print(f'init_info: {init_info}')
    return init_info

class Data(object):
    def __init__(self, train_paths, test_path, dataset, train_args):
        self.data_time = {}
        self.dataset = dataset
        self.train_args = train_args
        self.device = train_args.device
        batch_size, k_g = train_args.batch_size, train_args.k_g
        dictionary={
                "traces_pos": [],
                "traces_neg": [],
                "traces_unlabel": []
        }
        dic = None
        for path in train_paths:
            d = json.load(open(path))
            dictionary["traces_pos"]+=d["traces_pos"]
            if dic is None: dic = d
        self.vocab = dic['vocab']
        self.index2word = dic['vocab']
        self.word2index = {v: k for k, v in enumerate(self.index2word)}

        self.ltlfs, self.formulaes, self.accept, self.neighbor, self.states = gen_formulae_and_fsa(dictionary['traces_pos'], self.vocab, dataset, train=True, data_time=self.data_time, args=train_args)
        self.has_ltlfs = (len(self.ltlfs) > 0)
        if self.has_ltlfs:
            self.init_info = neighbor_init(self.ltlfs, self.vocab, self.word2index)
            self.formulaes_info = {'formulaes': self.formulaes,
                                   'states': self.states,
                                   'accept': self.accept,
                                   'neighbor': self.neighbor,
                                   'init_info': self.init_info}
        else:
            self.init_info = None
            self.formulaes_info = None

        self.fsm_states = [i for i in range(k_g)]

        # ---- train ----
        self.train_traces_pos = dictionary['traces_pos']
        self.train_traces_neg = []
        self.train_traces_unlabel = []
        self.data_time['train_gen_unlabel_time'] = 0
        self.data_time['train_gen_neg_time'] = 0
        print('---- generate neg traces for training set ----\n')
        if (train_args.noise > 0) and (train_args.noise <= 1):
            self.train_traces_pos = inject_noise(self.train_traces_pos, train_args.noise, self.vocab, silent=train_args.silent)
        if train_args.generate_neg and self.has_ltlfs:
            self.train_traces_neg = generate_neg_traces(self.train_traces_pos, self.vocab, dataset, self.ltlfs, train=True, valid=False, data_time=self.data_time, args=train_args)
        if train_args.noise_statistics:
            noise_statistics(self.train_traces_pos, self.train_traces_neg, self.train_traces_unlabel, dataset, train_args)
        if not train_args.silent:
            print('train, number of original traces:',len(self.train_traces_pos),' number of generated neg traces:',len(self.train_traces_neg),' number of generated potential pos traces:',len(self.train_traces_unlabel))
        print('\n----------\n')

        self.train_pos = [(self._read_trace(trace), 1.0, 1.0) for trace in self.train_traces_pos]
        self.train_neg = [(self._read_trace(trace), 0.0, 1.0) for trace in self.train_traces_neg]
        self.train_unlabel = [(self._read_trace(trace), 0.0, 0.0) for trace in self.train_traces_unlabel]
        self.train = self.train_pos + self.train_neg + self.train_unlabel
        # ---- train ----

        # ---- valid ----
        print('---- generate neg traces for validation set ----\n')
        self.valid_traces_pos = copy.deepcopy(self.train_traces_pos)
        self.valid_traces_neg = []
        self.valid_traces_unlabel = []
        if train_args.generate_valid_neg and self.has_ltlfs:
            self.valid_traces_neg = generate_neg_traces(self.valid_traces_pos, self.vocab, dataset, self.ltlfs, train=False, valid=True, data_time=self.data_time, args=train_args)
        if self.has_ltlfs:
            self.valid_traces_unlabel = gen_potential_pos_traces(self.valid_traces_pos, self.vocab, dataset, train=False, valid=True, data_time=self.data_time, args=train_args)
        if train_args.noise_statistics:
            noise_statistics(self.valid_traces_pos, self.valid_traces_neg, self.valid_traces_unlabel, dataset, train_args)
        if not train_args.silent:
            print('valid, number of original traces:',len(self.valid_traces_pos),' number of generated neg traces:',len(self.valid_traces_neg),' number of generated potential pos traces:',len(self.valid_traces_unlabel))
        print('\n----------\n')

        self.valid_pos = [(self._read_trace(trace), 1.0, 1.0) for trace in self.valid_traces_pos]
        self.valid_neg = [(self._read_trace(trace), 0.0, 1.0) for trace in self.valid_traces_neg]
        self.valid_unlabel = [(self._read_trace(trace), 0.0, 0.0) for trace in self.valid_traces_unlabel]
        self.valid = self.valid_pos + self.valid_neg + self.valid_unlabel
        # ---- valid ----

        # ---- test ----
        dictionary = json.load(open(test_path))
        self.test_traces_pos = dictionary['traces_pos']
        self.test_traces_neg = []
        self.test_traces_unlabel = []

        if train_args.generate_test_neg:
            print('---- generate neg traces for test set ----\n')
            self.test_traces_neg = generate_neg_traces_test(self.test_traces_pos, self.vocab, dataset, data_time=self.data_time, args=train_args)

            if not train_args.silent:
                print('test, number of original traces:',len(self.test_traces_pos),' number of generated neg traces:',len(self.test_traces_neg),' number of generated potential pos traces:',len(self.test_traces_unlabel))
            print('\n----------\n')

        self.test_pos = [(self._read_trace(trace), 1.0) for trace in self.test_traces_pos]
        self.test_neg = [(self._read_trace(trace), 0.0) for trace in self.test_traces_neg]
        self.test_unlabel = [(self._read_trace(trace), 0.0) for trace in self.test_traces_unlabel]
        self.test = self.test_pos + self.test_neg + self.test_unlabel
        # ---- test ----

        self._batch_size = batch_size
        pin_memory = train_args.pin_memory if torch.cuda.is_available() and train_args.gpu != -1 else False
        self.valid_dataloader = DataLoader(self.valid, batch_size=self._batch_size, shuffle=True, num_workers=train_args.num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
        self.train_dataloader = DataLoader(self.train, batch_size=self._batch_size, shuffle=True, num_workers=train_args.num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
        self.train_dataloaders = {'train_dataloader': self.train_dataloader,
                                  'valid_dataloader': self.valid_dataloader}

    def _read_trace(self, trace):
        trace_vec = torch.zeros(len(trace) * len(self.vocab), len(self.fsm_states), len(self.fsm_states))
        trace_indices = torch.LongTensor([i * len(self.vocab) + self.word2index[trace[i][0]] for i in range(len(trace))])
        trace_vec.index_copy_(0, trace_indices, torch.eye(len(self.fsm_states)).unsqueeze(0).repeat(len(trace), 1, 1))
        trace_vec = trace_vec.reshape((len(trace), len(self.vocab) * len(self.fsm_states), len(self.fsm_states)))
        return trace_vec
    
    def _get_trace_from_tensor(self, trace_vec: torch.Tensor):
        trace = []
        for i in range(trace_vec.size(0)):
            for word in self.vocab:
                k = self.word2index[word] * len(self.fsm_states)
                if trace_vec[i][k][0] == 1.0:
                    trace.append([word])
                    break
        return trace

    def resize(self, train_args):
        def _resize(dataset, k_g, vocab):
            ret = []
            for trace, label in dataset:
                trace = trace.reshape(trace.size(0), len(vocab), trace.size(2), trace.size(2))
                trace = trace[:, :, :k_g, :k_g]
                trace = trace.reshape(trace.size(0), k_g * len(vocab), k_g)
                ret.append((trace, label))
            return ret

        k_g = train_args.k_g
        self.fsm_states = [i for i in range(k_g)]
        self.train_pos = _resize(self.train_pos, k_g, self.vocab)
        self.train_neg = _resize(self.train_neg, k_g, self.vocab)
        self.train_unlabel = _resize(self.train_unlabel, k_g, self.vocab)
        self.train = self.train_pos + self.train_neg + self.train_unlabel

        self.valid_pos = _resize(self.valid_pos, k_g, self.vocab)
        self.valid_neg = _resize(self.valid_neg, k_g, self.vocab)
        self.valid_unlabel = _resize(self.valid_unlabel, k_g, self.vocab)
        self.valid = self.valid_pos + self.valid_neg + self.valid_unlabel

        self.test_pos = _resize(self.test_pos, k_g, self.vocab)
        self.test_neg = _resize(self.test_neg, k_g, self.vocab)
        self.test_unlabel = _resize(self.test_unlabel, k_g, self.vocab)
        self.test = self.test_pos + self.test_neg + self.test_unlabel

        # self.train_dataloader = DataLoader(self.train, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        self.valid_dataloader = DataLoader(self.valid, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        self.train_dataloader_pos = DataLoader(self.train_pos, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        self.train_dataloader_neg = DataLoader(self.train_neg, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        if len(self.train_unlabel) > 0:
            self.train_dataloader_unlabel = DataLoader(self.train_unlabel, batch_size=self._batch_size, shuffle=True, collate_fn=collate_fn)
        self.train_dataloaders = {# 'train_dataloader': self.train_dataloader,
                                  'train_dataloader_pos': self.train_dataloader_pos,
                                  'train_dataloader_neg': self.train_dataloader_neg,
                                  'train_dataloader_unlabel': self.train_dataloader_unlabel,
                                  'valid_dataloader': self.valid_dataloader}
        return

def collate_fn_dynamic(dataset):
    # dataset: [(_x, _y, _mask)] * batch_size
    # _x: Tensor(3 * max_trace_len * (num_state num_word) * num_state)
    # _y: Tensor(3 * 1)
    # _mask: Tensor(3 * max_trace_len * 1)
    x = [dataset[i][0] for i in range(len(dataset))]
    y = [dataset[i][1] for i in range(len(dataset))]
    mask = [dataset[i][2] for i in range(len(dataset))]
    x = torch.stack(x)
    y = torch.stack(y)
    mask = torch.stack(mask)
    x = torch.transpose(x, 0, 1)
    y = torch.transpose(y, 0, 1)
    mask = torch.transpose(mask, 0, 1)
    x = torch.reshape(x, (-1, x.size(2), x.size(3), x.size(4)))
    y = torch.reshape(y, (-1, y.size(2)))
    mask = torch.reshape(mask, (-1, mask.size(2), mask.size(3)))
    # x: Tensor((3 * batch_size) * max_trace_len * (num_state num_word) * num_state)
    # y: Tensor((3 * batch_size) * 1)
    # mask: Tensor((3 * batch_size) * max_trace_len * 1)
    return x, y, mask

def collate_fn(dataset):
    # dataset: [(trace,label)]
    # trace: Tensor(trace_len * (num_state num_word) * num_state)
    # label: 0.0/1.0
    # device = dataset[0][0].device

    max_trace_len = -1
    for data in dataset:
        if len(data[0]) > max_trace_len:
            max_trace_len = len(data[0])
    # print('max_trace_len:',max_trace_len)

    # new_data_x: Tensor(batch_size * max_trace_len * (num_state num_word) * num_state)
    new_data_x = torch.zeros(len(dataset), max_trace_len, len(dataset[0][0][0]), len(dataset[0][0][0][0]))
    # new_data_y: Tensor(batch_size * 1)
    new_data_y = torch.zeros(len(dataset), 1)
    # predict_mask: Tensor(batch_size * max_trace_len * 1)
    predict_mask = torch.zeros(len(dataset), max_trace_len, 1)
    # flags: Tensor(batch_size * 1)
    flags = torch.zeros(len(dataset), 1)

    cnt = 0
    for data in dataset:
        data_x = data[0]
        if len(data[0]) < max_trace_len:
            data_x = torch.cat((torch.zeros(max_trace_len-len(data[0]), len(data[0][0]), len(data[0][0][0])), data_x))
        new_data_x[cnt] = data_x
        new_data_y[cnt][0] = data[1]
        predict_mask[cnt][max_trace_len-len(data[0])][0] = 1
        flags[cnt][0] = data[2]
        cnt += 1

    return new_data_x, new_data_y, predict_mask, flags


def getLTLscore(traces_pos,traces_neg,vocab):
    ltlfs=[]
    stime=time.time()
    cnt=0
    # cache={}
    trace_num=len(traces_pos)+len(traces_neg)
    for v1 in vocab:
        print('ltl miner processing:%d/%d' % (cnt, len(vocab) ** 2))
        for v2 in vocab:
            ltlf_trees=[('G',('->',v1,('X',('F',v2)))),
                        ('G', ('->', v1, ('X', ('G', ('!', v2))))),
                        ('G',('->',v1,('X',v2))),
                        ('G',('->',v1,('X',('!',v2)))),
                        ('->',('F',v1),('U',('!',v1),('&',v2,('X',v1))))]
            cnt+=1
            if v1!=v2:
                ltlf_trees.append(('W', ('!', v1), v2))
            for ltlf_tree in ltlf_trees:
                ltlf = LTLf(vocab, ltlf_tree)
                check_pos_mark, check_neg_mark = ltlf.evaluate(traces_pos, traces_neg)
                ltlfs.append((check_pos_mark,check_neg_mark,(sum(check_pos_mark)+len(check_neg_mark)-sum(check_neg_mark))/trace_num,ltlf_tree))
    ltlfs.sort(key=lambda x:x[2],reverse=True)
    # for _,_,score,ltlf_tree in ltlfs:
    #     print(score,ltlf_tree)
    print('ltl process time:',time.time()-stime)
    traces_pos_ltl_score=[0]*len(traces_pos)
    traces_neg_ltl_score=[0]*len(traces_neg)
    ltlfs_=[]
    for ltlf in ltlfs:
        if ltlf[2]>0.5:
            ltlfs_.append(ltlf)
        elif ltlf[2]<0.5:
            ltlfs_.append(([1-i for i in ltlf[0]],[1-i for i in ltlf[1]],1-ltlf[2],ltlf[3]))
        else:
            continue
    for ltlf in ltlfs_:
        for j in range(len(traces_pos)):
            traces_pos_ltl_score[j]+=ltlf[0][j]/len(ltlfs_)
        for j in range(len(traces_neg)):
            traces_neg_ltl_score[j]+=(1-ltlf[1][j])/len(ltlfs_)
    # print(traces_pos_ltl_label)
    # traces_pos_ltl_label.sort()
    # print(traces_pos_ltl_label)
    return traces_pos_ltl_score,traces_neg_ltl_score




