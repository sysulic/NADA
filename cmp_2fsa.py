import json
from pathlib import Path
from typing import Dict, List, Set
import random
import re
import argparse
from FSM_for_cmp import FSM
class timeout_exception(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def random_trace(fsm: FSM, length: int, black_list: Set[str], pos_trace: bool, vocab: List[str]):
    flag = True
    count = 0
    s = ''
    while flag:
        flag = False
        count += 1
        if count > 10000:
            raise timeout_exception(f'Timeout, {"pos trace" if pos_trace else "neg trace"}, length={length}')
        pos_length = random.randint(1, length)
        trace = fsm.sample_pos_trace(pos_length)
        if fsm.check(trace) != pos_trace:
            flag = True
        else:
            s = ','.join(trace)
            if s in black_list:
                flag = True
    return trace, s

def get_transition_pro(fsm: FSM,traces_pos):
    for trace_idx,_ in enumerate(traces_pos):
        for state_idx,_ in enumerate(traces_pos[trace_idx]):
            traces_pos[trace_idx][state_idx]=_[0]
    # traces_pos=traces_pos[:2]
    # print(traces_pos)
    edge2pro = {}  # (start,action,end):pro
    for trace in traces_pos:
        father={} # (state,timestep)->[state]
        current_states=fsm.init_states
        for timestep,action in enumerate(trace):
            n_current_states=set()
            for current_state in current_states:
                if action in fsm.G[current_state].keys():
                    for next_state in fsm.G[current_state][action]:
                        n_current_states.add(next_state)
                        if (next_state,timestep) not in father.keys():
                            father[(next_state,timestep)]=[]
                        father[(next_state,timestep)].append(current_state)
            current_states=n_current_states
        acc_states=n_current_states.intersection(fsm.final_states)

        state2pro={(state,len(trace)-1):1/len(acc_states) for state in acc_states} # (state,timestep):pro
        # print('trace',trace)
        # print('state2pro',state2pro)
        # print('father',father)
        for timestep in range(len(trace)-1,-1,-1):
            # print('timestep',timestep)
            n_acc_states=set()
            for state in acc_states:
                for pre_state in father[(state,timestep)]:
                    n_acc_states.add(pre_state)
                    key=(pre_state,trace[timestep],state)
                    if key not in edge2pro.keys():
                        edge2pro[key]=0
                    # print('state2pro',state2pro)

                    edge2pro[key]+=state2pro[(state,timestep)]/len(father[(state,timestep)])
                    key=(pre_state,timestep-1)
                    if key not in state2pro.keys():
                        state2pro[key]=0
                    state2pro[key]+=state2pro[(state,timestep)]/len(father[(state,timestep)])
            acc_states=n_acc_states
    # for trace in traces_pos:
    #     print(trace)
    # print(edge2pro)
    state2cnt={}
    for s_state,_1,_2 in edge2pro.keys():
        if s_state not in state2cnt.keys():
            state2cnt[s_state]=0
        state2cnt[s_state]+=edge2pro[(s_state,_1,_2)]
    ret_dic={}
    for key in edge2pro.keys():
        edge2pro[key]/=state2cnt[key[0]]
        if key[0] not in ret_dic.keys():
            ret_dic[key[0]]=[]
        ret_dic[key[0]].append((key[1],key[2],edge2pro[key]))
        ret_dic[key[0]].sort()
    # print(edge2pro)
    # print(ret_dic)



    return ret_dic





def traces_have_end(traces):
    for trace in traces:
        if '<END>' in trace:
            return True
    else:
        return False

def remove_end(traces):
    ntraces=[]
    for trace in traces:
        ntrace=[]
        for state in trace:
            if state=='<END>':
                break
            ntrace.append(state)
        ntraces.append(ntrace)
    return ntraces

def add_end(traces):
    ntraces=[trace+['<END>'] for trace in traces]
    return ntraces

def cmp_2fsas(fsa1path,fsa2path,test_case=500,max_length=50, args=None, pos_name='input.json'):
    # print(f'read fsa from file {fsa1path}')
    fsm1 = FSM(Path(fsa1path))
    # print(fsm1.G)
    if not fsm1.satisfiable():
        print('No positive trace for fsm1! abort')
        return 0,0,[],[]
    # print(f'read fsa from file {fsa2path}')
    fsm2 = FSM(Path(fsa2path))
    # print(fsm2.G)
    # print(f'generate traces of {fsa1path}')
    fsm1_traces_pos = []
    fsm1_traces_set=set()
    while len(fsm1_traces_pos)<test_case:
        try:
            flag = True
            count = 0
            s = ''
            while flag:
                flag = False
                count += 1
                if count > 10000:
                    raise timeout_exception(f'Timeout')
                    # raise timeout_exception(f'Timeout, {"pos trace" if pos_trace else "neg trace"}, length={length}')
                pos_length = random.randint(1, max_length)
                trace = fsm1.sample_pos_trace(pos_length)
                if not fsm1.check(trace):
                    flag = True
                else:
                    s = ','.join(trace)
                    if s in fsm1_traces_set:
                        flag = True
            fsm1_traces_set.add(s)
            fsm1_traces_pos.append([state for state in trace])
        except timeout_exception as e:
            # print(f'{fsa1path} failed, trace_len{t_len}')
            break
    # print(f'generate traces of {fsa2path}')

    transition_pro = get_transition_pro(fsm2, json.load(open(fsa2path.replace('gt_fsm.txt', pos_name), 'r'))[
        'traces_pos'])
    fsm2_traces_pos=[]
    fsm2_traces_set = set()
    while len(fsm2_traces_pos)<test_case:
        try:
            flag = True
            count = 0
            s = ''
            while flag:
                flag = False
                count += 1
                if count > 10000:
                    raise timeout_exception(f'Timeout')
                    # raise timeout_exception(f'Timeout, {"pos trace" if pos_trace else "neg trace"}, length={length}')
                pos_length = random.randint(1, max_length)
                trace = fsm2.sample_pos_trace_by_pro(pos_length,transition_pro)
                if not fsm2.check(trace):
                    flag = True
                else:
                    s = ','.join(trace)
                    if s in fsm2_traces_set:
                        flag = True
            fsm2_traces_set.add(s)
            fsm2_traces_pos.append([state for state in trace])
        except timeout_exception as e:
            # print(f'{fsa1path} failed, trace_len{t_len}')
            break


    TP1=TP2=0
    if traces_have_end(fsm1_traces_pos) and not traces_have_end(fsm2_traces_pos):
        fsm1_traces_pos=remove_end(fsm1_traces_pos)
        fsm2_traces_pos=add_end(fsm2_traces_pos)
    elif traces_have_end(fsm2_traces_pos) and not traces_have_end(fsm1_traces_pos):
        fsm1_traces_pos=add_end(fsm1_traces_pos)
        fsm2_traces_pos=remove_end(fsm2_traces_pos)
    for i in fsm1_traces_pos:
        if fsm2.check(i):
            TP1+=1
        else:
            if args.print_traces: print('FP1 trace: ', i)
    for i in fsm2_traces_pos:
        if fsm1.check(i):
            TP2+=1
        else:
            if args.print_traces: print('FP2 trace: ', i)
    return TP1,TP2,fsm1_traces_pos,fsm2_traces_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test fsa')
    parser.add_argument('--datasets', type=str, nargs='+', default=[])
    parser.add_argument('--tag', type=int, required=False, help='tag',default=10005)
    parser.add_argument('--test_case', type=int, required=False,default=100)
    parser.add_argument('--seed', type=int, required=False, default=-1)
    parser.add_argument('--print_traces', type=int, required=False, default=0)
    parser.add_argument('--init', action='store_true', required=False)
    parser.add_argument('--init_postprocess', action='store_true', required=False)
    parser.add_argument('--val_k', type=int, required=False, default=0)
    args = parser.parse_args()
    # python cmp_2fsa.py --tag=1 --test_case=500

    if args.seed != -1:
        # torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.datasets == []:
        datasets = ['ArrayList', 'HashMap', 'HashSet', 'Hashtable', 'LinkedList', 'NumberFormatStringTokenizer',
                    'Signature', 'Socket', 'StringTokenizer', 'ZipOutputStream', 'StackAr']
    else:
        datasets = args.datasets
    
    for dataset in datasets:
        if args.init:
            TP1,TP2,fsm1_traces_pos,fsm2_traces_pos=cmp_2fsas(f'model/init/valk{args.val_k}/{dataset}/init_fsa.txt',f'data_ori_split10_neg3/{dataset}/gt_fsm.txt',args.test_case, args=args)
        elif args.init_postprocess:
            TP1,TP2,fsm1_traces_pos,fsm2_traces_pos=cmp_2fsas(f'model/init_postprocess/valk{args.val_k}/{dataset}/init_fsa.txt',f'data_ori_split10_neg3/{dataset}/gt_fsm.txt',args.test_case, args=args)
        else:
            TP1,TP2,fsm1_traces_pos,fsm2_traces_pos=cmp_2fsas(f'model/{args.tag}/{dataset}/final_fsa.txt',f'data_ori_split10_neg3/{dataset}/gt_fsm.txt',args.test_case, args=args)
        n1, n2 = len(fsm1_traces_pos), len(fsm2_traces_pos)
        print(dataset,TP1,TP2,n1,n2)
