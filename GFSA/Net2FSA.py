import csv
import pathlib
import time
import torch

def fsa_from_neighbor_list(num_state, vocab, accept_list, neighbor_list):
    ret_fsa={'accept':[],'neighbor':[],'states':['C%d'%i for i in range(num_state)],'start':['C0'],'neighbormap':{}}
    for acc in accept_list:
      ret_fsa['accept'].append(f'C{acc}')
    for _from, label, _to in neighbor_list:
      word = label
      if (f'C{_from}', word) not in ret_fsa['neighbormap'].keys():
        ret_fsa['neighbormap'][(f'C{_from}', word)] = []
      ret_fsa['neighbormap'][(f'C{_from}', word)].append(f'C{_to}')
      ret_fsa['neighbor'].append([f'C{_from}', f'C{_to}', word])
    return ret_fsa

def print_fsa(fsa):
    print(len(fsa['start']))
    for i in fsa['start']:
        print(i)
    print(len(fsa['accept']))
    for i in fsa['accept']:
        print(i)
    print(len(fsa['neighbor']))
    for i in fsa['neighbor']:
        print(i[0],'\t',i[1],'\t',i[2])

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

def accept(fsa,trace,t,state,cache:dict):
    if (t,state) in cache:
        return cache[(t,state)]
    if t==len(trace):
        cache[(t, state)]=state in fsa['accept']
        return state in fsa['accept']
    if (state,trace[t][0]) not in fsa['neighbormap'].keys():
        cache[(t, state)] = False
        return False
    for to_state in fsa['neighbormap'][(state,trace[t][0])]:
        if accept(fsa,trace,t+1,to_state,cache):
            cache[(t, state)] = True
            return True
    cache[(t, state)] = False
    return False

def test_fsa(fsa,traces_pos,traces_neg):
    test_results=[]
    TP = FN = FP = TN = 0
    # cnt=0
    for trace in traces_pos:
        # print('\rtesting trace (%d/%d)'%(cnt,len(traces_neg)+len(traces_pos)),end='')
        # cnt+=1
        cache={}
        if accept(fsa,trace,0,fsa['start'][0],cache):
            TP+=1
            test_results.append([1,1])
        else:
            FN+=1
            test_results.append([1,0])
    for trace in traces_neg:
        # print('\rtesting trace (%d/%d)' % (cnt, len(traces_neg) + len(traces_pos)), end='')
        # cnt+=1
        cache={}
        if accept(fsa,trace,0,fsa['start'][0],cache):
            FP+=1
            test_results.append([0,1])
        else:
            TN+=1
            test_results.append([1,0])
    acc = (TP + TN) / (TP + FN + FP + TN)
    if TP + FP == 0.0:
        pre = -1.0
    else:
        pre = (TP) / (TP + FP)
    if TP + FN == 0.0:
        rec = -1.0
    else:
        rec = (TP) / (TP + FN)
    if pre + rec == 0.0:
        F1 = -1.0
    else:
        F1 = (2 * pre * rec) / (pre + rec)
    return TP, FN, FP, TN, acc, pre, rec, F1, test_results



def net2fsa(file,d,state_num):
    net=torch.load(file)
    # print(net.state_dict()['_neighbor'])
    neighbor=torch.sigmoid(net._neighbor)
    # neighbor=net.state_dict()['_neighbor']
    accept=torch.sigmoid(net._accept)
    # accept=net.state_dict()['_accept']

    ret_fsa={'accept':[],'neighbor':[],'states':['C%d'%i for i in range(state_num)],'start':['C0'],'neighbormap':{}}
    for i in range(state_num):
        if accept[i]>0.5:
            ret_fsa['accept'].append('C%d'%i)
    for word_idx in range(len(d.index2word)):
        for from_state in range(state_num):
            for to_state in range(state_num):
                if neighbor[from_state][to_state+word_idx*state_num]>0.5:
                    if ('C%d'%from_state,d.index2word[word_idx]) not in ret_fsa['neighbormap'].keys():
                        ret_fsa['neighbormap'][('C%d'%from_state,d.index2word[word_idx])]=[]
                    ret_fsa['neighbormap'][('C%d' % from_state, d.index2word[word_idx])].append('C%d'%to_state)
                    ret_fsa['neighbor'].append(['C%d'%from_state,'C%d'%to_state,d.index2word[word_idx]])

    # print_fsa(ret_fsa)
    return ret_fsa

def add_ele(best_k,i,top_num):
    t=[]
    for k in best_k:
        t.append([k[0]+(1-i),k[1]+[1]])
        t.append([k[0]+i,k[1]+[0]])
    t.sort(key=lambda x:x[0])
    if len(t)>top_num:
        t=t[:top_num]
    return t


def get_best_fsa(ret_fsas,traces_pos,traces_neg):
    best_acc=0
    best_fsa=ret_fsas[0]
    cnt=0
    for fsa in ret_fsas:
        # print('testing fsa (%d/%d)'%(cnt,len(ret_fsas)))
        if len(fsa['accept'])==0:
            continue
        cnt+=1
        # print(fsa)
        TP, FN, FP, TN, acc, pre, rec, F1, _ = test_fsa(fsa, traces_pos, traces_neg)
        if acc>best_acc:
            best_acc=acc
            best_fsa=fsa
        # print('(%d) TP: %d, FN: %d, FP: %d, TN: %d, acc: %0.3f, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (
        #     TP + FN + FP + TN, TP, FN, FP, TN, acc, pre, rec, F1))
    return best_fsa


def net2fsa_topk(file,d,state_num,top_num,theta=0.5):
    int_time=time.time()
    net=torch.load(file)
    # print(net.state_dict()['_neighbor'])
    neighbor=torch.sigmoid(net._neighbor).cpu().detach().numpy()
    # neighbor=net.state_dict()['_neighbor']
    accept=torch.sigmoid(net._accept).cpu().detach().numpy()
    # accept=net.state_dict()['_accept']


    best_k=[[0,[]]]
    for i in range(state_num):
        best_k=add_ele(best_k,accept[i],top_num)
    for word_idx in range(len(d.index2word)):
        for from_state in range(state_num):
            for to_state in range(state_num):
                best_k = add_ele(best_k, neighbor[from_state][to_state+word_idx*state_num], top_num)

    def list2fsa(convert_list):
        cnt=0
        ret_fsa={'accept':[],'neighbor':[],'states':['C%d'%i for i in range(state_num)],'start':['C0'],'neighbormap':{}}
        for i in range(state_num):
            if convert_list[cnt]>=theta:
                ret_fsa['accept'].append('C%d'%i)
            cnt+=1
        for word_idx in range(len(d.index2word)):
            for from_state in range(state_num):
                for to_state in range(state_num):
                    if convert_list[cnt]>=theta:
                        if ('C%d'%from_state,d.index2word[word_idx]) not in ret_fsa['neighbormap'].keys():
                            ret_fsa['neighbormap'][('C%d'%from_state,d.index2word[word_idx])]=[]
                        ret_fsa['neighbormap'][('C%d' % from_state, d.index2word[word_idx])].append('C%d'%to_state)
                        ret_fsa['neighbor'].append(['C%d'%from_state,'C%d'%to_state,d.index2word[word_idx]])
                    cnt+=1
        return ret_fsa

    ret_fsas=[]
    for i in best_k:
        # print('loss of fsa:',i[0])
        ret_fsas.append(list2fsa(i[1]))
    int_time=time.time()-int_time
    rf_time=time.time()
    ret_fsa=get_best_fsa(ret_fsas,d.train_traces_pos,d.train_traces_neg)
    rf_time=time.time()-rf_time
    # print_fsa(ret_fsa)
    return ret_fsa,int_time,rf_time