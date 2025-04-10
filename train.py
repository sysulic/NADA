from multiprocessing import Pool
import csv
import pathlib
import json
import torch
import time
import argparse
import random
import os
import copy
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from GFSA.DataSets import Data
from GFSA.FSANet import FSANet
from GFSA.Net2FSA import test_fsa,write_fsa
from cmp_2fsa import cmp_2fsas
from convert_fsa_to_dot import write_dot_img
from FSM_for_cmp import FSM

work_dir = os.getcwd()

def one_train(dataset,train_args):
    stime=time.time()
    torch.set_num_threads(1)
    torch.set_printoptions(1, threshold=1e5, sci_mode=False)

    summary_name = Path(f"./Tensorboard/{train_args.tag}/{dataset}")
    if not os.path.exists(summary_name):
        os.makedirs(summary_name)
    else:
        os.popen(f'rm {summary_name}/events.*').close()
    writer = SummaryWriter(summary_name)

    if train_args.K==0:
        if train_args.data_dir == 'data_synthesis':
            train_paths = [f'./{train_args.data_dir}/{dataset}/input_train.json']
            test_path = f'./{train_args.data_dir}/{dataset}/input_test.json'
        else:
            train_paths = [f'./{train_args.data_dir}/{dataset}/input.json']
            test_path = f'./{train_args.data_dir}/{dataset}/input.json'
    else:
        train_paths=[]
        if train_args.data_dir.find("data_event") == -1:
            for i in range(train_args.K):
                if i==train_args.val_k:
                    continue
                train_paths.append(f'./{train_args.data_dir}/{dataset}/input_{i}.json')
        else:
            train_paths = [f'./{train_args.data_dir}/{dataset}/input_train_valk{train_args.val_k}.json']
        test_path = f'./{train_args.data_dir}/{dataset}/input_{train_args.val_k}.json'
    if train_args.enhanced_traces:
        if os.path.exists(f'./{train_args.data_dir}/{dataset}/input_enhanced.json'):
            train_paths.append(f'./{train_args.data_dir}/{dataset}/input_enhanced.json')
        else:
            print(f'enhanced traces for {train_args.data_dir}/{dataset} does not exist!')
    start_time = time.time()
    d = Data(train_paths, test_path, dataset, train_args)
    end_time = time.time() - start_time
    d.data_time['get_datasets_total_time'] = end_time
    print(f'get datasets total time: {end_time:0.3f}')

    if train_args.save_data:
        dic = {"traces_pos": d.train_traces_pos, 
            "traces_neg": d.train_traces_neg,
            "vocab": d.vocab}
        if not os.path.exists(f'./data_gen/{dataset}'):
            os.makedirs(f'./data_gen/{dataset}')
        file_name = f'./data_gen/{dataset}/input_train_valk{train_args.val_k}_neg_seed{train_args.seed}.json'
        if train_args.noise > 0 and train_args.noise <= 1:
            file_name = f'./data_gen/{dataset}/input_train_valk{train_args.val_k}_neg_seed{train_args.seed}_noise{train_args.noise}.json'
        with open(file_name, 'w') as f:
            json.dump(dic, f)

        dic = {"traces_pos": d.valid_traces_pos, 
            "traces_neg": d.valid_traces_neg,
            "traces_unlabel": d.valid_traces_unlabel,
            "vocab": d.vocab}
        file_name = f'./data_gen/{dataset}/input_valid_valk{train_args.val_k}_neg_seed{train_args.seed}.json'
        if train_args.noise > 0 and train_args.noise <= 1:
            file_name = f'./data_gen/{dataset}/input_valid_valk{train_args.val_k}_neg_seed{train_args.seed}_noise{train_args.noise}.json'
        with open(file_name, 'w') as f:
            json.dump(dic, f)

        dic = {"traces_pos": d.test_traces_pos, 
            "traces_neg": d.test_traces_neg,
            "vocab": d.vocab}
        file_name = f'./data_gen/{dataset}/input_test_valk{train_args.val_k}_neg_seed{train_args.seed}.json'
        if train_args.noise > 0 and train_args.noise <= 1:
            file_name = f'./data_gen/{dataset}/input_test_valk{train_args.val_k}_neg_seed{train_args.seed}_noise{train_args.noise}.json'
        with open(file_name, 'w') as f:
            json.dump(dic, f)
        
        if train_args.save_data_only:
            exit(0)

    if train_args.sample_unlabel and d.has_ltlfs:
        fsa_path = f'model/init/{train_args.tag}/valk{train_args.val_k}/{dataset}/init_fsa.txt'
        dot_file_path = f'graph/dot/init/{train_args.tag}/valk{train_args.val_k}/{dataset}_init_fsa.dot'
        fsa_img_path = f'graph/img/init/{train_args.tag}/valk{train_args.val_k}/{dataset}_init_fsa.png'
        write_dot_img(fsa_path, dot_file_path, fsa_img_path, train_args.silent, only_dot=(train_args.k_g > 10))

    reinit = False
    k_g = train_args.k_g
    print(f'({dataset}) net num_state: {k_g}')
    net = FSANet(k_g, len(d.vocab), train_args.batch_size, train_args.lr, d.formulaes_info, train_args, reinit)

    result_path = f'./model/{train_args.tag}/{dataset}'
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    epoch = train_args.epoch
    if train_args.epoch < 0:
        num_batches = len(d.train_dataloader)
        epoch = train_args.samples // num_batches // train_args.batch_size
    print_frequence = train_args.print_frequence
    if print_frequence <= 0:
        print_frequence = epoch//10+1
    step_per_epoch = len(d.train_dataloader)
    
    start_time = time.time()
    for j in range(1, epoch + 1):
        model_path = f'{result_path}/{j}.mdl'

        debug_args = {'vocab': d.vocab}
        loss, regular_loss, score, train_time, valid_time = net.net_train_batch(d.train_dataloaders, j, writer, debug_args=debug_args)

        if j%(print_frequence)==0:
            print(f'({dataset}) Epoch: {j}/{epoch}, Loss: {loss:.3f}, regular_loss:{regular_loss:.3f}, Valid Loss: {score:.3f}, Best Valid Loss: {net.best_score:.3f}, hashset len: {len(net.fsa_hashtable)}, Conflict steps: {net.conflicts}, Train Time: {train_time:0.3f}, Valid Time: {valid_time:0.3f}, Total Time: {time.time()-stime:0.3f}')
            # torch.save(net, model_path)
    
    end_time = time.time() - start_time
    d.data_time['train_time'] = end_time
    print(f'train time: {end_time:0.3f}')

    etime=str(time.time()-stime)
    one_res={'dataset':dataset}

    end_time = time.time() - stime
    d.data_time['total_time'] = end_time
    print(f'total time: {end_time:0.3f}')
    one_res['datatime'] = copy.deepcopy(d.data_time)

    if train_args.test_final_net:
        TP, FN, FP, TN, acc, pre, rec, F1, net_test_results = net.net_test(d.test[:])
        print('%s final net (mean):(%d) TP: %d, FN: %d, FP: %d, TN: %d, acc: %0.3f, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (
        dataset,(TP + FN + FP + TN).mean(), TP.mean(), FN.mean(), FP.mean(), TN.mean(), acc.mean(), pre.mean(), rec.mean(), F1.mean()))
        best_fsa_idx = int(torch.argmax(F1))
        print('%s final net (best F1):(%d) TP: %d, FN: %d, FP: %d, TN: %d, acc: %0.3f, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (
        dataset,(TP + FN + FP + TN)[best_fsa_idx], TP[best_fsa_idx], FN[best_fsa_idx], FP[best_fsa_idx], TN[best_fsa_idx], acc[best_fsa_idx], pre[best_fsa_idx], rec[best_fsa_idx], F1[best_fsa_idx]))
        one_res['final_net']={'TP':TP.tolist(),'FN':FN.tolist(),'FP':FP.tolist(),'TN':TN.tolist(),'acc':acc.tolist(),'pre':pre.tolist(),'rec':rec.tolist(),'F1':F1.tolist(),'time':etime,'test_results':net_test_results}

    if train_args.save_best_net:
        if net.best_accept is not None and net.best_neighbor is not None:
            TP, FN, FP, TN, acc, pre, rec, F1, net_test_results = net.net_test_best(d.test[:])
            print('%s best net:(%d) TP: %d, FN: %d, FP: %d, TN: %d, acc: %0.3f, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (
            dataset,(TP + FN + FP + TN), TP, FN, FP, TN, acc, pre, rec, F1))
            one_res['best_net']={'TP':TP,'FN':FN,'FP':FP,'TN':TN,'acc':acc,'pre':pre,'rec':rec,'F1':F1,'time':etime,'test_results':net_test_results}

    # save
    print(f'step per epoch: {step_per_epoch}, total steps: {epoch * step_per_epoch}')
    model_mdl = f'{result_path}/final.mdl'
    save_best = (train_args.save_best) and net.best_fsa is not None
    if save_best:
        torch.save(net,f'{result_path}/final.mdl')
        if train_args.save_best_allpos and net.best_fsa_allpos is not None:
            fsa = net.best_fsa_allpos
            print(f'best step idx: {net.best_step_allpos}, best fsa idx: {net.best_fsa_idx_allpos}')
        else:
            fsa = net.best_fsa
            print(f'best step idx: {net.best_step}, best fsa idx: {net.best_fsa_idx}')
        if train_args.save_best_allpos:
            print(f'best fsa satisfies all pos traces: {net.best_fsa_allpos is not None}')
    else:
        net.best_score = float('inf')
        net.net_valid(d.valid_dataloader, step_per_epoch * epoch, writer)
        print(f'best step idx: {net.best_step}, best fsa idx: {net.best_fsa_idx}')
        fsa = net.to_fsa_round_index(net.best_fsa_idx)
        torch.save(net,f'{result_path}/final.mdl')
    if train_args.postprocess1 and net.init_info is not None:
        fsa.neighbor_postprocess()
    if train_args.postprocess2:
        fsa.remove_unreachable(init=(not train_args.postprocess3))
    if train_args.postprocess3:
        fsa.remove_not_lead_accept(init=True)
    ret_fsa = fsa.to_dict(d.vocab)
    print(f'({dataset}) final fsa num_states: {fsa.num_state}, neighbor len: {len(ret_fsa["neighbor"])}')
    TP, FN, FP, TN, acc, pre, rec, F1, fsa_test_results = test_fsa(ret_fsa, d.test_traces_pos, d.test_traces_neg)

    if train_args.generate_test_neg and not train_args.silent:
        print('%s fsa(dev):(%d) TP: %d, FN: %d, FP: %d, TN: %d, acc: %0.3f, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (dataset, TP + FN + FP + TN, TP, FN, FP, TN, acc, pre, rec, F1))
    one_res['fsa']={'TP':TP,'FN':FN,'FP':FP,'TN':TN,'acc':acc,'pre':pre,'rec':rec,'F1':F1,'test_results':fsa_test_results}
    write_fsa(ret_fsa, f'{result_path}/final_fsa.txt')

    if train_args.data_dir.find("data_event") == -1:
        start_time = time.time()
        gt_path = f'data_ori_split10_neg3/{dataset}/gt_fsm.txt'
        pos_name = 'input.json'
        if train_args.data_dir.find("data_synthesis") != -1:
            gt_path = f'data_synthesis/{dataset}/gt_fsm.txt'
            pos_name = 'input_train.json'
        TP1,TP2,fsm1_traces_pos,fsm2_traces_pos=cmp_2fsas(f'{result_path}/final_fsa.txt', gt_path, len(d.test_traces_pos), train_args.max_length, args=train_args, pos_name=pos_name)
        n1, n2 = len(fsm1_traces_pos), len(fsm2_traces_pos)
        end_time = str(time.time() - start_time)
        pre = TP1 / n1 if n1 > 0 else -1
        F1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else -1
        if not train_args.silent:
            print('%s fsa:(%d) TP1: %d, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (dataset, n1, TP1, pre, rec, F1))
        one_res['fsa_cmp2fsas']={'TP1': TP1, 'TP2': TP2, 'n1': n1, 'n2': n2,
                                'pre': pre, 'rec': rec, 'F1': F1, 'time': etime}
    else:
        if not train_args.silent:
            print('%s fsa:(%d) TP: %d, FN: %d, FP: %d, TN: %d, acc: %0.3f, pre: %0.3f, rec: %0.3f, F1: %0.3f' % (dataset, TP + FN + FP + TN, TP, FN, FP, TN, acc, pre, rec, F1))
    fsa_path = f'model/{train_args.tag}/{dataset}/final_fsa.txt'
    dot_file_path = f'graph/dot/{train_args.tag}/{dataset}_final_fsa.dot'
    fsa_img_path = f'graph/img/{train_args.tag}/{dataset}_final_fsa.png'
    write_dot_img(fsa_path, dot_file_path, fsa_img_path, only_dot=(train_args.k_g > 10))
    return one_res

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Main script for parallel')
    parser.add_argument('--tag', type=int, required=False, help='tag',default=0)
    parser.add_argument('--data_dir', type=str, required=False, help='tag', default='data_ori_split10_neg3')
    parser.add_argument('--samples', type=int, required=False, default=500000)
    parser.add_argument('--epoch', type=int, required=False, default=-1)
    parser.add_argument('--k_g', type=int, required=False, default=5)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, required=False, default=1e-2)
    parser.add_argument('--K', type=int, required=False, default=10)
    parser.add_argument('--val_k', type=int, required=False, default=0)
    parser.add_argument('--print_frequence', type=int, required=False, default=0)
    parser.add_argument('--datasets', type=str, nargs='+', default=[])
    parser.add_argument('--print_traces', type=int, required=False, default=0)
    parser.add_argument('--save_best', type=int, required=False, default=1)
    parser.add_argument('--save_data', type=int, required=False, default=0)
    parser.add_argument('--save_data_only', type=int, required=False, default=0)
    parser.add_argument('--save_best_allpos', type=int, required=False, default=0)
    parser.add_argument('--save_best_net', type=int, required=False, default=0)
    parser.add_argument('--gpu', type=int, required=False, default=-1) # gpu, -1 refers to the cpu
    parser.add_argument('--num_workers', type=int, required=False, default=0)
    parser.add_argument('--pin_memory', type=int, required=False, default=0)

    # generate_neg_trace
    parser.add_argument('--num_models', type=int, required=False, default=1)
    parser.add_argument('--enhanced_traces', type=int, required=False, default=0, help='use enhanced traces')
    # 使用反例
    parser.add_argument('--generate_neg', type=int, required=False, default=1, help='generate neg traces for training set')
    parser.add_argument('--generate_neg_simple', type=int, required=False, default=0, help='randomly mutate positive examples')
    parser.add_argument('--generate_valid_neg', type=int, required=False, default=0, help='generate neg traces for validation set')
    parser.add_argument('--generate_test_neg', type=int, required=False, default=0, help='generate neg traces for test set')
    parser.add_argument('--sample_unlabel', type=int, required=False, default=0, help='sample unlabel traces')
    parser.add_argument('--max_length', type=int, required=False, default=50, help='max length for sampled traces')
    parser.add_argument('--silent', type=int, required=False, default=0)
    parser.add_argument('--test_final_net', type=int, required=False, default=0)
    parser.add_argument('--threshold1', type=float, required=False, default=1.0, help='confidence threshold')
    parser.add_argument('--threshold2', type=float, required=False, default=0.0, help='reserve formulas whose condition appears in positive traces')
    parser.add_argument('--fml_topk', type=int, required=False, default=-1)
    # optimization: use the relation between formula templates
    parser.add_argument('--fml_opt1', type=int, required=False, default=1)
    # optimization: reserve formulas whose condition appears in positive traces
    parser.add_argument('--fml_opt2', type=int, required=False, default=1)
    # noise rate
    parser.add_argument('--noise', type=float, required=False, default=0.0)
    parser.add_argument('--noise_statistics', type=int, required=False, default=0)

    parser.add_argument('--restart_steps', type=int, required=False, default=0)
    parser.add_argument('--seed', type=int, required=False, default=-1)
    # weight of classification loss
    parser.add_argument('--w_loss_pos', type=float, required=False, default=128)
    parser.add_argument('--unlabel', type=float, required=False, default=0)
    parser.add_argument('--no_valid', type=int, required=False, default=0)
    # post processing: force that the initial states can only transfer to other states
    parser.add_argument('--postprocess1', type=int, required=False, default=0)
    # post processing: remove the unreachable states
    parser.add_argument('--postprocess2', type=int, required=False, default=0)
    # post processing: those states not leading to accepting states
    parser.add_argument('--postprocess3', type=int, required=False, default=0)

    # at least one accepting state
    parser.add_argument('--faithfula1', type=float, required=False, default=0)
    # get all values away from 0.5
    parser.add_argument('--faithfula2', type=float, required=False, default=1)
    parser.add_argument('--faithfula2_old', type=int, required=False, default=1)
    # active function，0: minmax, 1: bidirectional leaky relu
    parser.add_argument('--active', type=int, required=False, default=1)
    # loss function，0: BCELoss, 1: MSE
    parser.add_argument('--criterion', type=int, required=False, default=1)

    # force that the initial states can only transfer to other states
    parser.add_argument('--neighbor_init', type=int, required=False, default=0)
    train_args = parser.parse_args()

    if train_args.gpu == -1:
        train_args.device = torch.device('cpu')
    else:
        train_args.device = torch.device(f'cuda:{train_args.gpu}')

    if train_args.datasets == []:
        datasets = ['ArrayList', 'HashMap', 'HashSet', 'Hashtable', 'LinkedList', 'NumberFormatStringTokenizer',
                    'Signature', 'Socket', 'StringTokenizer', 'ZipOutputStream', 'StackAr']
        if train_args.data_dir.find("data_event") != -1:
            datasets = ['LOAN', 'REIMB', 'ROAD', 'SEPSIS', 'TRAVEL']
        if train_args.data_dir.find("data_synthesis") != -1:
            datasets = [f'state{i}_vocab{j}' for j in [20,22,24,26,28] for i in [7,9,11,13,15]]
    else:
        datasets = train_args.datasets

    if train_args.seed != -1:
        torch.manual_seed(train_args.seed)
        random.seed(train_args.seed)

    result=[]
    for dataset in datasets:
        result.append(one_train(dataset,train_args))
    ret_list = result

    res_path = f'res/res_tag{train_args.tag}.json'
    if len(datasets) == 1:
        if not os.path.exists(f'res/{train_args.datasets[0]}'):
            os.makedirs(f'res/{train_args.datasets[0]}')
        res_path = f'res/{train_args.datasets[0]}/res_tag{train_args.tag}.json'
    with open(res_path,'w') as f:
        json.dump(ret_list,f)

