import json
from pathlib import Path
from typing import Dict, List, Set
import random
import re
import argparse

class FSM:
    def __init__(self, input_path: Path) -> None:
        d = []
        with input_path.open('r', encoding='utf-8') as fin:
            for line in fin:
                d.extend(line.strip().split())
        m = int(d[0])
        assert m > 0
        self.init_states = set()
        for i in range(1, m + 1):
            self.init_states.add(d[i])

        n = int(d[m + 1])
        self.final_states = set()
        for i in range(m + 2, m + n + 2):
            self.final_states.add(d[i])

        l = int(d[m + n + 2])
        self.G = dict()
        # self.state_set = set()
        self.state_set = self.init_states.union(self.final_states)
        self.vocab=set()

        for i in range(m + n + 3, 3 * l + m + n + 3, 3 ):
            s = d[i]
            t = d[i + 1]
            self.state_set.add(s)
            self.state_set.add(t)

            a = re.sub('_EXIT[0-9]+', '', d[i + 2])
            # print(d[i + 2],a)
            if a not in self.vocab:
                self.vocab.add(a)
            if s not in self.G:
                self.G[s] = dict()
            # if t not in self.G:
            #     self.G[t] = dict()
            if a not in self.G[s]:
                self.G[s][a] = set()
            self.G[s][a].add(t)
        # print('self.init_states',self.init_states)
        #
        # print('self.final_states', self.final_states)
        # print('self.G',self.G)

    def num_states(self) -> int:
        return len(self.state_set)

    def get_state_set(self) -> Set[int]:
        return self.state_set

    def dfs_sat(self,state,visited:set,dep:int=0):
        if state in self.final_states and dep>0:
            return True
        if state not in self.G.keys():
            return False
        for key in self.G[state].keys():
            for to in self.G[state][key]:
                if to not in visited:
                    visited.add(to)
                    if self.dfs_sat(to,visited,dep+1):
                        return True
        return False

    def satisfiable(self):
        for i in self.init_states:
            visited=set()
            # visited.add(i)
            if self.dfs_sat(i,visited):
                return True
        return False



    def _dfs(self, current_state: int, trace: List[str], i: int,cache:dict) -> bool:
        if (current_state,i) in cache:
            return cache[(current_state,i)]
        if i == len(trace):
            if current_state in self.final_states:
                cache[(current_state,i)]=True
            else:
                cache[(current_state,i)]=False
            return cache[(current_state,i)]
        if current_state in self.G.keys() and trace[i] in self.G[current_state]:
            for next_state in self.G[current_state][trace[i]]:
                if self._dfs(next_state, trace, i + 1,cache):
                    # G_cache[(current_state,trace[i],next_state)]+=1
                    cache[(current_state, i)]=True
                    return cache[(current_state,i)]
        cache[(current_state, i)]=False
        return cache[(current_state,i)]

    def check(self, trace: List[str]) -> bool:
        for i in self.init_states:
            cache={}
            if self._dfs(i, trace, 0,cache):
                return True
        return False

    def sample_pos_trace(self, length: int, states: List[str] = None) -> List[str]:
        flag = True
        cnt=0
        while flag:
            trace = []
            cnt += 1
            if cnt>100:
                break
            current_state = random.sample(list(self.init_states), k=1)[0]
            if states is not None:
                states.append(current_state)
            flag = False
            while len(trace) < length and not flag:
                if current_state not in self.G:
                    flag = True
                else:
                    event = random.sample(list(self.G[current_state].keys()), k=1)[0]
                    current_state = random.sample(list(self.G[current_state][event]), k=1)[0]
                    if states is not None:
                        states.append(current_state)
                    trace.append(event)
        return trace
    
    def sample_neg_trace(self, length: int) -> List[str]:
        if length == 1:
            init_token = set()
            for s in self.init_states:
                init_token.update(set(self.G[s].keys()))
            trace = random.sample(list(self.vocab.difference(init_token)), k=1)
        else:
            others = set()
            while len(others) == 0:
                states = []
                trace = self.sample_pos_trace(length-1, states=states)
                last_state = states[-1]
                others = self.vocab.difference(set(self.G[last_state].keys()))
            event = random.sample(list(others), k=1)[0]
            trace.append(event)
        return trace

    # transition_pro current_state:[(action,next_state,weight)]
    def sample_pos_trace_by_pro(self, length: int,transition_pro) -> List[str]:
        flag = True
        cnt=0
        while flag:
            trace = []
            cnt += 1
            if cnt>100:
                break
            current_state = random.sample(list(self.init_states), k=1)[0]
            flag = False
            while len(trace) < length and not flag:
                if current_state not in self.G:
                    flag = True
                else:
                    rand_num=random.random()
                    # print(rand_num)
                    cur_sum=0
                    for event,next_state,weight in transition_pro[current_state]:
                        cur_sum+=weight
                        if cur_sum>=rand_num:
                            current_state = next_state
                            trace.append(event)
                            break

        return trace

if __name__ == '__main__':
    file_path = 'data_ori_split10_neg3/HashMap/gt_fsm.txt'
    fsm = FSM(Path(file_path))
    trace = fsm.sample_neg_trace(10)
    print(trace)
