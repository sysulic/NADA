from pathlib import Path
from typing import Dict, List, Set
import random
import re

class FSA:
    def __init__(self, input_path: Path, vocab: Dict[str, str]) -> None:
        d = []
        with input_path.open('r', encoding='utf-8') as fin:
            for line in fin:
                d.extend(line.strip().split())
        m = int(d[0])
        assert m > 0
        if d[1].startswith('S'):
            prefix_len = 1
        else:
            prefix_len = 0
        self.init_states = set()
        for i in range(1, m+1):
            self.init_states.add(int(d[i][prefix_len:]))

        n = int(d[m+1])
        self.final_states = set()
        for i in range(m+2, m+n+2):
            self.final_states.add(int(d[i][prefix_len:]))
        
        l = int(d[m+n+2])
        self.G = dict()
        self.state_set = set()
        for i in range(m+n+3, (3+prefix_len)*l+m+n+3, 3+prefix_len):
            s = int(d[i][prefix_len:])
            t = int(d[i+1][prefix_len:])
            self.state_set.add(s)
            self.state_set.add(t)
            a = re.sub('_EXIT[0-9]+', '', d[i+2])
            if a in vocab:
                a = vocab[a]
                if s not in self.G:
                    self.G[s] = dict()
                if a not in self.G[s]:
                    self.G[s][a] = set()
                self.G[s][a].add(t)
            # else:
            #     print(f'Unknown input {a}')

    def num_states(self) -> int:
        return len(self.state_set)
    def get_state_set(self) -> Set[int]:
        return self.state_set

    def _dfs(self, current_state: int, trace: List[str], i: int) -> bool:
        if i == len(trace):
            if current_state in self.final_states:
                return True
            else:
                return False
        if trace[i] in self.G[current_state]:
            for next_state in self.G[current_state][trace[i]]:
                if self._dfs(next_state, trace, i+1):
                    return True
        return False

    def check(self, trace: List[str]) -> bool:
        for i in self.init_states:
            if self._dfs(i, trace, 0):
                return True
        return False

    def sample_pos_trace(self, length: int) -> List[str]:
        flag = True
        while flag:
            trace = []
            current_state = random.sample(self.init_states, k = 1)[0]
            flag = False
            while len(trace) < length and not flag:
                if current_state not in self.G:
                    flag = True
                else:
                    event = random.sample(self.G[current_state].keys(), k = 1)[0]
                    current_state = random.sample(self.G[current_state][event], k = 1)[0]
                    trace.append(event)
        return trace