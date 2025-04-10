import psutil
import sys

process = psutil.Process()

class LTLf():

    def __init__(self, vocab, LTLf_tree,cache=None):
        self.vocab = vocab
        self.LTLf_tree = LTLf_tree
        if cache is not None:
            self.cache = cache
        else:
            self.cache = {}

    def _checkLTL(self, f, t, trace, vocab, c=None, v=False, orif=None):
        """ Checks satisfaction of a LTL formula on an execution trace

            NOTES:
            * This works by using the semantics of LTL and forward progression through recursion
            * Note that this does NOT require using any off-the-shelf planner

            ARGUMENTS:
                f       - an LTL formula (must be in TREE format using nested tuples
                        if you are using LTL dict, then use ltl['str_tree'])
                t       - time stamp where formula f is evaluated
                trace   - execution trace (a dict containing:
                            trace['name']:    trace name (have to be unique if calling from a set of traces)
                            trace['trace']:   execution trace (in propositions format)
                            trace['plan']:    plan that generated the trace (unneeded)
                vocab   - vocabulary of propositions
                c       - cache for checking LTL on subtrees
                v       - verbosity

            OUTPUT:
                satisfaction  - true/false indicating ltl satisfaction on the given trace
        """
        if orif == None:
            orif = f
        if v:
            print('\nCurrent t = ' + str(t))
            print('Current f =', f)

        ###################################################

        # Check if first operator is a proposition
        if type(f) is str:
            if f in vocab:
                return f in trace['trace'][t]
            if f == 'true':
                return True
            if f=='false':
                return False

        # Check if sub-tree info is available in the cache
        key = (f, t, trace['name'])
        if c is not None:
            if key in c:
                if v: print('Found subtree history')
                return c[key]

        # Check for standard logic operators
        # if len(f)==0:
        # 	print('f', f, 't', t, 'trace', trace, 'vocab', vocab, 'c', c, 'v', v)
        if f[0] in ['not', '!']:
            value = not self._checkLTL(f[1], t, trace, vocab, c, v, orif)
        elif f[0] in ['and', '&', '&&']:
            value = all((self._checkLTL(f[i], t, trace, vocab, c, v, orif) for i in range(1, len(f))))
        elif f[0] in ['or', '|', '||']:
            value = any((self._checkLTL(f[i], t, trace, vocab, c, v, orif) for i in range(1, len(f))))
        elif f[0] in ['imp', '->']:
            value = not (self._checkLTL(f[1], t, trace, vocab, c, v, orif)) or self._checkLTL(f[2], t, trace, vocab, c,
                                                                                              v, orif)

        # Check if t is at final time step
        elif t == len(trace['trace']) - 1:
            # Confirm what your interpretation for this should be.
            if f[0] in ['G', 'F']:
                value = self._checkLTL(f[1], t, trace, vocab, c, v,
                                       orif)  # Confirm what your interpretation here should be
            elif f[0] == 'U':
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif)
            elif f[0] == 'W':  # weak-until
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or self._checkLTL(f[1], t, trace, vocab, c, v,
                                                                                            orif)
            elif f[0] == 'R':  # release (weak by default)
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif)
            elif f[0] == 'X':
                value = False
            elif f[0] == 'N':
                value = True
            else:
                # Does not exist in vocab, nor any of operators
                print('f', f, 't', t, 'trace', trace, 'vocab', vocab, 'c', c, 'v', v)
                sys.exit('LTL check - something wrong 1')

        else:
            # Forward progression rules
            if f[0] == 'X' or f[0] == 'N':
                value = self._checkLTL(f[1], t + 1, trace, vocab, c, v, orif)
            elif f[0] == 'G':
                value = self._checkLTL(f[1], t, trace, vocab, c, v, orif) and self._checkLTL(('G', f[1]), t + 1, trace,
                                                                                             vocab, c, v, orif)
            elif f[0] == 'F':
                value = self._checkLTL(f[1], t, trace, vocab, c, v, orif) or self._checkLTL(('F', f[1]), t + 1, trace,
                                                                                            vocab, c, v, orif)
            elif f[0] == 'U':
                # Basically enforces f[1] has to occur for f[1] U f[2] to be valid.
                if t == 0:
                    if not self._checkLTL(f[1], t, trace, vocab, c, v, orif) and not self._checkLTL(f[2], t, trace,
                                                                                                    vocab, c, v,
                                                                                                    orif):  # if f[2] is ture at time 0,then it is true
                        value = False
                    else:
                        value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or (
                                    self._checkLTL(f[1], t, trace, vocab, c, v) and self._checkLTL(('U', f[1], f[2]),
                                                                                                   t + 1, trace, vocab,
                                                                                                   c, v, orif))
                else:
                    value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or (
                                self._checkLTL(f[1], t, trace, vocab, c, v) and self._checkLTL(('U', f[1], f[2]), t + 1,
                                                                                               trace, vocab, c, v,
                                                                                               orif))

            elif f[0] == 'W':  # weak-until
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or (
                            self._checkLTL(f[1], t, trace, vocab, c, v, orif) and self._checkLTL(('W', f[1], f[2]),
                                                                                                 t + 1, trace, vocab, c,
                                                                                                 v, orif))
            elif f[0] == 'R':  # release (weak by default)
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) and (
                            self._checkLTL(f[1], t, trace, vocab, c, v, orif) or self._checkLTL(('R', f[1], f[2]),
                                                                                                t + 1, trace, vocab, c,
                                                                                                v, orif))
            else:
                # Does not exist in vocab, nor any of operators
                print('f', f, 't', t, 'trace', trace, 'vocab', vocab, 'c', c, 'v', v, ' orif', orif)
                sys.exit('LTL check - something wrong 2 ' + f[0])

        if v: print('Returned: ' + str(value))

        # Save result
        if c is not None and type(c) is dict:
            key = (f, t, trace['name'])
            c[key] = value  # append

        return value

    def pathCheck(self, trace, trace_name):

        trace_dir = {'name': trace_name, 'trace': tuple(trace)}
        return self._checkLTL(self.LTLf_tree, 0, trace_dir, self.vocab, self.cache)

    def evaluate(self, cluster1, cluster2):
        # print(self.LTLf_tree)
        check_pos_mark = []
        for i in range(len(cluster1)):
            st = self.pathCheck(cluster1[i], 'pos' + str(i))
            check_pos_mark.append(st)

        check_neg_mark = []
        for i in range(len(cluster2)):
            st = self.pathCheck(cluster2[i], 'neg' + str(i))
            check_neg_mark.append(st)
        # print(sys.getrefcount(self.cache))
        self.cache={}
        return check_pos_mark, check_neg_mark

