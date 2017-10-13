#!/usr/bin/env python

import nltk
import numpy as np
import six
import pdb

grammar_file = '../../dropbox/context_free_grammars/mol_zinc.grammar'

def load_gram(f):
    rules = [line.strip() for line in open(grammar_file).readlines()]
    rules = [line for line in rules if line]
    # now rules can be X -> Y | Z
    # split to X -> Y\nX->Z
    rules2 = []
    for r in rules:
        head, tail = r.split('->')
        tails = tail.split('|')
        for t in tails:
            rules2.append('%s -> %s' % (head.strip(' '), t.strip(' ')))
    rules2 += ['Nothing -> None']
    return '\n'.join(rules2)

gram = load_gram(grammar_file)

# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

# this map tells us the rhs symbol indices for each production rule
rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list),D))
count = 0

# this tells us for each lhs symbol which productions rules should be masked
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

# this tells us the indices where the masks are equal to 1
index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:,i]==1)[0][0])
ind_of_ind = np.array(index_array)

max_rhs = max([len(l) for l in rhs_map])

# rules 29 and 31 aren't used in the zinc data so we
# 0 their masks so they can never be selected
#masks[:,29] = 0
#masks[:,31] = 0
