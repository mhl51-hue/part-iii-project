'''
Sampling the second quantised Hamiltonian before mapping

Each sampled Hamiltonian is built as a FermionicOp from the chosen fermionic terms
'''

import math
from dataclasses import dataclass
import numpy as np
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

from PauliSampling import counts, selection, inclusion_probabilities
from FermiHubbard import fermihubbard_subspace, second_quantised_H
from NewSequenceofCode import average_over_runs_new

'''
Groups of fermionic terms
'''

@dataclass
class Group:
    name: str
    terms: dict
    weight: float

@dataclass
class FermionicGroups:
    n_spin_orbitals: int
    mapper: object
    constant: float
    groups: list                                              

    @property
    def total_weight(self):
        return sum(g.weight for g in self.groups)

    def build(self, fermionic_terms):
        '''
        sampled second quantised H: constant term + the sampled terms
        returns a FermionicOp only (no mapping)
        '''
        terms = dict(fermionic_terms)
        terms[""] = terms.get("", 0.0) + self.constant
        return FermionicOp(terms, num_spin_orbitals=self.n_spin_orbitals)

    def to_qubit(self, fermionic_op):
        return self.mapper.map(fermionic_op).simplify()

    def full_fermionic_op(self):
        return self.build({l: c for g in self.groups for l, c in g.terms.items()})

    def full_H(self):
        return self.to_qubit(self.full_fermionic_op())

    def show(self):
        print(f"{len(self.groups)} groups, total weight W = {self.total_weight:.4f}, constant = {self.constant}")
        for g in self.groups:
            print(f"  {g.name:20s} weight {g.weight:.3f}   " + "  ".join(f"'{l}':{c:+.3f}" for l, c in g.terms.items()))

def fermionic_groups(fermionic_op, mapper, tol=1e-12):
    '''
    fermionic_op: the second quantised Hamiltonian
    mapper: the mapper applied to each sampled Hamiltonian

    Terms acting on the same spin orbitals with the same number of ladder operators form one group,
    so a hopping term and its conjugate stay together
    '''
    n = fermionic_op.num_spin_orbitals
    constant, grouped = 0.0, {}
    for label, coef in fermionic_op.items():
        if abs(coef) < tol:
            continue
        if label == "":
            constant += coef
            continue
        orbitals = [int(token.split("_")[1]) for token in label.split()]
        # key = (tuple(sorted(set(orbitals))), len(orbitals)) # using this leads to 4 fermionic groups, 2 hopping and 2 onsite
        sites = tuple(sorted({o // 2 for o in orbitals}))
        key = (sites, len(orbitals)) # using this leads to 3 fermionic groups, 1 hopping and 2 onsite
        grouped.setdefault(key, {})[label] = coef

    groups = []
    for (sites, _), terms in grouped.items():
        mapped = mapper.map(FermionicOp(terms, num_spin_orbitals=n)).simplify()
        w = float(sum(abs(c) for l, c in mapped.to_list() if set(l) != {"I"}))
        if w > tol:
            groups.append(Group(f"sites {list(sites)}", terms, w))
    return FermionicGroups(n, mapper, float(np.real(constant)), groups)

'''
Building a sampled fermionic Hamiltonian from the groups, then mapping it
'''

def scaled(group, factor):
    return {l: c * factor for l, c in group.terms.items()}

def combine(dicts):
    out = {}
    for d in dicts:
        for l, c in d.items():
            out[l] = out.get(l, 0.0) + c
    return out

def drawn_terms(groups, n_draws, rand_num_gen):
    '''
    n_draws draws with replacement over a list of groups, group g drawn with probability w_g / W
    a group drawn k times has every fermionic coefficient multiplied by k W / (n_draws w_g), so the average is the full group
    '''
    w = np.array([g.weight for g in groups]); W = w.sum()
    k = counts(w / W, n_draws, rand_num_gen)
    return combine(scaled(g, kg * W / (n_draws * g.weight)) for g, kg in zip(groups, k) if kg > 0)

def inclusion(w, n):
    '''
    chance that each group is among the first n chosen without replacement, summed over all orderings
    '''
    pi = np.zeros(len(w))
    def rec(chosen, prob, remaining):
        if len(chosen) == n:
            pi[chosen] += prob
            return
        for j in range(len(w)):
            if j not in chosen:
                rec(chosen + [j], prob * w[j] / remaining, remaining - w[j])
    rec([], 1.0, w.sum())
    return pi

'''
Samplers: each returns a function that gives a new sampled Hamiltonian every time it is run
The sampled Hamiltonian is a FermionicOp, it is mapped to a SparsePauliOp later (in evaulate, mapper=groups.mapper)
'''

def rand_fermionic_sampler_without_replacement(groups, n_terms, reweight=False, seed=None, n_mc=4000):
    '''
    n_terms: number of fermionic terms (groups) kept in each sampled hamiltonian
    reweight: if False the kept groups keep their true coefficients (biased, smaller alpha)
              if True they are multiplied by 1/πg, the chance that group g is kept, so the average is the full H
    πg is exact when there are few orderings, otherwise estimated with n_mc repeats as in Randomsampling.py
    '''
    rand_num_gen = np.random.default_rng(seed)
    w = np.array([g.weight for g in groups.groups])
    kept_terms = lambda order: order[:n_terms] # first n_terms of a drawn ordering

    if not reweight or n_terms >= len(w):
        scale = np.ones(len(w))
    elif math.perm(len(w), n_terms) <= 300000:
        scale = 1 / inclusion(w, n_terms)
    else:
        scale = 1 / inclusion_probabilities(w, kept_terms, rand_num_gen, n_mc)

    def sampled_H():
        keep = kept_terms(selection(w, rand_num_gen))
        return groups.build(combine(scaled(groups.groups[j], scale[j]) for j in keep))
    return sampled_H

def rand_fermionic_sampler_with_replacement(groups, n_draws, seed=None):
    '''
    n_draws draws of fermionic terms with replacement
    '''
    rand_num_gen = np.random.default_rng(seed)

    def sampled_H():
        return groups.build(drawn_terms(groups.groups, n_draws, rand_num_gen))
    return sampled_H

def fermionic_hybrid_sampling(groups, n_keep, n_draws, seed=None):
    '''
    the n_keep heaviest fermionic terms are kept exactly, the rest of the weight is randomly drawn
    '''
    rand_num_gen = np.random.default_rng(seed)
    order = np.argsort([-g.weight for g in groups.groups], kind="stable")
    kept = combine(groups.groups[i].terms for i in order[:n_keep])
    rest = [groups.groups[i] for i in order[n_keep:]]

    def sampled_H():
        sampled = drawn_terms(rest, n_draws, rand_num_gen) if rest else {}
        return groups.build(combine([kept, sampled]))
    return sampled_H

def strings_per_sample(sampled_H, mapper, n=200):
    '''
    Mean number of distinct non-identity Pauli strings in a sampled H, to compare samplers at the same budget.
    mapper: the mapper the sampled FermionicOp is mapped with before counting, use groups.mapper
    '''
    def n_strings(H):
        H = mapper.map(H)
        return sum(1 for label, c in H.simplify().to_list() if set(label) != {"I"} and abs(c) > 1e-12)
    return float(np.mean([n_strings(sampled_H()) for _ in range(n)]))

if __name__ == "__main__":

    U, m, exact = 2, 20, True
    subspace = fermihubbard_subspace(U)
    H_full = subspace.qubit_H

    groups = fermionic_groups(second_quantised_H(U), JordanWignerMapper())
    groups.show()

    n_terms = 2
    n_draws = 3
    n_keep_hybrid = 1
    n_draws_hybrid = 1

    samplers = {
        "fermionic without rep, reweighted": rand_fermionic_sampler_without_replacement(groups, n_terms=n_terms, reweight=True, seed=0),
        "fermionic with rep": rand_fermionic_sampler_with_replacement(groups, n_draws=n_draws, seed=0),
        "fermionic hybrid": fermionic_hybrid_sampling(groups, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0),
    }
    print(f"U={U}, wall-Chebyshev product of LCUs (eq. 35), m={m}, S={'E0' if exact else 'HF'}")
    for name, sampled_H in samplers.items():
        a = average_over_runs_new(sampled_H, subspace, exact, m, factors_per_sample=1, n_runs=50, mapper=groups.mapper)
        print(f"{name:36s} p_mean={a['p_mean']:.2e}  F={a['F_mean']:.3f} +/- {a['F_std']:.3f}")

    print(f"without rep {strings_per_sample(rand_fermionic_sampler_without_replacement(groups, n_terms=n_terms, reweight=True, seed=0), groups.mapper)}")
    print(f"with rep {strings_per_sample(rand_fermionic_sampler_with_replacement(groups, n_draws=n_draws, seed=0), groups.mapper)}")
    print(f"hybrid {strings_per_sample(fermionic_hybrid_sampling(groups, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0), groups.mapper)}")
