'''
Sampling of Pauli H

First two methods will truncate H leading to it being biased, will reduce alpha
1. Deterministic truncation, keep n largest
2. Random sampling without replacement of terms, no reweighting

Stochastic methods that keep the same alpha
3. Random sampling without replacement of terms, with reweighting
4. Random sampling with replacement of terms
5. Hybrid sampling
'''

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from DeterministicSampling import pauli_terms, evaluate, pauli_truncate
from FermiHubbard import fermihubbard_subspace
from NewSequenceofCode import average_over_runs_new


'''
2. + 3. random sampling of terms without replacement
'''

def selection(weights, rand_num_gen):
    '''
    selecting terms without replacement with probabiltiy proportional to the weight
    '''
    w = np.asarray(weights, dtype=float)
    return rand_num_gen.choice(len(w), size=len(w), replace=False, p=w / w.sum())

def inclusion_probabilities(weights, kept_terms, rand_num_gen, n_mc=4000):
    '''
    After N draws the total probability of Pi is returned (πi)
    kept_terms: a function takes drawn ordering and returns indices of terms kept
    n_mc: how many times to repeat the random draw to estimate πi
    '''
    counts = np.zeros(len(weights))
    for _ in range(n_mc):
        counts[kept_terms(selection(weights, rand_num_gen))] += 1
    return np.maximum(counts / n_mc, 1 / n_mc)

def rand_pauli_sampler_without_replacement(full_H_op, n_terms=None, weight_fraction=None, reweight=False, seed=None, n_mc=4000):
    '''
    Returns one sampled H each time function is run
    full_H_op: full hamiltonian as a SparsePauliOp
    n_terms: number of non-identity terms kept in each sampled hamiltonian
    reweight: if False, method 2. 
              if True the kept terms are multiplied by 1/πi, method 3.
    '''
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    weights = np.array([abs(c) for _, c in terms])

    kept_terms = lambda order: order[:n_terms] # takes first n_terms of a drawn ordering

    scale = 1 / inclusion_probabilities(weights, kept_terms, rand_num_gen, n_mc) if reweight else np.ones(len(terms)) # reweighting 1/π factor only applied if reweight=True

    def sampled_H():
        keep = kept_terms(selection(weights, rand_num_gen))
        return SparsePauliOp.from_list(ident + [(terms[i][0], terms[i][1] * scale[i]) for i in keep]) # sampled Hamiltonian
    return sampled_H

'''
4. Random sampling with replacement of terms, terms are selected and replaced
'''

def rand_pauli_sampler_with_replacement(full_H_op, n_draws, seed=None):
    '''
    Returns one sampled H each time function is run
    n_draws: number of draws done on each Hamiltonian
    '''
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    w = np.array([abs(c) for _, c in terms])
    p, alpha = w / w.sum(), w.sum()

    def sampled_H():
        picks = rand_num_gen.choice(len(w), size=n_draws, p=p)

        coefs = {}
        for j in picks:
            label, c = terms[j]
            coefs[label] = coefs.get(label, 0.0) + alpha / len(picks) * np.sign(c.real)
        return SparsePauliOp.from_list(ident + list(coefs.items())) # sampled Hamiltonian
    return sampled_H

'''
5. Hybrid sampling
Keeps the first few terms of the Pauli string and then samples the rest
'''

def counts(p, n_draws, rand_num_gen):
    '''
    Number of times each string is drawn: n_draws with probabilities p
    '''
    return rand_num_gen.multinomial(n_draws, p)

def rand_sampling(terms, w, n_draws, rand_num_gen):
    '''
    label coefficient from n_draws over terms weights: sum w / N * k_j * sign(h_j)
    '''
    alpha = w.sum()
    count = counts(w/ alpha, n_draws, rand_num_gen)
    return [(label, alpha * k / n_draws * np.sign(c.real)) for (label, c), k in zip(terms, count) if k>0]

def hybrid_sampling(full_H_op, n_keep, n_draws, seed=None):
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    w = np.array([abs(c) for _, c in terms])
    order = np.argsort(-w, kind="stable")
    kept = [terms[i] for i in order[:n_keep]]
    rest = [terms[i] for i in order[n_keep:]]
    w_rest = w[order[n_keep:]]
    def sample_H():
        sampled = rand_sampling(rest, w_rest, n_draws, rand_num_gen) if rest else []
        return SparsePauliOp.from_list(ident + kept + sampled)
    return sample_H

if __name__ == "__main__":
    U, m, exact = 2, 20, True
    subspace = fermihubbard_subspace(U)
    H_full = subspace.qubit_H

    n_keep = 7 
    n_draws = 11
    n_keep_hybrid = 4
    n_draws_hybrid = 4

    hybrid_H = hybrid_sampling(H_full, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0)
    with_replacement_H = rand_pauli_sampler_with_replacement(H_full, n_draws=n_draws, seed=0)
    without_replacement_H = rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, reweight=True, seed=0)

    samplers = {"hybrid": hybrid_H, "with rep": with_replacement_H, "without rep": without_replacement_H}

    for name, sampled_H in samplers.items():
        string_counts = [len(pauli_terms(sampled_H())[1]) for _ in range(200)]
        print(f"{name:12s} strings per sample: {np.mean(string_counts):.2f}")

    def run(name, o):
        out = evaluate(o, subspace, exact, m=m, deg=50)
        r = out["wall_cheb_product"]
        mean_alpha = np.prod(out["alpha_product"]) ** (1 / m) # geometric mean of the alpha_v
        print(f"{name:60s} mean alpha_v={mean_alpha:.3f}  p={r['p']:.2e}  F={r['fidelity']:.3f}")

    # new method
    def avg_new(name, sampled_H, per_sample):
        a = average_over_runs_new(sampled_H, subspace, exact, m, factors_per_sample=per_sample, n_runs=50)
        print(f"{name:73s} p_mean={a['p_mean']:.2e}  F={a['F_mean']:.3f} +/- {a['F_std']:.3f}")
 
    print(f"U={U}, wall-Chebyshev product of LCUs (eq. 35), m={m}, S={'E0' if exact else 'HF'}")
    run("no sampling", H_full)
    run(f"deterministic truncate n={n_keep}", pauli_truncate(H_full, n_keep))
    avg_new(f"random sampling with no replacement of terms n={n_keep}", rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, seed=0), 1)
    avg_new(f"random sampling with no replacement of terms n={n_keep}, reweighted", rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, reweight=True, seed=0), 1)
    avg_new(f"Random sampling with replacement of terms, {n_draws} draws", rand_pauli_sampler_with_replacement(H_full, n_draws=n_draws, seed=0), 1) 
    avg_new(f"hybrid: {n_keep_hybrid} exact + {n_draws_hybrid} draws", hybrid_sampling(H_full, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0), 1)