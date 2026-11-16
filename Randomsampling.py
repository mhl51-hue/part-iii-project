'''
Sampling of Pauli H

First two methods will truncate H leading to it being biased, will reduce alpha
1. Deterministic truncation, keep n largest -> imported from DeterministicSampling
2. Random sampling without replacement of terms, no reweighting

3. Random sampling without replacement of terms, with reweighting
4. Random sampling with replacement of terms, (qdrift style)
5. Hybrid sampling
'''

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from DeterministicSampling import pauli_terms, evaluate, pauli_truncate
from FermiHubbard import fermihubbard_subspace


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
    reweight: if False this is method 2. 
              if True the kept terms are multiplied by 1/πi, this is method 3.
    
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
4. Random sampling with replacement of terms (qdrift style), terms are selected and replaced
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

def counts(p, n_draws, rand_num_gen, stratify):
    '''
    Number of times each string is drawn: n_draws with probabilities p
    '''
    if not stratify:
        return rand_num_gen.multinomial(n_draws, p)
    expected = n_draws * p
    base = np.floor(expected).astype(int)
    left = n_draws - base.sum()
    return base + (rand_num_gen.multinomial(left, (expected - base) / left) if left > 0 else 0)

def rand_sampling(terms, w, n_draws, rand_num_gen, stratify):
    '''
    label coefficient from n_draws over terms weights: sum w / N * k_j * sign(h_j)
    '''
    alpha = w.sum()
    count = counts(w/ alpha, n_draws, rand_num_gen, stratify)
    return [(label, alpha * k / n_draws * np.sign(c.real)) for (label, c), k in zip(terms, count) if k>0]

def hybrid_sampling(full_H_op, n_keep, n_draws, stratify=False, seed=None):
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    w = np.array([abs(c) for _, c in terms])
    order = np.argsort(-w, kind="stable")
    kept = [terms[i] for i in order[:n_keep]]
    rest = [terms[i] for i in order[n_keep:]]
    w_rest = w[order[n_keep:]]
    def sample_H():
        sampled = rand_sampling(rest, w_rest, n_draws, rand_num_gen, stratify) if rest else []
        return SparsePauliOp.from_list(ident + kept + sampled)
    return sample_H

'''
Need to average sampled Hamiltonian over multiple instances
'''

def average_over_runs(sampled_H, subspace, exact, m, projector="wall_cheb", n_runs=100):
    results = [evaluate(sampled_H(), subspace, exact, m=m, deg=500) for _ in range(n_runs)]
    p = np.array([r[projector]["p"] for r in results])
    F = np.array([r[projector]["fidelity"] for r in results])
    alpha = np.array([r["alpha_wall"] for r in results])

# p is calculated using a geometric mean 
    return dict(alpha=alpha.mean(), p=float(np.exp(np.mean(np.log(p)))), p_mean=p.mean(), F_mean=float(np.nanmean(F)), F_std=float(np.nanstd(F)))

if __name__ == "__main__":
    U, m, exact = 2, 20, True
    subspace = fermihubbard_subspace(U)
    H_full = subspace.qubit_H

    n_keep = 5 
    n_draws = 7
    n_keep_hybrid = 3
    n_draws_hybrid = 2

    hybrid_H = hybrid_sampling(H_full, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0)
    with_replacement_H = rand_pauli_sampler_with_replacement(H_full, n_draws=n_draws, seed=0)
    without_replacement_H = rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, reweight=True, seed=0)

    samplers = {"hybrid": hybrid_H, "with rep": with_replacement_H, "without rep": without_replacement_H}

    for name, sampled_H in samplers.items():
        string_counts = [len(pauli_terms(sampled_H())[1]) for _ in range(200)]
        print(f"{name:12s} strings per sample: mean {np.mean(string_counts):.2f}, "
            f"min {min(string_counts)}, max {max(string_counts)}")

    def run(name, o):
        r = evaluate(o, subspace, exact, m=m, deg=50)["wall_cheb"]
        print(f"{name:60s} alpha={evaluate(o, subspace, exact, m=m, deg=50)['alpha_wall']:.3f}  p={r['p']:.2e}  F={r['fidelity']:.3f}")
 
    def avg(name, sampled_H):
        a = average_over_runs(sampled_H, subspace, exact, m)
        print(f"{name:60s} alpha={a['alpha']:.3f}  p={a['p']:.2e}  F={a['F_mean']:.3f} +/- {a['F_std']:.3f}")
 
    print(f"U={U}, wall-Chebyshev P2, m={m}, S={'E0' if exact else 'HF'}")
    run("no sampling", H_full)
    run(f"deterministic truncate n={n_keep}", pauli_truncate(H_full, n_keep))
    avg(f"random sampling with no replacement of terms n={n_keep}", rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, seed=0))
    avg(f"random sampling with no replacement of terms n={n_keep}, reweighted", rand_pauli_sampler_without_replacement(H_full, n_terms=n_keep, reweight=True, seed=0))
    avg(f"Random sampling with replacement of terms, {n_draws} draws", rand_pauli_sampler_with_replacement(H_full, n_draws=n_draws, seed=0)) 
    avg(f"hybrid: {n_keep_hybrid} exact + {n_draws_hybrid} draws", hybrid_sampling(H_full, n_keep=n_keep_hybrid, n_draws=n_draws_hybrid, seed=0))