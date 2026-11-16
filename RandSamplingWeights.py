'''
Sampling of Pauli H by WEIGHT FRACTION only (no fixed number of terms or draws)
 
The weight of a term is |coefficient|; the total weight of the non-identity terms is alpha(H). Every method stops
once the terms it has used hold a chosen fraction of that weight, so the number of terms varies from sample to sample.
 
First two methods will truncate H leading to it being biased, will reduce alpha
1. Deterministic weight fraction, keep the largest terms holding a fraction f -> pauli_weight imported from DeterministicSampling
2. Random sampling without replacement of terms until they hold a fraction f of the weight, no reweighting
 
3. Random sampling without replacement of terms, with reweighting
4. Random sampling with replacement of terms (qdrift style) until the draws hold a fraction f of alpha
5. Hybrid: the largest terms holding keep_fraction of the weight are kept exactly, the remaining terms are sampled
   with replacement until the draws hold draw_fraction of the weight that is left
'''
 
import numpy as np
from qiskit.quantum_info import SparsePauliOp
 
from DeterministicSampling import pauli_terms, evaluate, pauli_weight
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
    The probability that each term is kept in a sample (πi)
 
    kept_terms: a function takes drawn ordering and returns indices of terms kept
    n_mc: how many times to repeat the random draw to estimate πi
    '''
    counts = np.zeros(len(weights))
    for _ in range(n_mc):
        counts[kept_terms(selection(weights, rand_num_gen))] += 1
    return np.maximum(counts / n_mc, 1 / n_mc)
 
def rand_pauli_sampler_without_replacement(full_H_op, weight_fraction, reweight=False, seed=None, n_mc=4000):
    '''
    Returns one sampled H each time function is run
 
    full_H_op: full hamiltonian as a SparsePauliOp
    weight_fraction: terms are drawn one at a time until the drawn terms hold at least this fraction of the total weight
    reweight: if False this is method 2.
              if True the kept terms are multiplied by 1/πi, this is method 3.
    '''
 
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    weights = np.array([abs(c) for _, c in terms])
 
    def kept_terms(order): # takes the first terms of a drawn ordering until they hold the weight fraction
        cumulative_weight = np.cumsum(weights[order])
        return order[: np.searchsorted(cumulative_weight, weight_fraction * weights.sum() - 1e-12) + 1]
 
    scale = 1 / inclusion_probabilities(weights, kept_terms, rand_num_gen, n_mc) if reweight else np.ones(len(terms)) # reweighting 1/π factor only applied if reweight=True
 
    def sampled_H():
        keep = kept_terms(selection(weights, rand_num_gen))
        return SparsePauliOp.from_list(ident + [(terms[i][0], terms[i][1] * scale[i]) for i in keep]) # sampled Hamiltonian
    return sampled_H
 
'''
4. random sampling with replacement of terms (qdrift style), terms are selected and replaced
'''
 
def draw_until_weight(w, weight_fraction, rand_num_gen):
    '''
    Draw terms with replacement, probability proportional to the weight w, until the weights of the draws add up to at
    least weight_fraction * sum(w). Returns the indices drawn (the number of draws N is random).
    '''
    p = w / w.sum()
    picks, drawn_weight = [], 0.0
    while drawn_weight < weight_fraction * w.sum() - 1e-12:
        j = rand_num_gen.choice(len(w), p=p)
        picks.append(j)
        drawn_weight += w[j]
    return picks
 
def qdrift_coefficients(terms, w, picks):
    '''
    Coefficient of each drawn string: (sum w / N) * k_j * sign(h_j), N = number of draws, k_j = times string j was drawn.
    A string drawn several times appears once, with the combined coefficient.
    '''
    coefs = {}
    for j in picks:
        label, c = terms[j]
        coefs[label] = coefs.get(label, 0.0) + w.sum() / len(picks) * np.sign(c.real)
    return list(coefs.items())
 
def rand_pauli_sampler_with_replacement(full_H_op, weight_fraction, seed=None):
    '''
    Returns one sampled H each time function is run
 
    weight_fraction: keep drawing until the weights of the drawn terms (each draw counted at its |h_j|) add up to at
                     least this fraction of alpha. The number of draws N is then random, so the average sampled
                     Hamiltonian is only approximately H.
    '''
 
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    w = np.array([abs(c) for _, c in terms])
 
    def sampled_H():
        picks = draw_until_weight(w, weight_fraction, rand_num_gen)
        return SparsePauliOp.from_list(ident + qdrift_coefficients(terms, w, picks)) # sampled Hamiltonian
    return sampled_H
 
'''
5. Hybrid sampling
Keeps the largest terms of the Pauli Hamiltonian exactly and samples the rest
'''
 
def hybrid_sampling(full_H_op, keep_fraction, draw_fraction, seed=None):
    '''
    keep_fraction: the largest terms are kept exactly, with their true coefficients, until they hold at least this
                   fraction of the total weight
    draw_fraction: the remaining terms are sampled with replacement until the draws hold at least this fraction of
                   the weight that is left over
    '''
    rand_num_gen = np.random.default_rng(seed)
    ident, terms = pauli_terms(full_H_op)
    w = np.array([abs(c) for _, c in terms])
    order = np.argsort(-w, kind="stable")
    n_keep = int(np.searchsorted(np.cumsum(w[order]), keep_fraction * w.sum() - 1e-12) + 1) # smallest number of terms holding keep_fraction
    kept = [terms[i] for i in order[:n_keep]]
    rest = [terms[i] for i in order[n_keep:]]
    w_rest = w[order[n_keep:]]
    def sampled_H():
        sampled = qdrift_coefficients(rest, w_rest, draw_until_weight(w_rest, draw_fraction, rand_num_gen)) if rest else []
        return SparsePauliOp.from_list(ident + kept + sampled)
    return sampled_H
 
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
 
    frac = 0.5 # weight fraction (at U=2 all strings have equal weight, so 0.5 is 5 of the 10 strings)
    frac_keep = 0.3 # weight fraction the hybrid keeps exactly
    frac_draw_hybrid = 0.2 # fraction of the leftover weight the hybrid draws
    frac_draw = 0.7
 
    hybrid_H = hybrid_sampling(H_full, keep_fraction=frac_keep, draw_fraction=frac_draw_hybrid, seed=0)
    with_replacement_H = rand_pauli_sampler_with_replacement(H_full, weight_fraction=frac_draw, seed=0)
    without_replacement_H = rand_pauli_sampler_without_replacement(H_full, weight_fraction=frac, reweight=True, seed=0)
 
    samplers = {"hybrid": hybrid_H, "with rep": with_replacement_H, "without rep": without_replacement_H}
 
    for name, sampled_H in samplers.items():
        string_counts = [len(pauli_terms(sampled_H())[1]) for _ in range(200)]
        print(f"{name:12s} strings per sample: mean {np.mean(string_counts):.2f}, "
            f"min {min(string_counts)}, max {max(string_counts)}")
 
    def run(name, o):
        r = evaluate(o, subspace, exact, m=m, deg=50)["wall_cheb"]
        print(f"{name:80s} alpha={evaluate(o, subspace, exact, m=m, deg=50)['alpha_wall']:.3f}  p={r['p']:.2e}  F={r['fidelity']:.3f}")
 
    def avg(name, sampled_H):
        a = average_over_runs(sampled_H, subspace, exact, m)
        print(f"{name:80s} alpha={a['alpha']:.3f}  p={a['p']:.2e}  F={a['F_mean']:.3f} +/- {a['F_std']:.3f}")
 
    print(f"U={U}, wall-Chebyshev P2, m={m}, S={'E0' if exact else 'HF'}")
    run("no sampling", H_full)
    run(f"deterministic weight fraction f={frac}", pauli_weight(H_full, frac))
    avg(f"random sampling with no replacement of terms, weight fraction f={frac}", rand_pauli_sampler_without_replacement(H_full, weight_fraction=frac, seed=0))
    avg(f"random sampling with no replacement of terms, weight fraction f={frac}, reweighted", rand_pauli_sampler_without_replacement(H_full, weight_fraction=frac, reweight=True, seed=0))
    avg(f"Random sampling with replacement of terms, weight fraction f={frac_draw}", rand_pauli_sampler_with_replacement(H_full, weight_fraction=frac_draw, seed=0))
    avg(f"hybrid: {frac_keep} of the weight exact + draws holding {frac_draw_hybrid} of the rest", hybrid_sampling(H_full, keep_fraction=frac_keep, draw_fraction=frac_draw_hybrid, seed=0))