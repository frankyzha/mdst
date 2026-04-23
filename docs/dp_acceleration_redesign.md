# Contiguous-DP Acceleration Redesign for Nonlinear MSPLIT

This note records the final four-case optimization picture after switching the impurity family from joint impurity to hard-label Gini impurity.

The four cases are:

1. binary misclassification loss
2. multiclass misclassification loss
3. binary Gini impurity
4. multiclass Gini impurity

The hard-loss cases now have exact closed-form accelerations in the code. The Gini cases are harder. For them, this note separates:

- what is already implemented exactly and safely in nonlinear MSPLIT
- what I believe is the cleanest exact theoretical limit

The short summary is:

- binary hard loss admits an exact `O(Km)` DP by two running minima
- multiclass hard loss admits an exact `O(KCm)` DP by one running minimum per class
- binary Gini is exactly a one-parameter weighted least-squares segmentation problem
- multiclass Gini is exactly a `(C-1)`-parameter weighted least-squares segmentation problem
- there is no exact finite-running-minima analogue for Gini in general
- the strongest exact ceiling for Gini is envelope pruning, not a Monge or divide-and-conquer shortcut

Here `m` is the number of occupied bins, `C` is the number of classes, and `K` is the maximum arity considered by the contiguous DP.

## 1. Standard Notation

Fix one node and one feature. Let the occupied bins for that feature be indexed by

`1, 2, ..., m`.

For bin `u`:

- `w_u > 0` is the total sample weight in bin `u`
- `h_u in R_+^C` is the class-count vector in bin `u`
- `p_u = h_u / w_u in Delta^{C-1}` is the empirical class-probability vector in bin `u`

Define prefix sums

- `W_t = sum_{u=1}^t w_u`
- `H_t = sum_{u=1}^t h_u`

with `W_0 = 0` and `H_0 = 0`.

For a segment `(k, t] = {k+1, ..., t}`, define

- `W(k,t) = W_t - W_k`
- `H(k,t) = H_t - H_k`

The fixed-arity contiguous DP has the form

`F_q(t) = min_{k in A_q(t)} F_{q-1}(k) + L(k,t) + B_q(k)`

where

- `F_q(t)` is the optimal cost of partitioning bins `1, ..., t` into `q` contiguous children
- `L(k,t)` is the segment cost on `(k,t]`
- `B_q(k)` is any additive cut-dependent term, such as the boundary penalty already tracked in the code
- `A_q(t) = { k < t : rows(k,t) >= min_child_size }`

The acceleration problem is therefore about the segment cost `L(k,t)`.

## 2. Four-Case Snapshot

| Case | Exact segment cost | Best exact acceleration story | Current implementation |
| --- | --- | --- | --- |
| Binary hard loss | `min(pos, neg) = [W - |D|]/2` | two running minima | implemented |
| Multiclass hard loss | `W - max_c H_c` | one running minimum per class | implemented |
| Binary Gini | `W/2 - D^2/(2W)` | one-parameter envelope pruning | exact delayed dominance pruning |
| Multiclass Gini | `W - ||H||_2^2 / W` | `(C-1)`-parameter geometric or dual pruning | exact delayed dominance pruning |

Here `D` denotes the binary hard margin on the segment.

The hard-loss rows are exact closed forms. The Gini rows do not collapse to finitely many scalar minima in general.

## 3. Cases I and II: Misclassification Loss

### 3.1 Binary Misclassification

Let

`D_t = H_{t,1} - H_{t,0}`.

Then

`L_hard(k,t) = min(pos(k,t), neg(k,t)) = [W(k,t) - |D_t - D_k|] / 2`.

Therefore

`F_q(t) = W_t/2 + min { M_q^+(t) - D_t/2, M_q^-(t) + D_t/2 }`

with two running minima

- `M_q^+(t) = min_{k in A_q(t)} [ F_{q-1}(k) - W_k/2 + D_k/2 + B_q(k) ]`
- `M_q^-(t) = min_{k in A_q(t)} [ F_{q-1}(k) - W_k/2 - D_k/2 + B_q(k) ]`

This yields an exact `O(Km)` algorithm.

### 3.2 Multiclass Misclassification

For multiclass labels,

`L_hard(k,t) = W(k,t) - max_c H_c(k,t)`.

Using `-max_c x_c = min_c (-x_c)`,

`F_q(t) = W_t + min_c { -H_{t,c} + M_{q,c}(t) }`

where

`M_{q,c}(t) = min_{k in A_q(t)} [ F_{q-1}(k) - W_k + H_{k,c} + B_q(k) ]`.

Thus we need one running minimum per class, which gives an exact `O(KCm)` algorithm.

These two kernels are already implemented in the nonlinear solver.

## 4. Case III: Binary Gini Impurity

### 4.1 Exact Segment Formula

For binary labels, define the centered bin profile

`x_u = p_{u,1} - p_{u,0} in [-1,1]`

and the prefix margin

`M_t = sum_{u=1}^t w_u x_u = D_t`.

The segment Gini impurity is

`L_gini(k,t) = W(k,t)/2 - [M_t - M_k]^2 / [2 W(k,t)]`.

Equivalently, define the prefix constant

`A_t = (1/2) sum_{u=1}^t w_u (1 - x_u^2)`.

Then

`L_gini(k,t) = A_t - A_k + min_{mu in [-1,1]} [ (W_t - W_k) mu^2 / 2 - (M_t - M_k) mu ]`.

So binary Gini is exactly a one-parameter weighted least-squares segmentation problem.

### 4.2 Exact Envelope Formulation

Plugging the previous identity into the DP gives

`F_q(t) = A_t + min_{mu in [-1,1]} [ W_t mu^2 / 2 - M_t mu + E_q(t, mu) ]`

where

`E_q(t, mu) = min_{k in A_q(t)} [ F_{q-1}(k) - A_k + M_k mu - W_k mu^2 / 2 + B_q(k) ]`.

For each predecessor `k`, the inner contribution is a concave quadratic function of `mu`. The exact DP is therefore a lower-envelope problem over one scalar parameter.

This is the cleanest exact ceiling for binary Gini:

- it is stronger than generic inequality pruning
- it is cleaner than the old joint-impurity two-dimensional formulation
- it explains why binary Gini is still harder than binary hard loss

### 4.3 Why There Is No Exact Finite-Minima Collapse

The binary hard-loss trick works because the absolute value is polyhedral. Binary Gini is curved:

`- z^2 / (2w) = min_{lambda in R} [ - lambda z + (w/2) lambda^2 ]`.

So exact binary Gini requires a continuum of dual directions. Any finite set of running minima would only approximate the parabola.

That is why there is no exact two-minima analogue for binary Gini.

### 4.4 Exact Structural Compression: Hard-Profile Blocks

The exact Gini-side analogue of a majority run is a hard-profile block: a maximal consecutive run of bins with identical empirical class profile.

If `p_a = ... = p_b = p`, then every subsegment `S` inside that block satisfies

`H_S = W_S p`

and therefore

`L_gini(S) = W_S (1 - ||p||_2^2)`.

So inside one hard-profile block the segment cost is linear in the segment weight.

Moreover, if a single cut moves inside such a block while the left and right outside contexts stay fixed, the resulting two-sided Gini objective is concave in the moved weight. Hence the optimum over a feasible interval is attained at one of the two feasible endpoints.

This is the exact Gini-side snapping lemma. It justifies hard-profile blocks as the right structural object for safe compression and pruning arguments. It also shows why pure same-class blocks are sufficient but not theoretically tight.

### 4.5 What Is Implemented Today

The current nonlinear MSPLIT code does not yet implement the full one-parameter envelope solver above. Instead, it uses an exact delayed dominance-pruning rule:

- predecessor states become active only when the last child-size constraint allows them
- an impurity witness at endpoint `t` can prune an older predecessor only from the first future endpoint at which `t` itself becomes a feasible predecessor
- this preserves exactness under `min_child_size`

This is weaker than full functional pruning, but it is exact and it already reduces runtime without changing the selected trees.

## 5. Case IV: Multiclass Gini Impurity

### 5.1 Exact Least-Squares Reduction

Let `U in R^{C x (C-1)}` be an orthonormal basis for the hyperplane orthogonal to the all-ones vector, and define

- `z_u = U^T p_u in R^{C-1}`
- `Z_t = sum_{u=1}^t w_u z_u`

Then, up to a partition-independent additive constant, multiclass Gini is exactly weighted least-squares segmentation of the ordered vectors `z_u`.

Equivalently, if

`A_t = sum_{u=1}^t w_u (1 - ||p_u||_2^2)`,

then

`L_gini(k,t) = A_t - A_k + min_{mu in R^{C-1}} [ (W_t - W_k) ||mu||_2^2 - 2 (Z_t - Z_k)^T mu ]`.

So the effective parameter dimension is `C-1`, not `C`.

### 5.2 Exact Multiclass Envelope Formulation

The fixed-arity DP becomes

`F_q(t) = A_t + min_{mu in R^{C-1}} [ W_t ||mu||_2^2 - 2 Z_t^T mu + E_q(t, mu) ]`

with

`E_q(t, mu) = min_{k in A_q(t)} [ F_{q-1}(k) - A_k + 2 Z_k^T mu - W_k ||mu||_2^2 + B_q(k) ]`.

Again, every predecessor contributes a concave quadratic function of `mu`. The exact DP is therefore a lower-envelope problem, now in `C-1` dimensions.

This is the cleanest exact multiclass story I know:

- the reduction is exact
- the dimension is fixed by the number of classes
- but the pruning object is geometric, not a finite family of scalar minima

### 5.3 Practical Ceiling

For multiclass Gini, the exact limit is low-dimensional geometric or dual pruning:

- when `C` is small, exact geometric envelope maintenance is conceptually appropriate
- when `C` grows, exact region maintenance becomes expensive
- in that regime, exact delayed dominance pruning is a reasonable robust fallback

So multiclass Gini is cleaner than the old joint impurity, but it still does not collapse to anything as simple as `O(KCm)` running minima.

## 6. Shortcuts That Fail

### 6.1 Majority-Shift RUSH Is Not Exact for Gini

Binary counterexample with bins `(pos, neg)`:

- `b1 = (0,1)`
- `b2 = (4,0)`
- `b3 = (3,2)`
- `b4 = (6,7)`

Bins `b2` and `b3` have the same majority class, but the best Gini cut is inside that majority run rather than at its boundary. So majority-shift RUSH is not exact for Gini.

### 6.2 Monge and Divide-and-Conquer Shortcuts Fail

For the binary sequence `(1,0,1)`, let `C(i,j)` be the one-segment binary Gini cost on `[i,j)`. Then

- `C(0,2) = 1/2`
- `C(1,3) = 1/2`
- `C(0,3) = 2/3`
- `C(1,2) = 0`

Thus

`C(0,2) + C(1,3) > C(0,3) + C(1,2)`,

so the Monge inequality fails.

There are also explicit binary examples in which the optimal predecessor for `F_2(t)` decreases as `t` grows, so the standard monotone-argmin divide-and-conquer optimization is not valid either.

### 6.3 Penalized Search Alone Cannot Recover Every Fixed-Arity Optimum

For the binary sequence `(0,1,0)`, the exact fixed-arity Gini costs are:

- `F_1 = 2/3`
- `F_2 = 1/2`
- `F_3 = 0`

The point `(2, F_2)` lies above the lower convex hull of `(1, F_1)` and `(3, F_3)`, so there is no penalty value whose penalized optimum coincides with the exact `K=2` solution.

Therefore optimal-partitioning methods are good inspiration, but penalty search alone cannot replace the fixed-arity DP exactly.

## 7. Final Redesign

The cleanest final design is:

1. Binary hard loss: exact `O(Km)` running-minima DP.
2. Multiclass hard loss: exact `O(KCm)` running-minima DP.
3. Binary Gini: exact one-parameter envelope pruning as the theoretical limit; exact delayed dominance pruning as the current robust implementation.
4. Multiclass Gini: exact `(C-1)`-parameter geometric or dual pruning as the theoretical limit; exact delayed dominance pruning as the current robust implementation.

This is the strongest exact story I can defend without overstating what the present code already does.

## 8. Complexity Summary

Let `m` be the number of occupied bins, `C` the number of classes, and `K` the largest arity considered by the contiguous DP.

Current exact implementation:

- binary hard loss: `O(Km)`
- multiclass hard loss: `O(KCm)`
- binary Gini: worst-case `O(Km^2)`, with exact delayed dominance pruning
- multiclass Gini: worst-case `O(Km^2 C)` segment evaluation cost if implemented naively, but the current code uses prefix-based `O(C)` impurity queries plus exact delayed dominance pruning

Theoretical exact ceiling:

- binary Gini: one-parameter output-sensitive envelope pruning
- multiclass Gini: `(C-1)`-parameter output-sensitive geometric or dual pruning

Compared with joint impurity, the key simplification is dimensional:

- binary impurity drops from a two-parameter envelope problem to a one-parameter problem
- multiclass impurity drops from `2C-2` parameters to `C-1`

## 9. Literature Anchors

- Guillem Rigaill. *A pruned dynamic programming algorithm to recover the best segmentations with 1 to Kmax change-points*. 2015. https://arxiv.org/abs/1004.0887
- Robert Maidstone, Toby Hocking, Guillem Rigaill, Paul Fearnhead. *On optimal multiple changepoint algorithms for large data*. 2017. https://tdhock.github.io/assets/papers/Maidstone2017optimal.pdf
- Liudmila Pishchagina, Guillem Rigaill, Vincent Runge. *Geometric-Based Pruning Rules for Change Point Detection in Multiple Independent Time Series*. 2024. https://computo-journal.org/published-202406-pishchagina-change-point/published-202406-pishchagina-change-point.pdf
- Vincent Runge, Charles Truong, Simon Querne. *DUST: A Duality-Based Pruning Method For Exact Multiple Change-Point Detection*. 2025. https://arxiv.org/abs/2507.02467

Those papers support the final interpretation:

- functional pruning is the natural exact ceiling in one parameter
- multivariate exact pruning is a geometric or dual problem
- inequality pruning is simpler but weaker

That is the final optimization picture I trust for the four contiguous-DP cases.
