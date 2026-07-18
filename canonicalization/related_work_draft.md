# Related-work draft: three sections for the literature review

Draft prose for the thesis lit review, written 2026-07-17. Citation keys are
`[Author Year, id]` placeholders to map onto BibTeX. Claims marked **[ours]**
are backed by this repo's measurements; claims marked **[self-reported]** come
from a tool's own paper and were not independently verified.

Suggested placement: these three sections form the related-work chapter, in
this order — they walk from "structure discovered" to "structure declared" to
"the convex-specific instance of the question", which is exactly the thesis
argument. The existing intro sections stay as background: "Computing
derivatives" introduces AD/coloring machinery that §A then surveys per tool;
"Matrix computations" covers the sparse-matrix substrate §B's assembly systems
rely on; "Canonicalization in CVXPY" defines the workload that §C situates
among its peers.

---

## A. Generic sparse automatic differentiation: detect, color, sweep

The classical pipeline for computing a sparse Jacobian or Hessian with
automatic differentiation has three stages: *detect* the sparsity pattern,
*compress* structurally orthogonal columns (or rows) via graph coloring
[Curtis, Powell & Reid 1974, cpr74; Coleman & Moré 1983, coleman83], and
*accumulate* the compressed derivative in one AD sweep per color
[Gebremedhin, Manne & Pothen 2005, gebremedhin05; Griewank & Walther 2008,
griewank08]. The three stages appear, with different engineering, in every
generic sparse-AD system in production use.

ADOL-C [Griewank, Juedes & Utke 1996, adolc] pairs taped
operator-overloading AD with pattern propagation ("safe"/"tight" modes) and
delegates coloring and seeding to ColPack [Gebremedhin et al. 2013, colpack].
CasADi [Andersson et al. 2019, casadi] detects patterns by hierarchical
bitvector dependency propagation (64 columns per sweep) and colors with a
greedy distance-2 heuristic, tried column-wise and row-wise with a penalty
factor for reverse sweeps; Hessians use star coloring. The most recent entrant
is the Julia stack of SparseConnectivityTracer.jl (operator-overloading
index-set tracing, a "binarization of the chain rule"),
SparseMatrixColorings.jl, and DifferentiationInterface.jl, which reports up to
three orders of magnitude improvement over the previous Julia state of the art
(Symbolics.jl + SparseDiffTools.jl) and improved star/acyclic/bicoloring
implementations relative to ColPack [Hill & Dalle 2025, sbfs;
Montoison, Dalle & Gebremedhin 2025, smc]. Notably, the mainstream ML
frameworks never built this machinery: PyTorch, TensorFlow and JAX offer no
sparsity detection or coloring; the only JAX option, sparsejac
[Schubert, sparsejac], requires the user to supply the pattern — itself
evidence for this thesis's premise. CppAD's "subgraph" drivers [Bell, cppad]
are an interesting outlier that skips coloring by extracting per-row
dependency subgraphs.

The pipeline's cost model is well understood in the compressed-sweep count:
for a Jacobian whose columns cannot be compressed — a single dense m×n block
suffices, since all its columns pairwise conflict — the sweep count degenerates
to min(m, n) and the total cost to O(n · nnz). What the literature does not
emphasize is how routinely this worst case arises in *practice*: any affine
expression carrying a dense data matrix (a regression design matrix, a
scenario matrix, a dense quadratic form) triggers it, even though the
"derivative" being reconstructed is a constant sitting in the expression graph.
**[ours]** We measure this mechanism identically in two independent
implementations: on the Jacobian of x ↦ Ax with dense A of size 4000×2000,
CasADi 3.7.2 spends 16.3 s (2000 forward colors), and the 2025 Julia stack
~53 s — of which 40 s is the greedy distance-2 coloring itself — while the
same shape at 10 nonzeros per row collapses to ~0.03 s in both. The pathology
is a property of the detect–color–sweep paradigm, not of any one
implementation. **[ours]** No peer-reviewed benchmark of CasADi's Jacobian
*construction* cost appears to exist (issue-tracker reports aside); the
mechanism probes in this work fill that gap.

## B. Modeling systems with declared derivative structure

A second family of systems never discovers structure, because their input
format declares it. AMPL [Fourer, Gay & Kernighan 1990, ampl] is the archetype:
the translator emits a `.nl` file in which the linear part of every objective
and constraint is stored as explicit sparse coefficient lists, so the Jacobian
pattern — and for affine rows, the values — exist the moment the AMPL Solver
Library reads the file [Gay, hooking]. Nonlinear parts are per-row expression
graphs differentiated by reverse AD on tapes allocated at read time, and
Hessian structure is recovered once, at read time, by detecting (group)
partial separability of the objective graph [Gay 1996, gay96]. There is no
sparsity detection pass and no coloring anywhere in the pipeline. **[ours]**
Our measurements confirm the corollary: driving the same dense-block probe
through the real ASL (via Pyomo's `.nl` writer and PyNumero's `AslNLP`), cost
is linear in nnz for dense and sparse data alike; the dense 4000×2000 case
costs 16.6 s of `.nl` writing and 16.5 s of parsing — text I/O, CPU-bound,
linear — after which the Jacobian evaluates in 0.2 s. The contrast with §A is
the thesis in miniature: the same matrix that costs CasADi 2000 AD sweeps and
the Julia stack 40 s of coloring is, to ASL, a list of numbers to be read.

JuMP [Dunning, Huchette & Lubin 2017, jump; Lubin et al. 2023, jump1] sits in
the same family for problem-data assembly: affine and quadratic coefficients
flow from user arrays into MathOptInterface's `MatrixOfConstraints` storage
during `copy_to`, with no AD involved [Legat, Dowson, Dias Garcia & Lubin
2022, moi]. (For *nonlinear* programs JuMP does run sparse reverse AD with
acyclic-coloring-based Hessians, descending from ReverseDiffSparse — structure
there is derived from the restricted expression format rather than traced from
a black box.) The MOI paper also contains the closest published precedent for
our cross-family comparison: its §5.4 P-median benchmark times problem
*generation* and *load* for MOI against solver C APIs and CVXPY 1.1.3,
finding CVXPY 10–20× over the direct C API and up to 455 s vs 1.17 s of
generation time at the largest size [Legat et al. 2022, moi, Table 1]. That
experiment is a sparse LP with CVXPY as of 2020; the present work updates the
comparison with a modern CVXPY, verified bit-identical outputs across systems,
dense-data and quadratic instances where the interesting regimes live, and a
cold/warm split.

The declared-structure idea has sharper recent incarnations. Gravity
[Hijazi et al. 2018, gravity] symbolically differentiates repeated constraint
*templates* once and instantiates over data, reporting ~5× faster derivative
evaluation than JuMP **[self-reported]**. ExaModels.jl
[Shin, Pacaud & Anitescu 2023, examodels] pushes the same idea to a SIMD
abstraction — derivative kernels generated once per computational pattern and
materialized over data arrays, reporting up to two orders of magnitude faster
derivative evaluation than JuMP or AMPL on AC-OPF (on GPU; structure
exploitation and parallelism contribute jointly) **[self-reported]**.
MathOptSymbolicAD, now MOI's `SymbolicAD` submodule [Dowson, symbolicad],
retrofits template detection onto JuMP models. SparseDiffEngine belongs to
this family — its unit of declared structure is the *atom*: each CVXPY atom
carries a closed-form sparse Jacobian block, and canonicalization assembles
them without ever discovering a pattern.

## C. Canonicalization of convex programs

Disciplined convex programming [Grant, Boyd & Ye 2006, dcp] made convex
modeling languages possible: CVX [Grant & Boyd 2014, cvx], CVXPY
[Diamond & Boyd 2016, cvxpy; Agrawal et al. 2018, rewriting], and Convex.jl
[Udell et al. 2014, convexjl] lower an atom-tree expression to conic standard
form. The lowering — canonicalization — is itself the sparse-linear-algebra
workload this thesis studies: producing (P, c, A, b) is exactly evaluating the
(constant) Jacobian of the lowered affine map. CVXPY's `cvxcore` backends and
the disciplined-parametrized-programming machinery [Agrawal et al. 2019, dpp]
define the cold and warm regimes; the benchmark suite of 25 problems used
throughout this work is CVXPY's own.

The cost of canonicalization is a recognized pain point rather than a
published benchmark topic. The MOI paper's Table 1 (above) is the only
cross-language measurement we are aware of; within the CVXPY ecosystem the
clearest evidence is CVXPYgen [Schaller et al. 2022, cvxpygen; 2025 follow-up,
cvxpygen25], which concedes the point architecturally: it caches
canonicalization offline and generates C code that re-canonicalizes only the
parameter-dependent entries, reporting up to three orders of magnitude
end-to-end speedup for embedded re-solves **[self-reported]**. CVXPYgen is
therefore the natural amortized comparison point for the engine's warm
re-solve path, while the present work targets the un-amortized question: what
does a *single* cold extraction cost, and why. Positioned against §A and §B,
the answer this thesis documents is that CVXPY's canonicalization workload
lands precisely on generic sparse AD's worst case (dense data blocks in affine
maps) and precisely on declared-structure systems' best case — and that
SparseDiffEngine, ASL and JuMP/MOI, which never trace or color, behave
uniformly linearly where CasADi and the Julia stack degenerate.

---

### Notes / verification caveats for the author

- Gay 1996 details (exact "group partial separability" terminology) should be
  checked against the PDF (ampl.com/REFS/ad96.pdf) before quoting — summarized
  here from secondary sources.
- Gravity's and ExaModels' headline factors are self-reported; label them as
  such or reproduce.
- MOI Table 1 numbers quoted above: Julia 1.5 / Python 3.8 / CVXPY 1.1.3 /
  GLPK 4.64 / SCS 2.1.2, solvers set to terminate immediately; their CVXPY
  implementation pre-assembled coefficient matrices ("many times slower"
  otherwise, their words) — worth noting when citing.
- The §A/§B measured numbers are this repo's probe results
  (results_probe_dense_block_{julia,asl}.json, probes/probe_why_slow.py);
  final Tier C timing tables land in results_{julia,jump,asl}_compare.json.
