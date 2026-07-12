#include <adolc/adolc.h>
#include <array>
#include <cmath>
#include <iostream>

// Same tridiagonal system as casadi/casadi_sparse_jacobian.py and
// jax/jax_sparse_jacobian.py:
//   f_i = sin(x_i) + x_{i-1}*x_i + exp(-x_i * x_{i+1})
//
// ADOL-C uses a tape-based approach:
//   1. Record the computation on a "tape" using adouble types
//   2. Replay the tape to compute derivatives
//   3. sparse_jac() automatically detects sparsity and uses ColPack
//      graph coloring to minimize the number of forward passes

constexpr int n = 10;

template <typename T>
void eval(const std::array<T, n> &x, std::array<T, n> &y) {
    for (int i = 0; i < n; i++) {
        y[i] = sin(x[i]);
        if (i > 0)
            y[i] += x[i - 1] * x[i];
        if (i < n - 1)
            y[i] += exp(-x[i] * x[i + 1]);
    }
}

int main() {
    auto tapeId = createNewTape();

    std::array<double, n> x, y;
    for (int i = 0; i < n; i++) x[i] = 1.0;

    // --- Step 1: Record the tape ---
    trace_on(tapeId);

    std::array<adouble, n> ax, ay;
    for (int i = 0; i < n; i++)
        ax[i] <<= x[i];

    eval(ax, ay);

    for (int i = 0; i < n; i++)
        ay[i] >>= y[i];

    trace_off();

    // --- Step 2: Detect sparsity pattern ---
    // Propagates index domains through the tape to find which outputs
    // depend on which inputs — without computing any numerical derivatives.
    std::vector<uint *> JP(n);
    std::span<uint *> JP_span(JP);
    ADOLC::Sparse::jac_pat<ADOLC::Sparse::SparseMethod::IndexDomains,
                           ADOLC::Sparse::ControlFlowMode::Safe>(
        tapeId, n, n, x.data(), JP_span);

    std::cout << "Sparsity pattern (row -> nonzero columns):\n";
    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        std::cout << "  row " << i << ": ";
        for (uint j = 1; j <= JP[i][0]; j++)
            std::cout << JP[i][j] << " ";
        std::cout << "\n";
        total_nnz += JP[i][0];
    }
    std::cout << "Non-zeros: " << total_nnz << " out of " << n * n << "\n";

    // --- Step 3: Graph coloring (via ColPack) ---
    // generate_seed_jac builds the seed matrix from the sparsity pattern.
    // Each column of the seed matrix = one "color" = one forward-mode pass.
    // Columns of J assigned the same color are structurally orthogonal.
    double **seed = nullptr;
    int p = 0;  // number of colors
    ADOLC::Sparse::generate_seed_jac<ADOLC::Sparse::CompressionMode::Column>(
        n, n, JP_span, &seed, &p);

    std::cout << "\nGraph coloring: " << p << " colors (forward passes) vs " << n << " without coloring\n";
    std::cout << "Seed matrix (rows=input columns, cols=colors, 1=assigned):\n";
    for (int i = 0; i < n; i++) {
        std::cout << "  col " << i << ": ";
        for (int j = 0; j < p; j++)
            printf("%.0f ", seed[i][j]);
        std::cout << "\n";
    }

    // --- Step 4: Compute sparse Jacobian ---
    int nnz;
    unsigned int *rind = nullptr, *cind = nullptr;
    double *values = nullptr;

    ADOLC::Sparse::sparse_jac<
        ADOLC::Sparse::SparseMethod::IndexDomains,
        ADOLC::Sparse::CompressionMode::Column,
        ADOLC::Sparse::ControlFlowMode::Safe,
        ADOLC::Sparse::BitPatternPropagationDirection::Auto>(
        tapeId, n, n, 0, x.data(), &nnz, &rind, &cind, &values);

    std::cout << "\nSparse Jacobian (" << nnz << " non-zeros):\n";
    for (int i = 0; i < nnz; i++) {
        printf("  J[%d,%d] = %f\n", rind[i], cind[i], values[i]);
    }

    for (int i = 0; i < n; i++) delete[] JP[i];
    for (int i = 0; i < n; i++) delete[] seed[i];
    delete[] seed;
    delete[] rind;
    delete[] cind;
    delete[] values;
    return 0;
}
