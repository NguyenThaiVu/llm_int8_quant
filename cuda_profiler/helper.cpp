#include <cmath>
#include <cfloat>
#include <cstdint>

// x: [rows, cols] row-major
// y: [rows, cols] row-major
void softmax_lastdim_2d_cpu(const float* x, float* y, int64_t rows, int64_t cols)
{
    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float* yr       = y + r * cols;

        // 1) max
        float m = -FLT_MAX;
        for (int64_t c = 0; c < cols; ++c) {
            if (xr[c] > m) m = xr[c];
        }

        // 2) sum exp(x - max)
        double sum = 0.0;  // double for a slightly more stable reference
        for (int64_t c = 0; c < cols; ++c) {
            sum += std::exp((double)xr[c] - (double)m);
        }

        // 3) normalize
        double inv = 1.0 / sum;
        for (int64_t c = 0; c < cols; ++c) {
            yr[c] = (float)(std::exp((double)xr[c] - (double)m) * inv);
        }
    }
}

void softmax_lastdim_3d_cpu(const float* x, float* y, int64_t dim0, int64_t dim1, int64_t dim2)
{
    for (int64_t i = 0; i < dim0; ++i) {
        for (int64_t j = 0; j < dim1; ++j) {
            const float* xij = x + (i * dim1 + j) * dim2;
            float* yij       = y + (i * dim1 + j) * dim2;

            // 1) max
            float m = -FLT_MAX;
            for (int64_t c = 0; c < dim2; ++c) {
                if (xij[c] > m) m = xij[c];
            }

            // 2) sum exp(x - max)
            double sum = 0.0;  // double for a slightly more stable reference
            for (int64_t c = 0; c < dim2; ++c) {
                sum += std::exp((double)xij[c] - (double)m);
            }

            // 3) normalize
            double inv = 1.0 / sum;
            for (int64_t c = 0; c < dim2; ++c) {
                yij[c] = (float)(std::exp((double)xij[c] - (double)m) * inv);
            }
        }
    }
}

bool allclose(const float* a, const float* b, int64_t n, float rtol=1e-5f, float atol=1e-6f)
{
    for (int64_t i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float tol  = atol + rtol * std::fabs(b[i]);
        if (diff > tol) {
            std::printf("Mismatch at %lld: a=%g b=%g diff=%g tol=%g\n",
                        (long long)i, a[i], b[i], diff, tol);
            return false;
        }
    }
    return true;
}

void init_1d_vector_int8(int8_t* x_q, int64_t n, int8_t value) {
    for (int64_t i = 0; i < n; ++i) {
        x_q[i] = value;
    }
}

void init_1d_vector_float(float* x, int64_t n, float value) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] = value;
    }
}