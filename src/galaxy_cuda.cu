// Native CUDA implementation tuned for NVIDIA RTX 5080 (Blackwell, sm_120).
// Same two-point angular correlation as galaxy_hip.cpp, restructured for
// NVIDIA:
//   - block-tiled pair loop (each thread owns one i, loops a shared-memory
//     j-tile) instead of one-thread-per-pair -> massive reuse / fewer loads
//   - precomputed per-galaxy sin/cos(decl) (no per-pair sincos)
//   - fast polynomial acos approximation (measured |err| <= 6.77e-5 rad;
//     this does shift a small number of counts between bins, see fast_acosf)
//   - DD/RR symmetry with whole-block diagonal skipping
//   - inner-loop unrolling; shared-memory atomic histograms (fast on Blackwell)
// Measured on an RTX 5080 (driver 610.57.04, CUDA 13.3, sm_120, performance
// governor), median of runs 2-9: 22.7 ms kernel, 0.032 s wall clock. The same
// code with the block tiling, the fast acos, the symmetry and the unrolling
// removed one at a time measures 188.7 ms, so the tuning is worth 8.3x. See
// the Makefile (`make cuda`, `make bench`) for the build and run protocol.
//
// The host side (catalog reader, timing, histogram assertions) is duplicated
// in galaxy_hip.cpp rather than shared through a header. That is deliberate:
// each backend is one self-contained translation unit that compiles with a
// single command and can be read end to end. The two share no device code.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static float *real_rasc;
static float *real_decl;
static float *rand_rasc;
static float *rand_decl;
static float *real_sin; // precomputed sin(decl) for real catalog
static float *real_cos; // precomputed cos(decl) for real catalog
static float *rand_sin; // precomputed sin(decl) for rand catalog
static float *rand_cos; // precomputed cos(decl) for rand catalog
static constexpr long int N = 100000L;
static long *histogram_DR;
static long *histogram_DD;
static long *histogram_RR;
static constexpr float PI = 3.14159265358979323846f;
static long int CPUMemory = 0L;
static long int GPUMemory = 0L;
static constexpr int totaldegrees = 360;
static constexpr int binsperdegree = 4;

// 1440 bins padded to 1536, carried over from the RDNA 2 tuning pass where the
// intent was to avoid LDS bank conflicts. Note 1536 is 3 * 512, not a power of
// two. On this GPU the padding measures as a no-op: an interleaved A/B of 1536
// against 1440 agrees to within 0.06 ms (0.3%) on every statistic, because the
// atomics scatter across bins by data rather than by thread index. It is kept
// only so the output stays comparable with the HIP build; 1440 is equally fine.
const int num_bins = binsperdegree * totaldegrees; // 1440
const int num_bins_padded = 1536;

// Tile size = threads per block. Each block computes a TILE x TILE sub-block of
// the pair matrix: TILE threads each own one i and loop a shared-memory j-tile.
#ifndef TILE
#define TILE 512
#endif

#define CUDA_ERR_CHECK(ans)                                                                                            \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "   GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ __forceinline__ void hist_add(unsigned int *histogram, int bin_index, unsigned int increment = 1U) {
    atomicAdd(&histogram[bin_index], increment);
}

__device__ __forceinline__ float fast_acosf(float x) {
    // Handbook-of-Math-Functions minimax approximation. Measured maximum error
    // against acosf is 6.77e-5 rad (0.0039 degrees) over a 2e8-point sweep of
    // [-1, 1], which is far below the 0.25-degree bin width.
    //
    // That bound does NOT make the binning identical, and an earlier version of
    // this comment wrongly claimed it did. A sample only has to sit within
    // 0.0039 degrees of a bin edge to move, and 0.65% of the sweep does exactly
    // that. Over the real catalogs it shifts counts in 353 of the 360 populated
    // bins, at most 0.03% of any one bin, with a largest omega change of
    // 0.002257 in the bin at 89.75 degrees. The histogram totals are unaffected,
    // so the N*N asserts below cannot see this: a total is invariant to
    // redistribution. Use acosf instead if you need exact bin agreement.
    float negate = (float)(x < 0.0f);
    x = fabsf(x);
    float ret = -0.0187293f;
    ret = ret * x + 0.0742610f;
    ret = ret * x - 0.2121144f;
    ret = ret * x + 1.5707288f;
    ret = ret * sqrtf(1.0f - x);
    ret = ret - 2.0f * negate * ret;
    return negate * 3.14159265358979f + ret;
}

__device__ __forceinline__ int compute_histogram_index(float expr) {
    constexpr float angle = 57.29577951308232f;
    expr = fminf(fmaxf(expr, -1.0f), 1.0f);

    int histogram_index = int(fast_acosf(expr) * angle * binsperdegree);
    return (histogram_index < 0) ? 0 : ((histogram_index >= num_bins) ? num_bins - 1 : histogram_index);
}

// Block-tiled kernel with precomputed sin/cos(decl).
// Grid is 2D over TILE-sized i-blocks (x) and j-blocks (y). Each of the TILE
// threads owns one i; it keeps its real_i/rand_i data in registers and loops a
// shared-memory tile of TILE j-galaxies. DD/RR use symmetry (j >= i); blocks
// fully below the diagonal skip DD/RR entirely.
__global__ void __launch_bounds__(TILE)
    fill_histograms(const float *__restrict__ d_real_rasc, const float *__restrict__ d_real_sin,
                    const float *__restrict__ d_real_cos, const float *__restrict__ d_rand_rasc,
                    const float *__restrict__ d_rand_sin, const float *__restrict__ d_rand_cos,
                    unsigned long long int *d_histogram_DR, unsigned long long int *d_histogram_DD,
                    unsigned long long int *d_histogram_RR) {
    const int tid = threadIdx.x;
    const int i_base = blockIdx.x * TILE;
    const int j_base = blockIdx.y * TILE;
    const int i = i_base + tid;

    extern __shared__ unsigned int s_mem[];
    unsigned int *s_hist_DR = s_mem;
    unsigned int *s_hist_DD = s_hist_DR + num_bins_padded;
    unsigned int *s_hist_RR = s_hist_DD + num_bins_padded;
    // Coordinate tiles for the j-block (real and rand catalogs).
    float *s_coord = (float *)(s_hist_RR + num_bins_padded);
    float *sj_real_sin = s_coord;
    float *sj_real_cos = sj_real_sin + TILE;
    float *sj_real_ra = sj_real_cos + TILE;
    float *sj_rand_sin = sj_real_ra + TILE;
    float *sj_rand_cos = sj_rand_sin + TILE;
    float *sj_rand_ra = sj_rand_cos + TILE;

    for (int b = tid; b < num_bins_padded; b += TILE) {
        s_hist_DR[b] = 0;
        s_hist_DD[b] = 0;
        s_hist_RR[b] = 0;
    }

    // Load this thread's i-galaxy data into registers.
    const bool valid_i = i < N;
    const int li = valid_i ? i : 0;
    const float ri_sin = d_real_sin[li];
    const float ri_cos = d_real_cos[li];
    const float ri_ra = d_real_rasc[li];
    const float di_sin = d_rand_sin[li];
    const float di_cos = d_rand_cos[li];
    const float di_ra = d_rand_rasc[li];

    const int j_end = min(j_base + TILE, (int)N);
    const int j_count = j_end - j_base;
    // Whole block below the diagonal -> no j >= i anywhere -> skip DD/RR.
    const bool do_sym = (j_base + j_count - 1) >= i_base;

    // Cooperatively load the j-tile into shared memory.
    const int jload = j_base + tid;
    if (tid < j_count) {
        sj_real_sin[tid] = d_real_sin[jload];
        sj_real_cos[tid] = d_real_cos[jload];
        sj_real_ra[tid] = d_real_rasc[jload];
        sj_rand_sin[tid] = d_rand_sin[jload];
        sj_rand_cos[tid] = d_rand_cos[jload];
        sj_rand_ra[tid] = d_rand_rasc[jload];
    }
    __syncthreads();

    if (valid_i) {
#pragma unroll 8
        for (int jj = 0; jj < j_count; ++jj) {
            const int j = j_base + jj;

            // DR: real_i vs rand_j (not symmetric, always counted).
            const float dr_expr = ri_sin * sj_rand_sin[jj] + ri_cos * sj_rand_cos[jj] * cosf(ri_ra - sj_rand_ra[jj]);
            hist_add(s_hist_DR, compute_histogram_index(dr_expr));

            // DD/RR: symmetric, only j >= i.
            if (do_sym && j >= i) {
                const unsigned int inc = 1U + static_cast<unsigned int>(j != i);

                const float dd_expr =
                    ri_sin * sj_real_sin[jj] + ri_cos * sj_real_cos[jj] * cosf(ri_ra - sj_real_ra[jj]);
                hist_add(s_hist_DD, compute_histogram_index(dd_expr), inc);

                const float rr_expr =
                    di_sin * sj_rand_sin[jj] + di_cos * sj_rand_cos[jj] * cosf(di_ra - sj_rand_ra[jj]);
                hist_add(s_hist_RR, compute_histogram_index(rr_expr), inc);
            }
        }
    }
    __syncthreads();

    for (int b = tid; b < num_bins; b += TILE) {
        if (s_hist_DR[b] > 0)
            atomicAdd(&d_histogram_DR[b], (unsigned long long int)s_hist_DR[b]);
        if (s_hist_DD[b] > 0)
            atomicAdd(&d_histogram_DD[b], (unsigned long long int)s_hist_DD[b]);
        if (s_hist_RR[b] > 0)
            atomicAdd(&d_histogram_RR[b], (unsigned long long int)s_hist_RR[b]);
    }
}

// Forward declarations
static int get_device();
static int parseargs_readinput(int argc, char *argv[]);

static inline bool is_ascii_whitespace(char c) {
    return (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f');
}

static inline void skip_ascii_whitespace(const char *&cursor, const char *end) {
    while (cursor < end && is_ascii_whitespace(*cursor))
        ++cursor;
}

static bool parse_int_fast(const char *&cursor, const char *end, int *value) {
    skip_ascii_whitespace(cursor, end);
    if (cursor >= end)
        return false;

    int sign = 1;
    if (*cursor == '+' || *cursor == '-') {
        sign = (*cursor == '-') ? -1 : 1;
        ++cursor;
    }

    if (cursor >= end || *cursor < '0' || *cursor > '9')
        return false;

    int parsed_value = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        parsed_value = parsed_value * 10 + (*cursor - '0');
        ++cursor;
    }

    *value = sign * parsed_value;
    return true;
}

static bool parse_float_fast(const char *&cursor, const char *end, float *value) {
    skip_ascii_whitespace(cursor, end);
    if (cursor >= end)
        return false;

    int sign = 1;
    if (*cursor == '+' || *cursor == '-') {
        sign = (*cursor == '-') ? -1 : 1;
        ++cursor;
    }

    double result = 0.0;
    bool has_digits = false;

    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        has_digits = true;
        result = result * 10.0 + (double)(*cursor - '0');
        ++cursor;
    }

    if (cursor < end && *cursor == '.') {
        ++cursor;
        double place = 0.1;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            has_digits = true;
            result += (double)(*cursor - '0') * place;
            place *= 0.1;
            ++cursor;
        }
    }

    if (!has_digits)
        return false;

    if (cursor < end && (*cursor == 'e' || *cursor == 'E')) {
        ++cursor;

        int exp_sign = 1;
        if (cursor < end && (*cursor == '+' || *cursor == '-')) {
            exp_sign = (*cursor == '-') ? -1 : 1;
            ++cursor;
        }

        if (cursor >= end || *cursor < '0' || *cursor > '9')
            return false;

        int exponent = 0;
        while (cursor < end && *cursor >= '0' && *cursor <= '9') {
            exponent = exponent * 10 + (*cursor - '0');
            ++cursor;
        }

        exponent *= exp_sign;
        if (exponent > 0) {
            while (exponent--)
                result *= 10.0;
        } else if (exponent < 0) {
            while (exponent++)
                result *= 0.1;
        }
    }

    *value = (float)(sign * result);
    return true;
}

static int read_catalog_mmap(const char *file_path, float *output_rasc, float *output_decl, int expected_galaxies,
                             float arcmin2rad) {
    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        printf("   ERROR: Cannot open data file %s\n", file_path);
        return (EXIT_FAILURE);
    }

    struct stat file_info;
    if (fstat(fd, &file_info) != 0 || file_info.st_size <= 0) {
        printf("   ERROR: Cannot stat data file %s\n", file_path);
        close(fd);
        return (EXIT_FAILURE);
    }

    void *mapped_data = mmap(NULL, (size_t)file_info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped_data == MAP_FAILED) {
        printf("   ERROR: Cannot memory-map data file %s\n", file_path);
        return (EXIT_FAILURE);
    }

    const char *cursor = (const char *)mapped_data;
    const char *end = cursor + file_info.st_size;

    int number_of_galaxies = 0;
    if (!parse_int_fast(cursor, end, &number_of_galaxies)) {
        printf("   ERROR: Cannot read galaxy count in %s\n", file_path);
        munmap(mapped_data, (size_t)file_info.st_size);
        return (EXIT_FAILURE);
    }

    if (number_of_galaxies < expected_galaxies) {
        printf("   ERROR: File %s has %d galaxies, expected at least %d\n", file_path, number_of_galaxies,
               expected_galaxies);
        munmap(mapped_data, (size_t)file_info.st_size);
        return (EXIT_FAILURE);
    }

    for (int i = 0; i < expected_galaxies; ++i) {
        float rasc = 0.0f;
        float decl = 0.0f;

        if (!parse_float_fast(cursor, end, &rasc) || !parse_float_fast(cursor, end, &decl)) {
            printf("   ERROR: Cannot read line %d in data file %s\n", i + 1, file_path);
            munmap(mapped_data, (size_t)file_info.st_size);
            return (EXIT_FAILURE);
        }

        output_rasc[i] = rasc * arcmin2rad;
        output_decl[i] = decl * arcmin2rad;
    }

    munmap(mapped_data, (size_t)file_info.st_size);
    return (EXIT_SUCCESS);
}

int main(int argc, char **argv) {
    printf("        Native CUDA Galaxy Correlation - RTX 5080 (sm_120)\n");
    long int histogramDRsum, histogramDDsum, histogramRRsum;
    double walltime;
    double inputReadTimeMs = 0.0;
    double kernelExecutionTimeMs = 0.0;
    double outputWriteTimeMs = 0.0;
    struct timeval _ttime;
    struct timeval inputStart, inputEnd;
    get_device();

    gettimeofday(&_ttime, NULL);
    walltime = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    // Allocate host memory
    real_rasc = (float *)calloc(100000L, sizeof(float));
    real_decl = (float *)calloc(100000L, sizeof(float));
    rand_rasc = (float *)calloc(100000L, sizeof(float));
    rand_decl = (float *)calloc(100000L, sizeof(float));
    real_sin = (float *)calloc(100000L, sizeof(float));
    real_cos = (float *)calloc(100000L, sizeof(float));
    rand_sin = (float *)calloc(100000L, sizeof(float));
    rand_cos = (float *)calloc(100000L, sizeof(float));
    CPUMemory += 8L * 100000L * sizeof(float);

    // Read input data from files
    gettimeofday(&inputStart, NULL);
    if (parseargs_readinput(argc, argv) != 0) {
        printf("   Program stopped.\n");
        return (EXIT_FAILURE);
    }
    // Precompute per-galaxy sin/cos(decl) once (removes redundant per-pair trig).
    for (long int k = 0; k < N; ++k) {
        real_sin[k] = sinf(real_decl[k]);
        real_cos[k] = cosf(real_decl[k]);
        rand_sin[k] = sinf(rand_decl[k]);
        rand_cos[k] = cosf(rand_decl[k]);
    }
    gettimeofday(&inputEnd, NULL);
    inputReadTimeMs = (inputEnd.tv_sec - inputStart.tv_sec) * 1000.0;
    inputReadTimeMs += (inputEnd.tv_usec - inputStart.tv_usec) / 1000.0;

    printf("   Input data read, now calculating histograms\n");

    FILE *outfile;

    if (argc != 4) {
        printf("Usage: ./galaxy_cuda data_100k_arcmin.txt flat_100k_arcmin.txt "
               "omega.out\n");
        return (EXIT_FAILURE);
    }

    histogram_DR = (long int *)calloc(totaldegrees * binsperdegree + 1ULL, sizeof(long int));
    histogram_DD = (long int *)calloc(totaldegrees * binsperdegree + 1ULL, sizeof(long int));
    histogram_RR = (long int *)calloc(totaldegrees * binsperdegree + 1ULL, sizeof(long int));
    CPUMemory += 3L * (totaldegrees * binsperdegree + 1L) * sizeof(long int);

    // Allocate GPU device memory (precomputed sin/cos + rasc per catalog)
    float *d_real_sin, *d_real_cos, *d_real_rasc;
    float *d_rand_sin, *d_rand_cos, *d_rand_rasc;

    struct timeval t1, t2;
    double gpuPhaseTimeMs;
    gettimeofday(&t1, NULL);

    int deviceCount = 0;
    CUDA_ERR_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("   \nRunning on %d GPU(s)\n", deviceCount);

    const int tiles_per_dim = (N + TILE - 1) / TILE;
    dim3 threadsInBlock(TILE, 1, 1);
    dim3 threadBlocks(tiles_per_dim, tiles_per_dim, 1);

    // Widen before multiplying: dim3 members are unsigned int, so evaluating the
    // whole product in 32 bits and casting afterwards overflows for large grids.
    const long int number_of_threads = (long int)threadsInBlock.x * threadsInBlock.y * threadsInBlock.z *
                                       threadBlocks.x * threadBlocks.y * threadBlocks.z;
    const long int threads_per_block = threadsInBlock.x;

    CUDA_ERR_CHECK(cudaSetDevice(0));

    printf("====================================================================\n");

    printf("    Using GPU 0 (RTX 5080, sm_120)\n");
    printf("    Tile: %d (%ld threads/block), grid %dx%d blocks\n", TILE, threads_per_block, threadBlocks.x,
           threadBlocks.y);

    // Allocate device memory
    size_t inputDataArrayBytes = N * sizeof(float);
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_real_sin, inputDataArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_real_cos, inputDataArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_real_rasc, inputDataArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_rand_sin, inputDataArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_rand_cos, inputDataArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_rand_rasc, inputDataArrayBytes));
    GPUMemory += 6L * inputDataArrayBytes;

    // Allocate histogram arrays
    unsigned long long int *d_histogram_DR, *d_histogram_DD, *d_histogram_RR;
    size_t histogramArrayBytes = (totaldegrees * binsperdegree + 1ULL) * sizeof(unsigned long long);

    CUDA_ERR_CHECK(cudaMalloc((void **)&d_histogram_DR, histogramArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_histogram_DD, histogramArrayBytes));
    CUDA_ERR_CHECK(cudaMalloc((void **)&d_histogram_RR, histogramArrayBytes));
    CUDA_ERR_CHECK(cudaMemset(d_histogram_DR, 0, histogramArrayBytes));
    CUDA_ERR_CHECK(cudaMemset(d_histogram_DD, 0, histogramArrayBytes));
    CUDA_ERR_CHECK(cudaMemset(d_histogram_RR, 0, histogramArrayBytes));
    GPUMemory += 3L * histogramArrayBytes;

    // Copy input data to device
    CUDA_ERR_CHECK(cudaMemcpy(d_real_sin, real_sin, inputDataArrayBytes, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(d_real_cos, real_cos, inputDataArrayBytes, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(d_real_rasc, real_rasc, inputDataArrayBytes, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(d_rand_sin, rand_sin, inputDataArrayBytes, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(d_rand_cos, rand_cos, inputDataArrayBytes, cudaMemcpyHostToDevice));
    CUDA_ERR_CHECK(cudaMemcpy(d_rand_rasc, rand_rasc, inputDataArrayBytes, cudaMemcpyHostToDevice));

    printf("    threadBlocks:\t\t{%d, %d, %d} blocks.\n    threadsInBlock:\t\t%d "
           "threads.\n",
           threadBlocks.x, threadBlocks.y, threadBlocks.z, threadsInBlock.x * threadsInBlock.y * threadsInBlock.z);
    printf("    Total number of threads:\t%ld\n", number_of_threads);

    // Launch kernel with padded shared histograms + j-tile coordinate buffers.
    size_t sharedMemSize = 3 * num_bins_padded * sizeof(unsigned int) + 6 * TILE * sizeof(float);
    printf("    Shared memory per block:\t%zu bytes (padded: %d bins, tile %d)\n", sharedMemSize, num_bins_padded,
           TILE);

    // CUDA-event kernel timing (finer than gettimeofday).
    cudaEvent_t kstart, kstop;
    CUDA_ERR_CHECK(cudaEventCreate(&kstart));
    CUDA_ERR_CHECK(cudaEventCreate(&kstop));
    CUDA_ERR_CHECK(cudaEventRecord(kstart));

    fill_histograms<<<threadBlocks, threadsInBlock, sharedMemSize, 0>>>(d_real_rasc, d_real_sin, d_real_cos,
                                                                        d_rand_rasc, d_rand_sin, d_rand_cos,
                                                                        d_histogram_DR, d_histogram_DD, d_histogram_RR);

    CUDA_ERR_CHECK(cudaGetLastError());
    CUDA_ERR_CHECK(cudaEventRecord(kstop));
    CUDA_ERR_CHECK(cudaEventSynchronize(kstop));
    float kernelMs = 0.0f;
    CUDA_ERR_CHECK(cudaEventElapsedTime(&kernelMs, kstart, kstop));
    kernelExecutionTimeMs = (double)kernelMs;
    CUDA_ERR_CHECK(cudaEventDestroy(kstart));
    CUDA_ERR_CHECK(cudaEventDestroy(kstop));

    // Copy results back to host
    CUDA_ERR_CHECK(cudaMemcpy(histogram_DR, d_histogram_DR, histogramArrayBytes, cudaMemcpyDeviceToHost));
    CUDA_ERR_CHECK(cudaMemcpy(histogram_DD, d_histogram_DD, histogramArrayBytes, cudaMemcpyDeviceToHost));
    CUDA_ERR_CHECK(cudaMemcpy(histogram_RR, d_histogram_RR, histogramArrayBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_ERR_CHECK(cudaFree(d_real_rasc));
    CUDA_ERR_CHECK(cudaFree(d_real_sin));
    CUDA_ERR_CHECK(cudaFree(d_real_cos));
    CUDA_ERR_CHECK(cudaFree(d_rand_rasc));
    CUDA_ERR_CHECK(cudaFree(d_rand_sin));
    CUDA_ERR_CHECK(cudaFree(d_rand_cos));
    CUDA_ERR_CHECK(cudaFree(d_histogram_DR));
    CUDA_ERR_CHECK(cudaFree(d_histogram_DD));
    CUDA_ERR_CHECK(cudaFree(d_histogram_RR));

    gettimeofday(&t2, NULL);
    gpuPhaseTimeMs = (t2.tv_sec - t1.tv_sec) * 1000.0;
    gpuPhaseTimeMs += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Kernel execution time: %f ms.\n", kernelExecutionTimeMs);

    // Free host memory
    free(real_rasc);
    free(real_decl);
    free(rand_rasc);
    free(rand_decl);
    free(real_sin);
    free(real_cos);
    free(rand_sin);
    free(rand_cos);

    // Verify histogram sums
    histogramDRsum = 0L;
    for (int i = 0; i < binsperdegree * totaldegrees; ++i)
        histogramDRsum += histogram_DR[i];
    printf("results:\n");
    printf("   DR histogram sum = %ld\n", histogramDRsum);

    if (histogramDRsum != 10000000000L) {
        printf("   Incorrect histogram sum, exiting.. histogramDRsum: %ld\t\n   "
               "percentage of target: %15f\n",
               histogramDRsum, ((float)histogramDRsum / (float)(N * N)));
        return (EXIT_FAILURE);
    }

    histogramDDsum = 0L;
    for (int i = 0; i < binsperdegree * totaldegrees; ++i)
        histogramDDsum += histogram_DD[i];
    printf("   DD histogram sum = %ld\n", histogramDDsum);
    if (histogramDDsum != 10000000000L) {
        printf("   Incorrect histogram sum, exiting.. histogramDDsum: %ld\n", histogramDDsum);
        return (EXIT_FAILURE);
    }

    histogramRRsum = 0L;
    for (int i = 0; i < binsperdegree * totaldegrees; ++i)
        histogramRRsum += histogram_RR[i];
    printf("   RR histogram sum = %ld\n", histogramRRsum);
    if (histogramRRsum != 10000000000L) {
        printf("   Incorrect histogram sum, exiting..histogramRRsum: %ld\n", histogramRRsum);
        return (EXIT_FAILURE);
    }

    struct timeval outputStart, outputEnd;
    gettimeofday(&outputStart, NULL);

    // Write results to output file
    outfile = fopen(argv[3], "w");
    if (outfile == NULL) {
        printf("Cannot open output file %s\n", argv[3]);
        return (-1);
    }

    fprintf(outfile, "bin start\t\tomega\t        hist_DD\t        hist_DR\t     "
                     "   hist_RR\n");

    for (int i = 0; i < binsperdegree * totaldegrees; ++i) {
        if (histogram_RR[i] > 0) {
            float omega = (histogram_DD[i] - 2 * histogram_DR[i] + histogram_RR[i]) / ((float)(histogram_RR[i]));
            fprintf(outfile, "%6.3f\t%15f\t%15ld\t%15ld\t%15ld\n", ((float)i) / binsperdegree, omega, histogram_DD[i],
                    histogram_DR[i], histogram_RR[i]);
            if (i < 5)
                printf("   %6.4f", omega);
        } else {
            if (i < 5)
                printf("         ");
        }
    }
    printf("\n");

    fclose(outfile);

    gettimeofday(&outputEnd, NULL);
    outputWriteTimeMs = (outputEnd.tv_sec - outputStart.tv_sec) * 1000.0;
    outputWriteTimeMs += (outputEnd.tv_usec - outputStart.tv_usec) / 1000.0;

    // Free host memory
    free(histogram_DR);
    free(histogram_DD);
    free(histogram_RR);

    printf("   Results written to file %s\n", argv[3]);
    printf("   CPU memory allocated  = %.2lf MB\n", CPUMemory / 1000000.0);
    printf("   GPU memory allocated  = %.2lf MB\n", GPUMemory / 1000000.0);
    printf("   Timing breakdown      = input %.3f ms | kernel %.3f ms | output "
           "%.3f ms\n",
           inputReadTimeMs, kernelExecutionTimeMs, outputWriteTimeMs);
    printf("   GPU phase time        = %.3f ms\n", gpuPhaseTimeMs);

    gettimeofday(&_ttime, NULL);
    walltime = (double)(_ttime.tv_sec) + (double)(_ttime.tv_usec / 1000000.0) - walltime;

    printf("   Total wall clock time = %.3lf s\n", walltime);

    return (EXIT_SUCCESS);
}

static int get_device() {
    int deviceCount;
    CUDA_ERR_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("   Found %d CUDA devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128)
        return (EXIT_FAILURE);

    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        CUDA_ERR_CHECK(cudaGetDeviceProperties(&deviceProp, device));
        printf("      Device %s | device %d\n", deviceProp.name, device);
        printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
        int clockRateKHz = 0;
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device);
        printf("         clockRate                    =    %8.2lf MHz\n", clockRateKHz / 1000.0);
        printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
    }

    CUDA_ERR_CHECK(cudaSetDevice(0));
    CUDA_ERR_CHECK(cudaGetDevice(&device));
    if (device != 0)
        printf("   Unable to set device 0, using %d instead", device);
    else
        printf("   Using CUDA device %d\n\n", device);

    return (EXIT_SUCCESS);
}

static int parseargs_readinput(int argc, char *argv[]) {
    FILE *out_file;
    constexpr int expected_galaxies = 100000;
    const float arcmin2rad = 1.0f / 60.0f / 180.0f * PI;

    if (argc != 4) {
        printf("   Usage: galaxy real_data random_data output_file\n");
        return (EXIT_FAILURE);
    }

    printf("   Running galaxy_cuda %s %s %s\n", argv[1], argv[2], argv[3]);

    if (read_catalog_mmap(argv[1], real_rasc, real_decl, expected_galaxies, arcmin2rad) != 0) {
        return (EXIT_FAILURE);
    }
    printf("   Successfully read %d lines from %s\n", expected_galaxies, argv[1]);

    if (read_catalog_mmap(argv[2], rand_rasc, rand_decl, expected_galaxies, arcmin2rad) != 0) {
        return (EXIT_FAILURE);
    }
    printf("   Successfully read %d lines from %s\n", expected_galaxies, argv[2]);

    out_file = fopen(argv[3], "w");
    if (out_file == NULL) {
        printf("   ERROR: Cannot open output file %s\n", argv[3]);
        return (EXIT_FAILURE);
    }
    fclose(out_file);

    return (EXIT_SUCCESS);
}
