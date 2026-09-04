// Native HIP implementation tuned for AMD RDNA 2 (RX 6900 XT, gfx1030).
// Same two-point angular correlation as galaxy_cuda.cu, structured for RDNA 2:
//   - one thread per pair over a 2D grid, wave64 (-mwavefrontsize64)
//   - LDS histograms padded 1440 -> 1536 to avoid bank conflicts
//   - DD/RR symmetry: the j >= i half only, incrementing by 1 or 2
// Unlike the CUDA path this calls the real acosf and computes sin/cos(decl)
// per pair, so the two backends do not produce bit-identical output;
// results/omega.out is the CUDA output. See the Makefile (`make hip`).
//
// The AMD figures in README.md are historical records from the RX 6900 XT and
// have not been re-measured; they are not comparable with the RTX 5080 run.
//
// The host side is duplicated from galaxy_cuda.cu rather than shared through
// a header - see the note there. Each backend is one self-contained
// translation unit.
#include <fcntl.h>
#include <hip/hip_runtime.h>
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
static constexpr long int N = 100000L;
static long *histogram_DR;
static long *histogram_DD;
static long *histogram_RR;
static constexpr float PI = 3.14159265358979323846f;
static long int CPUMemory = 0L;
static long int GPUMemory = 0L;
static constexpr int totaldegrees = 360;
static constexpr int binsperdegree = 4;

// RDNA 2 optimization: Padded bins to avoid LDS bank conflicts
// 1440 bins padded to 1536 (power of 2 aligned for 32 banks)
const int num_bins = binsperdegree * totaldegrees;
const int num_bins_padded = 1536;

// Configurable block sizes for RDNA 2 occupancy tuning
// Test: 16x32 (512 threads, 8 wave64), 32x16, or 16x16 (256 threads, 4 wave64)
#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 16
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 32
#endif

#define HIP_ERR_CHECK(ans)                                                                                             \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
static inline void gpuAssert(hipError_t code, const char *file, int line) {
    if (code != hipSuccess) {
        fprintf(stderr, "   GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        exit(code);
    }
}

__device__ __forceinline__ int compute_histogram_index(float expr) {
    constexpr float angle = 57.29577951308232f;
    expr = fminf(fmaxf(expr, -1.0f), 1.0f);

    int histogram_index = int(acosf(expr) * angle * binsperdegree);
    return (histogram_index < 0) ? 0 : ((histogram_index >= num_bins) ? num_bins - 1 : histogram_index);
}

// Optimized kernel for RDNA 2 with DD/RR symmetry and reduced shared-memory
// pressure
__global__ void fill_histograms(const float *__restrict__ d_real_rasc, const float *__restrict__ d_real_decl,
                                const float *__restrict__ d_rand_rasc, const float *__restrict__ d_rand_decl,
                                unsigned long long int *d_histogram_DR, unsigned long long int *d_histogram_DD,
                                unsigned long long int *d_histogram_RR) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int local_threads = blockDim.x * blockDim.y;

    extern __shared__ unsigned int s_hist[];
    unsigned int *s_hist_DR = s_hist;
    unsigned int *s_hist_DD = s_hist_DR + num_bins_padded;
    unsigned int *s_hist_RR = s_hist_DD + num_bins_padded;

    for (int b = local_tid; b < num_bins_padded; b += local_threads) {
        s_hist_DR[b] = 0;
        s_hist_DD[b] = 0;
        s_hist_RR[b] = 0;
    }
    __syncthreads();

    if (i < N && j < N) {
        const float real_decl_i = d_real_decl[i];
        const float real_decl_j = d_real_decl[j];
        const float rand_decl_i = d_rand_decl[i];
        const float rand_decl_j = d_rand_decl[j];
        const float real_rasc_i = d_real_rasc[i];
        const float real_rasc_j = d_real_rasc[j];
        const float rand_rasc_i = d_rand_rasc[i];
        const float rand_rasc_j = d_rand_rasc[j];

        const float sin_real_i = sin(real_decl_i);
        const float cos_real_i = cos(real_decl_i);
        const float sin_real_j = sin(real_decl_j);
        const float cos_real_j = cos(real_decl_j);
        const float sin_rand_i = sin(rand_decl_i);
        const float cos_rand_i = cos(rand_decl_i);
        const float sin_rand_j = sin(rand_decl_j);
        const float cos_rand_j = cos(rand_decl_j);

        // The LDS atomics below need no explicit ballot aggregation: RDNA 2
        // resolves same-address atomics within a wave in hardware.
        // -mwavefrontsize64 is a codegen flag, not an aggregation strategy.
        const float dr_expr = sin_real_i * sin_rand_j + cos_real_i * cos_rand_j * cos(real_rasc_i - rand_rasc_j);
        atomicAdd(&s_hist_DR[compute_histogram_index(dr_expr)], 1U);

        if (j >= i) {
            const unsigned int symmetric_increment = 1U + static_cast<unsigned int>(j != i);

            const float dd_expr = sin_real_i * sin_real_j + cos_real_i * cos_real_j * cos(real_rasc_i - real_rasc_j);
            atomicAdd(&s_hist_DD[compute_histogram_index(dd_expr)], symmetric_increment);

            const float rr_expr = sin_rand_i * sin_rand_j + cos_rand_i * cos_rand_j * cos(rand_rasc_i - rand_rasc_j);
            atomicAdd(&s_hist_RR[compute_histogram_index(rr_expr)], symmetric_increment);
        }
    }
    __syncthreads();

    for (int b = local_tid; b < num_bins; b += local_threads) {
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

// Sums hist[0..num_bins) and checks it against target. Every histogram must sum
// to N*N; a variant that fails this is a failed variant, not a fast one.
static int verify_histogram_sum(const long *hist, int num_bins, long target, const char *label) {
    long sum = 0L;
    for (int i = 0; i < num_bins; ++i)
        sum += hist[i];
    printf("   %s histogram sum = %ld\n", label, sum);
    if (sum != target) {
        printf("   Incorrect %s histogram sum, exiting.. expected %ld, got %ld (%.6f of target)\n", label, target, sum,
               (double)sum / (double)target);
        return (EXIT_FAILURE);
    }
    return (EXIT_SUCCESS);
}

static inline void skip_ascii_whitespace(const char *&cursor, const char *end) {
    while (cursor < end && (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t' ||
                            *cursor == '\v' || *cursor == '\f'))
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

// Accepts [+-]?digits[.digits] - the fixed-point form the catalogs use (see
// data/README.md). Exponent notation is not accepted: neither catalog contains
// any, and an unparsed 'e' makes the *next* field fail, so read_catalog_mmap
// reports the line and exits rather than silently truncating a value.
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
    printf("        Native HIP Galaxy Correlation - Optimized for AMD RDNA 2\n");
    double walltime;
    double inputReadTimeMs = 0.0;
    double kernelExecutionTimeMs = 0.0;
    double outputWriteTimeMs = 0.0;
    struct timeval _ttime;
    struct timeval inputStart, inputEnd;
    struct timeval kernelStart, kernelEnd;
    struct timeval outputStart, outputEnd;
    get_device();

    gettimeofday(&_ttime, NULL);
    walltime = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    // Allocate host memory
    real_rasc = (float *)calloc(100000L, sizeof(float));
    real_decl = (float *)calloc(100000L, sizeof(float));
    rand_rasc = (float *)calloc(100000L, sizeof(float));
    rand_decl = (float *)calloc(100000L, sizeof(float));
    CPUMemory += 4L * 100000L * sizeof(float);

    // Read input data from files
    gettimeofday(&inputStart, NULL);
    if (parseargs_readinput(argc, argv) != 0) {
        printf("   Program stopped.\n");
        return (EXIT_FAILURE);
    }
    gettimeofday(&inputEnd, NULL);
    inputReadTimeMs = (inputEnd.tv_sec - inputStart.tv_sec) * 1000.0;
    inputReadTimeMs += (inputEnd.tv_usec - inputStart.tv_usec) / 1000.0;

    printf("   Input data read, now calculating histograms\n");

    FILE *outfile;

    histogram_DR = (long int *)calloc(totaldegrees * binsperdegree + 1ULL, sizeof(long int));
    histogram_DD = (long int *)calloc(totaldegrees * binsperdegree + 1ULL, sizeof(long int));
    histogram_RR = (long int *)calloc(totaldegrees * binsperdegree + 1ULL, sizeof(long int));
    CPUMemory += 3L * (totaldegrees * binsperdegree + 1L) * sizeof(long int);

    // Allocate GPU device memory
    float *d_real_decl, *d_real_rasc, *d_rand_decl, *d_rand_rasc;

    struct timeval t1, t2;
    double gpuPhaseTimeMs;
    gettimeofday(&t1, NULL);

    // RDNA 2 optimized block configuration
    dim3 threadsInBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 threadBlocks((N + threadsInBlock.x - 1) / threadsInBlock.x, (N + threadsInBlock.y - 1) / threadsInBlock.y);

    // Widen before multiplying: dim3 members are unsigned int, so evaluating the
    // whole product in 32 bits and casting afterwards overflows. The 32x32 RDNA 2
    // configuration is 3125*3125*1024 = 1e10 threads, which wrapped to
    // 1410065408.
    const long int number_of_threads = (long int)threadsInBlock.x * threadsInBlock.y * threadsInBlock.z *
                                       threadBlocks.x * threadBlocks.y * threadBlocks.z;
    const long int threads_per_block = (long int)threadsInBlock.x * threadsInBlock.y * threadsInBlock.z;

    printf("====================================================================\n");

    printf("    Using GPU 0 with RDNA 2 optimizations\n");
    printf("    Block size: %dx%d (%ld threads/block)\n", BLOCK_SIZE_X, BLOCK_SIZE_Y, threads_per_block);
    printf("    Wavefront size: 64 (wave64)\n");

    // Allocate device memory
    size_t inputDataArrayBytes = N * sizeof(float);
    HIP_ERR_CHECK(hipMalloc((void **)&d_real_decl, inputDataArrayBytes));
    HIP_ERR_CHECK(hipMalloc((void **)&d_real_rasc, inputDataArrayBytes));
    HIP_ERR_CHECK(hipMalloc((void **)&d_rand_decl, inputDataArrayBytes));
    HIP_ERR_CHECK(hipMalloc((void **)&d_rand_rasc, inputDataArrayBytes));
    GPUMemory += 4L * inputDataArrayBytes;

    // Allocate histogram arrays
    unsigned long long int *d_histogram_DR, *d_histogram_DD, *d_histogram_RR;
    size_t histogramArrayBytes = (totaldegrees * binsperdegree + 1ULL) * sizeof(unsigned long long);

    HIP_ERR_CHECK(hipMalloc((void **)&d_histogram_DR, histogramArrayBytes));
    HIP_ERR_CHECK(hipMalloc((void **)&d_histogram_DD, histogramArrayBytes));
    HIP_ERR_CHECK(hipMalloc((void **)&d_histogram_RR, histogramArrayBytes));
    HIP_ERR_CHECK(hipMemset(d_histogram_DR, 0, histogramArrayBytes));
    HIP_ERR_CHECK(hipMemset(d_histogram_DD, 0, histogramArrayBytes));
    HIP_ERR_CHECK(hipMemset(d_histogram_RR, 0, histogramArrayBytes));
    GPUMemory += 3L * histogramArrayBytes;

    // Copy input data to device
    HIP_ERR_CHECK(hipMemcpy(d_real_decl, real_decl, inputDataArrayBytes, hipMemcpyHostToDevice));
    HIP_ERR_CHECK(hipMemcpy(d_real_rasc, real_rasc, inputDataArrayBytes, hipMemcpyHostToDevice));
    HIP_ERR_CHECK(hipMemcpy(d_rand_decl, rand_decl, inputDataArrayBytes, hipMemcpyHostToDevice));
    HIP_ERR_CHECK(hipMemcpy(d_rand_rasc, rand_rasc, inputDataArrayBytes, hipMemcpyHostToDevice));

    printf("    threadBlocks:\t\t{%d, %d, %d} blocks.\n    threadsInBlock:\t\t%d "
           "threads.\n",
           threadBlocks.x, threadBlocks.y, threadBlocks.z, threadsInBlock.x * threadsInBlock.y * threadsInBlock.z);
    printf("    Total number of threads:\t%ld\n", number_of_threads);

    // Launch kernel with padded shared memory for LDS optimization
    size_t sharedMemSize = 3 * num_bins_padded * sizeof(unsigned int);
    printf("    Shared memory per block:\t%zu bytes (padded: %d bins)\n", sharedMemSize, num_bins_padded);

    gettimeofday(&kernelStart, NULL);
    hipLaunchKernelGGL(fill_histograms, threadBlocks, threadsInBlock, sharedMemSize, 0, d_real_rasc, d_real_decl,
                       d_rand_rasc, d_rand_decl, d_histogram_DR, d_histogram_DD, d_histogram_RR);

    HIP_ERR_CHECK(hipGetLastError());
    HIP_ERR_CHECK(hipDeviceSynchronize());
    gettimeofday(&kernelEnd, NULL);
    kernelExecutionTimeMs = (kernelEnd.tv_sec - kernelStart.tv_sec) * 1000.0;
    kernelExecutionTimeMs += (kernelEnd.tv_usec - kernelStart.tv_usec) / 1000.0;

    // Copy results back to host
    HIP_ERR_CHECK(hipMemcpy(histogram_DR, d_histogram_DR, histogramArrayBytes, hipMemcpyDeviceToHost));
    HIP_ERR_CHECK(hipMemcpy(histogram_DD, d_histogram_DD, histogramArrayBytes, hipMemcpyDeviceToHost));
    HIP_ERR_CHECK(hipMemcpy(histogram_RR, d_histogram_RR, histogramArrayBytes, hipMemcpyDeviceToHost));

    // Free device memory
    HIP_ERR_CHECK(hipFree(d_real_rasc));
    HIP_ERR_CHECK(hipFree(d_real_decl));
    HIP_ERR_CHECK(hipFree(d_rand_rasc));
    HIP_ERR_CHECK(hipFree(d_rand_decl));
    HIP_ERR_CHECK(hipFree(d_histogram_DR));
    HIP_ERR_CHECK(hipFree(d_histogram_DD));
    HIP_ERR_CHECK(hipFree(d_histogram_RR));

    gettimeofday(&t2, NULL);
    gpuPhaseTimeMs = (t2.tv_sec - t1.tv_sec) * 1000.0;
    gpuPhaseTimeMs += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Kernel execution time: %f ms.\n", kernelExecutionTimeMs);

    // Free host memory
    free(real_rasc);
    free(real_decl);
    free(rand_rasc);
    free(rand_decl);

    // Verify histogram sums
    printf("results:\n");
    if (verify_histogram_sum(histogram_DR, num_bins, N * N, "DR") != EXIT_SUCCESS)
        return (EXIT_FAILURE);
    if (verify_histogram_sum(histogram_DD, num_bins, N * N, "DD") != EXIT_SUCCESS)
        return (EXIT_FAILURE);
    if (verify_histogram_sum(histogram_RR, num_bins, N * N, "RR") != EXIT_SUCCESS)
        return (EXIT_FAILURE);

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
    HIP_ERR_CHECK(hipGetDeviceCount(&deviceCount));

    printf("   Found %d HIP devices\n", deviceCount);
    if (deviceCount < 0 || deviceCount > 128)
        return (EXIT_FAILURE);

    int device;
    for (device = 0; device < deviceCount; ++device) {
        hipDeviceProp_t deviceProp;
        HIP_ERR_CHECK(hipGetDeviceProperties(&deviceProp, device));
        printf("      Device %s | device %d\n", deviceProp.name, device);
        printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem / 1000000000.0);
        printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
        printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
        printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
        printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
        printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
        printf("         clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate / 1000.0);
        printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
    }

    HIP_ERR_CHECK(hipSetDevice(0));
    HIP_ERR_CHECK(hipGetDevice(&device));
    if (device != 0)
        printf("   Unable to set device 0, using %d instead", device);
    else
        printf("   Using HIP device %d\n\n", device);

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

    printf("   Running galaxy_hip %s %s %s\n", argv[1], argv[2], argv[3]);

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
