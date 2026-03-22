#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

typedef struct {
    int32_t x;
    int32_t z;
    uint32_t count;
} slimy_cuda_result;

typedef struct {
    int device_index;
    uint32_t max_width;
    uint32_t max_height;
    uint32_t max_out_capacity;
    uint32_t map_width;
    uint32_t map_height;
    slimy_cuda_result* d_results;
    uint32_t* d_count;
    int* d_overflow;
    uint8_t* d_slime_map;
    slimy_cuda_result* h_results;
    uint32_t* h_count;
    int* h_overflow;
    cudaStream_t stream;
} slimy_cuda_context;

static __device__ __forceinline__ int32_t mul_wrap_i32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a * (uint32_t)b);
}

static __device__ __forceinline__ int64_t chunk_seed(int64_t world_seed, int32_t x, int32_t z) {
    int32_t t1 = mul_wrap_i32(mul_wrap_i32(x, x), 4987142);
    int32_t t2 = mul_wrap_i32(x, 5947611);
    // Match Java/Zig semantics:
    // (z *% z) is 32-bit wrapping, then multiplied in 64-bit by 4392871.
    int64_t t3 = (int64_t)mul_wrap_i32(z, z) * 4392871LL;
    int32_t t4 = mul_wrap_i32(z, 389711);

    uint64_t u = (uint64_t)world_seed;
    u += (uint64_t)(int64_t)t1;
    u += (uint64_t)(int64_t)t2;
    u += (uint64_t)t3;
    u += (uint64_t)(int64_t)t4;
    u ^= (uint64_t)(int64_t)987234911;
    return (int64_t)u;
}

static __device__ __forceinline__ int32_t java_next(int64_t* seed, int32_t bits) {
    const uint64_t multiplier = 0x5deece66dULL;
    const uint64_t addend = 0xbULL;
    const uint64_t mask = (1ULL << 48) - 1;

    uint64_t s = (uint64_t)(*seed);
    s = (s * multiplier + addend) & mask;
    *seed = (int64_t)s;
    return (int32_t)(s >> (48 - bits));
}

static __device__ __forceinline__ int32_t java_next_int10(int64_t seed_input) {
    const uint64_t multiplier = 0x5deece66dULL;
    const uint64_t mask = (1ULL << 48) - 1;

    int64_t seed = (int64_t)(((uint64_t)seed_input ^ multiplier) & mask);

    while (true) {
        int32_t bits = java_next(&seed, 31);
        int32_t val = bits % 10;
        // Match Java's 32-bit signed overflow behavior explicitly:
        // if ((bits - val + (bound - 1)) >= 0) return val;
        // C/C++ signed overflow is UB, so do the arithmetic in uint32_t.
        const uint32_t wrapped = (uint32_t)bits - (uint32_t)val + 9u;
        if ((wrapped & 0x80000000u) == 0) {
            return val;
        }
    }
}

static __device__ __forceinline__ bool is_slime(int64_t world_seed, int32_t x, int32_t z) {
    return java_next_int10(chunk_seed(world_seed, x, z)) == 0;
}

static __global__ void slime_map_kernel(
    int64_t world_seed,
    int32_t map_x0,
    int32_t map_z0,
    uint32_t map_width,
    uint32_t map_height,
    uint8_t* slime_map
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)map_width * (uint64_t)map_height;
    if (idx >= total) return;

    int32_t rel_x = (int32_t)(idx % map_width);
    int32_t rel_z = (int32_t)(idx / map_width);
    int32_t x = map_x0 + rel_x;
    int32_t z = map_z0 + rel_z;
    slime_map[idx] = is_slime(world_seed, x, z) ? 1 : 0;
}

static __global__ void search_kernel(
    const uint8_t* slime_map,
    uint32_t map_width,
    uint8_t threshold,
    int32_t x0,
    int32_t z0,
    uint32_t width,
    uint32_t height,
    slimy_cuda_result* out_results,
    uint32_t out_capacity,
    uint32_t* out_count,
    int* overflow
) {
    const uint32_t bx = blockIdx.x * blockDim.x;
    const uint32_t bz = blockIdx.y * blockDim.y;
    const uint32_t tx = threadIdx.x;
    const uint32_t tz = threadIdx.y;
    const uint32_t rel_x = bx + tx;
    const uint32_t rel_z = bz + tz;

    __shared__ uint8_t tile[32][32];
    const uint32_t tile_x0 = bx;
    const uint32_t tile_z0 = bz;

    for (uint32_t lz = tz; lz < 32; lz += blockDim.y) {
        for (uint32_t lx = tx; lx < 32; lx += blockDim.x) {
            const uint32_t map_x = tile_x0 + lx;
            const uint32_t map_z = tile_z0 + lz;
            if (map_x < map_width && map_z < height + 16) {
                tile[lz][lx] = slime_map[(uint64_t)map_z * map_width + map_x];
            } else {
                tile[lz][lx] = 0;
            }
        }
    }
    __syncthreads();

    if (rel_x >= width || rel_z >= height) return;

    const int32_t x = x0 + (int32_t)rel_x;
    const int32_t z = z0 + (int32_t)rel_z;

    uint32_t count = 0;
    for (int32_t dx = -8; dx <= 8; ++dx) {
        for (int32_t dz = -8; dz <= 8; ++dz) {
            int32_t d2 = dx * dx + dz * dz;
            if (!(d2 > 1 && d2 <= 64)) continue;
            const int32_t lx = (int32_t)tx + dx + 8;
            const int32_t lz = (int32_t)tz + dz + 8;
            uint8_t slime = tile[(uint32_t)lz][(uint32_t)lx];
            if (slime != 0) {
                ++count;
            }
        }
    }

    if (count >= threshold) {
        uint32_t out_idx = atomicAdd(out_count, 1);
        if (out_idx < out_capacity) {
            out_results[out_idx].x = x;
            out_results[out_idx].z = z;
            out_results[out_idx].count = count;
        } else {
            atomicExch(overflow, 1);
        }
    }
}

int slimy_cuda_get_device_count(void) {
    int count = 0;
    cudaError_t st = cudaGetDeviceCount(&count);
    if (st != cudaSuccess) return -1;
    return count;
}

int slimy_cuda_get_device_score(int device_index, float* score, uint64_t* free_mem, uint64_t* total_mem) {
    cudaError_t st = cudaSetDevice(device_index);
    if (st != cudaSuccess) return 1;

    cudaDeviceProp prop;
    st = cudaGetDeviceProperties(&prop, device_index);
    if (st != cudaSuccess) return 1;

    size_t free_b = 0;
    size_t total_b = 0;
    st = cudaMemGetInfo(&free_b, &total_b);
    if (st != cudaSuccess) return 1;

    *free_mem = (uint64_t)free_b;
    *total_mem = (uint64_t)total_b;

    float perf = (float)prop.multiProcessorCount * (float)prop.clockRate;
    float cap = (float)(prop.major * 10 + prop.minor);
    float mem = (float)((double)free_b / (double)(1ull << 30));
    *score = perf * (1.0f + cap * 0.02f) * (1.0f + mem * 0.02f);

    return 0;
}

int slimy_cuda_context_init(
    int device_index,
    uint32_t max_width,
    uint32_t max_height,
    uint32_t max_out_capacity,
    slimy_cuda_context** out_ctx
) {
    if (!out_ctx) return 1;
    *out_ctx = nullptr;

    cudaError_t st = cudaSetDevice(device_index);
    if (st != cudaSuccess) return 1;

    slimy_cuda_context* ctx = (slimy_cuda_context*)malloc(sizeof(slimy_cuda_context));
    if (!ctx) return 1;

    ctx->device_index = device_index;
    ctx->max_width = max_width;
    ctx->max_height = max_height;
    ctx->max_out_capacity = max_out_capacity;
    ctx->map_width = max_width + 16;
    ctx->map_height = max_height + 16;
    ctx->d_results = nullptr;
    ctx->d_count = nullptr;
    ctx->d_overflow = nullptr;
    ctx->d_slime_map = nullptr;
    ctx->h_results = nullptr;
    ctx->h_count = nullptr;
    ctx->h_overflow = nullptr;
    ctx->stream = nullptr;

    st = cudaMalloc((void**)&ctx->d_results, (size_t)ctx->max_out_capacity * sizeof(slimy_cuda_result));
    if (st != cudaSuccess) goto fail;
    st = cudaMalloc((void**)&ctx->d_count, sizeof(uint32_t));
    if (st != cudaSuccess) goto fail;
    st = cudaMalloc((void**)&ctx->d_overflow, sizeof(int));
    if (st != cudaSuccess) goto fail;
    st = cudaMalloc((void**)&ctx->d_slime_map, (size_t)ctx->map_width * (size_t)ctx->map_height * sizeof(uint8_t));
    if (st != cudaSuccess) goto fail;
    st = cudaMallocHost((void**)&ctx->h_results, (size_t)ctx->max_out_capacity * sizeof(slimy_cuda_result));
    if (st != cudaSuccess) goto fail;
    st = cudaMallocHost((void**)&ctx->h_count, sizeof(uint32_t));
    if (st != cudaSuccess) goto fail;
    st = cudaMallocHost((void**)&ctx->h_overflow, sizeof(int));
    if (st != cudaSuccess) goto fail;
    st = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (st != cudaSuccess) goto fail;

    *out_ctx = ctx;
    return 0;

fail:
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    if (ctx->h_overflow) cudaFreeHost(ctx->h_overflow);
    if (ctx->h_count) cudaFreeHost(ctx->h_count);
    if (ctx->h_results) cudaFreeHost(ctx->h_results);
    if (ctx->d_slime_map) cudaFree(ctx->d_slime_map);
    if (ctx->d_overflow) cudaFree(ctx->d_overflow);
    if (ctx->d_count) cudaFree(ctx->d_count);
    if (ctx->d_results) cudaFree(ctx->d_results);
    free(ctx);
    return 1;
}

int slimy_cuda_context_deinit(slimy_cuda_context* ctx) {
    if (!ctx) return 0;

    cudaSetDevice(ctx->device_index);
    cudaStreamDestroy(ctx->stream);
    cudaFreeHost(ctx->h_overflow);
    cudaFreeHost(ctx->h_count);
    cudaFreeHost(ctx->h_results);
    cudaFree(ctx->d_slime_map);
    cudaFree(ctx->d_overflow);
    cudaFree(ctx->d_count);
    cudaFree(ctx->d_results);
    free(ctx);
    return 0;
}

int slimy_cuda_search_batch_ctx(
    slimy_cuda_context* ctx,
    int64_t world_seed,
    uint8_t threshold,
    int32_t x0,
    int32_t z0,
    uint32_t width,
    uint32_t height,
    slimy_cuda_result* out_results,
    uint32_t out_capacity,
    uint32_t* out_count
) {
    if (!ctx || !out_results || !out_count) return 1;
    if (width > ctx->max_width || height > ctx->max_height || out_capacity > ctx->max_out_capacity) return 3;

    cudaError_t st = cudaSetDevice(ctx->device_index);
    if (st != cudaSuccess) return 1;

    st = cudaMemsetAsync(ctx->d_count, 0, sizeof(uint32_t), ctx->stream);
    if (st != cudaSuccess) return 1;
    st = cudaMemsetAsync(ctx->d_overflow, 0, sizeof(int), ctx->stream);
    if (st != cudaSuccess) return 1;

    const uint64_t total = (uint64_t)width * (uint64_t)height;
    if (total > 0 && width > 0 && height > 0) {
        const uint32_t map_threads = 256;
        const uint64_t map_total = (uint64_t)(width + 16) * (uint64_t)(height + 16);
        const uint32_t map_blocks = (uint32_t)((map_total + map_threads - 1) / map_threads);
        slime_map_kernel<<<map_blocks, map_threads, 0, ctx->stream>>>(
            world_seed,
            x0 - 8,
            z0 - 8,
            width + 16,
            height + 16,
            ctx->d_slime_map
        );
        st = cudaGetLastError();
        if (st != cudaSuccess) return 1;

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        search_kernel<<<grid, block, 0, ctx->stream>>>(
            ctx->d_slime_map,
            width + 16,
            threshold,
            x0,
            z0,
            width,
            height,
            ctx->d_results,
            out_capacity,
            ctx->d_count,
            ctx->d_overflow
        );
        st = cudaGetLastError();
        if (st != cudaSuccess) return 1;
    }

    st = cudaMemcpyAsync(ctx->h_count, ctx->d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, ctx->stream);
    if (st != cudaSuccess) return 1;
    st = cudaMemcpyAsync(ctx->h_overflow, ctx->d_overflow, sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
    if (st != cudaSuccess) return 1;
    st = cudaStreamSynchronize(ctx->stream);
    if (st != cudaSuccess) return 1;

    uint32_t copy_count = *(ctx->h_count);
    if (copy_count > out_capacity) copy_count = out_capacity;
    if (copy_count > 0) {
        st = cudaMemcpyAsync(ctx->h_results, ctx->d_results, (size_t)copy_count * sizeof(slimy_cuda_result), cudaMemcpyDeviceToHost, ctx->stream);
        if (st != cudaSuccess) return 1;
        st = cudaStreamSynchronize(ctx->stream);
        if (st != cudaSuccess) return 1;
        memcpy(out_results, ctx->h_results, (size_t)copy_count * sizeof(slimy_cuda_result));
    }

    *out_count = copy_count;
    if (*(ctx->h_overflow)) return 2;
    return 0;
}

int slimy_cuda_search_batch(
    int device_index,
    int64_t world_seed,
    uint8_t threshold,
    int32_t x0,
    int32_t z0,
    uint32_t width,
    uint32_t height,
    slimy_cuda_result* out_results,
    uint32_t out_capacity,
    uint32_t* out_count
) {
    cudaError_t st = cudaSetDevice(device_index);
    if (st != cudaSuccess) return 1;

    slimy_cuda_result* d_results = nullptr;
    uint32_t* d_count = nullptr;
    int* d_overflow = nullptr;
    uint8_t* d_slime_map = nullptr;
    uint64_t total = 0;
    uint32_t map_width = 0;
    uint32_t map_height = 0;
    uint32_t count = 0;
    int overflow = 0;
    uint32_t copy_count = 0;

    st = cudaMalloc((void**)&d_results, (size_t)out_capacity * sizeof(slimy_cuda_result));
    if (st != cudaSuccess) return 1;

    st = cudaMalloc((void**)&d_count, sizeof(uint32_t));
    if (st != cudaSuccess) {
        cudaFree(d_results);
        return 1;
    }

    st = cudaMalloc((void**)&d_overflow, sizeof(int));
    if (st != cudaSuccess) {
        cudaFree(d_count);
        cudaFree(d_results);
        return 1;
    }

    map_width = width + 16;
    map_height = height + 16;
    st = cudaMalloc((void**)&d_slime_map, (size_t)map_width * (size_t)map_height * sizeof(uint8_t));
    if (st != cudaSuccess) {
        cudaFree(d_overflow);
        cudaFree(d_count);
        cudaFree(d_results);
        return 1;
    }

    st = cudaMemset(d_count, 0, sizeof(uint32_t));
    if (st != cudaSuccess) goto fail;
    st = cudaMemset(d_overflow, 0, sizeof(int));
    if (st != cudaSuccess) goto fail;

    total = (uint64_t)width * (uint64_t)height;
    if (total > 0 && width > 0 && height > 0) {
        const uint32_t map_threads = 256;
        uint64_t map_total = (uint64_t)map_width * (uint64_t)map_height;
        uint32_t map_blocks = (uint32_t)((map_total + map_threads - 1) / map_threads);
        slime_map_kernel<<<map_blocks, map_threads>>>(
            world_seed,
            x0 - 8,
            z0 - 8,
            map_width,
            map_height,
            d_slime_map
        );

        st = cudaGetLastError();
        if (st != cudaSuccess) goto fail;

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        search_kernel<<<grid, block>>>(
            d_slime_map,
            map_width,
            threshold,
            x0,
            z0,
            width,
            height,
            d_results,
            out_capacity,
            d_count,
            d_overflow
        );

        st = cudaGetLastError();
        if (st != cudaSuccess) goto fail;
        st = cudaDeviceSynchronize();
        if (st != cudaSuccess) goto fail;
    }

    st = cudaMemcpy(&count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) goto fail;
    st = cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) goto fail;

    copy_count = count;
    if (copy_count > out_capacity) copy_count = out_capacity;

    if (copy_count > 0) {
        st = cudaMemcpy(out_results, d_results, (size_t)copy_count * sizeof(slimy_cuda_result), cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) goto fail;
    }

    *out_count = copy_count;

    cudaFree(d_slime_map);
    cudaFree(d_overflow);
    cudaFree(d_count);
    cudaFree(d_results);

    if (overflow) return 2;
    return 0;

fail:
    cudaFree(d_slime_map);
    cudaFree(d_overflow);
    cudaFree(d_count);
    cudaFree(d_results);
    return 1;
}

} // extern "C"
