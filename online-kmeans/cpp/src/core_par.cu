#include "core_par.hpp"

#include "error.cuh"

#include <cuda.h>
#include <cuda_runtime.h>


namespace signals {

namespace cuda {

__global__
void k_means_kernel(
    uint8_t* d_img, uint8_t* d_assigned_img,
    uint8_t* d_prototypes, uint64_t *d_sums, uint64_t *d_counts,
    size_t img_height, size_t img_width, uint64_t k) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if core in image boundary
    if (i >= img_height || j >= img_width)
        return;
    
    uint8_t b = d_img[i * img_width * 3 + j * 3];
    uint8_t g = d_img[i * img_width * 3 + j * 3 + 1];
    uint8_t r = d_img[i * img_width * 3 + j * 3 + 2];

    float min_distance = MAXFLOAT;
    int assigned_prototype_index = -1;
    for (int p = 0; p < k; p++)
    {
        uint8_t prot_b = d_prototypes[p * 3];
        uint8_t prot_g = d_prototypes[p * 3 + 1];
        uint8_t prot_r = d_prototypes[p * 3 + 2];

        float distance_squared = (r - prot_r) * (r - prot_r) + (g - prot_g) * (g - prot_g) + (b - prot_b) * (b - prot_b);
        if (distance_squared < min_distance) {
            min_distance = distance_squared;
            assigned_prototype_index = p;
        }
    }
    d_assigned_img[i * img_width + j] = assigned_prototype_index;

    // Use atomic operations to safely update sums and counts
    atomicAdd((unsigned long long int*) &d_sums[assigned_prototype_index * 3], b);
    atomicAdd((unsigned long long int*) &d_sums[assigned_prototype_index * 3 + 1], g);
    atomicAdd((unsigned long long int*) &d_sums[assigned_prototype_index * 3 + 2], r);
    atomicAdd((unsigned long long int*) &d_counts[assigned_prototype_index], 1);
}


__global__
void k_means_kernel_shared(
    uint8_t* d_img, uint8_t* d_assigned_img,
    uint8_t* d_prototypes, uint64_t *d_sums, uint64_t *d_counts,
    size_t img_height, size_t img_width, uint64_t k)
{
    extern __shared__ uint8_t prot_shared[];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Collaborative loading of prototypes into the shared memory
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (thread_idx < k * 3) {
        prot_shared[thread_idx] = d_prototypes[thread_idx];
    }
    __syncthreads();

    // Check if core in image boundary
    if (i >= img_height || j >= img_width)
        return;

    // Algorithm as normal but with shared prototypes
    uint8_t b = d_img[i * img_width * 3 + j * 3];
    uint8_t g = d_img[i * img_width * 3 + j * 3 + 1];
    uint8_t r = d_img[i * img_width * 3 + j * 3 + 2];

    float min_distance = MAXFLOAT;
    int assigned_prototype_index = -1;
    for (int p = 0; p < k; p++)
    {
        uint8_t prot_b = prot_shared[p * 3];
        uint8_t prot_g = prot_shared[p * 3 + 1];
        uint8_t prot_r = prot_shared[p * 3 + 2];

        float distance_squared = (r - prot_r) * (r - prot_r) + (g - prot_g) * (g - prot_g) + (b - prot_b) * (b - prot_b);
        if (distance_squared < min_distance) {
            min_distance = distance_squared;
            assigned_prototype_index = p;
        }
    }
    d_assigned_img[i * img_width + j] = assigned_prototype_index;

    // Use atomic operations to safely update sums and counts
    atomicAdd((unsigned long long int*) &d_sums[assigned_prototype_index * 3], b);
    atomicAdd((unsigned long long int*) &d_sums[assigned_prototype_index * 3 + 1], g);
    atomicAdd((unsigned long long int*) &d_sums[assigned_prototype_index * 3 + 2], r);
    atomicAdd((unsigned long long int*) &d_counts[assigned_prototype_index], 1);
}


void k_means(
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    uint64_t k, float_t stab_error, int max_iterations,
    const KmeansInfo& device_info, bool use_shared_mem,
    uint8_t* stored_prototypes, bool use_stored_prototypes)
{   
    // Copy data to CUDA (initial)
    SAFE_CALL( cudaMemcpy(device_info.d_img, img, sizeof(uint8_t) * img_height * img_width * 3, cudaMemcpyHostToDevice));

    srand((unsigned) time(NULL));

    // Create k prototypes with random values
    uint8_t* prototypes;
    
    if (use_stored_prototypes)
    {
        prototypes = stored_prototypes;
    }
    else
    {
        prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * 3);
        for (int i = 0; i < k * 3; i++) 
            prototypes[i] = rand() % 256;
    }
    
    uint8_t* assigned_img = (uint8_t*) malloc (sizeof(uint8_t) * img_height * img_width);  // Map : pixels -> cluster number

    // Array for calculating means
    uint64_t* sums = (uint64_t*) malloc (sizeof(uint64_t) * k * 3);
    uint64_t* counts = (uint64_t*) malloc (sizeof(uint64_t) * k);

    uint8_t* old_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * 3);

    // Zero array for copying into sums and counts
    uint64_t* zeros = (uint64_t*) calloc (k * 3, sizeof(uint64_t));
    
    DeviceInfo dev_info;
    find_best_grid(dev_info, img_height, img_width);

    bool bound_reached = false;

    // Loop until prototypes are stable
    for (int iteration_count = 0; !bound_reached && iteration_count < max_iterations; iteration_count++)
    {
        memcpy(old_prototypes, prototypes, k * 3 * sizeof(uint8_t));    // Save old values for calculating differences
        
        // Copy to CUDA
        SAFE_CALL( cudaMemcpy(device_info.d_prototypes, old_prototypes, sizeof(uint8_t) * k * 3, cudaMemcpyHostToDevice));
        SAFE_CALL( cudaMemset(device_info.d_sums, 0, sizeof(uint64_t) * k * 3));
        SAFE_CALL( cudaMemset(device_info.d_counts, 0, sizeof(uint64_t) * k));

        // Kernel call
        dim3 dim_grid = dim3(dev_info.grid.x, dev_info.grid.y, dev_info.grid.z);
        dim3 dim_block = dim3(dev_info.block.x, dev_info.block.y, dev_info.block.z);

        if (use_shared_mem) {
            k_means_kernel_shared <<<dim_grid, dim_block, k * 3 * sizeof(uint8_t)>>> (
                device_info.d_img, device_info.d_assigned_img, device_info.d_prototypes, 
                device_info.d_sums, device_info.d_counts,
                img_height, img_width, k
            );
        }
        else {
            k_means_kernel <<<dim_grid, dim_block>>> (
                device_info.d_img, device_info.d_assigned_img, device_info.d_prototypes, 
                device_info.d_sums, device_info.d_counts,
                img_height, img_width, k
            );
        }
        CHECK_CUDA_ERROR;

        // Copy data back to CPU
        SAFE_CALL( cudaMemcpy(assigned_img, device_info.d_assigned_img, sizeof(uint8_t) * img_height * img_width, cudaMemcpyDeviceToHost ));
        SAFE_CALL( cudaMemcpy(sums, device_info.d_sums, sizeof(uint64_t) * k * 3, cudaMemcpyDeviceToHost ));
        SAFE_CALL( cudaMemcpy(counts, device_info.d_counts, sizeof(uint64_t) * k, cudaMemcpyDeviceToHost ));

        // Update values of the prototypes to the means of the associated pixels
        for (int i = 0; i < k; i++)
        {
            if (counts[i] != 0)
            {
                prototypes[i * 3] = sums[i * 3] / counts[i];
                prototypes[i * 3 + 1] = sums[i * 3 + 1] / counts[i];
                prototypes[i * 3 + 2] = sums[i * 3 + 2] / counts[i];
            }
        }

        // Calculate differences
        bound_reached = true;

        for (int i = 0; i < k; i++)
        {
            uint8_t prot_b = prototypes[i * 3];
            uint8_t prot_g = prototypes[i * 3 + 1];
            uint8_t prot_r = prototypes[i * 3 + 2];
            uint8_t old_b = old_prototypes[i * 3];
            uint8_t old_g = old_prototypes[i * 3 + 1];
            uint8_t old_r = old_prototypes[i * 3 + 2];

            float distance_squared = pow(prot_r - old_r, 2) + pow(prot_g - old_g, 2) + pow(prot_b - old_b, 2);

            if (distance_squared > stab_error)
            {
                bound_reached = false;
                break;
            }
        }
    }

    // Substitute each pixel with the corresponding prototype value
    for (int i = 0; i < img_height; i++)
    {
        for (int j = 0; j < img_width; j++)
        {
            int index = assigned_img[i * img_width + j];
            dst[i * img_width * 3 + j * 3] = prototypes[index * 3];
            dst[i * img_width * 3 + j * 3 + 1] = prototypes[index * 3 + 1];
            dst[i * img_width * 3 + j * 3 + 2] = prototypes[index * 3 + 2];
        }
    }

    // Copy prototypes to stored prototypes to return them
    if (!use_stored_prototypes)
    {
        memcpy(stored_prototypes, prototypes, k * 3 * sizeof(uint8_t));

        free(prototypes);
    }

    // Free memory
    free(assigned_img);
    free(counts);
    free(sums);
    free(old_prototypes);
}


KmeansInfo init_k_means (size_t img_height, size_t img_width, uint64_t k)
{
    KmeansInfo device_info;
    SAFE_CALL( cudaMalloc(&device_info.d_img, sizeof(uint8_t) * img_height * img_width * 3) );
    SAFE_CALL( cudaMalloc(&device_info.d_assigned_img, sizeof(uint8_t) * img_height * img_width) );
    SAFE_CALL( cudaMalloc(&device_info.d_prototypes, sizeof(uint8_t) * k * 3) );
    SAFE_CALL( cudaMalloc(&device_info.d_sums, sizeof(uint64_t) * k * 3) );
    SAFE_CALL( cudaMalloc(&device_info.d_counts, sizeof(uint64_t) * k) );
    find_best_grid(device_info.dim, img_height, img_width);
    return device_info;
}


void deinit_k_means (KmeansInfo& device_info)
{
    SAFE_CALL( cudaFree( device_info.d_img ) );
    SAFE_CALL( cudaFree( device_info.d_assigned_img ) );
    SAFE_CALL( cudaFree( device_info.d_prototypes ) );
    SAFE_CALL( cudaFree( device_info.d_sums ) );
    SAFE_CALL( cudaFree( device_info.d_counts ) );
}


static cudaDeviceProp find_best_gpu()
{
    // Save properties of GPUs
    int dev_count;
    SAFE_CALL( cudaGetDeviceCount(&dev_count) );

    // Save best GPU
    cudaDeviceProp dev_prop;
    cudaDeviceProp best_device_prop;

    for (int i = 0; i < dev_count; i++)
    {
        SAFE_CALL( cudaGetDeviceProperties(&dev_prop, i) );
        
        if (i == 0 
            || (dev_prop.maxThreadsPerMultiProcessor * dev_prop.multiProcessorCount > 
                best_device_prop.maxThreadsPerMultiProcessor * best_device_prop.multiProcessorCount))
        {
            best_device_prop = dev_prop;
        }
    }
    return best_device_prop;
}


DeviceInfo& find_best_grid(DeviceInfo& device_info, std::size_t height, std::size_t width)
{
    auto best_device = find_best_gpu();

    std::size_t threadsPerBlockTemp;

    if(height * width > best_device.maxThreadsPerBlock / 2)
        threadsPerBlockTemp = best_device.maxThreadsPerBlock;
    else
        threadsPerBlockTemp = pow(2, ceil(log2(height * width)));
    
    std::size_t threadsPerBlockRow = pow(2, ceil(log2(sqrt(threadsPerBlockTemp))));

    device_info.block = {
        threadsPerBlockRow, 
        threadsPerBlockRow, 
        1,
    };
    device_info.grid = {
        width / threadsPerBlockRow, 
        height / threadsPerBlockRow,
        1
    };
    if (width % threadsPerBlockRow) device_info.grid.x++;
    if (height % threadsPerBlockRow) device_info.grid.y++;
    return device_info;
}



} // namespace cuda

} // namespace signals