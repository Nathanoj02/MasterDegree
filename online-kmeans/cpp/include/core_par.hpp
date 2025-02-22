#ifndef SIGNALS_CORE_PAR_HPP
#define SIGNALS_CORE_PAR_HPP

#include <opencv2/imgcodecs.hpp>

namespace signals {

namespace cuda {

struct DimInfo {
    std::size_t x;
    std::size_t y;
    std::size_t z;
};

struct DeviceInfo {
    DimInfo block;
    DimInfo grid;
};

struct KmeansInfo {
    uint8_t *d_img;
    uint8_t *d_assigned_img;
    uint8_t *d_prototypes;
    uint64_t *d_sums;
    uint64_t *d_counts; 
    struct DeviceInfo dim;
};

/**
 * Find best grid for current GPU
 * 
 * @param device_info Struct in which the best block and grid dimensions are memorized 
 * @param height Height of the input matrix
 * @param width Width of the input matrix
 * 
 * @return Device info with best block and grid dimensions
 */
DeviceInfo& find_best_grid(
    DeviceInfo& device_info, std::size_t height, std::size_t width
);

/**
 * Initialization for k-means algorithm, allocating memory on the GPU
 * 
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * 
 * @return Struct with pointers to GPU memory and block and grid dimensions
 */
KmeansInfo init_k_means(
    size_t img_height, size_t img_width, uint64_t k
);

/**
 * K-means clustering algorithm
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param stab_error Error bound to reach to end the algorithm
 * @param use_shared_mem Flag, use shared memory in CUDA kernel
 * @param stored_prototypes Memory zone where prototypes can be upload and also where the resulting prototypes are returned
 * @param use_stored_prototypes Use stored_prototypes instead of random ones
 */
void k_means(
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    uint64_t k, float_t stab_error, int max_iterations,
    const KmeansInfo& device_info, bool use_shared_mem,
    uint8_t* stored_prototypes, bool use_stored_prototypes
);

/**
 * De-initialization for k-means algorithm, deallocating memory from the GPU
 * 
 * @param device_info Struct with pointers to GPU memory and block and grid dimensions
 */
void deinit_k_means(
    KmeansInfo& device_info
);



} // namespace cuda

} // namespace signals

#endif // SIGNALS_CORE_PAR_HPP