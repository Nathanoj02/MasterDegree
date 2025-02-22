#ifndef SIGNALS_CORE_HPP
#define SIGNALS_CORE_HPP

#include <opencv2/imgcodecs.hpp>

namespace signals
{

/**
 * K-means clustering algorithm
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 */
void k_means (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    uint64_t k, float_t stab_error, int max_iterations
);

} // namespace signals

#endif // SIGNALS_CORE_HPP