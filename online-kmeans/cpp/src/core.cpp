#include "core.hpp"

#include <iostream>

namespace signals
{

void k_means (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    uint64_t k, float_t stab_error, int max_iterations) 
{
    srand((unsigned) time(NULL));

    // Create k prototypes with random values
    uint8_t* prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * 3);
    for (int i = 0; i < k * 3; i++) 
    {
        prototypes[i] = rand() % 256;
    }
    
    uint8_t* assigned_img = (uint8_t*) calloc (img_height * img_width, sizeof(uint8_t));  // Map : pixels -> cluster number

    // Array for calculating means
    uint64_t* sums = (uint64_t*) malloc (sizeof(uint64_t) * k * 3);
    uint64_t* counts = (uint64_t*) malloc (sizeof(uint64_t) * k);

    uint8_t* old_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * 3);

    bool bound_reached = false;

    // Loop until prototypes are stable
    for (int iteration_count = 0; !bound_reached && iteration_count < max_iterations; iteration_count++)
    {
        memcpy(old_prototypes, prototypes, k * 3 * sizeof(uint8_t));    // Save old values for calculating differences

        // Resetting sums and counts
        for (int i = 0; i < k * 3; i++) sums[i] = 0;
        for (int i = 0; i < k; i++) counts[i] = 0;

        // Associate each pixel to nearest prototype (with Euclidian distance)
        for (int i = 0; i < img_height; i++)
        {
            for (int j = 0; j < img_width; j++)
            {
                uint8_t b = img[i * img_width * 3 + j * 3];
                uint8_t g = img[i * img_width * 3 + j * 3 + 1];
                uint8_t r = img[i * img_width * 3 + j * 3 + 2];

                float min_distance = MAXFLOAT;
                int assigned_prototype_index = -1;
                for (int p = 0; p < k; p++)
                {
                    uint8_t prot_b = prototypes[p * 3];
                    uint8_t prot_g = prototypes[p * 3 + 1];
                    uint8_t prot_r = prototypes[p * 3 + 2];

                    float distance = sqrt(pow(r - prot_r, 2) + pow(g - prot_g, 2) + pow(b - prot_b, 2));
                    if (distance < min_distance) {
                        min_distance = distance;
                        assigned_prototype_index = p;
                    }
                }
                assigned_img[i * img_width + j] = assigned_prototype_index;

                sums[assigned_prototype_index * 3] += b;
                sums[assigned_prototype_index * 3 + 1] += g;
                sums[assigned_prototype_index * 3 + 2] += r;
                counts[assigned_prototype_index]++;
            }
        }

        // Update values of the prototypes to the means of the associated pixels
        for (int i = 0; i < k; i++)
        {
            if (counts[i] != 0)
            {
                prototypes[i * 3] = (uint8_t) ((float) sums[i * 3] / counts[i]);
                prototypes[i * 3 + 1] = (uint8_t) ((float) sums[i * 3 + 1] / counts[i]);
                prototypes[i * 3 + 2] = (uint8_t) ((float) sums[i * 3 + 2] / counts[i]);
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
    
    // Free memory
    free (prototypes);
    free (assigned_img);
    free (sums);
    free (counts);
    free (old_prototypes);
}

} // namespace signals