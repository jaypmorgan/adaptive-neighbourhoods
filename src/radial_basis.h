#ifndef RADIAL_BASIS_H
#define RADIAL_BASIS_H

#include <vector>
#include "vector_ops.h"


template <typename T>
T rbf_basis(std::vector<T> u, std::vector<T> v)
{
    return vector_norm<T>(vector_sub<T>(u, v));
}


template <typename T>
T rbf_inverse_multiquadric(std::vector<T> u, std::vector<T> v, T epsilon)
{   
    return 1.0 / sqrt(1.0 + pow(epsilon * rbf_basis<T>(u, v), 2));
}

template <typename T>
T rbf_gaussian(std::vector<T> u, std::vector<T> v, T epsilon) 
{
    return exp(-(epsilon * pow(rbf_basis<T>(u, v), 2)));
}

#endif