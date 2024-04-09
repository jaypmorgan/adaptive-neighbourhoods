#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <algorithm>

template <typename T>
std::vector<T> vector_sub(std::vector<T> u, std::vector<T> v)
{
    // check that both vectors have the same length
    if (u.size() != v.size())
        throw std::invalid_argument("Vectors must have the same length.");

    std::vector<T> result = {};
    for (int i = 0; i < u.size(); i++) {
        result.push_back(u[i]-v[i]);
    }
    return result;
}

template <typename T>
T vector_sum(std::vector<T> u)
{
  T sum = 0;
  for (const auto& el : u)
    sum += el;
  return sum;
}

template <typename T>
std::string vector_print(std::vector<T> u)
{
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < u.size(); i++) {
        oss << u[i];
        if ((i+1) != u.size()) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

template <typename T>
bool vector_in(std::vector<T> u, T element) 
{
    for (auto& el : u)
        if (el == element)
            return true;
    return false;
}

template <typename T>
std::vector<T> vector_copy(std::vector<T> u) 
{
    std::vector<T> copy = {};
    for (auto& el : u)
        copy.push_back(el);
    return copy;
}

template <typename T>
T vector_max(std::vector<T> u) 
{
    std::vector<T> sorted = vector_copy(u);
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size()-1];
}

template <typename T>
T vector_min(std::vector<T> u) 
{
    std::vector<T> sorted = vector_copy(u);
    std::sort(sorted.begin(), sorted.end());
    return sorted[0];
}

template <typename T>
T vector_mean(std::vector<T> u) 
{
    T mean = 0;
    for (auto& el : u)
        mean += el;
    return mean / u.size();
}

template <typename T>
std::vector<T> vector_unique(std::vector<T> u)
{
    std::vector<T> result = {};
    for (auto& el : u)
        if (!vector_in(result, el))
            result.push_back(el);
    return result;
}

template <typename T>
std::vector<int> vector_find_element(std::vector<T> u, T element)
{
    std::vector<int> result = {};
    for (int i = 0; i < u.size(); i++)
        if (u[i] == element)
            result.push_back(i);
    return result;
}

template <typename T>
T vector_norm(std::vector<T> u, int power = 2)
{
    T result = 0;
    for (auto& el : u)
        result += pow(el, power);
    return result;
}

template <typename T>
std::vector<T> vector_fill(size_t size, T value)
{
    std::vector<T> result{};
    for (size_t i = 0; i < size; i++) {
        result.push_back(value);
    }
    return result;
}

template <typename T>
std::vector<T> vector_zeros(size_t size) 
{
    return vector_fill<T>(size, 0);
}

template <typename T>
std::vector<T> vector_scale(std::vector<T> u, T scalar)
{
    std::vector<T> result{};
    for (auto& el : u)
        result.push_back(el * scalar);
    return result;
}

template <typename T>
std::vector<T> vector_scale(std::vector<T> u, std::vector<T> v)
{
    std::vector<T> result{};
    for (int i = 0; i < u.size(); i++)
        result.push_back(u[i]*v[i]);
    return result;
}

#endif
