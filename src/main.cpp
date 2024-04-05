#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// namespace alias
namespace py = pybind11;

// type alias
using Vec = std::vector<float>;
using Mat = std::vector<std::vector<float>>;

Vec vector_sub(Vec u, Vec v)
{
    // check that both vectors have the same length
    if (u.size() != v.size())
        throw std::invalid_argument("Vectors must have the same length.");

    Vec result = {};
    for (int i = 0; i < u.size(); i++) {
        result.push_back(u[i]-v[i]);
    }
    return result;
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
T vector_norm(Vec u, int power = 2)
{
    T result = 0;
    for (auto& el : u)
        result += pow(el, power);
    return result;
}

template <typename T>
T rbf_basis(Vec u, Vec v)
{
    return vector_norm<T>(vector_sub(u, v));
}

bool is_intersecting(Vec u, Vec v, float r1, float r2)
{
    float distance = rbf_basis<float>(u, v);
    return distance <= (r1+r2);
}

template <typename T>
T rbf_inverse_multiquadric(Vec u, Vec v, T epsilon)
{   
    return 1.0 / sqrt(1.0 + pow(epsilon * rbf_basis<T>(u, v), 2));
}

template <typename T>
T rbf_gaussian(Vec u, Vec v, T epsilon) 
{
    return exp(-(epsilon * pow(rbf_basis<T>(u, v), 2)));
}

Vec density(Mat x, std::vector<int> y, float epsilon)
{
    size_t n_samples = x.size();
    Vec d = {};
    float tmp_d = 0;
    Vec xi;
    Vec xj;

    std::vector<std::vector<int>> same_class_indexes;
    std::vector<int> classes = vector_unique<int>(y);

    for (auto& cls : classes) {
        same_class_indexes.push_back(vector_find_element(y, cls));
    }

    for (size_t i = 0; i < n_samples; i++) {
        tmp_d = 0;
        auto xi = x[i];
        auto yi = y[i];
        auto same_class_index = same_class_indexes[yi];
        for (auto& j : same_class_index) {
            if (i != j)
                tmp_d += rbf_inverse_multiquadric(xi, x.at(j), epsilon); 
        }
        tmp_d = 1 - (tmp_d / x.size());
        d.push_back(tmp_d);
    }

    return d;
}

std::vector<float> epsilon_expand(
    Mat x,
    std::vector<int> y,
    float step_size = 0.0001,
    float epsilon   = 5.0)
{

    std::vector<float> d{1, 2, 3, 4, 5};
    return d;

    // std::vector<float> d = density(x, y, epsilon);

    // std::cout << "Density Information:" 
    //           << "\n- maximum: " << vector_max<float>(d)
    //           << "\n- minimum: " << vector_min<float>(d)
    //           << "\n- mean:    " << vector_mean<float>(d)
    //           << std::endl;

    // return d;
}

/* 
 * PYTHON BINDINGS
 */
py::array_t<float> wrapper(
    Mat x, 
    std::vector<int> y,
    float step_size = 0.0001,
    float epsilon   = 5.0) 
{
    py::array result = py::cast(
        epsilon_expand(x, y, step_size, epsilon));
    return result;
}


PYBIND11_MODULE(adaptive_neighbourhoods, m)
{
    m.doc() = "Adaptive Neighbourhoods package";
    m.def("epsilon_expand", &wrapper, "Generate the neighbourhoods",
          py::arg("x"), py::arg("y"), py::arg("step_size") = 0.0001, py::arg("epsilon") = 5.0);
}


int main(void)
{

    //read_csv("../../../../iris.data");

    Mat x = {
        {4.0, 1.0, 4.0, 2.5},
        {4.0, 1.0, 4.0, 2.5},
        {4.0, 1.0, 4.0, 2.5},
    };
    Vec y = {0, 1, 1};
    auto normed = vector_norm<float>(y);
    std::cout << "Normed: " << normed << std::endl;

    std::cout << "Vector operations:"
              << "\n- min: " << vector_min<float>(y)
              << "\n- max: " << vector_max<float>(y)
              << "\n- find elements: " << vector_print(vector_find_element<float>(y, 3.0))
              << "\n- unique elements: " << vector_print(vector_unique(y))
              << std::endl;

    std::vector<int> classes(y.begin(), y.end());
    std::cout << vector_print(classes) << std::endl;


    // epsilon_expand(x, classes);

    return 0;
}