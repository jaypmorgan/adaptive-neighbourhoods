#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vector_ops.h"
#include "radial_basis.h"

#define DEBUG

// namespace alias
namespace py = pybind11;

// type alias
using Vec = std::vector<float>;
using Mat = std::vector<std::vector<float>>;

bool is_intersecting(Vec u, Vec v, float r1, float r2)
{
    float distance = rbf_basis<float>(u, v);
    return distance <= (r1+r2);
}

Vec compute_density(Mat x, std::vector<int> y, float epsilon, std::vector<std::vector<int>> same_class_indexes)
{
    size_t n_samples = x.size();
    Vec d = {};
    float tmp_d = 0;
    Vec xi;
    Vec xj;

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

bool intersects(std::vector<float> u, std::vector<float> v, float r1, float r2)
{
    float distance = rbf_basis(u, v);
    return distance < (r1 + r2);
}

std::vector<bool> is_touching(
    Mat x,
    std::vector<int> y,
    std::vector<float> radius,
    std::vector<bool> touching,
    std::vector<std::vector<int>> diff_class_indexes)
{
    std::vector<float> xi;
    std::vector<float> xj;
    float r1;
    float r2;
    int cls;
    std::vector<int> cls_indexes;

    for (int i = 0; i < x.size(); i++) {
        if (!touching[i]) {  // skip those that are already touching something
            cls = y[i];
            cls_indexes = diff_class_indexes[cls];
            for (auto& j : cls_indexes) {
                if (i != j) {   // skip self (always touching self...)
                    xi = x[i];
                    xj = x[j];
                    r1 = radius[i];
                    r2 = radius[j];
                    if (intersects(xi, xj, r1, r2)) {
                        touching[i] = true;
                        touching[j] = true;
                    }
                }
            }
        }
    }
    return touching;
}

std::vector<float> epsilon_expand(
    Mat x,
    std::vector<int> y,
    float step_size = 0.0001,
    float epsilon   = 5.0)
{

    // compute the indexes of each of the different classes
    // e.g. {
    //  {0, 1, 2},  // class 0 is on index 0, 1, 2
    //  {3, 4, 5},  // class 1 is on index 3, 4, 5
    //  {6},        // class 2 is on index 6
    // }
    std::vector<std::vector<int>> same_class_indexes;
    std::vector<std::vector<int>> diff_class_indexes;
    std::vector<int> classes = vector_unique<int>(y);
    std::vector<int> this_class;
    std::vector<int> diff_this_class;

    for (auto& cls : classes) {
        this_class = vector_find_element(y, cls);
        same_class_indexes.push_back(this_class);
        diff_this_class.clear();
        for (int i = 0; i < x.size(); i++)
            if (!vector_in(this_class, i))
                diff_this_class.push_back(i);
        diff_class_indexes.push_back(diff_this_class);
    }

    // Calculate the density value for each point
    std::vector<float> density = compute_density(x, y, epsilon, same_class_indexes);

#ifdef DEBUG
    std::cout << "Density Information:" 
              << "\n- maximum: " << vector_max<float>(density)
              << "\n- minimum: " << vector_min<float>(density)
              << "\n- mean:    " << vector_mean<float>(density)
              << std::endl;
#endif

    // Expand the radii
    std::vector<bool> touching{ vector_zeros<bool>(x.size()) };
    std::vector<float> radius{ vector_zeros<float>(x.size()) };
    std::vector<float> step_sizes{ vector_fill<float>(x.size(), step_size) };
    std::vector<int> non_touching{};
    int step = 0;

    while (vector_find_element<bool>(touching, false).size() > 0) {
        step++;
        step_sizes = vector_scale(step_sizes, density);    // scale step by density
        touching   = is_touching(x, y, radius, touching, diff_class_indexes);  // find which data points overlap

        // set touching for those with step below threshold
        for (int i = 0; i < touching.size(); i++)
            if (step_sizes[i] < 1e-15)
                touching[i] = true;

        // expand neighbourhood for data points that are not touching
        non_touching = vector_find_element<bool>(touching, false);
        for (auto& idx : non_touching) {
            radius[idx] += step_sizes[idx];
        }

        std::cout << "\rStep: " << step 
                  << " Number of touching " << vector_find_element(touching, true).size()
                  <<"  Max radius " << vector_max(radius) 
                  << std::flush;
    }

    return radius;
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


std::vector<std::vector<std::string>> read_csv(std::string filename, std::string delimiter = ",")
{
    std::ifstream file{ filename };
    std::string line;
    std::string token;
    size_t pos{ 0 };

    std::vector<std::vector<std::string>> data{};
    std::vector<std::string> record{};

    if (file.is_open()) {
        while (std::getline(file, line)) {
            record.clear();
            while ((pos = line.find(delimiter)) != std::string::npos) {
                token = line.substr(0, pos);
                record.push_back(token);
                line.erase(0, pos+delimiter.length());
            }
            record.push_back(line);
            data.push_back(record); 
        }
    }

    return data;
}


int main(void)
{

    auto data = read_csv("../../../../iris.data");
    data = std::vector<std::vector<std::string>>(data.begin(), data.end()-1);
    auto record = data[0];
    std::cout << "First record:"
              << "\n" << vector_print(record)
              << std::endl;

    Mat x{};
    for (int i = 0; i < data.size(); i++) {
        std::vector<float> row{};
        for (int j = 0; j < data[i].size()-1; j++) {
            row.push_back(0);
        }
        x.push_back(row);
    }

    for (int row = 0; row < data.size(); row++)
        for (int col = 0; col < data[row].size()-1; col++)
            x[row][col] = std::stof(data[row][col]);

    std::cout << "First record of x"
              << "\n" << vector_print(x[0])
              << std::endl;


    const std::unordered_map<std::string, int> mapper {
        {"Iris-setosa", 0},
        {"Iris-versicolor", 1},
        {"Iris-virginica", 2},
    };

    std::vector<int> y{};
    for (int i = 0; i < data.size(); i++) {
        y.push_back(mapper.at(data[i][data[i].size()-1]));
    }

    auto normed = vector_norm<int>(y);
    std::cout << "Normed: " << normed << std::endl;

    std::cout << "Vector operations:"
              << "\n- min: " << vector_min<int>(y)
              << "\n- max: " << vector_max<int>(y)
              << "\n- find elements: " << vector_print(vector_find_element<int>(y, 3.0))
              << "\n- unique elements: " << vector_print(vector_unique(y))
              << std::endl;

    std::vector<float> neighbourhoods = epsilon_expand(x, y);
    std::cout << "Neighbourhoods:"
              << "\n" << vector_print(neighbourhoods)
              << std::endl;

    return 0;
}