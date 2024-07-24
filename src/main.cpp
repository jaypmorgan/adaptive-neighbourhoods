#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
// #include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
//#include "Eigen/src/Core/products/Parallelizer.h"
#include "vector_ops.h"
#include "radial_basis.h"

// # define DEBUG

// namespace alias
namespace py = pybind11;

// type alias
using Vec = Eigen::VectorXf;
using Mat = Eigen::MatrixXf;


template <typename T>
T euclidean_distance(Vec u, Vec v) {
  return (u-v).norm();
}


bool is_intersecting(Vec u, Vec v, float r1, float r2)
{
  float distance =  euclidean_distance<float>(u, v);
  return distance <= (r1+r2);
}

std::vector<float> compute_density(
                    Mat x,
                    std::vector<int> y,
                    float epsilon,
                    std::vector<std::vector<int>> diff_class_indexes)
{
  size_t n_samples = y.size();
  Vec d(n_samples);
  int done = 0;

  //  Eigen::initParallel();
  //Eigen::setNbThreads(omp_get_max_threads());

  Mat distance(n_samples, n_samples);
  //#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < n_samples; i++)
    for (size_t j = 0; j < n_samples; j++)
      //#pragma omp atomic write
      distance(i, j) = (1.0 / sqrt(1.0 + pow(epsilon * (x.row(i)-x.row(j)).squaredNorm(), 2)));

  // set the diagonal to zero so that it doesn't affect the sum
  distance.diagonal().array() = 0.0;
  
  //#pragma omp parallel for
  for (size_t i = 0; i < n_samples; i++) {
    auto other_class_index = diff_class_indexes[y[i]];
    for (const auto& j : other_class_index) {
      //#pragma omp critical
      distance(i, j) = 0;
    }
  }

  //  Eigen::setNbThreads(0);

  auto summed = distance.rowwise().sum();
  std::vector<float> result{};
  for (size_t i = 0; i < n_samples; i++) {
    result.push_back(summed(i) / (n_samples - diff_class_indexes[y[i]].size()));
  }

  return result;
}


float distance(Vec u, Vec v, float r1, float r2)
{
  return euclidean_distance<float>(u, v) - (r1 + r2);
}

bool intersects(Vec u, Vec v, float r1, float r2)
{
  float distance = euclidean_distance<float>(u, v);
  return distance < (r1 + r2);
}

std::vector<bool> is_touching(Mat x,
                              std::vector<int> y,
                              std::vector<float> radius,
                              std::vector<bool> touching,
                              std::vector<std::vector<int>> diff_class_indexes)
{
  float r1;
  float r2;
  int cls;
  std::vector<int> cls_indexes;

  for (int i = 0; i < x.rows(); i++) {
    if (!touching[i]) {  // skip those that are already touching something
      cls = y[i];
      cls_indexes = diff_class_indexes[cls];
      for (const auto& j : cls_indexes) {
        r1 = radius[i];
        r2 = radius[j];
        if (intersects(x.row(i), x.row(j), r1, r2)) {
          touching[i] = true;
          touching[j] = true;
        }
      }
    }
  }
  return touching;
}

float decay_density(float density, int steps)
{
  return exp(-(density + steps)); // TODO: Add decay e^(-p_c(i) * xi * n)
}

std::vector<float> epsilon_expand(
                                  Mat x,
                                  std::vector<int> y,
                                  float max_step_size = 0.0001,
                                  float epsilon   = 5.0,
                                  bool show_progress = false)
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
    for (int i = 0; i < x.rows(); i++)
      if (!vector_in(this_class, i))
        diff_this_class.push_back(i);
    diff_class_indexes.push_back(diff_this_class);
  }

  // Calculate the density value for each point
  auto density = compute_density(x, y, epsilon, diff_class_indexes);

#ifdef DEBUG
  std::cout << "Density Information:" 
            << "\n- maximum: " << density.minCoeff()
            << "\n- minimum: " << density.maxCoeff()
            << "\n- mean:    " << density.mean()
            << std::endl;
#endif

  // Expand the radii
  std::vector<bool> touching{ vector_zeros<bool>(x.rows()) };
  std::vector<float> radius{ vector_zeros<float>(x.rows()) };
  std::vector<int> non_touching{};
  std::vector<float> step_sizes{ vector_fill<float>(x.rows(), max_step_size) };

  // set the step size for each point to half the min distance
  //#pragma omp parallel for
  for (int i = 0; i < x.rows(); i++) {
    if (show_progress)
      std::cout << "\rComputing step size: " << i << std::flush;
    float min_dist = step_sizes[i];
    auto other_class_points = diff_class_indexes[y[i]];
    for (const auto& j : other_class_points) {
      if (i != j) {
        float dist = euclidean_distance<float>(x.row(i), x.row(j));
        if (dist < min_dist)
          min_dist = dist / 2.;
      }
    }
    //#pragma atomic write
    step_sizes[i] = min_dist;
  }
  
  int step = 0;

  while (vector_find_element<bool>(touching, false).size() > 0) {
    touching = is_touching(x, y, radius, touching, diff_class_indexes);  // find which data points overlap
    // set touching for those with step below threshold
    for (int i = 0; i < touching.size(); i++)
      if (step_sizes [i] < 1e-15)
        touching[i] = true;

    // expand neighbourhood for data points that are not touching
    non_touching = vector_find_element<bool>(touching, false);
    for (const auto& idx : non_touching) {
      float s = step_sizes[idx];
      bool increase = true;
      // TODO: Make quicker than n^2
      //Eigen::setNbThreads(omp_get_max_threads());

      //#pragma omp parallel for
      for (const auto& jdx : non_touching) {
        if (idx != jdx && y[idx] != y[jdx]) {
          float dist = distance(x.row(idx), x.row(jdx), radius[idx], radius[jdx]);
          if (dist <= s) {
            //#pragma atomic write
            touching[idx] = true, touching[jdx] = true, increase = false, s = dist;
          }
        }
      }

      //Eigen::setNbThreads(0);
      
      if (increase)
        radius[idx] += s;  // TODO check that neighbourhoods wouldn't overlap subject to increases.
    }

    
    // decay the step sizes using exponetial decay
    for (size_t i = 0; i < density.size(); i++) {
      step_sizes[i] = step_sizes[i] * density[i]; //(decay(density[i], step));
    }

    step++;
    if (show_progress)
      std::cout << "\rExpanding Neighbourhoods Step: " << step 
                << " Number of touching " << vector_find_element(touching, true).size()
                <<"  Max radius " << vector_max(radius) 
                << std::flush;
  }

  if (show_progress)
    std::cout << std::endl;

  return radius;
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

struct Iris {
  Mat x;
  std::vector<int> y;
};

Iris read_iris(std::string path)
{
  auto data = read_csv(path);
  
  data = std::vector<std::vector<std::string>>(data.begin(), data.end()-1);  // remove the empty line

  Mat x(150, 4);

  for (int row = 0; row < data.size(); row++)
    for (int col = 0; col < data[row].size()-1; col++)
      x(row,col) = std::stof(data[row][col]);

  const std::unordered_map<std::string, int> mapper {
    {"Iris-setosa", 0},
    {"Iris-versicolor", 1},
    {"Iris-virginica", 2},
  };

  std::vector<int> y{};
  for (int i = 0; i < data.size(); i++)
    y.push_back(mapper.at(data[i][data[i].size()-1]));

  Iris iris;
  iris.x = x;
  iris.y = y;
  return iris;
}

int main(int argc, char** argv)
{
  auto iris = read_iris(argv[1]);

  std::cout << iris.x << std::endl;

  auto normed = vector_norm<int>(iris.y);
  std::cout << "Normed: " << normed << std::endl;

  std::cout << "Vector operations:"
            << "\n- min: " << vector_min<int>(iris.y)
            << "\n- max: " << vector_max<int>(iris.y)
            << "\n- find elements: " << vector_print(vector_find_element<int>(iris.y, 2.0))
            << "\n- unique elements: " << vector_print(vector_unique(iris.y))
            << std::endl;

  std::vector<float> neighbourhoods = epsilon_expand(iris.x, iris.y, 0.0001, 5, true);
  std::cout << "Neighbourhoods:"
            << "\n" << vector_print(neighbourhoods)
            << std::endl;

  return 0;
}

/* 
 * PYTHON BINDINGS
 */
py::array_t<float> wrapper_epsilon(
                                   Mat x,
                                   std::vector<int> y,
                                   float max_step_size = 0.0001,
                                   float epsilon   = 5.0,
                                   bool show_progress = false) 
{
  py::array result = py::cast(epsilon_expand(x, y, max_step_size, epsilon, show_progress));
  return result;
}

py::array_t<float> wrapper_compute_density(
                                           Mat x,
                                           std::vector<int> y,
                                           float epsilon,
                                           std::vector<std::vector<int>> diff_class_indexes)
{
  py::array result = py::cast(compute_density(x, y, epsilon, diff_class_indexes));
  return result;
}


PYBIND11_MODULE(adaptive_neighbourhoods, m)
{
  m.doc() = "Adaptive Neighbourhoods package";
  m.def("epsilon_expand", &wrapper_epsilon, "Generate the neighbourhoods",
        py::arg("x"), py::arg("y"), py::arg("max_step_size") = 0.0001, py::arg("epsilon") = 5.0,
        py::arg("show_progress") = false);
  m.def("compute_density", &wrapper_compute_density, "Compute the density value for each point.",
        py::arg("x"), py::arg("y"), py::arg("epsilon"), py::arg("diff_class_indexes"));
}

