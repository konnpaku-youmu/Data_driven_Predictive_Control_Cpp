#ifndef HELPER_H
#define HELPER_H

#include <eigen3/Eigen/Eigen>

using namespace Eigen;

void forward_euler(MatrixXd *A, MatrixXd *B, double Ts);

VectorXd mulvar_noise_vec(VectorXd &mean, MatrixXd &cov);

#endif
