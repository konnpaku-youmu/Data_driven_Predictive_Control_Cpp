#include <helper.h>

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

void forward_euler(MatrixXd *A, MatrixXd *B, double Ts)
{
    *A = MatrixXd::Identity(A->rows(), A->cols()) + Ts * (*A);
    *B = Ts * (*B);
    
    return;
}

VectorXd mulvar_noise_vec(VectorXd &mean, MatrixXd &cov)
{   
    int n_outputs = mean.rows();
    // Obtain random normal variables for each output dimension
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Initialize output noise vector
    VectorXd noise(n_outputs);

    // Use Cholesky decomposition of covariance matrix to generate correlated noise
    LLT<MatrixXd> llt_covariance(cov);
    MatrixXd L = llt_covariance.matrixL();

    for (int i = 0; i < n_outputs; i++) {
        noise(i) = dist(gen);
    }

    VectorXd mulvar_noise = mean + L * noise;

    return mulvar_noise;
}
