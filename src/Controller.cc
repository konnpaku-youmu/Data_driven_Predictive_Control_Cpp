#include "Controller.h"
#include "SysModels.h"

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

template <typename T>
Controller<T>::Controller(T *model_ptr)
{
    model = model_ptr;
}

template <typename T>
auto Controller<T>::get_policy() -> std::function<VectorXd(VectorXd &, VectorXd &)>
{
    return std::bind(&Controller<T>::_policy, this, std::placeholders::_1, std::placeholders::_2);
}

template <typename T>
LQR<T>::LQR(T *model_ptr, const MatrixXd &Q, const MatrixXd &R) : Controller<T>(model_ptr)
{
    this->Q = Q;
    this->R = R;

    _compute_K();
}

template <typename T>
void LQR<T>::_compute_K()
{
    MatrixXd A = *(this->model->getA());
    MatrixXd B = *(this->model->getB());

    MatrixXd P = Q;

    for (int i = 0; i < 100; i++)
    {
        P = Q + A.transpose() * P * A - A.transpose() * P * B * (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;
    }

    K = -(R + B.transpose() * P * B).inverse() * B.transpose() * P * A;
}

template <typename T>
VectorXd LQR<T>::_policy(const VectorXd &xk, const VectorXd &rk)
{
    return K * (xk - rk);
}

template <typename T>
MPC<T>::MPC(T *model_ptr, const MatrixXd &Q, const MatrixXd &R) : Controller<T>(model_ptr)
{
    this->Q = Q;
    this->R = R;
}

template <typename T>
void MPC<T>::_build_opti_problem()
{

}

template <typename T>
VectorXd MPC<T>::_policy(VectorXd &xk, VectorXd &rk)
{
    return VectorXd();
}

// declare all the template classes
template class Controller<InvertedPendulum>;
template class LQR<InvertedPendulum>;
template class MPC<InvertedPendulum>;
