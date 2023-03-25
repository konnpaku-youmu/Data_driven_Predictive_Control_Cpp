#include "SysBase.h"

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

template <typename T>
class Controller
{
protected:
    T* model;

    virtual VectorXd _policy(const VectorXd &, const VectorXd &) = 0;

public:
    Controller(T* model_ptr);

    auto get_policy() -> std::function<VectorXd(VectorXd &, VectorXd &)>;
};

template <typename T>
class LQR : public Controller<T>
{
protected:
    MatrixXd Q, R;

    MatrixXd K;

    VectorXd _policy(const VectorXd &xk, const VectorXd &rk);

    void _compute_K();

public:
    LQR(T* model_ptr, const MatrixXd &Q, const MatrixXd &R);

};

template <typename T>
class MPC : public Controller<T>
{
protected:
    MatrixXd Q, R;

    VectorXd _policy(VectorXd &xk, VectorXd &rk);

    void _build_opti_problem();

public:
    MPC(T* model_ptr, const MatrixXd &Q, const MatrixXd &R);
};
