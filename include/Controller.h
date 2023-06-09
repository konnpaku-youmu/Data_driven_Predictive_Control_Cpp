#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "SysBase.h"
#include "helper.h"

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

enum
{
    Hankel,
    PartialHankel,
    Page
} typedef SMStruct;

enum
{
    CANONICAL,
    REGULARIZED
} typedef OCPType;

template <typename T>
class Controller
{
protected:
    T *model;

    VectorSeq ref_traj;

    virtual VectorXd _policy(const VectorXd &, const VectorXd &) = 0;

public:
    Controller(T *model_ptr);

    auto get_policy() -> ControlLaw;

    void show_reference_traj();
};

template <typename T>
class RandInput : public Controller<T>
{
protected:
    double input_cov;

    VectorXd _policy(const VectorXd &xk, const VectorXd &rk);

public:
    RandInput(T *model_ptr, double cov);
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
    LQR(T *model_ptr, const MatrixXd &Q, const MatrixXd &R);
};

template <typename T>
class MPC : public Controller<T>
{
protected:
    MatrixXd Q, R;

    VectorXd _policy(const VectorXd &xk, const VectorXd &rk);

    void _build_opti_problem();

public:
    MPC(T *model_ptr, const MatrixXd &Q, const MatrixXd &R);
};

template <typename T>
class DeePC : public Controller<T>
{
private:
    casadi::MX _unpack_opti_g();

protected:
    casadi::DM Q, R;

    uint32_t T_ini, horizon;

    ControlLaw init_controller;

    SMStruct signal_mat;

    OCPType ocp_type;

    casadi::MX loss, traj_constraint;

    std::vector<double> lbx, ubx, lbu, ubu;

    casadi::Matrix<double> U_p, U_f, Y_p, Y_f; // Signal Matrix Blocks

    casadi::Function *solver;

    casadi::MXDict opti_vars, opti_params, problem, result;

    void _build_controller();

    void _set_constraints();

    void _build_loss_func();

    VectorXd _policy(const VectorXd &xk, const VectorXd &rk);

public:
    DeePC(T *model_ptr, const MatrixXd &Q, const MatrixXd &R,
          uint32_t T_ini, uint32_t horizon,
          SMStruct sm_struct, const ControlLaw &init_policy,
          const std::vector<std::vector<double>> &state_bounds,
          const std::vector<std::vector<double>> &input_bounds);
};

#endif // CONTROLLER_H
