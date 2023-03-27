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

template <typename T>
class Controller
{
protected:
    T* model;

    VectorSeq ref_traj;

    virtual VectorXd _policy(const VectorXd &, const VectorXd &) = 0;


public:
    Controller(T* model_ptr);

    auto get_policy() -> ControlLaw;

    void show_reference_traj();
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

    VectorXd _policy(const VectorXd &xk, const VectorXd &rk);

    void _build_opti_problem();

public:
    MPC(T* model_ptr, const MatrixXd &Q, const MatrixXd &R);
};

template <typename T>
class DeePC
{
private:
    ControlLaw init_controller;

    casadi::Function solver;

public:
    DeePC(T *model_ptr, const MatrixXd &Q, const MatrixXd &R,
        uint32_t T_ini, uint32_t horizon, SMStruct sm_struct,
        ControlLaw init_policy, VectorLst init_ref_traj_bounds);
};


#endif // CONTROLLER_H