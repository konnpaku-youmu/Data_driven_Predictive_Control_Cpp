#ifndef SYS_BASE_H
#define SYS_BASE_H

#include <iostream>
#include <functional>
#include <eigen3/Eigen/Eigen>
#include <casadi/casadi.hpp>
#include "helper.h"

using namespace Eigen;

class System
{
protected:
    double Ts;

    VectorSeq x;
    VectorSeq y;
    VectorSeq u;

    uint8_t n_states;
    uint8_t n_inputs;
    uint8_t n_outputs;

    bool isNoisy;
    VectorXd σ_y;

    VectorXd _meas_noise();

    void _set_init_states(VectorXd &);

    virtual void _build_system_model() {}

    virtual VectorXd _f(VectorXd &, VectorXd &) = 0;

    virtual VectorXd _output(VectorXd &, VectorXd &) = 0;

public:
    void simulate(uint32_t n_steps, ControlLaw policy,
                  const VectorSeq &ref_traj);

    double get_Ts();

    void plot_output();
};

class LinearSystem : public System
{
protected:
    void _build_system_model(double Ts, bool isNoisy, const VectorXd &noise_cov);

public:
    virtual MatrixXd *getA() = 0;

    virtual MatrixXd *getB() = 0;

    virtual MatrixXd *getC() = 0;

    virtual MatrixXd *getD() = 0;

    VectorXd _f(VectorXd &, VectorXd &);

    VectorXd _output(VectorXd &, VectorXd &);
};

class NonlinearSystem : public System
{
private:
    /* data */
public:
    NonlinearSystem(/* args */);

    ~NonlinearSystem();
};

#endif
