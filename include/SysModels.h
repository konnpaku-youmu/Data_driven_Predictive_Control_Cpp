#ifndef SYS_MODEL_H
#define SYS_MODEL_H

#include "SysBase.h"
#include <math.h>

using namespace Eigen;

class InvertedPendulum : public LinearSystem
{
private:
    MatrixXd A = MatrixXd::Zero(4, 4);
    MatrixXd B = MatrixXd::Zero(4, 1);
    MatrixXd C = MatrixXd::Zero(2, 4);
    MatrixXd D = MatrixXd::Zero(2, 1);

    VectorXd noise_cov = VectorXd::Zero(2);

    double Rm = 2.6;
    double Km = 0.00767;
    double Kb = 0.00767;
    double Kg = 3.7;
    double M = 0.455;
    double l = 0.305;
    double m = 0.210;
    double r = 0.635e-2;
    double g = 9.81;

    double a_21 = -m * g / M;
    double a_22 = -(Kg * Kg * Kb * Km) / (M * Rm * r * r);
    double a_31 = (M + m) * g / (l * M);
    double a_32 = (Kg * Kg * Kb * Km) / (l * M * Rm * r * r);

    double b_2 = (Kg * Km) / (M * Rm * r);
    double b_3 = -(Kg * Km) / (l * M * Rm * r);

public:
    InvertedPendulum(double Ts, bool isNoisy, VectorXd &x0)
    {
        A << 0, 0, 1, 0,
            0, 0, 0, 1,
            0, a_21, a_22, 0,
            0, a_31, a_32, 0;

        B << 0,
            0,
            b_2,
            b_3;

        C << 1, 0, 0, 0,
            0, 1, 0, 0;

        D << 0,
            0;

        noise_cov << 9e-6, 1e-4;

        lb_output << -1.0, -M_PI;
        ub_output << 1.0, M_PI;

        lb_input << -6.0;
        ub_input << 6.0;

        this->_build_system_model(Ts, isNoisy, noise_cov);
        this->_set_init_states(x0);
    }

    VectorXd lb_output = VectorXd::Zero(2);
    VectorXd ub_output = VectorXd::Zero(2);
    VectorXd lb_input = VectorXd::Zero(1);
    VectorXd ub_input = VectorXd::Zero(1);

    MatrixXd *getA() { return &A; }
    MatrixXd *getB() { return &B; }
    MatrixXd *getC() { return &C; }
    MatrixXd *getD() { return &D; }
};

class FlexJoint : public LinearSystem
{
private:
    MatrixXd A = MatrixXd::Zero(4, 4);
    MatrixXd B = MatrixXd::Zero(4, 1);
    MatrixXd C = MatrixXd::Zero(2, 4);
    MatrixXd D = MatrixXd::Zero(2, 1);

    VectorXd noise_cov = VectorXd::Zero(1);

    double Rm = 2.6;
    double Km = 0.00767;
    double Kb = 0.00767;
    double Kg = 70;
    double Jl = 0.0059;
    double Jh = 0.0021;
    double Ks = 1.60956;

    double a21 = Ks / Jh;
    double a22 = -(Kg * Kg * Km * Kb) / (Jh * Rm);
    double a31 = -(Jl + Jh) * Ks / (Jl * Jh);
    double a32 = (Kg * Kg * Km * Kb) / (Jh * Rm);
    double b2 = (Km * Kg) / (Jh * Rm);
    double b3 = (-Km * Kg) / (Rm * Jh);

public:
    FlexJoint(double Ts, bool isNoisy, VectorXd &x0)
    {
        A << 0, 0, 1, 0,
            0, 0, 0, 1,
            0, a21, a22, 0,
            0, a31, a32, 0;

        B << 0,
            0,
            b2,
            b3;

        C << 1, 0, 0, 0,
            0, 1, 0, 0;

        D << 0,
            0;

        this->_build_system_model(Ts, isNoisy, noise_cov);
        this->_set_init_states(x0);
    }

    VectorXd lb_output = VectorXd::Zero(2);
    VectorXd ub_output = VectorXd::Zero(2);
    VectorXd lb_input = VectorXd::Zero(1);
    VectorXd ub_input = VectorXd::Zero(1);

    MatrixXd *getA() { return &A; }
    MatrixXd *getB() { return &B; }
    MatrixXd *getC() { return &C; }
    MatrixXd *getD() { return &D; }
};

class IPNonlinear : public NonlinearSystem
{
    using NonlinearSystem::NonlinearSystem;
};

#endif
