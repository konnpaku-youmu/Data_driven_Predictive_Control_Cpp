#include <SysBase.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <boost/format.hpp>

#include "matplotlibcpp.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

VectorXd System::_meas_noise()
{
    VectorXd mean(n_outputs);
    MatrixXd cov;

    mean.setZero();
    cov = MatrixXd(σ_y.asDiagonal());

    VectorXd noise = mulvar_noise_vec(mean, cov);

    return noise;
}

void System::_set_init_states(VectorXd &x0)
{
    this->x.push_back(x0);

    VectorXd u0 = VectorXd::Zero(n_inputs, 1);
    this->u.push_back(u0);

    VectorXd y0 = this->_output(x0, u0);
    this->y.push_back(y0);

    return;
}

void System::simulate(uint32_t n_steps, ControlLaw policy,
                      const VectorSeq &ref_traj)
{
    VectorXd u_k, x_next, y_next;

    for (int k = 0; k < n_steps; k++)
    {
        u_k = policy(this->x.back(), ref_traj.at(k));
        x_next = _f(this->x.back(), u_k);
        y_next = _output(x_next, u_k);

        this->x.push_back(x_next);
        this->y.push_back(y_next);
        this->u.push_back(u_k);
    }

    return;
}

double System::get_Ts()
{
    return this->Ts;
}

void System::plot_output()
{
    std::vector<double> t, y;

    for (int k = 0; k < this->n_outputs; k++)
    {
        t.clear();
        y.clear();

        for (int i = 1; i < this->y.size(); i++)
        {
            t.push_back((i - 1) * Ts);
            y.push_back(this->y[i](k));
        }

        std::string label = (boost::format("$y_%1%$") % k).str();

        plt::plot(t, y,
                  {{"label", label},
                   {"linewidth", "1.2"},
                   {"drawstyle", "steps-mid"}});
    }

    plt::xlabel("Time (s)");
    plt::title("Output");
    plt::legend({{"loc", "best"}, {"fontsize", "medium"}});
    plt::show();

    return;
}

void LinearSystem::_build_system_model(double Ts, bool isNoisy,
                                       const VectorXd &noise_cov)
{
    this->Ts = Ts;

    // set the dimensions of the system
    this->n_states = getA()->rows();
    this->n_inputs = getB()->cols();
    this->n_outputs = getC()->rows();

    this->isNoisy = isNoisy;
    this->σ_y = (isNoisy) ? noise_cov : VectorXd::Zero(n_outputs);

    // discretize the system
    forward_euler(getA(), getB(), Ts);

    return;
}

VectorXd LinearSystem::_f(VectorXd &x, VectorXd &u)
{
    return (*getA() * x + *getB() * u);
}

VectorXd LinearSystem::_output(VectorXd &x,
                               VectorXd &u)
{
    return (*getC() * x + *getD() * u + _meas_noise());
}

NonlinearSystem::NonlinearSystem(/* args */)
{
}

NonlinearSystem::~NonlinearSystem()
{
}
