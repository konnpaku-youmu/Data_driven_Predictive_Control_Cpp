#include <SysBase.h>
#include <helper.h>
#include <eigen3/Eigen/Eigen>

#include <matplot/matplot.h>

using namespace Eigen;

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

void System::simulate(uint32_t n_steps,
                      std::function<VectorXd(VectorXd &, VectorXd &)> policy,
                      std::vector<VectorXd> &ref_traj)
{
    VectorXd u_k, x_next, y_next;

    for (int k = 1; k < n_steps; k++)
    {
        u_k = policy(this->x.back(), ref_traj[k]);
        x_next = _f(this->x.back(), u_k);
        y_next = _output(x_next, u_k);

        this->x.push_back(x_next);
        this->y.push_back(y_next);
        this->u.push_back(u_k);
    }

    return;
}

void System::plot_output(std::string xlabel, std::string ylabel)
{
    std::vector<double> x, y;

    // plot every state as a separate line
    for (int i = 0; i < this->n_outputs; i++)
    {
        x.clear();
        y.clear();

        for (int k = 0; k < this->y.size(); k++)
        {
            x.push_back(k * this->Ts);
            y.push_back(this->y[k](i));
        }

        matplot::plot(x, y, "-o")->line_width(2);
        matplot::hold(matplot::on);
    }
    
    matplot::title("Output");
    matplot::xlabel(xlabel);
    matplot::ylabel(ylabel);
    matplot::legend();
    matplot::show();

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
