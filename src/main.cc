#include <SysModels.h>
#include <Controller.h>
#include <helper.h>
#include <eigen3/Eigen/Eigen>
#include <memory>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

using namespace Eigen;

#define NOISY true

int main()
{
    set_plot_style();

    double Ts = 0.05;
    int n_steps = 200;
    VectorXd x0(4);

    x0 << 0.5, 0, 0, 0;

    auto joint = FlexJoint(Ts, false, x0);
    auto rand_ctrl = RandInput<FlexJoint>(&joint, 9.0);

    std::vector<uint8_t> id_output;
    id_output.push_back(0);
    VectorLst limits;
    limits.push_back(Vector2d(-0.5, 0.5));
    VectorSeq traj = gen_random_setpoints(2, n_steps, id_output, limits, 0.02);

    uint32_t T_ini = 4;
    uint32_t horizon = 20;

    std::vector<double> lb_states, ub_states, lb_inputs, ub_inputs;
    lb_states = {-Infinity, -Infinity};
    ub_states = {Infinity, Infinity};
    lb_inputs = {-5.0};
    ub_inputs = {5.0};

    std::vector<std::vector<double>> state_bounds, input_bounds;
    state_bounds = {lb_states, ub_states};
    input_bounds = {lb_inputs, ub_inputs};

    MatrixXd Q = MatrixXd::Zero(2, 2);
    MatrixXd R = MatrixXd::Zero(1, 1);

    Q << 2.5, 0,
         0,  10;

    R << 0.01;

    DeePC<FlexJoint> deePC(&joint, Q, R, T_ini, horizon,
                            SMStruct::Hankel,
                            rand_ctrl.get_policy(),
                            state_bounds, input_bounds);
    
    joint.simulate(n_steps, deePC.get_policy(), traj);
    joint.plot_output();

    return 0;
}
