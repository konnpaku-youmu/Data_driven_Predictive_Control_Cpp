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

    std::vector<uint8_t> id_output;
    id_output.push_back(0);
    VectorLst limits;
    limits.push_back(Vector2d(-0.5, 0.5));

    VectorSeq traj = gen_random_setpoints(4, n_steps, id_output, limits, 0.01);

    InvertedPendulum IP(Ts, NOISY, x0);

    MatrixXd Q = MatrixXd::Zero(4, 4);
    MatrixXd R = MatrixXd::Zero(1, 1);

    Q << 1.5, 0, 0, 0,
        0, 10, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

    R << 0.05;

    LQR<InvertedPendulum> lqr(&IP, Q, R);

    uint32_t T_ini = 4;
    uint32_t horizon = 20;

    DeePC<InvertedPendulum> deePC(&IP, Q, R, T_ini, horizon,
                                  SMStruct::Hankel, 
                                  lqr.get_policy(),
                                  limits, limits);

    IP.plot_output();

    return 0;
}
