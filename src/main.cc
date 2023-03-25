#include <SysModels.h>
#include <Controller.h>
#include <eigen3/Eigen/Eigen>

#include <matplot/matplot.h>

using namespace Eigen;

#define NOISY true

int main()
{
    double Ts = 0.05;
    int n_steps = 100;
    VectorXd x0(4);

    x0 << 0.5, 0, 0, 0;

    std::vector<VectorXd> traj(n_steps, VectorXd::Zero(4));

    InvertedPendulum IP(Ts, NOISY, x0);

    MatrixXd Q = MatrixXd::Zero(4, 4);
    MatrixXd R = MatrixXd::Zero(1, 1);

    Q << 6, 0, 0, 0,
         0, 10, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

    R << 0.05;

    LQR<InvertedPendulum> lqr(&IP, Q, R);

    IP.simulate(n_steps, lqr.get_policy(), traj);
    
    IP.plot_output("Time (s)", "Output");

    return 0;
}
