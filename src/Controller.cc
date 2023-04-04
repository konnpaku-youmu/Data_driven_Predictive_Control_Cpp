#include "Controller.h"
#include "SysModels.h"

#include <casadi/casadi.hpp>
#include <eigen3/Eigen/Eigen>

using namespace Eigen;

template <typename T>
Controller<T>::Controller(T *model_ptr)
{
    this->model = model_ptr;
}

template <typename T>
auto Controller<T>::get_policy() -> ControlLaw
{
    return std::bind(&Controller<T>::_policy, this,
                     std::placeholders::_1, std::placeholders::_2);
}

template <typename T>
void Controller<T>::show_reference_traj()
{
    plot_vecseq(ref_traj, this->model->get_Ts(),
                {{"drawstyle", "steps-mid"},
                 {"color", "grey"},
                 {"linewidth", "1.5"},
                 {"linestyle", "--"}});
    return;
}

template <typename T>
RandInput<T>::RandInput(T *model_ptr, double cov) : Controller<T>(model_ptr)
{
    this->input_cov = cov;
}

template <typename T>
VectorXd RandInput<T>::_policy(const VectorXd &xk, const VectorXd &rk)
{
    int nu = this->model->get_nu();
    VectorXd mean = VectorXd::Zero(nu);
    MatrixXd cov = VectorXd::Identity(nu, nu) * input_cov;
    return mulvar_noise_vec(mean, cov);
}

template <typename T>
LQR<T>::LQR(T *model_ptr, const MatrixXd &Q, const MatrixXd &R) : Controller<T>(model_ptr)
{
    this->Q = Q;
    this->R = R;

    _compute_K();
}

template <typename T>
void LQR<T>::_compute_K()
{
    MatrixXd A = *(this->model->getA());
    MatrixXd B = *(this->model->getB());

    MatrixXd P = Q;

    for (int i = 0; i < 100; i++)
    {
        P = Q + A.transpose() * P * A - A.transpose() * P * B * (R + B.transpose() * P * B).inverse() * B.transpose() * P * A;
    }

    K = -(R + B.transpose() * P * B).inverse() * B.transpose() * P * A;
}

template <typename T>
VectorXd LQR<T>::_policy(const VectorXd &xk, const VectorXd &rk)
{
    this->ref_traj.push_back(rk);
    return K * (xk - rk);
}

template <typename T>
MPC<T>::MPC(T *model_ptr, const MatrixXd &Q, const MatrixXd &R) : Controller<T>(model_ptr)
{
    this->Q = Q;
    this->R = R;
}

template <typename T>
void MPC<T>::_build_opti_problem()
{
}

template <typename T>
VectorXd MPC<T>::_policy(const VectorXd &xk, const VectorXd &rk)
{

    return VectorXd();
}

template <typename T>
DeePC<T>::DeePC(T *model_ptr, const MatrixXd &Q, const MatrixXd &R,
                uint32_t T_ini, uint32_t horizon,
                SMStruct sm_struct,
                const ControlLaw &init_policy,
                const std::vector<std::vector<double>> &state_bounds,
                const std::vector<std::vector<double>> &input_bounds) : Controller<T>(model_ptr)
{
    eigmat2DM(Q, this->Q);
    eigmat2DM(R, this->R);

    this->T_ini = T_ini;
    this->horizon = horizon;

    this->init_controller = init_policy;

    this->signal_mat = sm_struct;

    if (std::is_same<T, LinearSystem>::value)
    {
        this->ocp_type = OCPType::CANONICAL;
    }
    else
    {
        this->ocp_type = OCPType::REGULARIZED;
    }

    _build_controller();
    _set_constraints();
    _build_loss_func();

    // vertcat all opti_vars
    std::vector<casadi::MX> opti_vars_vec;
    for (auto &var : this->opti_vars)
    {
        opti_vars_vec.push_back(var.second);
    }
    auto opti_vars_MX = casadi::MX::vertcat(opti_vars_vec);

    std::vector<casadi::MX> opti_params_vec;
    for (auto &param : this->opti_params)
    {
        opti_params_vec.push_back(param.second);
    }
    auto opti_params_MX = casadi::MX::vertcat(opti_params_vec);

    problem["x"] = opti_vars_MX;
    problem["f"] = loss;
    problem["g"] = traj_constraint;
    problem["p"] = opti_params_MX;

    lbx = state_bounds[0];
    ubx = state_bounds[1];
    lbu = input_bounds[0];
    ubu = input_bounds[1];

    casadi::Dict opts;
    opts["ipopt.tol"] = 1e-5;
    opts["ipopt.max_iter"] = 100;
    opts["print_time"] = true;
    opts["expand"] = true;

    this->solver = new casadi::Function(casadi::nlpsol("solver", "ipopt",
                                                       this->problem, opts));
}

template <typename T>
void DeePC<T>::_build_controller()
{
    uint32_t L = T_ini + horizon;

    uint8_t nx = this->model->get_nx();
    uint8_t nu = this->model->get_nu();
    uint8_t ny = this->model->get_ny();

    uint32_t init_steps;

    if (this->signal_mat == SMStruct::Hankel)
    {
        init_steps = (nu + 1) * (nx + L) - 1;
    }

    // initializing
    std::vector<uint8_t> _nx_vec(1, 0);
    VectorLst _lims;
    _lims.push_back(Vector2d(this->model->lb_output(0), this->model->ub_output(0)));
    VectorSeq traj_init = gen_random_setpoints(nx, init_steps, _nx_vec, _lims, 0.1);

    this->model->simulate(init_steps, this->init_controller, traj_init);

    return;
}

template <typename T>
void DeePC<T>::_set_constraints()
{
    int L = T_ini + horizon;

    int nx = this->model->get_nx();
    int nu = this->model->get_nu();
    int ny = this->model->get_ny();

    VectorSeq u = this->model->get_input_seq(1, -1);
    VectorSeq y = this->model->get_output_seq(1, -1);

    // construct signal matrix
    MatrixXd M_u, M_y;
    if (this->signal_mat == SMStruct::Hankel)
    {
        M_u = hankelize(u, L);
        M_y = hankelize(y, L);
    }

    split_mat(M_u, nu * T_ini, U_p, U_f);
    split_mat(M_y, ny * T_ini, Y_p, Y_f);

    // construct OCP variables
    casadi::MX opt_u = casadi::MX::sym("u", nu * horizon, 1);
    casadi::MX opt_y = casadi::MX::sym("y", ny * horizon, 1);
    casadi::MX opt_g = casadi::MX::sym("g", this->U_f.size2());

    this->opti_vars = {{"u", opt_u},
                       {"y", opt_y},
                       {"g", opt_g}};

    // construct OCP parameters
    casadi::MX u_ini = casadi::MX::sym("u_ini", nu * T_ini, 1);
    casadi::MX y_ini = casadi::MX::sym("y_ini", ny * T_ini, 1);
    casadi::MX ref = casadi::MX::sym("ref", ny, 1);

    this->opti_params = {{"u_ini", u_ini},
                         {"y_ini", y_ini},
                         {"ref", ref}};

    // Trajectory constraint:
    casadi::MX A = casadi::MX::vertcat({U_p, U_f, Y_p, Y_f});
    casadi::MX b = casadi::MX::vertcat({opti_params.at("u_ini"),
                                        opti_params.at("y_ini"),
                                        opti_vars.at("u"),
                                        opti_vars.at("y")});

    this->traj_constraint = casadi::MX::mtimes(A, opti_vars.at("g")) - b;

    return;
}

template <typename T>
void DeePC<T>::_build_loss_func()
{
    loss = 0;

    casadi::MX yk, uk;

    for (int k = 0; k < horizon; k++)
    {
        yk = opti_vars.at("y")(k) - opti_params.at("ref");
        uk = opti_vars.at("u")(k);

        loss += (1 / 2) * casadi::MX::sum1(casadi::MX::mtimes({yk.T(), Q, yk})) +
                (1 / 2) * casadi::MX::sum1(casadi::MX::mtimes({uk.T(), R, uk}));
    }

    if (ocp_type == OCPType::REGULARIZED)
    {
        // Add regularization terms
        int λs = 160;
    }

    return;
}

template <typename T>
VectorXd DeePC<T>::_policy(const VectorXd &xk, const VectorXd &rk)
{
    casadi::MX y_ini, u_ini;
    VectorSeq y_tail = this->model->get_output_seq(-T_ini, -1);
    VectorSeq u_tail = this->model->get_input_seq(-T_ini, -1);

    vecseq2MX(u_tail, opti_params.at("u_ini"));
    vecseq2MX(y_tail, opti_params.at("y_ini"));
    vec2MX(rk, opti_params.at("ref"));

    this->ref_traj.push_back(rk);

    casadi::MXDict arg;

    arg["lbg"] = 0;
    arg["ubg"] = 0;
    arg["p"] = casadi::MX::vertcat({opti_params.at("u_ini"),
                                    opti_params.at("y_ini"),
                                    opti_params.at("ref")});

    this->solver->call(arg, result);

    casadi::MX opt_g = _unpack_opti_g();
    casadi::MX opt_u = casadi::MX::mtimes(this->U_f, opt_g);

    VectorXd u = VectorXd::Zero(horizon);
    MX2vec(opt_u, u);

    std::cout << u << std::endl;

    int nu = this->model->get_nu();
    return u.head(nu);
}

template <typename T>
casadi::MX DeePC<T>::_unpack_opti_g()
{
    int L = this->U_f.size2();

    casadi::MX opti = result.at("x");
    casadi::MX g = casadi::MX::zeros(L, 1);

    // last L elements are g
    for (int i = 0; i < L; i++)
    {
        g(i) = opti(opti.size1() - L + i);
    }

    return g;
}

// declare all the template classes
template class Controller<InvertedPendulum>;
template class Controller<FlexJoint>;

template class LQR<InvertedPendulum>;
template class MPC<InvertedPendulum>;
template class DeePC<InvertedPendulum>;

template class RandInput<FlexJoint>;
template class DeePC<FlexJoint>;
