#include <helper.h>

using namespace Eigen;

void forward_euler(MatrixXd *A, MatrixXd *B, double Ts)
{
    *A = MatrixXd::Identity(A->rows(), A->cols()) + Ts * (*A);
    *B = Ts * (*B);

    return;
}

VectorXd mulvar_noise_vec(VectorXd &mean, MatrixXd &cov)
{
    int n_outputs = mean.rows();
    // Obtain random normal variables for each output dimension
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // Initialize output noise vector
    VectorXd noise(n_outputs);

    // Use Cholesky decomposition of covariance matrix to generate correlated noise
    LLT<MatrixXd> llt_covariance(cov);
    MatrixXd L = llt_covariance.matrixL();

    for (int i = 0; i < n_outputs; i++)
    {
        noise(i) = dist(gen);
    }

    VectorXd mulvar_noise = mean + L * noise;

    return mulvar_noise;
}

VectorSeq gen_random_setpoints(uint8_t n_outputs, uint32_t L,
                               std::vector<uint8_t> id_output,
                               VectorLst limits,
                               double sw_prob)
{
    VectorSeq sp;

    // Obtain random normal variables for each output dimension
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Initialize output noise vector
    VectorXd setpoint = VectorXd::Zero(n_outputs);

    for (int i = 0; i < L; i++)
    {
        if (dist(gen) <= sw_prob)
        {
            for (int j = 0; j < n_outputs; j++)
            {
                if (std::find(id_output.begin(), id_output.end(), j) != id_output.end())
                {
                    std::uniform_real_distribution<double> distn(limits[j](0), limits[j](1));
                    setpoint(j) = distn(gen);
                }
            }
        }

        sp.push_back(setpoint);
    }

    return sp;
}

void set_plot_style()
{
    plt::rcparams({{"font.family", "serif"}});
    plt::rcparams({{"font.size", "14"}});
    plt::rcparams({{"text.usetex", "1"}});
    plt::grid(true);

    return;
}

void plot_vecseq(VectorSeq &vec_seq, double Ts,
                 const std::map<std::string, std::string> &kwargs)
{
    int n_outputs = vec_seq[0].rows();
    int n_steps = vec_seq.size();

    std::vector<double> t(n_steps);
    std::vector<std::vector<double>> y(n_outputs);

    for (int i = 0; i < n_steps; i++)
    {
        t[i] = i * Ts;
        for (int j = 0; j < n_outputs; j++)
        {
            y[j].push_back(vec_seq[i](j));
        }
    }

    for (int i = 0; i < n_outputs; i++)
    {
        if (std::all_of(y[i].begin(), y[i].end(), [](double x)
                        { return x == 0; }))
            continue;
        plt::plot(t, y[i], kwargs);
    }
    return;
}

MatrixXd hankelize(VectorSeq &vec_seq, uint32_t L)
{

    assert(L <= vec_seq.size());

    int n_outputs = vec_seq[0].rows();
    int n_steps = vec_seq.size();

    MatrixXd H(n_outputs * L, n_steps - L + 1);

    for (int i = 0; i < n_steps - L + 1; i++)
    {
        for (int j = 0; j < L; j++)
        {
            H.block(n_outputs * j, i, n_outputs, 1) = vec_seq[i + j];
        }
    }

    return H;
}

void split_mat(const MatrixXd &mat, uint32_t split_pos,
               casadi::Matrix<double> &upper,
               casadi::Matrix<double> &lower)
{
    int n_rows = mat.rows();
    int n_cols = mat.cols();

    upper = casadi::DM::zeros(split_pos, n_cols);
    lower = casadi::DM::zeros(n_rows - split_pos, n_cols);

    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            if (i < split_pos)
            {
                upper(i, j) = mat(i, j);
            }
            else
            {
                lower(i - split_pos, j) = mat(i, j);
            }
        }
    }

    return;
}

void vec2MX(const VectorXd &vec, casadi::MX &casadi_vec)
{
    int n_rows = vec.rows();

    casadi_vec = casadi::MX::zeros(n_rows, 1);

    for (int i = 0; i < n_rows; i++)
    {
        casadi_vec(i, 0) = vec(i);
    }

    return;
}

void MX2vec(const casadi::MX &casadi_vec, VectorXd &vec)
{
    int n_rows = casadi_vec.rows();

    vec = VectorXd::Zero(n_rows);

    for (int i = 0; i < n_rows; i++)
    {
        vec(i) = casadi::MX::abs();
    }

    return;
}

void vecseq2MX(const VectorSeq &seq, casadi::MX &casadi_mat)
{
    int n_rows = seq[0].rows();
    int n_cols = seq.size();

    casadi_mat = casadi::MX::zeros(n_rows * n_cols, 1);

    for (int i = 0; i < n_cols; i++)
    {
        for (int j = 0; j < n_rows; j++)
        {
            casadi_mat(i * n_rows + j, 0) = seq[i](j);
        }
    }

    return;
}

void eigmat2DM(const MatrixXd &mat, casadi::DM &casadi_mat)
{
    int n_rows = mat.rows();
    int n_cols = mat.cols();

    casadi_mat = casadi::DM::zeros(n_rows, n_cols);

    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            casadi_mat(i, j) = mat(i, j);
        }
    }

    return;
}
