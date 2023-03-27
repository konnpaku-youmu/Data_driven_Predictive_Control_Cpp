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
                               std::vector<Vector2d> limits,
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