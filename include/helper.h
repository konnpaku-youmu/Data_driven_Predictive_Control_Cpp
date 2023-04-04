#ifndef HELPER_H
#define HELPER_H

#include <memory>
#include <vector>
#include <math.h>
#include <eigen3/Eigen/Eigen>
#include <casadi/casadi.hpp>
#include <matplotlibcpp.h>

using namespace Eigen;
namespace plt = matplotlibcpp;

// type aliases for convenience
typedef std::vector<casadi::MX> MXVec;
typedef std::vector<casadi::SX> SXVec;
typedef std::vector<casadi::DM> DMVec;

typedef std::vector<VectorXd> VectorSeq;
typedef std::vector<VectorXd> VectorLst;
typedef std::function<VectorXd(const VectorXd &, const VectorXd &)> ControlLaw;

// helper functions
void forward_euler(MatrixXd *A, MatrixXd *B, double Ts);

VectorXd mulvar_noise_vec(VectorXd &mean, MatrixXd &cov);

VectorSeq gen_random_setpoints(uint8_t n_outputs, uint32_t L,
                               std::vector<uint8_t> id_output,
                               VectorLst limits,
                               double sw_prob);

void set_plot_style();

void plot_vecseq(VectorSeq &vec_seq, double Ts,
                 const std::map<std::string, std::string> &kwargs);

MatrixXd hankelize(VectorSeq &vec_seq, uint32_t L);

void split_mat(const MatrixXd &mat, uint32_t split_pos,
               casadi::Matrix<double> &upper,
               casadi::Matrix<double> &lower);

void vec2MX(const VectorXd &vec, casadi::MX &casadi_vec);

void MX2vec(const casadi::MX &casadi_vec, VectorXd &vec);

void vecseq2MX(const VectorSeq &seq, casadi::MX &casadi_mat);

void eigmat2DM(const MatrixXd &mat, casadi::DM &casadi_mat);

#endif
