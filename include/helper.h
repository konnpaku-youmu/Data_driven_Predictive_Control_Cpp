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

typedef std::vector<casadi::MX> MXVec;
typedef std::vector<casadi::SX> SXVec;
typedef std::vector<casadi::DM> DMVec;

typedef std::vector<VectorXd> VectorSeq;
typedef std::vector<VectorXd> VectorLst;
typedef std::function<VectorXd(const VectorXd &, const VectorXd &)> ControlLaw;

void forward_euler(MatrixXd *A, MatrixXd *B, double Ts);

VectorXd mulvar_noise_vec(VectorXd &mean, MatrixXd &cov);

VectorSeq gen_random_setpoints(uint8_t n_outputs, uint32_t L,
                               std::vector<uint8_t> id_output,
                               std::vector<Vector2d> limits,
                               double sw_prob);

void set_plot_style();

void plot_vecseq(VectorSeq &vec_seq, double Ts,
                 const std::map<std::string, std::string> &kwargs);

#endif
