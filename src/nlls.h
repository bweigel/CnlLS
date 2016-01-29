/*
 * nlls.h
 *
 *  Created on: Jan 28, 2016
 *      Author: mori
 */

#ifndef NLLS_H_
#define NLLS_H_

#include<vector>
#include<string>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_multifit_nlin.h>

struct data {
    size_t n;
    double * x;
    double * y;
    double * sigma;
};

struct simulation {
    std::vector<double> x, y, weights, sigma, param;
    size_t p;
    size_t n;
};

#define N 80

// Class definition
class Cnlm {
public:
    Cnlm(const simulation xy_data);
    int setXY(const simulation xy_data);
    const void printSummary();
    int fitModel(std::vector<double> x_init,
            int (*f)(const gsl_vector *, void *, gsl_vector *),
            int (*df)(const gsl_vector *, void *, gsl_matrix *));
    int fitModel(std::vector<double> x_init,
            int (*f)(const gsl_vector *, void *, gsl_vector *));
private:
    int fitModel_backend(std::vector<double> x_init, gsl_multifit_function_fdf &FUN);
    double m_sumsq, m_DOF, m_c;
    std::vector<double> m_param, m_ERR;
    struct resnorm {
        double chi, chi0;
    } m_resnorm;
    std::string m_solvername;
    unsigned niter, n_fun_eval, n_jacobian_eval;
    int m_info, m_status;
    simulation m_data;
};

void print_state(size_t iter, gsl_multifit_fdfsolver * s);

// function to simulate data
simulation expb_simulate(std::vector<double> x, std::vector<double> param,
        double si);
simulation MM_simulate(std::vector<double> x, std::vector<double> param,
        double si); // Y = Vmax*X/(Km+X)	--> parameters c0 = Vmax, c1 = Km
simulation Hill_simulate(std::vector<double> x, std::vector<double> param,
        double si);	// Y = Vmax*x^n/(Km + x^n) --> parameters c0 = Vmax, c1 = Km, c2 = n
simulation SubInh_simulate(std::vector<double> x, std::vector<double> param,
        double si);	// Y = Vmax'*x/(Km'+x+x^2/Ksi) --> parameters c0 = Vmax, c1 = Km, c2 = KSi
simulation CompInh_simulate(std::vector<double> x, std::vector<double> param, double si);

// fit functions
int expb_f(const gsl_vector * x, void *data, gsl_vector * f);
int expb_df(const gsl_vector * x, void *data, gsl_matrix * J);
int MM_f(const gsl_vector * x, void *data, gsl_vector * f);
int MM_df(const gsl_vector * x, void *data, gsl_matrix * J);
int Hill_f(const gsl_vector * x, void *data, gsl_vector * f);
int Hill_df(const gsl_vector * x, void *data, gsl_matrix * J);
int SubInh_f(const gsl_vector * x, void *data, gsl_vector * f);
int SubInh_df(const gsl_vector * x, void *data, gsl_matrix * J);

#endif /* NLLS_H_ */
