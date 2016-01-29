/*
 * class_function.cpp
 *
 *  Created on: Jan 28, 2016
 *      Author: mori
 */


#include"nlls.h"
#include<iostream>

// constructor
Cnlm::Cnlm(const simulation xy_data){
    m_data = xy_data;
}

int Cnlm::setXY(const simulation xy_data){
	m_data = xy_data;
	return 0;
}

int Cnlm::fitModel_backend(std::vector<double> x_init, gsl_multifit_function_fdf &FUN){
    const size_t n = m_data.x.size();
    const size_t p = m_data.p;
    m_info = 0;
    unsigned iter = 0;

    gsl_vector_view w = gsl_vector_view_array(&m_data.weights[0], n);

    const gsl_multifit_fdfsolver_type * T = gsl_multifit_fdfsolver_lmsder;
    gsl_multifit_fdfsolver *s;

    gsl_matrix *J = gsl_matrix_alloc(n, p);
    gsl_matrix *covar = gsl_matrix_alloc (p, p);

    gsl_vector *res_f;

    // initial values for determination of parameters
    gsl_vector_view x = gsl_vector_view_array (&x_init[0], p);

    // create new solver
    s = gsl_multifit_fdfsolver_alloc (T, n, p);

    /* initialize solver with starting point and weights */
    gsl_multifit_fdfsolver_wset (s, &FUN, &x.vector, &w.vector);

    /* compute initial residual norm */
    res_f = gsl_multifit_fdfsolver_residual(s);
    m_resnorm.chi0 = gsl_blas_dnrm2(res_f);

    print_state(iter, s);

    // solve by iteration
    do{
        iter++;
        m_status = gsl_multifit_fdfsolver_iterate (s);
        std::cout << "status = " << gsl_strerror (m_status) << std::endl;

        print_state (iter, s);

        if(m_status) break;

        m_status = gsl_multifit_test_delta (s->dx, s->x, 1e-4, 1e-4);

    }while (m_status == GSL_CONTINUE && iter < 500);

    // compute covariance matrix and best fit parameters
    gsl_multifit_fdfsolver_jac(s, J);
    gsl_multifit_covar (J, 0.0, covar);

    /* compute final residual norm */
    m_resnorm.chi = gsl_blas_dnrm2(res_f);

    m_solvername = gsl_multifit_fdfsolver_name(s);
    m_info = 0;
    niter = gsl_multifit_fdfsolver_niter(s);
    n_fun_eval = FUN.nevalf;
    n_jacobian_eval = FUN.nevaldf;
    m_DOF = n-p; //compute degrees of freedom
    m_c = GSL_MAX_DBL(1, m_resnorm.chi / sqrt(m_DOF));
    m_param.clear();
    m_ERR.clear();
    for(unsigned i = 0; i < p; i++){
        m_param.push_back(gsl_vector_get(s->x, i));
        m_ERR.push_back(gsl_matrix_get(covar,i,i));
    }
    gsl_multifit_fdfsolver_free (s);
    gsl_matrix_free (covar);
    gsl_matrix_free (J);
    return 0;
}

int Cnlm::fitModel(std::vector<double> x_init,
		int (*f)(const gsl_vector * , void *, gsl_vector *), int (*df)(const gsl_vector * , void *, gsl_matrix * )){

    const size_t n = m_data.x.size();
    const size_t p = m_data.p;

    //construct data struct to provide to gsl_multifit_function_fdf via function
    struct data d = { n, &m_data.x[0], &m_data.y[0], &m_data.sigma[0]}; // from simulated data

    gsl_multifit_function_fdf FUN;

    FUN.f = f;
    FUN.df = df;
    FUN.n = n;
    FUN.p = p;
    FUN.params = &d;

    this->fitModel_backend(x_init, FUN);

    return 0;
}

int Cnlm::fitModel(std::vector<double> x_init,
		int (*f)(const gsl_vector * , void *, gsl_vector *)){

    const size_t n = m_data.x.size();
    const size_t p = m_data.p;

    //construct data struct to provide to gsl_multifit_function_fdf via function
    struct data d = { n, &m_data.x[0], &m_data.y[0], &m_data.sigma[0]}; // from simulated data

    gsl_multifit_function_fdf FUN;

    FUN.f = f;
    FUN.df = NULL;
    FUN.n = n;
    FUN.p = p;
    FUN.params = &d;

    this->fitModel_backend(x_init, FUN);

    return 0;
}

const void Cnlm::printSummary(){
    std::cout << "# Data\nx\ty\tweight\tsigma\n";
    for(unsigned i = 0; i < m_data.x.size(); i++){
        std::cout << m_data.x[i] << "\t" << m_data.y[i]  << "\t" << m_data.weights[i]  << "\t" << m_data.sigma[i] << std::endl;
    }
    std::cout << "#\n# Summary\n# -------------------------------------------\n";
    std::cout << "# least squares fit estimates \n#\n";
    std::cout << "# \tvalue \tsd\n";
    for(unsigned i = 0; i < m_param.size(); i++){
    	std::cout << "# c" << i << "\t" << m_param[i] << "\t" << m_ERR[i] << std::endl;
    }
    std::cout << "#\n# Chisq/DOF: " << pow(m_resnorm.chi, 2.0)/m_DOF << std::endl;
}


// General functions
void print_state (size_t iter, gsl_multifit_fdfsolver * s){
    std::cout << "iteration: " << iter;
    for(unsigned i = 0; i < s->x->size; i++){
        std::cout << "\tc" << i << " = " << gsl_vector_get (s->x, i);
    }
    std::cout << "\n\t\t|f(x)| =  " << gsl_blas_dnrm2 (s->f) << std::endl;
}

