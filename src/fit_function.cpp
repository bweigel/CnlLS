/*
 * fit_function.cpp
 *
 *  Created on: Jan 28, 2016
 *      Author: mori
 */


#include"nlls.h"


int expb_f (const gsl_vector * x, void *data, gsl_vector * f)
{
    size_t n = ((struct data *)data)->n;
    double *y = ((struct data *)data)->y;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double A = gsl_vector_get (x, 0);
    double lambda = gsl_vector_get (x, 1);
    double b = gsl_vector_get (x, 2);
    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Model Yi = A * exp(-lambda * i) + b */
        double t = xin[i], s = sigma[i];
        double Yi = A * exp (-lambda * t) + b;
        gsl_vector_set (f, i, (Yi - y[i])/s);
    }
    return GSL_SUCCESS;
}

int expb_df (const gsl_vector * x, void *data,  gsl_matrix * J)
{
    size_t n = ((struct data *)data)->n;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double A = gsl_vector_get (x, 0);
    double lambda = gsl_vector_get (x, 1);
    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        /* where fi = (Yi - yi)/sigma[i],
         */
        /*
Yi = A * exp(-lambda * i) + b */
        /* and the xj are the parameters (A,lambda,b) */
        double t = xin[i], s = sigma[i];
        double e = exp(-lambda * t);
        gsl_matrix_set (J, i, 0, e/s);
        gsl_matrix_set (J, i, 1, (-t * A * e)/s);
        gsl_matrix_set (J, i, 2, 1.0/s);
    }
    return GSL_SUCCESS;
}

simulation expb_simulate(std::vector<double> x, std::vector<double> param, double si){

    gsl_rng * r;
    const gsl_rng_type * type;
    gsl_rng_env_setup();
    unsigned n = x.size();

    simulation out;
    out.p = param.size();
    out.n = n;
    out.x.assign(x.begin(), x.end());
    out.sigma.assign(n, si);

    for(unsigned i = 0; i < out.p; i++){
        out.param.push_back(param[i]);
    }

    // start RNG
    type = gsl_rng_default;
    r = gsl_rng_alloc (type);

    // simulate data with some noise
    for (unsigned i = 0; i < n; i++)
    {
        double yi = out.param[0]+ out.param[1] * exp (out.param[2] * x[i]);
        double s = si;
        double dy = gsl_ran_gaussian(r, s);

        out.y.push_back(yi + dy);
        out.weights.push_back(1.0/(s*s));
    }

    gsl_rng_free (r);
    return out;
}

/*
 * Michaelis Menten
 */

int MM_f (const gsl_vector * x, void *data, gsl_vector * f)
{
    size_t n = ((struct data *)data)->n;
    double *y = ((struct data *)data)->y;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double Vmax = gsl_vector_get (x, 0);
    double Km = gsl_vector_get (x, 1);

    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Model Yi = A * exp(-lambda * i) + b */
        double t = xin[i], s = sigma[i];
        double Yi = Vmax * t/(Km + t);
        gsl_vector_set (f, i, (Yi - y[i])/s);
    }
    return GSL_SUCCESS;
}

int MM_df (const gsl_vector * x, void *data,  gsl_matrix * J)
{
    size_t n = ((struct data *)data)->n;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double Vmax = gsl_vector_get (x, 0);
    double Km = gsl_vector_get (x, 1);
    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        double t = xin[i], s = sigma[i];
        double a = pow(Km + t, 2.0);
        gsl_matrix_set (J, i, 0, (t/(Km+t)) / s );
        gsl_matrix_set (J, i, 1, (-(t*Vmax)/a) / s);
    }
    return GSL_SUCCESS;
}

simulation MM_simulate(std::vector<double> x, std::vector<double> param, double si){

    gsl_rng * r;
    const gsl_rng_type * type;
    gsl_rng_env_setup();
    unsigned n = x.size();

    simulation out;
    out.p = param.size();
    out.n = n;
    out.x.assign(x.begin(), x.end());
    out.sigma.assign(n, si);

    for(unsigned i = 0; i < out.p; i++){
        out.param.push_back(param[i]);
    }

    // start RNG
    type = gsl_rng_default;
    r = gsl_rng_alloc (type);

    // simulate data with some noise
    for (unsigned i = 0; i < n; i++)
    {
        double yi = out.param[0] * x[i]/(out.param[1] + x[i]);
        double s = si;
        double dy = gsl_ran_gaussian(r, s);

        out.y.push_back(yi + dy);
        out.weights.push_back(1.0/(s*s));
    }

    gsl_rng_free (r);
    return out;
}

/*
 * Hill
 */

int Hill_f (const gsl_vector * x, void *data, gsl_vector * f)
{
    size_t n = ((struct data *)data)->n;
    double *y = ((struct data *)data)->y;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double Vmax = gsl_vector_get (x, 0);
    double Km = gsl_vector_get (x, 1);
    double h_coeff = gsl_vector_get (x, 2);

    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Model Yi = A * exp(-lambda * i) + b */
        double t = xin[i], s = sigma[i];
        double Yi = Vmax * pow(t, h_coeff)/(Km + pow(t, h_coeff));
        gsl_vector_set (f, i, (Yi - y[i])/s);
    }
    return GSL_SUCCESS;
}

int Hill_df (const gsl_vector * x, void *data,  gsl_matrix * J)
{
    size_t n = ((struct data *)data)->n;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double Vmax = gsl_vector_get (x, 0);
    double Km = gsl_vector_get (x, 1);
    double h_coeff = gsl_vector_get (x, 2);
    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        double t = xin[i], s = sigma[i];
        double a = pow(t, h_coeff);
        double b = pow(Km + a, 2.0);
        //double c = pow(t, h_coeff) * log(h_coeff);
        double c = log(h_coeff);
        double d = Km+a;
        gsl_matrix_set (J, i, 0, (a/(Km + a)) / s );
        gsl_matrix_set (J, i, 1, (-(Vmax*a)/b) / s);
      //  gsl_matrix_set (J, i, 2, c*(-Vmax * a/(b) + c/(Km+a)) / s );
        gsl_matrix_set (J, i, 2, (Vmax*a*c/d -Vmax*pow(a, 2.0)*c/pow(d, 2.0) )/ s );
    }
    return GSL_SUCCESS;
}

simulation Hill_simulate(std::vector<double> x, std::vector<double> param, double si){

    gsl_rng * r;
    const gsl_rng_type * type;
    gsl_rng_env_setup();
    unsigned n = x.size();

    simulation out;
    out.p = param.size();
    out.n = n;
    out.x.assign(x.begin(), x.end());
    out.sigma.assign(n, si);

    for(unsigned i = 0; i < out.p; i++){
        out.param.push_back(param[i]);
    }

    // start RNG
    type = gsl_rng_default;
    r = gsl_rng_alloc (type);

    // simulate data with some noise
    for (unsigned i = 0; i < n; i++)
    {
        double yi = out.param[0] * pow(x[i], out.param[2])/(out.param[1] + pow(x[i], out.param[2]));
        double s = si;
        double dy = gsl_ran_gaussian(r, s);

        out.y.push_back(yi + dy);
        out.weights.push_back(1.0/(s*s));
    }

    gsl_rng_free (r);
    return out;
}


/*
 * Substrate inhibition model v = Vmax'*x/(Km'+x+x^2/Ksi)
 */

int SubInh_f (const gsl_vector * x, void *data, gsl_vector * f)
{
    size_t n = ((struct data *)data)->n;
    double *y = ((struct data *)data)->y;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double Vmax = gsl_vector_get (x, 0);
    double Km = gsl_vector_get (x, 1);
    double Ksi = gsl_vector_get (x, 2);

    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Model Yi = A * exp(-lambda * i) + b */
        double t = xin[i], s = sigma[i];
        double Yi = Vmax * t/(Km+t+pow(t,2.0)/Ksi);
        gsl_vector_set (f, i, (Yi - y[i])/s);
    }
    return GSL_SUCCESS;
}

int SubInh_df (const gsl_vector * x, void *data,  gsl_matrix * J)
{
    size_t n = ((struct data *)data)->n;
    double *xin = ((struct data *)data)->x;
    double *sigma = ((struct data *)data)->sigma;
    double Vmax = gsl_vector_get (x, 0);
    double Km = gsl_vector_get (x, 1);
    double Ksi = gsl_vector_get (x, 2);
    size_t i;
    for (i = 0; i < n; i++)
    {
        /* Jacobian matrix J(i,j) = dfi / dxj, */
        double t = xin[i], s = sigma[i];
        double a = pow(Km+t+pow(t,2.0)/Ksi, 2.0);
        gsl_matrix_set (J, i, 0, (t/(Km + t + pow(t, 2.0)/Ksi)) / s );
        gsl_matrix_set (J, i, 1, -(Vmax*t/ a) /s );
        gsl_matrix_set (J, i, 2, (Vmax * pow(t, 3.0) / (a * pow(Ksi,2.0)))/ s );
    }
    return GSL_SUCCESS;
}

simulation SubInh_simulate(std::vector<double> x, std::vector<double> param, double si){

    gsl_rng * r;
    const gsl_rng_type * type;
    gsl_rng_env_setup();
    unsigned n = x.size();

    simulation out;
    out.p = param.size();
    out.n = n;
    out.x.assign(x.begin(), x.end());
    out.sigma.assign(n, si);

    for(unsigned i = 0; i < out.p; i++){
        out.param.push_back(param[i]);
    }

    // start RNG
    type = gsl_rng_default;
    r = gsl_rng_alloc (type);

    // simulate data with some noise
    for (unsigned i = 0; i < n; i++)
    {
        double yi = out.param[0] * x[i]/(out.param[1] + x[i] + pow(x[i],2.0)/out.param[2]);
        double s = si;
        double dy = gsl_ran_gaussian(r, s);

        out.y.push_back(yi + dy);
        out.weights.push_back(1.0/(s*s));
    }

    gsl_rng_free (r);
    return out;
}
