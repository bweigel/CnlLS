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



/*
 * Competetive Inhibition
 * inhibitor concentration should be given in the param vector (last value)
 */

