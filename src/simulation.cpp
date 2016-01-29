/*
 * simulation.cpp
 *
 *  Created on: Jan 29, 2016
 *      Author: mori
 */

#include"nlls.h"


/*
 * Wrapper for RNG initialization
 */

gsl_rng * RNG(){
    gsl_rng * r;
    const gsl_rng_type * type;
    gsl_rng_env_setup();

    // start RNG
    type = gsl_rng_default;
    r = gsl_rng_alloc (type);

	return r;
}


simulation MM_simulate(std::vector<double> x, std::vector<double> param, double si){

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
    gsl_rng * r = RNG();

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

simulation Hill_simulate(std::vector<double> x, std::vector<double> param, double si){

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
    gsl_rng * r = RNG();

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


simulation SubInh_simulate(std::vector<double> x, std::vector<double> param, double si){

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
    gsl_rng * r = RNG();


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

simulation CompInh_simulate(std::vector<double> x, std::vector<double> param, double si){


    unsigned n = x.size();

    simulation out;
    out.p = param.size();
    out.n = n;
    out.x.assign(x.begin(), x.end());
    out.sigma.assign(n, si);

    for(unsigned i = 0; i < out.p; i++){
        out.param.push_back(param[i]);
    }

    gsl_rng * r = RNG();
    // simulate data with some noise
    for (unsigned i = 0; i < n; i++)
    {
        double yi = out.param[0] * x[i]/(out.param[1] *(1 + out.param[3]/out.param[2]) + x[i]);	// v = Vmax*x/(Km*(1+i/Ki)+x)
        double s = si;
        double dy = gsl_ran_gaussian(r, s);

        out.y.push_back(yi + dy);
        out.weights.push_back(1.0/(s*s));
    }

    gsl_rng_free (r);
    return out;
}






