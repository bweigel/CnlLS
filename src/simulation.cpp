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

/*
 * Wrapper for data simulation
 */

simulation simulate(std::vector<double> x, std::vector<double> param, double si, double (*FUN)(const double, const std::vector<double>)){

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
    	// pass x with parameters to the function specified in the arguments
        double yi = FUN(x[i], out.param);
        double s = si;
        double dy = gsl_ran_gaussian(r, s);

        out.y.push_back(yi + dy);
        out.weights.push_back(1.0/(s*s));
    }
    // deallocate memory from RNG
    gsl_rng_free (r);
    return out;
}

/*
 * Simulation functions
 */

double MM(const double x, const std::vector<double> param){
	return param[0] * x/(param[1] + x);
}

double Hill(const double x, const std::vector<double> param){
	return param[0] * pow(x, param[2])/(param[1] + pow(x, param[2]));
}

double CompInh(const double x, const std::vector<double> param){
	return param[0] * x/(param[1] *(1 + param[3]/param[2]) + x);
}

double SubInh(const double x, const std::vector<double> param){
	return param[0] * x/(param[1] + x + pow(x,2.0)/param[2]);
}

double expb(const double x, const std::vector<double> param){
	return param[0]+ param[1] * exp (param[2] * x);
}




