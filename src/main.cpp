/*
 * main.cpp
 *
 *  Created on: Jan 28, 2016
 *      Author: mori
 */


#include<iostream>
#include"nlls.h"

int main() {
    // simulate some data
    std::vector<double> in;
    for(unsigned i = 0; i < N; i++){
        in.push_back(i/3.0);
    }
    std::vector<double> exp_par = {23,24};
    simulation exp_sim = simulate(in, exp_par, 0.05, MM);

    // create Clnm object
    Cnlm mynLM(exp_sim);
    // fit with initial parameters
    std::vector<double> initial_x = {1, 5};
    mynLM.fitModel(initial_x, &MM_f, &MM_df);
    mynLM.printSummary();

    std::vector<double> MM_par = {10, 5, 25};
    simulation MM_sim = simulate(in, MM_par, 0.05, SubInh);

    // create Clnm object
    mynLM.setXY(MM_sim);
    // fit with initial parameters
    std::vector<double> initial_MM = {9, 4, 1};
    mynLM.fitModel(initial_MM, &SubInh_f, &SubInh_df);
    mynLM.printSummary();

    return 0;
}

