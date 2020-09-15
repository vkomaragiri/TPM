//
// Created by vasundhara on 1/8/20.
//

#include "../include/HyperParameters.h"

int HyperParameters::max_height = 10;
bool HyperParameters::prune = false;
ldouble HyperParameters::tol=1e-10;
bool HyperParameters::tune = false;
bool HyperParameters::merge = false;
bool HyperParameters::run_mixture = false;
int HyperParameters::num_components = 10;
int HyperParameters::num_iterations_em = 200;
int HyperParameters::interval_for_structure_learning=10;
ldouble HyperParameters::mi_low_limit=1e-5;
int HyperParameters::num_samples=10000;
ORDERING_HEURISTIC HyperParameters::ord_heu=min_fill;
ldouble HyperParameters::laplace = 1.0;