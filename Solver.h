#pragma once

#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include "math.h"
#include <eigen/Core>
#include <eigen/Dense>
#include <eigen/SparseCore>

using namespace std;
using namespace Eigen;
using namespace placeholders;

#define GRADIENT_DESCENT 0
#define NEWTON 1
#define LBFGS 2

struct Handle {
    // typedefs
    typedef function<void(double&, const VectorXd&)> ComputeEnergy;
    typedef function<void(VectorXd&, const VectorXd&)> ComputeGradient;
    typedef function<void(SparseMatrix<double>&, const VectorXd&)> ComputeHessian;
    
    // constructor
    Handle() {}
    
    // member variables
    ComputeEnergy computeEnergy;
    ComputeGradient computeGradient;
    ComputeHessian computeHessian;
};

class Solver {
public:
    // constructor
    Solver() {}
    
    // solve
    void solve(int method, int n, Handle *handle, int m = 10);
    
    // member variables
    VectorXd x;
    vector<double> obj;

private:
    // gradient descent
    void gradientDescent();
    
    // newton
    void newton();
    
    // lbfgs
    void lbfgs(int m);
    
    // member variables
    int n;
    int k;
    Handle *handle;
};