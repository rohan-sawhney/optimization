#pragma once

#include <stdlib.h>
#include <vector>
#include <eigen/Dense>
#include <eigen/Sparse>

using namespace std;
using namespace placeholders;
using namespace Eigen;

#define GRADIENT_DESCENT 0
#define NEWTON 1
#define LBFGS 2

struct Handle {
    // typedefs
    typedef function<void(double&, const VectorXd&)> ComputeObjective;
    typedef function<void(VectorXd&, const VectorXd&)> ComputeGradient;
    typedef function<void(SparseMatrix<double>&, const VectorXd&)> ComputeHessian;
    
    // constructor
    Handle() {}
    
    // member variables
    ComputeObjective computeObjective;
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
