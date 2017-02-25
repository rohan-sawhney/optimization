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
#define INTERIOR_POINT 3

struct Handle {
    // typedefs
    typedef function<void(double&, const VectorXd&)> ComputeValue;
    typedef function<void(VectorXd&, const VectorXd&)> ComputeVector;
    typedef function<void(SparseMatrix<double>&, const VectorXd&)> ComputeMatrix;
    
    // constructor
    Handle() {}
    
    // member variables
    ComputeValue F;
    ComputeVector gradF;
    ComputeMatrix hessF;
    
    ComputeMatrix b; // r x 1
    ComputeMatrix A; // r x n
    
    ComputeVector H;     // [h1  h2  ... hm]
    ComputeMatrix gradH; // [∇h1 ∇h2 ... ∇hm]: n x m
    ComputeMatrix hessH; // [Δh1 Δh2 ... Δhm]: n x (mxn)
};

class Solver {
public:
    // constructor
    Solver() {}
    
    // solve
    void solve(int method, Handle *handle, int n, int m = 10, int r = 10);
    
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
    
    // interior point
    void interiorPoint(int m, int r);
    
    // member variables
    int n;
    int k;
    Handle *handle;
};
