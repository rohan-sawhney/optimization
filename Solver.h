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
    typedef function<void(MatrixXd&, const VectorXd&)> ComputeDenseMatrix;
    typedef function<void(SparseMatrix<double>&, const VectorXd&)> ComputeSparseMatrix;
    
    // constructor
    Handle() {}
    
    // member variables
    ComputeValue F;
    ComputeVector gradF;
    ComputeSparseMatrix hessF;
    
    // Inequality constraints H(x) <= 0
    ComputeVector H;            // [h1  h2  ... hm]
    ComputeDenseMatrix gradH;   // [∇h1 ∇h2 ... ∇hm]: n x m
    ComputeSparseMatrix hessH;  // [Δh1 Δh2 ... Δhm]: n x (mxn)
    
    // Equality constraints Ax = b
    ComputeVector b;            // r x 1
    ComputeDenseMatrix A;       // r x n
};

class Solver {
public:
    // constructor
    Solver() {}
    
    // solve
    void solve(int method, Handle *handle, int n, int m = 10, int r = 10, bool isFeasible = false);
    
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
    void interiorPoint(Handle *handle, int m, int r, bool isFeasible);
    
    // member variables
    int n;
    int k;
    Handle *handle;
};
