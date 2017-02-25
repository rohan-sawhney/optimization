#include "Solver.h"
#include <iostream>

struct Rosenbrock {
    void objective(double& f, const VectorXd& u)
    {
        double x = u(0);
        double y = u(1);
        
        f = 100*pow(y - x*x, 2) + pow(1 - x, 2);
    }
    
    void gradient(VectorXd& g, const VectorXd& u)
    {
        double x = u(0);
        double y = u(1);
        
        g(0) = -400*x*(y - x*x) - 2*(1 - x);
        g(1) = 200*(y - x*x);
    }
    
    void hessian(SparseMatrix<double>& H, const VectorXd& u)
    {
        double x = u(0);
        double y = u(1);
        
        H.insert(0, 0) = 1200*x*x - 400*y + 2;
        H.insert(0, 1) = -400*x;
        H.insert(1, 0) = -400*x;
        H.insert(1, 1) = 200;
        H.makeCompressed();
    }
};

void solveRosenbrock()
{
    // Link functions to compute objective, gradient and hessian
    Rosenbrock rosen;
    
    Handle handle;
    handle.F = bind(&Rosenbrock::objective, &rosen, _1, _2);
    handle.gradF = bind(&Rosenbrock::gradient, &rosen, _1, _2);
    handle.hessF = bind(&Rosenbrock::hessian, &rosen, _1, _2);
    
    Solver solver;
    
    // Solve using gradient descent
    solver.solve(GRADIENT_DESCENT, &handle, 2);
    cout << "x: " << solver.x(0) << " y: " << solver.x(1) << "\n" << endl;
        
    // Solve using newton's method
    solver.solve(NEWTON, &handle, 2);
    cout << "x: " << solver.x(0) << " y: " << solver.x(1) << "\n" << endl;

    // Solve using lbfgs
    solver.solve(LBFGS, &handle, 2);
    cout << "x: " << solver.x(0) << " y: " << solver.x(1) << "\n" << endl;
}

int main(int argc, char** argv)
{
    // Solve unconstrained minimization problem
    solveRosenbrock();
    
    return 0;
}
