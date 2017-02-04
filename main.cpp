#include "Solver.h"
#include <iostream>

struct Rosenbrock {
    void computeObjective(double& objective, const VectorXd& u)
    {
        double x = u(0);
        double y = u(1);
        
        objective = 100*pow(y - x*x, 2) + pow(1 - x, 2);
    }
    
    void computeGradient(VectorXd& gradient, const VectorXd& u)
    {
        double x = u(0);
        double y = u(1);
        
        gradient(0) = -400*x*(y - x*x) - 2*(1 - x);
        gradient(1) = 200*(y - x*x);
    }
    
    void computeHessian(SparseMatrix<double>& hessian, const VectorXd& u)
    {
        double x = u(0);
        double y = u(1);
        
        hessian.insert(0, 0) = 1200*x*x - 400*y + 2;
        hessian.insert(0, 1) = -400*x;
        hessian.insert(1, 0) = -400*x;
        hessian.insert(1, 1) = 200;
        hessian.makeCompressed();
    }
};

void solveRosenbrock()
{
    // Link functions to compute objective, gradient and hessian
    Rosenbrock rosen;
    
    Handle handle;
    handle.computeObjective = bind(&Rosenbrock::computeObjective, &rosen, _1, _2);
    handle.computeGradient = bind(&Rosenbrock::computeGradient, &rosen, _1, _2);
    handle.computeHessian = bind(&Rosenbrock::computeHessian, &rosen, _1, _2);
    
    Solver solver;
    
    // Solve using gradient descent
    solver.solve(GRADIENT_DESCENT, 2, &handle);
    cout << "x: " << solver.x(0) << " y: " << solver.x(1) << "\n" << endl;
    
    // Solve using gradient descent
    solver.solve(NEWTON, 2, &handle);
    cout << "x: " << solver.x(0) << " y: " << solver.x(1) << "\n" << endl;
    
    // Solve using gradient descent
    solver.solve(LBFGS, 2, &handle);
    cout << "x: " << solver.x(0) << " y: " << solver.x(1) << "\n" << endl;
}

int main(int argc, char** argv)
{
    // Solve unconstrained minimization problem
    solveRosenbrock();
    
    return 0;
}
