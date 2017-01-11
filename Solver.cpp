#include "Solver.h"
#include <deque>
#include <eigen/SparseCholesky>
#define beta 0.9
#define EPSILON 1e-8
#define MAX_ITERS 10000

void Solver::gradientDescent()
{
    double f = 0.0, tp = 1.0;
    handle->computeEnergy(f, x);
    obj.push_back(f);
    VectorXd xp = VectorXd::Zero(n);
    VectorXd v = VectorXd::Zero(n);

    while (true) {
        // compute momentum term
        v = x;
        if (k > 1) v += (k-2)*(x - xp)/(k+1);
        
        // compute update direction
        VectorXd g(n);
        handle->computeGradient(g, v);
        
        // compute step size
        double t = tp;
        double fp = 0.0;
        VectorXd xn = v - t*g;
        VectorXd xnv = xn - v;
        handle->computeEnergy(fp, v);
        handle->computeEnergy(f, xn);
        while (f > fp + g.dot(xnv) + xnv.dot(xnv)/(2*t)) {
            t = beta*t;
            xn = v - t*g;
            xnv = xn - v;
            handle->computeEnergy(f, xn);
        }
    
        // update
        tp = t;
        xp = x;
        x  = xn;
        obj.push_back(f);
        k++;

        // check termination condition
        if (g.norm() < EPSILON || fabs(f - fp) < EPSILON || k > MAX_ITERS) break;
    }
}

void solvePositiveDefinite(VectorXd& x,
                           const VectorXd& b,
                           const SparseMatrix<double>& A)
{
    SimplicialCholesky<SparseMatrix<double>> solver(A);
    x = solver.solve(b);
}

void Solver::newton()
{
    double f = 0.0;
    handle->computeEnergy(f, x);
    obj.push_back(f);

    const double alpha = 0.5;
    while (true) {
        // compute update direction
        VectorXd g(n);
        handle->computeGradient(g, x);

        SparseMatrix<double> H(n, n);
        handle->computeHessian(H, x);

        VectorXd p;
        solvePositiveDefinite(p, g, H);

        // compute step size
        double t = 1.0;
        double fp = f;
        handle->computeEnergy(f, x - t*p);
        while (f > fp - alpha*t*g.dot(p)) {
            t = beta*t;
            handle->computeEnergy(f, x - t*p);
        }
        
        // terminate if f is not finite
        if (!isfinite(f)) break;

        // update
        x -= t*p;
        obj.push_back(f);
        k++;

        // check termination condition
        if (g.norm() < EPSILON || fabs(f - fp) < EPSILON || k > MAX_ITERS) break;
    }
}

void Solver::lbfgs(int m)
{
    double f = 0.0;
    handle->computeEnergy(f, x);
    obj.push_back(f);
    VectorXd g(n);
    handle->computeGradient(g, x);
    deque<VectorXd> s;
    deque<VectorXd> y;
    
    const double alpha = 1e-4;
    while (true) {
        // compute update direction
        int l = min(k, m);
        VectorXd q = -g;
        
        VectorXd a(l);
        for (int i = l-1; i >= 0; i--) {
            a(i) = s[i].dot(q) / y[i].dot(s[i]);
            q -= a(i)*y[i];
        }
        
        VectorXd p = q;
        if (l > 0) p *= y[l-1].dot(s[l-1]) / y[l-1].dot(y[l-1]);
        
        for (int i = 0; i < l; i++) {
            double b = y[i].dot(p) / y[i].dot(s[i]);
            p += (a(i) - b)*s[i];
        }
        
        // compute step size
        double t = 1.0;
        double fp = f;
        handle->computeEnergy(f, x + t*p);
        while (f > fp + alpha*t*g.dot(p)) {
            t = beta*t;
            handle->computeEnergy(f, x + t*p);
        }
        
        // update
        VectorXd xp = x;
        VectorXd gp = g;
        x += t*p;
        handle->computeGradient(g, x);
        obj.push_back(f);
        k++;
        
        // update history
        if (k > m) {
            s.pop_front();
            y.pop_front();
        }
        s.push_back(x - xp);
        y.push_back(g - gp);
        
        // check termination condition
        if (g.norm() < EPSILON || fabs(f - fp) < EPSILON || k > MAX_ITERS) break;
    }
}

void Solver::solve(int method, int n, Handle *handle, int m)
{
    // initialize
    this->n = n;
    this->handle = handle;
    k = 0;
    x = VectorXd::Zero(n);
    obj.clear();
    
    // solve using selected method
    switch (method) {
        case GRADIENT_DESCENT:
            gradientDescent();
            break;
        case NEWTON:
            newton();
            break;
        case LBFGS:
            lbfgs(m);
            break;
        default:
            break;
    }
    
    // print objective and iterations
    cout << "Objective: " << obj[k-1] << endl;
    cout << "Iterations: " << k << endl;
}
