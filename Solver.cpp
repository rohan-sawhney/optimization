#include "Solver.h"
#include <deque>
#include <iostream>

#define beta 0.9
#define alpha 0.01
#define mu 10.0
#define EPSILON 1e-8
#define MAX_ITERS 10000

void Solver::gradientDescent()
{
    double f = 0.0, tp = 1.0;
    handle->F(f, x);
    obj.push_back(f);
    VectorXd xp = VectorXd::Zero(n);

    while (true) {
        // compute momentum term
        VectorXd v = x;
        if (k > 1) v += (k-2)*(x - xp)/(k+1);
        
        // compute update direction
        VectorXd g(n);
        handle->gradF(g, v);
        
        // compute step size
        double t = tp;
        double fp = 0.0;
        VectorXd xn = v - t*g;
        VectorXd xnv = xn - v;
        handle->F(fp, v);
        handle->F(f, xn);
        while (f > fp + g.dot(xnv) + xnv.dot(xnv)/(2*t)) {
            t = beta*t;
            xn = v - t*g;
            xnv = xn - v;
            handle->F(f, xn);
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

void solvePositiveDefinite(const SparseMatrix<double>& A,
                           const VectorXd& b,
                           VectorXd& x)
{
    SimplicialCholesky<SparseMatrix<double>> solver(A);
    x = solver.solve(b);
}

void Solver::newton()
{
    double f = 0.0;
    handle->F(f, x);
    obj.push_back(f);

    while (true) {
        // compute update direction
        VectorXd g(n);
        handle->gradF(g, x);

        SparseMatrix<double> H(n, n);
        handle->hessF(H, x);

        VectorXd p;
        solvePositiveDefinite(H, g, p);
        
        // compute step size
        double t = 1.0;
        double fp = f;
        handle->F(f, x - t*p);
        while (f > fp - alpha*t*g.dot(p)) {
            t = beta*t;
            handle->F(f, x - t*p);
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
    handle->F(f, x);
    obj.push_back(f);
    VectorXd g(n);
    handle->gradF(g, x);
    deque<VectorXd> s;
    deque<VectorXd> y;
    
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
        handle->F(f, x + t*p);
        while (f > fp + alpha*t*g.dot(p)) {
            t = beta*t;
            handle->F(f, x + t*p);
        }
        
        // update
        VectorXd xp = x;
        VectorXd gp = g;
        x += t*p;
        handle->gradF(g, x);
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

bool findFeasiblePoint()
{
    // TODO: Call interior point
    return false;
}

void computeDense(Handle *handle, VectorXd& g, VectorXd& h, MatrixXd& hg,
                  MatrixXd& A, VectorXd& b, VectorXd& x)
{
    handle->gradF(g, x);
    handle->H(h, x);
    handle->gradH(hg, x);
    handle->A(A, x);
    handle->b(b, x);
}

void computeSparse(Handle *handle, SparseMatrix<double> H,
                   SparseMatrix<double> hH, VectorXd& x)
{
    handle->hessF(H, x);
    handle->hessH(hH, x);
}

double dualityGap(const VectorXd& h, const VectorXd& u)
{
    return -h.dot(u);
}

void assembleResidual(VectorXd& res,
                      const VectorXd& x, const VectorXd& u, const VectorXd& v,
                      const VectorXd& g, const VectorXd& h, MatrixXd& hg,
                      const VectorXd& b, const MatrixXd& A,
                      int n, int m, int r)
{
    res.block(0, 0, n, 1) = g + hg*u + A.transpose()*v;
    res.block(n, 0, m, 1) = u.asDiagonal()*h + (1.0/mu)*(dualityGap(h, u)/m)*VectorXd::Ones(m);
    res.block(n+m, 0, r, 1) = A*x - b;
}

void assembleHessian(SparseMatrix<double>& hess,
                     const VectorXd& u, const SparseMatrix<double>& H,
                     const VectorXd& h, const MatrixXd& hg, SparseMatrix<double>& hH,
                     const MatrixXd& A, int n, int m, int r)
{
    int nm = n + m;
    int nmr = nm + r;
    vector<Triplet<double>> triplets;
    
    for (int i = 0; i < H.outerSize(); i++) {
        for (SparseMatrix<double>::InnerIterator it(H, i); it; ++it) {
            triplets.push_back(Triplet<double>(it.row(), it.col(), it.value()));
        }
    }
    
    for (int i = 0; i < hH.outerSize(); i++) {
        for (SparseMatrix<double>::InnerIterator it(hH, i); it; ++it) {
            int m = it.col()/n;
            triplets.push_back(Triplet<double>(it.row(), it.col()%n, u(m)*it.value()));
        }
    }
    
    for (int i = n; i < nm; i++) {
        triplets.push_back(Triplet<double>(i, i, h(i)));
        for (int j = 0; j < n; j++) {
            triplets.push_back(Triplet<double>(i, j, u(i)*hg(j, i)));
            triplets.push_back(Triplet<double>(j, i, hg(j, i)));
        }
    }

    for (int i = nm; i < nmr; i++) {
        for (int j = 0; j < n; j++) {
            triplets.push_back(Triplet<double>(i, j, A(i, j)));
            triplets.push_back(Triplet<double>(j, i, A(i, j)));
        }
    }
    
    hess.setFromTriplets(triplets.begin(), triplets.end());
}

double backtrackingStepSize(const VectorXd& u, const VectorXd& du, int m)
{
    double step = 1.0;
    for (int i = 0; i < m; i++) {
        if (du(i) < 0.0) step = min(step, -u(i)/du(i));
    }
    
    return step;
}

bool isInfeasible(const VectorXd& h, int m)
{
    for (int i = 0; i < m; i++) {
        if (h(i) >= 0.0) return true;
    }
    
    return false;
}

void Solver::interiorPoint(Handle *handle, int m, int r, bool isFeasible)
{
    if (!isFeasible) {
        if (!findFeasiblePoint()) {
            cout << "Problem is infeasible" << endl;
            return;
        }
    }
    
    int nm = n + m;
    int nmr = nm + r;
    double f = 0.0;
    handle->F(f, x);
    obj.push_back(f);
    VectorXd u = VectorXd::Ones(m);
    VectorXd v = VectorXd::Zero(r);
    
    while (true) {
        // compute update direction
        VectorXd g(n), h(m), b(r);
        MatrixXd hg(n, m), A(r, n);
        computeDense(handle, g, h, hg, A, b, x);
        
        SparseMatrix<double> H(n, n), hH(n, n*m);
        computeSparse(handle, H, hH, x);
        
        VectorXd res(nmr);
        assembleResidual(res, x, u, v, g, h, hg, b, A, n, m, r);
        
        SparseMatrix<double> hess(nmr, nmr);
        assembleHessian(hess, u, H, h, hg, hH, A, n, m, r);
        
        VectorXd p;
        solvePositiveDefinite(hess, -res, p);
        VectorXd dx = p.block(0, 0, n, 1);
        VectorXd du = p.block(n, 0, m, 1);
        VectorXd dv = p.block(nm, 0, r, 1);
        
        // compute step size
        double theta = backtrackingStepSize(u, du, m);
        VectorXd xn = x + theta*dx;
        VectorXd un = u + theta*du;
        VectorXd vn = v + theta*dv;
        VectorXd resp = res;
        
        // maintain feasibility
        handle->H(h, xn);
        while (isInfeasible(h, m)) {
            theta = beta*theta;
            
            xn = x + theta*dx;
            handle->H(h, xn);
        }
        
        // reduce residual
        computeDense(handle, g, h, hg, A, b, xn);
        assembleResidual(res, xn, un, vn, g, h, hg, b, A, n, m, r);
        while (res.norm() > (1.0 - alpha*theta)*resp.norm()) {
            theta = beta*theta;
            
            xn = x + theta*dx;
            un = u + theta*du;
            vn = v + theta*dv;
            computeDense(handle, g, h, hg, A, b, xn);
            assembleResidual(res, xn, un, vn, g, h, hg, b, A, n, m, r);
        }
        
        // update
        x = xn;
        u = un;
        v = vn;
        handle->F(f, x);
        obj.push_back(f);
        k++;
    
        // check termination condition
        double gap = dualityGap(h, u);
        double feasibility = sqrt(res.block(0, 0, n, 1).squaredNorm() +
                                  res.block(nm, 0, r, 1).squaredNorm());
        if ((feasibility < EPSILON && gap < 2*EPSILON) || k > MAX_ITERS) break;
    }
}

void Solver::solve(int method, Handle *handle, int n, int m, int r, bool isFeasible)
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
            cout << "Method: Accelerated Gradient Descent" << endl;
            gradientDescent();
            break;
        case NEWTON:
            cout << "Method: Newton's Method" << endl;
            newton();
            break;
        case LBFGS:
            cout << "Method: LBFGS" << endl;
            lbfgs(m);
            break;
        case INTERIOR_POINT:
            cout << "Method: Interior Point" << endl;
            interiorPoint(handle, m, r, isFeasible);
            break;
        default:
            break;
    }
    
    // print objective and iterations
    cout << "Objective: " << obj[k-1] << endl;
    cout << "Iterations: " << k << endl;
}
