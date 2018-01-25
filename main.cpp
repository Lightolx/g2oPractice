#include <iostream>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

class fitVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // needed when you have a member whose type is Eigen::Vector
    virtual void setToOriginImpl()
    {
        _estimate = Eigen::Vector3d(0,0,0);
    }

    virtual void oplusImpl (const double* delta)
    {
        _estimate += Eigen::Vector3d(delta);
    }

    virtual bool read(istream& fin) {}
    virtual bool write(ostream& fout) const {}
};

class fitEdge: public g2o::BaseUnaryEdge<1, double, fitVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    fitEdge(double x):BaseUnaryEdge()
    {
        x_ = x;
    }

    void computeError()
    {
        const fitVertex* pVertex = static_cast<const fitVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = pVertex->estimate();
        _error(0) = _measurement - exp(abc(0)*x_*x_ + abc(1)*x_ + abc(2));
    }

    virtual bool read(istream& fin) {}
    virtual bool write(ostream& fout) const {}

private:
    double x_;

};

int main()
{
    double a = 2.5, b = 4.0, c = 3, d = 8.2;
    int N = 100;
    double sigma = 2.0;
    cv::RNG rng;
    double abc[3] = {1, 1, 1};
    double cd[2] = {1, 1};

    vector<double> x_data, y_data, z_data;

    cout << "generaing data:" << endl;

    for (int i = 0; i < N; i++)
    {
        double x = i/100.0;
        x_data.push_back(x);
        double y = exp(a*x*x + b*x + c) + rng.gaussian(sigma);
        double z = exp(c*x*x + d*x) + rng.gaussian(sigma);
        y_data.push_back(y);
        z_data.push_back(z);
    }

//    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1> >;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    Block::LinearSolverType* LinearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
//    Block* solver_ptr = new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<Block>(std::move(linearSolver)));

//    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    fitVertex* vertex = new fitVertex();
    vertex->setEstimate(Eigen::Vector3d(abc));
    vertex->setId(0);
    optimizer.addVertex(vertex);

    for (int i = 0; i < N; i++)
    {
        fitEdge* edge = new fitEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, vertex);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity());
        optimizer.addEdge(edge);
    }

    cout << "start optimization" << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);

    Eigen::Vector3d estimate = vertex->estimate();
    cout << "estimated model: " << estimate.transpose() << endl;
}