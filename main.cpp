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

// BaseVertex <dimension of variable, type of variable>
class fitVertex: public g2o::BaseVertex<4, Eigen::Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // needed when you have a member whose type is Eigen::Vector
    virtual void setToOriginImpl()
    {
        _estimate = Eigen::Vector4d(0,0,0,0);
    }

    virtual void oplusImpl (const double* delta)
    {
        _estimate += Eigen::Vector4d(delta);
    }

    virtual bool read(istream& fin) {}
    virtual bool write(ostream& fout) const {}
};

// BaseEdge, <dimension of observation, type of observation, Vertex type>
// todo:: change the type of observation from double to Vector2d
class fitEdge: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, fitVertex>
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
        const Eigen::Vector4d abcd = pVertex->estimate();
        _error(0) = _measurement(0) - exp(abcd(0)*x_*x_ + abcd(1)*x_ );
        _error(1) = _measurement(1) - exp(abcd(2)*x_*x_ + abcd(3)*x_ );
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
    double abcd[4] = {1,1,1,1};
//    double cd[2] = {1, 1};

    vector<double> x_data, y_data, z_data;

    cout << "generaing data:" << endl;

    for (int i = 0; i < N; i++)
    {
        double x = i/100.0;
        x_data.push_back(x);
        double y = exp(a*x*x + b*x) + rng.gaussian(sigma);
        double z = exp(c*x*x + d*x) + rng.gaussian(sigma);
        y_data.push_back(y);
        z_data.push_back(z);
    }



//    using Block = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1> >;
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<4,2>> Block;
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
    vertex->setEstimate(Eigen::Vector4d(abcd));
    vertex->setId(0);
    optimizer.addVertex(vertex);

    for (int i = 0; i < N; i++)
    {
        fitEdge* edge = new fitEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, vertex);
        edge->setMeasurement(Eigen::Vector2d(y_data[i], z_data[i]));
//        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix<double,2,2>::Identity());
        optimizer.addEdge(edge);
    }


    cout << "start optimization" << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);

    Eigen::Vector4d estimate = vertex->estimate();
    cout << "estimated model: " << estimate.transpose() << endl;

}