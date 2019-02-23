//
// Created by lightol on 2/23/19.
//
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <Eigen/Eigen>

using std::cout;
using std::endl;
using std::cerr;

g2o::SE3Quat Converter2SE3Quat(const Eigen::Matrix4d & T) {
    Eigen::Matrix3d R;
    R = T.topLeftCorner(3, 3);

    Eigen::Vector3d t = T.topRightCorner(3, 1);

    return g2o::SE3Quat(R, t);
}

bool LoadGtPoses(const std::string &poseFile, std::vector<Eigen::Matrix4d> &vGtTcws, bool bSTCC, int endID) {
    std::ifstream fTime(poseFile);
    if (!fTime.is_open()) {
        cerr << "open file " << poseFile << " failed." << endl;
        return false;
    }
    std::string line;
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    int nPose = 0;
    while (getline(fTime, line)) {
        if (nPose++ >= endID) {
            break;
        }
        if (bSTCC) {
            line = line.substr(line.find(' ') + 1);
        }

        std::stringstream ss(line);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                ss >> Twc(i, j);
            }
        }

        vGtTcws.push_back(Twc.inverse());
    }

    return true;
}

bool LoadLoopEdges(const std::string &poseFile, std::vector<Eigen::Matrix4d> &vTcw12s,
                   std::vector<std::pair<int, int> > &vID12s) {
    std::ifstream fin(poseFile);
    if (!fin.is_open()) {
        cerr << "open file " << poseFile << " failed." << endl;
        return false;
    }
    std::string line;
    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
    int id1(0), id2(0);
    while (getline(fin, line)) {
        std::stringstream ss(line);

        ss >> id1 >> id2;
        vID12s.push_back(std::make_pair(id1, id2));

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                ss >> Tcw(i, j);
            }
        }

        vTcw12s.push_back(Tcw);
    }

    return true;
}

int main(int argc, char **argv) {
    // Step0: 造出两个pose
    std::string path_to_pose(argv[1]);
    std::string path_to_constraint(argv[2]);

    // step0.0: read pose ground truth
    int nImages = 6500;
    std::vector<Eigen::Matrix4d> vGtTcws;
    vGtTcws.reserve(nImages);
    if (!LoadGtPoses(path_to_pose, vGtTcws, true, nImages)) {
        cerr << "Load Pose Ground truth file failed" << endl;
        return 1;
    }

    //step0.1: read loop edge
    std::vector<Eigen::Matrix4d> vTcw12s;
    std::vector<std::pair<int, int> > vID12s;
    vTcw12s.reserve(10 * nImages);
    vID12s.reserve(10 * nImages);
    if (!LoadLoopEdges(path_to_constraint, vTcw12s, vID12s)) {
        cerr << "Load loop edge file failed" << endl;
        return 1;
    }

    Eigen::Matrix4d T0 = vGtTcws[400];
    Eigen::Matrix4d T1 = vGtTcws[600];
    cout << "T0 is\n" << T0 << endl;
    cout << "T1 is\n" << T1 << endl;

    // Step1: 构造一个g2o的optimizer
    // 在这里不要搞什么BlockSolver_6_3, 直接给BlockSolverX，免得出错
    typedef g2o::BlockSolverX Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    Block::LinearSolverType* LinearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<Block>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Step2: 添加两个pose为vertex和以及构造edge
    g2o::VertexSE3Expmap * vertex0= new g2o::VertexSE3Expmap();
    vertex0->setEstimate(Converter2SE3Quat(T0));
    vertex0->setId(0);
    vertex0->setFixed(true);
    optimizer.addVertex(vertex0);

    g2o::VertexSE3Expmap * vertex1= new g2o::VertexSE3Expmap();
    vertex1->setEstimate(Converter2SE3Quat(T1));
    vertex1->setId(1);
//    vertex1->setFixed(true);
    optimizer.addVertex(vertex1);

    g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();

    g2o::SE3Quat obs = Converter2SE3Quat(T1 * T0.inverse());
    e->setId(0);
    e->setMeasurement(obs);
    // 注意g2o的edge必须给Information Matrix，因为它计算代价函数时是 chi2 = error * Info * error.inv(),
    // 如果不把Info初始化成单位阵I，那么Eigen::Matrix的默认构造函数会把Info初始化成零阵，
    // 那么不论ComputeError()函数把Error计算成什么，到头来chi2()总会被计算成0,也就是优化器认为cost function永远都是零，
    // 它不会对变量有任何优化
    e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    e->setVertex(0, optimizer.vertex(0));
    e->setVertex(1, optimizer.vertex(1));

    optimizer.addEdge(e);

    // Step3: optimize and the print out
    cout << "start optimization" << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(100);

    g2o::VertexSE3Expmap* vSE32 =
            static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
    g2o::SE3Quat SE3quat2 = vSE32->estimate();

    Eigen::Matrix4d dif2 = SE3quat2.to_homogeneous_matrix();
    cout << "after optimize, T1 is\n" << dif2 << endl;
}