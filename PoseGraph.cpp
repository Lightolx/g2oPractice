//
// Created by lightol on 3/7/19.
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
#include <sophus/so3.h>

using std::cout;
using std::endl;
using std::cerr;

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

g2o::SE3Quat Converter2SE3Quat(const Eigen::Matrix4d & T) {
    Eigen::Matrix3d R;
    R = T.topLeftCorner(3, 3);

    Eigen::Vector3d t = T.topRightCorner(3, 1);

    return g2o::SE3Quat(R, t);
}

int main(int argc, char **argv) {
    std::string path_to_pose(argv[1]);
    std::string path_to_constraint(argv[2]);

    // Step0: Read in pose ground truth and loop constraints
    // step0.0: read pose ground truth
    int minImgID = 200;
    int nImages = 1000;
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

    // Step1: 使用g2o做一个pose graph, 优化所有KF的pose
    typedef g2o::BlockSolverX Block;
    std::unique_ptr<Block::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    Block::LinearSolverType* LinearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<Block>(std::move(linearSolver)));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // Step1.1: 把所有的KF pose构造成g2o的vertex
    for (int i = minImgID; i < nImages; ++i) {
        // 加一点噪声
        Eigen::Vector3d noise = Eigen::Vector3d::Random();
        noise.normalize();
        noise /= 30;
        Eigen::AngleAxisd rv_noise(noise.norm(), noise.normalized());
        Eigen::Matrix3d Rnoise(rv_noise);
        Eigen::Vector3d tnoise = Eigen::Vector3d::Random();
        tnoise.normalize();
//        tnoise *= 1;
//        tnoise /= 30;
        Eigen::Matrix4d Tcw_noise = Eigen::Matrix4d::Identity();
        Rnoise = Eigen::Matrix3d::Identity();
        Tcw_noise.topLeftCorner(3, 3) = Rnoise;
        Tcw_noise.topRightCorner(3, 1) = tnoise;

//        for (int i = 200; i < 205; ++i) {
        g2o::VertexSE3* vertex = new g2o::VertexSE3();
        Eigen::Matrix4d Tcw = vGtTcws[i];
        if (i != minImgID) {
            Tcw *=  Tcw_noise;
        }
//        cout << "before, Tcw is\n" << Tcw << endl;
        g2o::Isometry3 pose(Tcw);
        vertex->setEstimate(pose);
        g2o::Isometry3 V = vertex->estimate();
//        cout << "after, Tcw is\n" << V.matrix() << endl;

        vertex->setId(i);
        vertex->setFixed(i == minImgID);

        optimizer.addVertex(vertex);
    }

    // Step1.2: 构造normal edge, 也就是根据pose算出来的前后帧之间的相对变换
    for (int i = minImgID; i < nImages - 1; ++i) {
        g2o::VertexSE3* vSE32 = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        g2o::Isometry3 Isometry = vSE32->estimate();
        Eigen::Matrix4d Tcw1(Isometry.matrix());
        vSE32 = static_cast<g2o::VertexSE3*>(optimizer.vertex(i+1));
        Isometry = vSE32->estimate();
        Eigen::Matrix4d Tcw2(Isometry.matrix());

//        Eigen::Matrix4d Tcw1 = vGtTcws[i];
//        Eigen::Matrix4d Tcw2 = vGtTcws[i+1];
        Eigen::Matrix4d Tcw12 = Tcw2.inverse() * Tcw1;

        g2o::EdgeSE3 *e = new g2o::EdgeSE3;

        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i+1)));
        e->setMeasurement(g2o::Isometry3(Tcw12));

        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());

        optimizer.addEdge(e);
    }

    // Step1.2.5: 加一个首尾回环
//    {
//        Eigen::Matrix4d Tcw1 = vGtTcws[minImgID];
//        Eigen::Matrix4d Tcw2 = vGtTcws[nImages - 1];
//        Eigen::Matrix4d Tcw12 = Tcw2.inverse() * Tcw1;
//
//        g2o::EdgeSE3 *e = new g2o::EdgeSE3;
//
//        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(minImgID)));
//        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nImages - 1)));
//        e->setMeasurement(g2o::Isometry3(Tcw12));
//
//        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
//
//        optimizer.addEdge(e);
//    }

    // Step1.2.6: 多加几个回环
    for (int nLoop = 0; nLoop < 500; ++nLoop)
    {
        int i = rand() % (nImages - minImgID) + minImgID;
        int j = rand() % (nImages - minImgID) + minImgID;
        while (i == j) {
            j = rand() % (nImages - minImgID) + minImgID;
        }

        if (nLoop == 0) {
            i = minImgID;
            j = nImages - 1;
        }

        // 加一点噪声
        Eigen::Vector3d noise = Eigen::Vector3d::Random();
        noise.normalize();
        noise /= 50;
        Eigen::AngleAxisd rv_noise(noise.norm(), noise.normalized());
        Eigen::Matrix3d Rnoise(rv_noise);
        Eigen::Vector3d tnoise = Eigen::Vector3d::Random();
        tnoise.normalize();
//        tnoise *= 1;
        tnoise /= 0.5;
        Eigen::Matrix4d Tcw_noise = Eigen::Matrix4d::Identity();
        Rnoise = Eigen::Matrix3d::Identity();
        Tcw_noise.topLeftCorner(3, 3) = Rnoise;
        Tcw_noise.topRightCorner(3, 1) = tnoise;

        Eigen::Matrix4d Tcw1 = vGtTcws[i];
        Eigen::Matrix4d Tcw2 = vGtTcws[j];
        Eigen::Matrix4d Tcw12 = Tcw2.inverse() * Tcw1;
        Tcw12 = Tcw12 * Tcw_noise;

        g2o::EdgeSE3 *e = new g2o::EdgeSE3;

        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(j)));
        e->setMeasurement(g2o::Isometry3(Tcw12));

        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());

        optimizer.addEdge(e);
    }

    // Step1.3: 构造loop edge, 根据当前帧与回环帧之间的相对变换
//    int nConstriants = vID12s.size();
//    for (int i = 0; i < nConstriants; ++i) {
//        Eigen::Matrix4d Tcw12 = vTcw12s[i];
//        std::pair<int, int> ids = vID12s[i];
//
//        g2o::EdgeSE3 *e = new g2o::EdgeSE3;
//
//        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids.first)));
//        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ids.second)));
//        e->setMeasurement(g2o::Isometry3(Tcw12));
//
//        e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
//
//        optimizer.addEdge(e);
//    }

    // Step1.4: Optimize!
    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(200);

    // Step2: 输出所有KF的pose
    double error = 0.0;
    for (int i = minImgID; i < nImages; ++i) {
        g2o::VertexSE3* vSE32 = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        g2o::Isometry3 Isometry = vSE32->estimate();
        Eigen::Matrix4d Tcw(Isometry.matrix());

        error += (Tcw - vGtTcws[i]).topRightCorner(3, 1).norm();
    }
    cout << "error = " << error << endl;
}