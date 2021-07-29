#ifndef WINDMILLDETECTION_H
#define WINDMILLDETECTION_H
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <cstring>
#include<fstream>
#define Infinity 1e15

class WindMillDetection
{
private:
    cv::Mat show;             // the image to be showed for the predicted hit point.
    cv::Mat show_rect;        // the image to be showed for the rect on the hit fan.
    cv::Mat rawimage;         // the raw image of a frame
    cv::Mat mask;
    cv::Point2f center_point; // the center of windmill
    double radius;
    cv::Point2f armor_point; // the point in the center of armor
    cv::Point2f hat_point;   // the predicted point
    double param_omega;
    double param_a;
    double param_b;
    double param_fai;
    double param_theta0;
    double loss;
    double starttime;                   // the time when the program starts
    std::vector<double> time;           // this "time" is the "dt" in the function."dt" = nowtime - starttime
    std::vector<double> measured_theta; // the "real" theta measured from the hitfan.  Adopt radian system.
    std::vector<double> v;              //the v of angle .
    cv::Size imagesize;
    int max_datanum; // 训练数据的最大数量
    //Eigen::ArrayXXd time_; // converted to an Eigen array for easy computation
    //Eigen::ArrayXXd measured_theta_; // converted to an Eigen array for easy computation
    double zero_delta;
    double last_fai;
    double last_theta0;
    double end_condition;
    double huber_delta;
    double lr;            // learning rate
    int least_sample_num; // the least number for sample
    std::vector<double> error;
    //int cnt = 0;
    double out_angle;

public:
    WindMillDetection() {}
    WindMillDetection(double start, double omega, double a, double b, double fai, double end_cond = 1e-5, int maxnum = 3000, double delta = 1e-5, double huber =1.5, double lr_ = 5, int leastsamplenum = 500) : param_omega(omega), param_a(a), param_b(b), param_fai(fai),
                                                                                                                                                                                                                    max_datanum(maxnum), zero_delta(delta), end_condition(end_cond), huber_delta(huber)
    {
        last_fai = Infinity;
        last_theta0 = Infinity;
        lr = lr_;
        starttime = start;
        least_sample_num = leastsamplenum;
        param_theta0 = 0;
    }
    ~WindMillDetection() {}
    // get a new frame to be processed
    inline void SetFrame(const cv::Mat &frame,double out)
    {
        out_angle = out;
        std::ofstream fout;
        fout.open("./outangle.txt",std::ios::app);
        fout<<out_angle<<',';
        rawimage = frame;
        imagesize.width = frame.cols;
        imagesize.height = frame.rows;
    }
    void Process(double nowtime, int method); // choose a method  to fit
    cv::Mat GetShowMat(double now);           // get the result picture
    //double GetLoss();                         // get the loss value
    cv::Mat GetImage(int which);
    void WriteToTXT(std::string filename, int which);

private:
    void ML(const Eigen::ArrayXXd &time_, const Eigen::ArrayXXd &measured_theta_, const Eigen::ArrayXXd &v_);
    //void MAP(const Eigen::ArrayXXd &time_, const Eigen::ArrayXXd &measured_theta_);
    //void EKF(const Eigen::ArrayXXd &time_, const Eigen::ArrayXXd &measured_theta_);
    inline double ThetaFunction(double t);
    double FindHitFan();
    void V_Handle();
};

#endif