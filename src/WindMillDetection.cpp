#include "WindMillDetection.h"
#include <cmath>
#include <Eigen/Dense>
#include <cstring>
#include <fstream>
#include <iostream>
#define pi 3.1415926

void WindMillDetection::WriteToTXT(std::string filename, int which)
{
    std::ofstream fout;
    std::vector<std::vector<double>> datalist = {time, measured_theta, v, error};

    fout.open(filename, std::ios::app);
    int size = datalist[which].size();
    for (int i = 0; i < size; i++)
    {
        fout << double(datalist[which][i]) << ',';
        //std::cerr<<datalist[which][i]<<std::endl;
    }
}

inline double WindMillDetection::ThetaFunction(double t)
{
    double angle = -param_a / param_omega * cos(param_omega * t + param_fai) + param_b * t + param_theta0;
    double temp = angle/(2*pi);
    angle = (temp-double(int(temp)))*2*pi;
    return angle;
}

double WindMillDetection::FindHitFan()
{
    show_rect = rawimage.clone();
    cv::Mat processimage = rawimage.clone();
    cv::Mat searchcenterimage = rawimage.clone();
    // search twice.
    // first step for center point only
    cv::cvtColor(searchcenterimage, searchcenterimage, CV_BGR2GRAY);
    threshold(searchcenterimage, searchcenterimage, 50, 255, CV_THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarcy;
    findContours(searchcenterimage, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    std::vector<cv::RotatedRect> box(contours.size());
    int index = 0;
    double area = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        box[i] = cv::minAreaRect(cv::Mat(contours[i]));
        if (i == 0)
        {
            area = box[i].size.area();
            index = i;
        }
        else
        {
            if (box[i].size.area() < area)
            {
                area = box[i].size.area();
                index = i;
            }
        }
    }
    center_point = box[index].center;

    //second step for the point on the hit fan.
    contours.clear();
    hierarcy.clear();
    box.clear();

    cv::floodFill(processimage, cv::Point(0, 0), cv::Scalar(0, 0, 255));
    cv::cvtColor(processimage, processimage, CV_BGR2GRAY);
    threshold(processimage, processimage, 50, 255, CV_THRESH_BINARY);
    processimage = 255 - processimage;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
    cv::morphologyEx(processimage, processimage, cv::MORPH_DILATE, kernel);
    mask = processimage;
    findContours(processimage, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cv::Point2f rect[4];
    index = 0;
    //int second_index = -1;
    area = 1e15;
    //bool change =0;
    for (int i = 0; i < contours.size(); i++)
    {
        box[i] = cv::minAreaRect(cv::Mat(contours[i]));

        if (area > box[i].size.area())
        {
            if (box[i].size.area() < 90)
                continue;
            index = i;
            area = box[i].size.area();
        }
    }
    armor_point = box[index].center;
    radius = sqrt((armor_point - center_point).dot(armor_point - center_point));
    box[index].points(rect);
    //std::cout<<box[index].size.area()<<std::endl;
    cv::circle(show_rect, armor_point, 5, cv::Scalar(0, 255, 0), -1, 8);
    cv::circle(show_rect, center_point, 5, cv::Scalar(0, 255, 0), -1, 8);
    for (int j = 0; j < 4; j++)
    {
        cv::line(show_rect, rect[j], rect[(j + 1) % 4], cv::Scalar(255, 0, 0), 2, 8); //绘制最小外接矩形每条边
    }
    show = show_rect.clone();
    // get the theta now from the center and hit fan point
    cv::Point2f x_direction = cv::Point2f(1, 0);
    cv::Point2f y_direction = cv::Point2f(0, -1);
    cv::Point2f armor_direction = (armor_point - center_point) / radius;
    double costheta_x = double(armor_direction.dot(x_direction));

    double costheta_y = double(armor_direction.dot(y_direction));
    double theta;
    //std::cerr<<costheta_x<<' '<<costheta_y<<std::endl;
    if (costheta_x > 0 && abs(costheta_y) < zero_delta)
    {
        // the +x axis . theta = 0
        theta = 0;
        //std::cerr<<1<<std::endl;
    }
    else if (abs(costheta_x) < zero_delta && costheta_y > 0)
    {
        // the +y axis. theta = pi/2
        theta = pi / 2;
        //std::cerr<<2<<std::endl;
    }
    else if (abs(costheta_y) < zero_delta && costheta_x < 0)
    {
        // the -x axis. theta = pi
        theta = pi;
        //std::cerr<<3<<std::endl;
    }
    else if (abs(costheta_x) < zero_delta && costheta_y < 0)
    {
        // the -y axis.theta = 3pi/2
        theta = 3 * pi / 2;
        //std::cerr<<4<<std::endl;
    }
    else if (costheta_x > 0 && costheta_y > 0)
    {
        // the first quadrant (0,pi/2)
        theta = acos(costheta_x);
        //std::cerr<<5<<std::endl;
    }
    else if (costheta_x < 0 && costheta_y > 0)
    {
        // the second quadrant (pi/2,pi)
        theta = acos(costheta_x);
        //std::cerr<<6<<std::endl;
    }
    else if (costheta_x < 0 && costheta_y < 0)
    {
        // the third quadrant (pi,3pi/2)
        theta = acos(costheta_y) + pi / 2;
        //std::cerr<<7<<std::endl;
    }
    else if (costheta_x > 0 && costheta_y < 0)
    {
        // the forth quadrant (3pi/2,2pi)
        theta = 2 * pi - acos(costheta_x);
        //std::cerr<<8<<std::endl;
    }

    //std::cerr<<theta<<std::endl;

    return theta;
}

void WindMillDetection::V_Handle()
{
    int len = time.size();
    if (len == 2)
    {
        double dtheta = measured_theta[1] - measured_theta[0];
        if (abs(dtheta) > pi)
        {
            dtheta = -(2 * pi - abs(dtheta)) * dtheta / abs(dtheta);
        }
        double v_ = dtheta / (time[1] - time[0]);
        v.push_back(v_);
        return;
    }
    double dtheta = measured_theta[len - 1] - measured_theta[len - 2];
    if (abs(dtheta) > pi)
    {
        dtheta = -(2 * pi - abs(dtheta)) * dtheta / abs(dtheta);
    }
    double v_;
    v_ = dtheta / (time[len - 1] - time[len - 2]);
    v.push_back(v_);
}

void WindMillDetection::Process(double nowtime, int method)
{
    //std::cerr<<nowtime<<' '<<starttime<<std::endl;
    double dt = nowtime - starttime; // get the "dt",which is the variable in the theta function
    //std::cerr<<time.size()<<std::endl;
    int len = time.size();
    if (len >= max_datanum)
    {
        time.erase(time.begin());
        time.push_back(dt);
    }
    else
    {
        time.push_back(dt);
        len++;
    }

    double nowtheta = FindHitFan();

    if (len >= max_datanum)
    {
        measured_theta.erase(measured_theta.begin());
        measured_theta.push_back(nowtheta);
    }
    else
    {
        measured_theta.push_back(nowtheta);
    }
    if (len <= 1)
        return;

    V_Handle();
    Eigen::ArrayXXd time_(len-1, 1);
    Eigen::ArrayXXd measured_theta_(len-1, 1);
    Eigen::ArrayXXd v_(len - 1, 1);
    for (int i = 1; i < len; i++)
    {
        time_(i-1, 0) = time[i];
        measured_theta_(i-1, 0) = measured_theta[i];
        //if (i < len - 1)
        v_(i-1, 0) = v[i-1];
    }
    last_fai = param_fai;
    last_theta0 = param_theta0;
    ML(time_, measured_theta_, v_);
    std::cerr << param_fai << std::endl;
}

void WindMillDetection::ML(const Eigen::ArrayXXd &time_, const Eigen::ArrayXXd &measured_theta_, const Eigen::ArrayXXd &v_)
{

    double dfai = 0;
    //double dtheta0_ = 0;//平均值
    int size = time_.size();
    if (size <= least_sample_num)
        return;
    //std::cerr<<"gogogo!"<<std::endl;
    //Eigen::ArrayXXd dv_dfai(size,1);
    Eigen::ArrayXXd dL(size, 1);
    Eigen::ArrayXXd dtheta0(size, 1);
    int cnt = 0;

    for (int i = 0; i < 10; i++)
    {

        for (int j = 0; j < size; j++)
        {
            double tmp_v = param_a * sin(param_omega * time_(j, 0) + param_fai) + param_b;

            if (cnt == 0)
            {
                error.push_back(abs(tmp_v - v_(j, 0)));
                cnt++;
            }
            if (abs(tmp_v - v_(j, 0)) <= huber_delta)
            {
                double dv_dfai = param_a * cos(param_omega * time_(j, 0) + param_fai);
                dL(j, 0) = (tmp_v - v_(j, 0)) * dv_dfai;
            }
            else
            {
                if (tmp_v > v_(j, 0))
                {
                    double dv_dfai = param_a * cos(param_omega * time_(j, 0) + param_fai);
                    dL(j, 0) = dv_dfai * huber_delta;
                }
                else
                {
                    double dv_dfai = param_a * cos(param_omega * time_(j, 0) + param_fai);
                    dL(j, 0) = -dv_dfai * huber_delta;
                }
            }
        }
        dfai = dL.mean();
        param_fai -= lr * dfai;
    }

    //get theta0
    // for(int i =0;i<size;i++){
    //     double theta0_temp = ThetaFunction(time_(i, 0));
    //     double dtheta0_ = measured_theta_(i,0)-theta0_temp;
    //     if(abs(dtheta0_)>pi){
    //         dtheta0_ = -(2*pi-abs(dtheta0_))*dtheta0_/abs(dtheta0_);
    //     }
    //     dtheta0(i,0) = dtheta0_;
    // }
    // param_theta0 = dtheta0.mean();
    double theta0_hat;
    double latest_angle = measured_theta_(size-1,0);
    double latest_predicted_angle = ThetaFunction(time_(size-1,0));
    theta0_hat = latest_angle-latest_predicted_angle;//这样算出来的也只是两者的差值，等下还要加回来。
    if(theta0_hat>pi)
        theta0_hat-=2*pi;
    else if(theta0_hat<-pi)
        theta0_hat+=2*pi;
    theta0_hat+=param_theta0;
    if(abs(param_theta0)<0.001)
        param_theta0 = theta0_hat;
    param_theta0 = 0.1*param_theta0+0.9*theta0_hat;


}

cv::Mat WindMillDetection::GetImage(int which)
{
    if (which == 1)
    {
        return show;
    }
    else if (which == 2)
    {
        return show_rect;
    }
    else if(which == 3){
        return mask;
    }
    else
    {
        return rawimage;
    }
}

cv::Mat WindMillDetection::GetShowMat(double now)
{
    double t = now-starttime;
    double theta = ThetaFunction(t);
    float dx = double(radius * cos(-theta));
    float dy = double(radius * sin(-theta));
    cv::Point2f fitpoint = cv::Point2f(dx, dy) + center_point;
    cv::circle(show, fitpoint, 5, cv::Scalar(255, 0, 0), -1, 8);
    return show;
}
