#include<vector>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<cmath>
#include<chrono>
#include "windmill.hpp"
#include<iostream>
#include "WindMillDetection.h"
#include<cstring>

int main()
{
    double starttime = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
    WINDMILL::WindMill
    wm(starttime);
    WindMillDetection wmd(starttime,1.884,0.785,1.305,4);
    // cv::namedWindow("processimage");
    // cv::namedWindow("show_rect");
    char k ='a';
    do
    {
        double nowtime = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
        cv::Mat img =wm.getMat(nowtime);
        double outangle = wm.getNowAngle();

        wmd.SetFrame(img,outangle);
        try{
            wmd.Process(nowtime,1);
        }
        catch(std::string e){
            std::cerr<<e;
        }
        cv::Mat showrect = wmd.GetImage(2);
        cv::imshow("rect",showrect);
        cv::Mat show = wmd.GetShowMat(nowtime);
        cv::Mat mask = wmd.GetImage(3);

        /****************************/
        // cv::Mat show = img.clone();
        // //cv::floodFill(show,cv::Point(0,0),cv::Scalar(0,0,255));
        // cv::cvtColor(show,show,CV_BGR2GRAY);
        // threshold(show, show, 50, 255, CV_THRESH_BINARY);
        // //show =255-show;


        // //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6,6 ));
        // //cv::morphologyEx(show, show, cv::MORPH_DILATE, kernel);
        

        // std::vector<std::vector<cv::Point>> contours;
        // std::vector<cv::Vec4i> hierarcy;
        // findContours(show, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        // std::vector<cv::RotatedRect> box(contours.size());
        // cv::Point2f rect[4];
        // for(int i = 0;i<contours.size();i++){
        //     box[i] = cv::minAreaRect(cv::Mat(contours[i]));
        //     cv::circle(img, cv::Point(box[i].center.x, box[i].center.y), 5, cv::Scalar(0, 255, 0), -1, 8);
        //     box[i].points(rect);
        //      for(int j=0; j<4; j++)
        //     {
        //         cv::line(img, rect[j], rect[(j+1)%4], cv::Scalar(255, 0, 0), 2, 8);  //绘制最小外接矩形每条边
        //     }
        // }
        /***************************/

        cv::imshow("show",img);
        cv::imshow("result",show);
        cv::imshow("mask",mask);
        k = cv::waitKey(1);
        if(k==' ')
            k = cv::waitKey(0);
        if(k=='s'){
            cv::imwrite("./show.png",img);
            cv::imwrite("./result.png",show);
            cv::imwrite("./mask.png",mask);
            cv::imwrite("./rect.png",showrect);
        }
    } while (k!='q');
    // wmd.WriteToTXT("v.txt",2);
    // wmd.WriteToTXT("t.txt",0);
    // wmd.WriteToTXT("theta.txt",1);
    // wmd.WriteToTXT("error.txt",3);
    //std::cout<<wmd.GetImage(2);
        return 0;
}