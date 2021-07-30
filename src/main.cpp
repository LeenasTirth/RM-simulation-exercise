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

    char k ='a';
    do
    {
        double nowtime = std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
        cv::Mat img =wm.getMat(nowtime);


        wmd.SetFrame(img);
        try{
            wmd.Process(nowtime,1);
        }
        catch(std::string e){
            std::cerr<<e;
        }
        //cv::Mat showrect = wmd.GetImage(2);
        //cv::imshow("rect",showrect);
        cv::Mat show = wmd.GetShowMat(nowtime);
        cv::Mat mask = wmd.GetImage(3);

        cv::imshow("show",img);
        cv::imshow("result",show);
        cv::imshow("mask",mask);
        k = cv::waitKey(1);
        if(k==' ')
            k = cv::waitKey(0);
        if(k=='s'){
            // cv::imwrite("./show.png",img);
            // cv::imwrite("./result.png",show);
            // cv::imwrite("./mask.png",mask);
            // cv::imwrite("./rect.png",showrect);
        }
    } while (k!='q');
        return 0;
}