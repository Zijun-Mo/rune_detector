#include <bits/stdc++.h>
#include <opencv2/core/mat.hpp>
#include <opencv4/opencv2/opencv.hpp>
// 预处理函数：二值化和闭运算
cv::Mat preprocess(const cv::Mat& frame) {
    cv::Mat gray, binary, morph;

    // 转换为灰度图像
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);

    cv::imshow("binary", binary);

    return morph;
}
int main(){
    // 创建一个VideoCapture对象并打开视频文件
    cv::VideoCapture cap("/home/mozijun/Mycode_c/rune_detector/video.mp4");

    // 检查视频是否成功打开
    if(!cap.isOpened()){
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }
    cv::Mat frame;
    // 循环读取视频帧
    while(cap.read(frame)){
        if(frame.empty()) break;

        // 预处理当前帧
        cv::Mat aim_img = preprocess(frame);

        // 定义结构元素
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat flow_img; 
        cv::morphologyEx(aim_img, flow_img, cv::MORPH_CLOSE, element);

    }

    // 释放VideoCapture对象
    cap.release();
    // 关闭所有OpenCV窗口
    cv::destroyAllWindows();

    return 0; 
}