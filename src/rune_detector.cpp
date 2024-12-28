#include <bits/stdc++.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <sstream>
#include <vector>
cv::Mat preprocess(const cv::Mat& frame); // 预处理函数
double calculateAngleDifference(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算角度差
double calculateAxisLength(const cv::RotatedRect& ellipse, const cv::Point2f& direction); // 计算椭圆沿特定方向的轴的长度
double calculateRatioDifferenceHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算比例差
double calculateMatchScoreHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算匹配程度
double calculateRatioDifferenceHitted(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算比例差
double calculateMatchScoreHitted(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse); // 计算匹配程度
std::vector<cv::Point2f> ellipseIntersections(const cv::RotatedRect& ellipse, const cv::Point2f& dir); // 计算指定方向与椭圆的两个交点

std::vector<cv::Point2f> get_signal_points(const cv::RotatedRect& ellipse, const cv::RotatedRect& rect); // 提取 6 个 signal points
std::vector<cv::Point2f> processHittingLights(const cv::Mat& flow_img, cv::Mat& frame); // 处理击打灯条
std::vector<std::vector<cv::Point2f>> processHittedLights(const cv::Mat& arm_img, const cv::Mat& hited_img, cv::Mat& frame); // 处理 已击中灯条
int main(){
    // 创建一个VideoCapture对象并打开视频文件
    cv::VideoCapture cap("/home/mozijun/Mycode_c/rune_detector/video.mp4");

    // 检查视频是否成功打开
    if(!cap.isOpened()){
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    // 获取视频帧率和帧大小
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 创建VideoWriter对象
    cv::VideoWriter video("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));

    cv::Mat frame;
    // 循环读取视频帧
    while(cap.read(frame)){
        if(frame.empty()) break;

        // 预处理当前帧
        cv::Mat aim_img = preprocess(frame);

        // 定义结构元素
        cv::Mat element_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat element_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat flow_img;

        // 先进行膨胀操作
        cv::dilate(aim_img, flow_img, element_dilate);

        // 再进行侵蚀操作
        cv::erode(flow_img, flow_img, element_erode);
        std::vector<cv::Point2f> signal_points_hitting = processHittingLights(flow_img, frame);


        cv::Mat arm_img; // 转臂图像
        cv::erode(aim_img, arm_img, element_dilate);

        cv::Mat hited_img; // 已打击装甲板图像
        std::vector<cv::RotatedRect> hited_lights;
        cv::dilate(aim_img, hited_img, element_dilate);
        std::vector<std::vector<cv::Point2f>> signal_points_hitted = processHittedLights(arm_img, hited_img, frame);

        // 将处理后的帧写入视频
        video.write(frame);
    }

    // 释放VideoCapture对象
    cap.release();
    // 释放VideoWriter对象
    video.release();
    // 关闭所有OpenCV窗口
    cv::destroyAllWindows();

    return 0; 
}
cv::Mat preprocess(const cv::Mat& frame) {
    cv::Mat gray, binary;

    // 转换为灰度图像
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::threshold(gray, binary, 80, 255, cv::THRESH_BINARY);

    return binary;
}
// 计算角度差
double calculateAngleDifference(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    cv::Point2f rect_center = rect.center;
    cv::Point2f ellipse_center = ellipse.center;
    cv::Point2f rect_vertices[4];
    rect.points(rect_vertices);

    // 计算矩形长边方向向量
    cv::Point2f rect_long_edge = rect_vertices[1] - rect_vertices[0];
    if (cv::norm(rect_vertices[2] - rect_vertices[1]) > cv::norm(rect_long_edge)) {
        rect_long_edge = rect_vertices[2] - rect_vertices[1];
    }

    // 计算中心连线向量
    cv::Point2f center_line = ellipse_center - rect_center;

    // 计算角度
    double rect_angle = std::atan2(rect_long_edge.y, rect_long_edge.x);
    double center_line_angle = std::atan2(center_line.y, center_line.x);
    double angle_diff = std::abs(rect_angle - center_line_angle);
    angle_diff = std::min(angle_diff, CV_PI * 2 - angle_diff);
    return std::min(angle_diff, CV_PI - angle_diff);
}
// 计算椭圆沿特定方向的轴的长度
double calculateAxisLength(const cv::RotatedRect& ellipse, const cv::Point2f& direction) {
    // 椭圆的长轴和短轴长度
    double a = ellipse.size.width / 2.0;  // 长轴的一半
    double b = ellipse.size.height / 2.0; // 短轴的一半

    // 椭圆的旋转角度（以弧度表示）
    double theta = ellipse.angle * CV_PI / 180.0;

    // 方向向量的单位化
    cv::Point2f unit_direction = direction / cv::norm(direction);

    // 计算旋转后的方向向量
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);
    double x_prime = unit_direction.x * cos_theta + unit_direction.y * sin_theta;
    double y_prime = -unit_direction.x * sin_theta + unit_direction.y * cos_theta;

    // 计算沿特定方向的弦的长度
    double chord_length = 2 * a * b / std::sqrt(b * b * x_prime * x_prime + a * a * y_prime * y_prime);

    return chord_length;
}
// 计算比例差
double calculateRatioDifferenceHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double center_distance = cv::norm(rect.center - ellipse.center);
    // 计算椭圆沿特定方向的轴的长度
    cv::Point2f rect_vertices[4];
    rect.points(rect_vertices);
    cv::Point2f rect_long_edge = rect_vertices[1] - rect_vertices[0];
    if (cv::norm(rect_vertices[2] - rect_vertices[1]) > cv::norm(rect_long_edge)) {
        rect_long_edge = rect_vertices[2] - rect_vertices[1];
    }
    double ellipse_axis = calculateAxisLength(ellipse, rect_long_edge);

    double ratio1 = ellipse_axis / cv::norm(rect_long_edge);
    double ratio2 = cv::norm(rect_long_edge) / center_distance;

    double target_ratio1 = 310.0 / 330.0;
    double target_ratio2 = 330.0 / 355.0;

    double ratio_diff1 = std::abs(ratio1 - target_ratio1);
    double ratio_diff2 = std::abs(ratio2 - target_ratio2);

    return ratio_diff1 + ratio_diff2;
}

// 计算匹配程度
double calculateMatchScoreHitting(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double angle_diff = calculateAngleDifference(rect, ellipse);
    double ratio_diff = calculateRatioDifferenceHitting(rect, ellipse);

    // 归一化并计算总分
    double angle_score = angle_diff / CV_PI;
    double ratio_score = ratio_diff / 2.0; // 假设最大比例差为2

    if(angle_diff < CV_PI / 12 && ratio_score < 0.2) return angle_score + ratio_score;
    else return -1; // 不匹配
}
// 计算比例差
double calculateRatioDifferenceHitted(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double rect_long_edge = std::max(rect.size.width, rect.size.height);
    double center_distance = cv::norm(rect.center - ellipse.center);
    double ratio_score = rect_long_edge / center_distance;
    double target_ratio = 330.0 / 355.0;
    return std::abs(ratio_score - target_ratio);
}
// 计算匹配程度
double calculateMatchScoreHitted(const cv::RotatedRect& rect, const cv::RotatedRect& ellipse) {
    double angle_diff = calculateAngleDifference(rect, ellipse);
    double ratio_diff = calculateRatioDifferenceHitted(rect, ellipse);

    // 归一化并计算总分
    double angle_score = angle_diff / CV_PI;
    double ratio_score = ratio_diff; // 假设最大比例差为1
    if(angle_diff < CV_PI / 12 && ratio_score < 0.2) return angle_score + ratio_score;
    else return -1; // 不匹配
}
// 计算指定方向与椭圆的两个交点
std::vector<cv::Point2f> ellipseIntersections(const cv::RotatedRect& ellipse, const cv::Point2f& dir) {
    // 单位化方向
    cv::Point2f unit_dir = dir / cv::norm(dir);

    // 椭圆长短轴半径
    double a = ellipse.size.width * 0.5;
    double b = ellipse.size.height * 0.5;

    // 旋转角（弧度）
    double theta = ellipse.angle * CV_PI / 180.0;
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

    // 方向向量旋转到椭圆坐标系
    double x_ =  unit_dir.x * cos_t + unit_dir.y * sin_t;
    double y_ = -unit_dir.x * sin_t + unit_dir.y * cos_t;

    // 计算半弦长
    double half_len = (a * b) / std::sqrt(b * b * x_ * x_ + a * a * y_ * y_);

    // 椭圆中心
    cv::Point2f c = ellipse.center;

    // 原方向向量在图像坐标系下的分量（逆旋转）
    // 为了得到在图像坐标系下 ±half_len 的坐标，需要将 (±x_, ±y_) 再旋转回来
    // 可直接在单位化方向上乘以 half_len，分别正负即可
    cv::Point2f dir_n = unit_dir * static_cast<float>(half_len);

    // 交点1、交点2 = 椭圆中心 ± dir_n
    std::vector<cv::Point2f> pts(2);
    pts[0] = c + dir_n; 
    pts[1] = c - dir_n;
    return pts;
}
// 提取 6 个 signal points
std::vector<cv::Point2f> get_signal_points(const cv::RotatedRect& ellipse, const cv::RotatedRect& rect){
    // 最终返回的 76 个点
    // [0]、[1] = 矩形两条短边的中心；[2] ~ [5] = 椭圆交点；(共 6 个)
    std::vector<cv::Point2f> result(6, cv::Point2f(0,0));

    // 1. 找矩形的两条短边中心
    cv::Point2f pts[4];
    rect.points(pts);
    // 计算四条边长度
    std::vector<std::pair<float,int>> edges; // (边长度, 起点索引)
    for(int i=0; i<4; i++){
        float len = cv::norm(pts[(i+1)%4] - pts[i]);
        edges.push_back(std::make_pair(len, i));
    }
    // 按边长排序
    std::sort(edges.begin(), edges.end(),
              [](auto &a, auto &b){return a.first < b.first;});

    // edges[0], edges[1] 即为两条短边
    auto idx0 = edges[0].second; 
    auto idx1 = edges[1].second; 
    cv::Point2f mid0 = 0.5f * (pts[idx0] + pts[(idx0+1)%4]);
    cv::Point2f mid1 = 0.5f * (pts[idx1] + pts[(idx1+1)%4]);

    // 根据离椭圆中心距离判断谁放 0 号位
    float d0 = cv::norm(mid0 - ellipse.center);
    float d1 = cv::norm(mid1 - ellipse.center);
    if(d0 > d1){
        result[0] = mid0; 
        result[1] = mid1;
    } else {
        result[0] = mid1; 
        result[1] = mid0;
    }

    // 2. 计算椭圆中心与矩形中心之间的连线 dir
    cv::Point2f dir = rect.center - ellipse.center;
    // 垂直方向 dir_perp
    cv::Point2f dir_perp(-dir.y, dir.x);

    // 3. 分别计算这两条方向与椭圆的交点 (各自 2 个)
    std::vector<cv::Point2f> pts_dir  = ellipseIntersections(ellipse, dir);
    std::vector<cv::Point2f> pts_perp = ellipseIntersections(ellipse, dir_perp);

    // 合并成 4 个点
    std::vector<cv::Point2f> four_pts;
    four_pts.insert(four_pts.end(), pts_dir.begin(),  pts_dir.end());
    four_pts.insert(four_pts.end(), pts_perp.begin(), pts_perp.end());

    // 4. 找到距离矩形中心最近的点放在位置 [2]，其余按顺时针顺序放 [3]、[4]、[5]
    // 先找距离 rect.center 最近的点
    float min_dist = 1e9f;
    int   min_idx = 0;
    for(int i=0; i<4; i++){
        float dist = cv::norm(four_pts[i] - rect.center);
        if(dist < min_dist){
            min_dist = dist;
            min_idx = i;
        }
    }
    // 把最近的点放 2 号
    result[2] = four_pts[min_idx];
    
    // 剩下 3 个点，按顺时针顺序放 [3],[4],[5]
    // 可先把最近点移除，再以矩形中心为参考进行 atan2 排序
    cv::Point2f base = rect.center;
    std::vector<cv::Point2f> remain;
    for(int i=0; i<4; i++){
        if(i != min_idx) remain.push_back(four_pts[i]);
    }
    // 以矩形中心为参考，按顺时针(atan2)排序
    std::sort(remain.begin(), remain.end(), 
              [base](const cv::Point2f &p1, const cv::Point2f &p2){
                  double a1 = std::atan2(p1.y - base.y, p1.x - base.x);
                  double a2 = std::atan2(p2.y - base.y, p2.x - base.x);
                  return a1 < a2;
              });
    result[3] = remain[0];
    result[4] = remain[1];
    result[5] = remain[2];

    return result;
}
std::vector<cv::Point2f> processHittingLights(
    const cv::Mat& flow_img,  // 形态学处理后的图像
    cv::Mat& frame)           // 用于绘制的原图或当前帧
{
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(flow_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 筛选并保存长宽比在 2.5 到 7 之间的旋转矩形
    std::vector<cv::RotatedRect> flow_lights;
    for (const auto& contour : contours) {
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        double aspect_ratio = static_cast<double>(rotated_rect.size.width) / rotated_rect.size.height;
        if (aspect_ratio < 1.0) {
            aspect_ratio = 1.0 / aspect_ratio;
        }
        if (aspect_ratio >= 2.5 && aspect_ratio <= 7.0 && rotated_rect.size.area() >= 20) {
            flow_lights.push_back(rotated_rect);
        }
    }

    // 储存含有两个以上子轮廓的父轮廓并用椭圆拟合
    std::vector<cv::RotatedRect> aim_lights;
    for (size_t i = 0; i < contours.size(); i++) {
        int child_count = 0;
        for (int j = hierarchy[i][2]; j != -1; j = hierarchy[j][0]) {
            child_count++;
        }
        // 检查是否有两个以上子轮廓并且轮廓点数大于等于 5
        if (child_count >= 2 && contours[i].size() >= 5) {
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
            aim_lights.push_back(ellipse);
        }
    }

    // 进行匹配
    std::pair<cv::RotatedRect, cv::RotatedRect> matched_light; 
    bool matched = false;
    double min_score = 1e9;

    for (const auto& aim_light : aim_lights) {
        for(const auto& flow_light : flow_lights) {
            double score = calculateMatchScoreHitting(flow_light, aim_light);
            if (score < min_score && score != -1) {
                min_score = score;
                matched = true;
                matched_light.first = flow_light;
                matched_light.second = aim_light;
            }
        }
    }

    std::vector<cv::Point2f> signal_points_hitting;
    if (matched) {
        cv::RotatedRect flow_light = matched_light.first;
        cv::RotatedRect aim_light = matched_light.second;

        // 绘制匹配的灯条
        cv::Point2f flow_light_vertices[4];
        flow_light.points(flow_light_vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(frame, flow_light_vertices[i], flow_light_vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        // 绘制匹配的椭圆
        cv::ellipse(frame, aim_light, cv::Scalar(0, 255, 0), 2);

        // 获取关键点
        signal_points_hitting = get_signal_points(aim_light, flow_light);
    }

    return signal_points_hitting;
}
std::vector<std::vector<cv::Point2f>> processHittedLights(
    const cv::Mat& arm_img, 
    const cv::Mat& hited_img,
    cv::Mat& frame)
{
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // 处理 arm_img
    cv::findContours(arm_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> arm_lights;
    for (const auto& contour : contours) {
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        double aspect_ratio = static_cast<double>(rotated_rect.size.width) / rotated_rect.size.height;
        if (aspect_ratio < 1.0) {
            aspect_ratio = 1.0 / aspect_ratio;
        }
        if (aspect_ratio >= 2.5 && aspect_ratio <= 7.0 && rotated_rect.size.area() >= 20) {
            arm_lights.push_back(rotated_rect);
        }
    }

    // 清空并处理 hited_img
    contours.clear();
    cv::findContours(hited_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> hited_lights;
    for (size_t i = 0; i < contours.size(); i++) {
        int child_count = 0;
        for (int j = hierarchy[i][2]; j != -1; j = hierarchy[j][0]) {
            child_count++;
        }
        // 检查是否有一个子轮廓并且轮廓点数 >=5
        if (child_count == 1 && contours[i].size() >= 5) {
            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
            hited_lights.push_back(ellipse);
        }
    }

    // 进行匹配
    std::vector<std::pair<cv::RotatedRect, cv::RotatedRect>> matched_arm_lights;
    for (const auto& arm_light : arm_lights) {
        double local_min_score = 1e9;
        cv::RotatedRect matched_hited_light;
        for(const auto& hited_light : hited_lights){
            double score = calculateMatchScoreHitted(arm_light, hited_light);
            if (score < local_min_score && score != -1) {
                local_min_score = score;
                matched_hited_light = hited_light;
            }
        }
        if(local_min_score != 1e9) {
            matched_arm_lights.push_back(std::make_pair(arm_light, matched_hited_light));
        }
    }

    // 绘制结果并保存关键点
    std::vector<std::vector<cv::Point2f>> signal_points_hitted;
    for (const auto& matched_arm_light : matched_arm_lights) {
        cv::RotatedRect arm_light = matched_arm_light.first;
        cv::RotatedRect hited_light = matched_arm_light.second;

        // 绘制匹配的灯条
        cv::Point2f arm_light_vertices[4];
        arm_light.points(arm_light_vertices);
        for (int i = 0; i < 4; i++) {
            cv::line(frame, arm_light_vertices[i], arm_light_vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }
        // 绘制匹配的椭圆
        cv::ellipse(frame, hited_light, cv::Scalar(0, 0, 255), 2);

        // 获取关键点
        signal_points_hitted.push_back(get_signal_points(hited_light, arm_light));
    }

    return signal_points_hitted;
}