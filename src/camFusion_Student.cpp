#include "camFusion.hpp"
#include "dataStructures.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor,
                         cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx,
                         cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 2, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 1);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int) it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &currBoundingBox, std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {

    for (const auto &match: kptMatches) {
        auto kp = kptsCurr[match.trainIdx].pt;
        auto roi = currBoundingBox.roi;
        if (kp.x >= roi.x && kp.x < roi.x + roi.width && kp.y >= roi.y && kp.y < kp.y + roi.height)
            currBoundingBox.kptMatches.emplace_back(match);
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {

    auto dT = 1 / frameRate;

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {

            const double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // inner loop
    } // outer loop

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        TTC = NAN;
        return;
    }

    // compute median dist. ratio to reduce outlier influence
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0
                                                     : distRatios[medIndex];

    dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {

    // Fraction of points, closest along the x direction, to be discarded before finding the closest point
    const double p = .02;
    double laneWidth = 4.0; // assumed width of the ego lane, in meters
    double dT = 1. / frameRate; // time between two measurements, in seconds

    /* Returns the p-percentale of the lidar points distance along x, after ignoring those points
     * with a y component grater in absolute value than `max_allowed` (which is meant to be half the lane width).
     */
    auto percentile = [](const vector<LidarPoint> &points, double p, const double max_allowed) -> double {
        vector<double> xs;
        xs.reserve(points.size());
        for (const auto &point: points)
            if (std::abs(point.y) <= max_allowed)
                xs.push_back(point.x);

        sort(xs.begin(), xs.end());
        std::size_t pos = std::round(xs.size() * p);
        double res = xs[pos];
        return res;
    };

    /* Find closest distances to Lidar points within ego lane in current and previous frame, trying to discard
     * enough points to filter out outliers */
    auto minXPrev = percentile(lidarPointsPrev, p, laneWidth / 2.);
    auto minXCurr = percentile(lidarPointsCurr, p, laneWidth / 2.);

    // Finally compute the the TTC
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {

    // Return a vector of bounding box indices (in `frame.boundingBoxes`) that contain the given keypoints
    auto find_bboxes_with_keypoint = [](const DataFrame &frame, const cv::KeyPoint &keypoint) -> vector<int> {
        vector<int> bbox_indices;
        for (int i = 0; i < frame.boundingBoxes.size(); ++i) {
            const auto roi = frame.boundingBoxes[i].roi;
            const auto point = keypoint.pt;
            if (point.x >= roi.x && point.x < roi.x + roi.width &&
                point.y >= roi.y && point.y < roi.y + roi.height)
                bbox_indices.push_back(i);
        }
        return bbox_indices;
    };

    // matches of bounding boxes with an IOU below this thresholds will be discarded and not returned
    double iou_threshold = .2;

    /* Compute and store the the number of matches between every pair of matched bounding boxes and the number
     * of matched keypoints every bounding box contains; it will be used to compute the IOU */
    map<pair<int, int>, int> intersection;
    map<int, int> union_prev;
    map<int, int> union_curr;
    for (const auto &match: matches) {
        /* For each match, find all bboxes in the previous frame that contain its first keypoint,
         * and all bboxes in the current frame that contain its second keypoin */
        auto kp1 = prevFrame.keypoints[match.queryIdx];
        auto prev_bbox_indices = find_bboxes_with_keypoint(prevFrame, kp1);
        auto kp2 = currFrame.keypoints[match.trainIdx];
        auto curr_bbox_indices = find_bboxes_with_keypoint(currFrame, kp2);
        // For each pair of bboxes in the two respective sequences, update their intersection and union calculations
        for (const auto prev_i: prev_bbox_indices)
            for (const auto curr_i: curr_bbox_indices) {
                intersection[make_pair(prev_i, curr_i)] = intersection[make_pair(prev_i, curr_i)] + 1;
                union_prev[prev_i] = union_prev[prev_i] + 1;
                union_curr[curr_i] = union_curr[curr_i] + 1;
            }
    }

    // Calculate and store the IOU of every pair of bboxes, where the first bbox in the pair is in the prev. frame
    // and the second bbox is in the curr. frame
    vector<tuple<int, int, double>> iou;
    for (int prev_bbox_i = 0; prev_bbox_i < prevFrame.boundingBoxes.size(); ++prev_bbox_i) {
        for (int curr_bbox_i = 0; curr_bbox_i < currFrame.boundingBoxes.size(); ++curr_bbox_i) {
            auto the_IOU = static_cast<double>(2. * intersection[make_pair(prev_bbox_i, curr_bbox_i)]) /
                           (union_prev[prev_bbox_i] + union_curr[curr_bbox_i]);
            if (the_IOU >= iou_threshold)
                iou.emplace_back(make_tuple(prev_bbox_i, curr_bbox_i, the_IOU));
        }
    }

    /* List the pairs of bboxes (one in the prev. frame and one in the current frame) in descending order of
     * IOU, until you have paired all the bboxes in the prev. frame or all the bboxes in the curr. frame */

    sort(iou.begin(), iou.end(),
         [](const tuple<int, int, double> &a, const tuple<int, int, double> &b) -> bool {
             return get<2>(a) > get<2>(b);
         });

    // Compile the best matches into `bbBestMatches` (each bounding box has at most one match)
    vector<bool> used_bboxes_prev_frame(prevFrame.boundingBoxes.size(), false);
    vector<bool> used_bboxes_curr_frame(currFrame.boundingBoxes.size(), false);
    map<int, double> bbox_iou;
    for (const auto &item: iou) {
        auto prev_bb_i = get<0>(item);
        auto curr_bb_i = get<1>(item);
        auto the_IOU = get<2>(item);
        if (!used_bboxes_prev_frame[prev_bb_i] && !used_bboxes_curr_frame[curr_bb_i]) {
            used_bboxes_prev_frame[prev_bb_i] = true;
            used_bboxes_curr_frame[curr_bb_i] = true;
            bbBestMatches[prevFrame.boundingBoxes[prev_bb_i].boxID] = currFrame.boundingBoxes[curr_bb_i].boxID;
            bbox_iou[prevFrame.boundingBoxes[prev_bb_i].boxID] = the_IOU;
        }
    }
}
