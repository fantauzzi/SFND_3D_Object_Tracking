#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <unistd.h>


using namespace std;


int main(int argc, const char *argv[]) {

    vector<string> detectorTypes{"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    // Check validity of provided arguments. Exit with usage message if not valid.

    auto print_usage = [&]() {

        auto print_vector = [](const vector<string> &v) {
            for (const auto &item: v)
                cerr << item << " ";
        };

        cerr << "USAGE:" << endl;
        cerr << "       3D_object_tracking <detector> <descriptor> [on]" << endl;
        cerr << "              <detector> -one of ";
        print_vector(detectorTypes);
        cerr << ";" << endl;
        cerr << "              <descriptors> -one of ";
        print_vector(descriptorTypes);
        cerr << ";" << endl;
        cerr << "              on -optional, it turns on graphic visualization." << endl;
        exit(-1);
    };

    if (argc < 3 || argc > 4)
        print_usage();

    string detectorType = argv[1];
    if (find(detectorTypes.begin(), detectorTypes.end(), detectorType) == detectorTypes.end()) {
        cerr << "Unknown detector " << detectorType << endl;
        print_usage();
    }

    string descriptorType = argv[2];
    if (find(descriptorTypes.begin(), descriptorTypes.end(), descriptorType) == descriptorTypes.end()) {
        cerr << "Unknown descriptor " << descriptorType << endl;
        print_usage();
    }

    bool on_option = false;

    if (argc == 4) {
        if (strcmp(argv[3], "on") != 0) {
            cerr << "Unrecognized argument " << argv[3] << endl;
            print_usage();
        }
        on_option = true;
    }

    const bool bVis = on_option;

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4, 4, cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    vector<string> sensorTypes = {"LIDAR", "CAMERA"};


    // Will store TTC data: <detector, descriptor, camera/lidar> -> vector of TTC
    map<tuple<string, string, string>, vector<double>> stats;

    string stats_file_base_name = "stats/stats";  // File name to contain the performance report

    string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    cout << "Working on detector " << detectorType << ", descriptor " << descriptorType << " ";

    // The Akaze detector only works with the Akaze descriptor, and vice-versa
    // The SIFT detector doesn't work with the ORB descriptor
    if ((detectorType == "AKAZE" && descriptorType != "AKAZE") ||
        (descriptorType == "AKAZE" && detectorType != "AKAZE") ||
        (detectorType == "SIFT" && descriptorType == "ORB")) {
        cerr << "Combination of detector " << detectorType << " and descriptor " << descriptorType << " not supported."
             << endl;
        exit(0);
    }

    // The SIFT descriptor does not work with BF matcher
    if (descriptorType == "SIFT" && matcherType == "MAT_BF") {
        cerr << "Combination of descriptor " << descriptorType << " and matcher " << matcherType << " not supported."
             << endl;
        exit(0);
    }

    string descriptorKind = (descriptorType == "SIFT") ? "DES_HOG" : "DES_BINARY";

    // Loop over all images
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {
        cout << "." << flush;

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold,
                      nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        // cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        // cout << "#3 : CROP LIDAR POINTS done" << endl;

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints,
                            shrinkFactor,
                            P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        if (bVis) {
            show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1400, 1400),
                          true);
        }

        // cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        if (detectorType == "SHITOMASI")
            detGoodFeaturesToTrack(keypoints, imgGray, bVis);
        else if (detectorType == "HARRIS")
            detKeypointsHarris(keypoints, imgGray, bVis);
        else if (detectorType == "FAST")
            detKeypointsModern(keypoints, img, "FAST", bVis);
        else if (detectorType == "BRISK")
            detKeypointsModern(keypoints, img, "BRISK", bVis);
        else if (detectorType == "ORB")
            detKeypointsModern(keypoints, img, "ORB", bVis);
        else if (detectorType == "AKAZE")
            detKeypointsModern(keypoints, img, "AKAZE", bVis);
        else if (detectorType == "SIFT")
            detKeypointsModern(keypoints, img, "SIFT", bVis);
        else {
            cerr << "Unknown detector type: " << detectorType << endl;
            exit(-1);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false; // Remember to remove before collecting final output
        if (bLimitKpts) {
            int maxKeypoints = 50;
            // there is no response info, so keep the first 50 as they are sorted in descending quality order
            if (detectorType == "SHITOMASI" || detectorType == "HARRIS")
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            else
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        // cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors,
                      descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorKind, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            if (bVis) {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg,
                                (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName =
                        "Matching keypoints with " + detectorType + " " + descriptorType + " " + matcherType;
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }

            // cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;


            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;

            matchBoundingBoxes(matches,
                               bbBestMatches,
                               *(dataBuffer.end() - 2),
                               *(dataBuffer.end() -
                                 1)); // associate bounding boxes between current and previous frame using keypoint matches

            // store matches in current data frame
            (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

            // cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
                 it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
                     it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
                    if (it1->second == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
                     it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
                    if (it1->first == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                // compute TTC for current match
                if (!currBB->lidarPoints.empty() &&
                    !prevBB->lidarPoints.empty()) // only compute TTC if we have Lidar points
                {
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar;
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    auto header_stats = make_tuple(detectorType, descriptorType, "LIDAR");
                    stats[header_stats].push_back(ttcLidar);

                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints,
                                             (dataBuffer.end() - 1)->keypoints,
                                             (dataBuffer.end() - 1)->kptMatches);
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                     currBB->kptMatches, sensorFrameRate, ttcCamera);
                    header_stats = make_tuple(detectorType, descriptorType, "CAMERA");
                    stats[header_stats].push_back(ttcCamera);

                    if (bVis) {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                                      cv::Point(currBB->roi.x + currBB->roi.width,
                                                currBB->roi.y + currBB->roi.height),
                                      cv::Scalar(0, 255, 0), 2);

                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2,
                                cv::Scalar(0, 0, 255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                } // TTC computation
            } // loop over all BB matches
        } // Keypoints matching
    } // loop over all images
    cout << endl;

    // Saves stats to a file named stats_<descriptor>_<detector>.txt
    string complete_file_name = dataPath + stats_file_base_name + "_" + detectorType + "_" + descriptorType + ".txt";
    ofstream stats_file(complete_file_name);
    if (!stats_file.is_open()) {
        cerr << "Unable to open file " << complete_file_name << " for writing." << endl;
        exit(-1);
    }

    for (const auto &detector: detectorTypes)
        for (const auto &descriptor: descriptorTypes)
            for (const auto &sensor: sensorTypes) {
                auto data_header = make_tuple(detector, descriptor, sensor);
                auto data_i = stats.find(data_header);
                if (data_i == stats.end())
                    continue;
                stats_file << detector << " " << descriptor << " " << sensor << " ";
                for (const auto item: stats[data_header])
                    stats_file << item << " ";
                stats_file << endl;
            }

    return 0;
}
