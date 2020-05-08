# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 3.1
* make >= 4.1 
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT detector
* gcc/g++ >= 5.4
* python >= 3.6 and its [`matplotlib`](https://matplotlib.org) package (optional, to launch the application with all combinations of detector/descriptor and to display performance charts)

## Build Instructions

1. Clone this repo `git clone https://github.com/fantauzzi/SFND_3D_Object_Tracking.git`
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`

## Downloading the Dataset

Image detection relies on the [Yolo](https://pjreddie.com/darknet/yolo/) pre-trained neural network, whose weights need to be downloaded into directory `dat/yolo` in order to run the application.

From directory `dat/yolo` run:

```shell script
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

## Running Instructions

To **run the application** with a choice of keypoints detector and descriptor, in the `build` directory:
```shell script
./3D_object_tracking <detector> <descriptor> [on]
```

* `<detector>` is one of:
* `<descriptor>` is one of:
* `on` is an optional argument, which requests to display matched keypoints, bounding boxes and a top view of the lidar data.

The application will save the computed time-to-collision (TTC) for camera and lidar in a text file named `stats_<detector>_<descriptor>.txt` under the `stats` directory, overwriting existing files as needed. Entries in the file are space separated. 

A Python script is available to **automate the launching of the application** with all the combinations of detectors and descriptors. In directory `script` run:
```shell script
python run_all
```  

To **display charts** of the TTC from data saved by the application run, in the `script` directory:
```shell script
python charts.py
```

## Project Rubric

Traceability of requirements for Udacity's Nanodegree project.

 * **FP.0 Final Report** -This README.
 * **FP.1 Match 3D Objects** 
 * **FP.2 Compute Lidar-based TTC**
 * **FP.3 Associate Keypoint Correspondences with Bounding Boxes**
 * **FP.4 Compute Camera-based TTC**
 * **FP.5 Performance Evaluation 1**
 * **FP.6 Performance Evaluation 2**
 
