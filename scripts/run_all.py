import subprocess;


def main():
    detectors = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
    descriptors = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]

    for detector in detectors:
        for descriptor in descriptors:
            subprocess.call(['../build/3D_object_tracking', detector, descriptor]);


if __name__ == '__main__':
    main()
