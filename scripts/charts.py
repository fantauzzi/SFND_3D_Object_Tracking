import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import glob


def main():
    stats = {}
    detectors = ["SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"]
    descriptors = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]
    sensors = ['LIDAR', 'CAMERA']
    file_names = glob.glob('../stats/stats_*.txt')
    for file_name in file_names:
        with open(file_name) as file:
            for line in file:
                items = line.split(sep=' ')
                detector = items[0]
                descriptor = items[1]
                sensor = items[2]
                assert sensor in sensors
                assert detector in detectors
                assert descriptor in descriptors
                stats[(detector, descriptor, sensor)] = [float(data) for data in items[3:len(items) - 4]]

    fig, axs = plt.subplots(len(detectors), len(descriptors))
    plt.subplots_adjust(hspace=.5, left=.04, right=.97, top=.95, bottom=.05)
    for row, detector in enumerate(detectors):
        for col, descriptor in enumerate(descriptors):
            ax = axs[row][col]
            for sensor in sensors:
                plotme = stats.get((detector, descriptor, sensor))
                if plotme is None:
                    continue
                ax.plot(plotme, label=sensor)
                ax.set_title(detector + " " + descriptor, size=10, fontweight='bold')
                ax.set_xticks(range(len(plotme)))

    ax = axs[0][0]
    ax.legend()
    ax.set_ylabel('TTC (sec)')

    plt.show()


if __name__ == '__main__':
    main()
