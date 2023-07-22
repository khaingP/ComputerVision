import cv2 as cv
import numpy as np
from sklearn.cluster import AffinityPropagation

SOURCE_IMAGE_PATH = 'saw1.jpg'
# TARGET_IMAGE_PATH = 'simple.png'
Video_Path = 'sawmovie1.mp4'
FLANN_INDEX_KDTREE = 1
FLANN_TREES = 5
FLANN_CHECKS = 50
FLANN_K = 2
KEYPOINT_VALIDITY_THRESHOLD = 0.6
AFFINITY_DAMPING = 0.9


def main():
    video = cv.VideoCapture(Video_Path)
    sift = cv.SIFT_create()
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    source_image = cv.imread(SOURCE_IMAGE_PATH)
    source_image = cv.resize(source_image, (frame_width, frame_height), interpolation=cv.INTER_LINEAR)
    gray_source = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

    source_keypoints, source_descriptors = sift.detectAndCompute(gray_source, None)
    # marked_source = cv.drawKeypoints(source_image, source_keypoints, None)
    # cv.imshow('Source', marked_source)

    while video.isOpened():
        ret, frame = video.read()
        frame = cv.resize(frame, (source_image.shape[1], source_image.shape[0]),
                          interpolation=cv.INTER_LINEAR)

        gray_target = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        target_keypoints, target_descriptors = sift.detectAndCompute(gray_target, None)
        # marked_target = cv.drawKeypoints(frame, target_keypoints, None)
        # cv.imshow('Target', marked_target)

        index_params = {'algorithm': FLANN_INDEX_KDTREE, 'trees': FLANN_TREES}
        search_params = {'checks': FLANN_CHECKS}
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(source_descriptors, target_descriptors, k=FLANN_K)

        print(matches[0])

        matches_mask = []
        valid_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < KEYPOINT_VALIDITY_THRESHOLD * n.distance:
                matches_mask.append([m])
                valid_matches.append(target_keypoints[matches[i][0].trainIdx].pt)
        valid_matches = np.asarray(valid_matches, dtype=np.int32)
        valid_matches_2 = np.reshape(valid_matches, (-1, 2))

        draw_parameters = {'matchColor': (0, 0, 255), 'singlePointColor': (255, 0, 0),
                           'matchesMask': matches_mask, 'flags': cv.DrawMatchesFlags_DEFAULT}
        matches_visualisation = cv.drawMatchesKnn(source_image, source_keypoints,
                                                  frame, target_keypoints,
                                                  matches_mask, None,
                                                  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('Matches', matches_visualisation)

        detected_visualisation = frame.copy()
        for point in valid_matches:
            cv.circle(detected_visualisation, tuple(point), 5, (0, 0, 255))

        af = AffinityPropagation(damping=AFFINITY_DAMPING).fit(valid_matches_2)
        cluster_center_indices = af.cluster_centers_indices_
        labels = af.labels_
        cluster_count = len(cluster_center_indices)
        for cluster in range(cluster_count):
            cluster_points = valid_matches_2[labels == cluster]
            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)
            cv.rectangle(detected_visualisation, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

        cv.imshow('Detected', detected_visualisation)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
