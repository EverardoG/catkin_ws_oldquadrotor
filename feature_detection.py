import numpy as np
import cv2
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# Image size is proportional to algorithm frame processing speed - bigger pictures cause more frame lag
ap.add_argument("-i",
                "--image",
                required=True,
	            help="path to the static image that will be processed for keypoints")
ap.add_argument("-l",
                "--label",
                help="string to label the object found in the camera feed" )
args = vars(ap.parse_args())


def feat_det(image):

    MIN_MATCH_COUNT = 10

    # Reads in source for cam feed
    # if VideoCapture(0) doesn't work, try -1, 1, 2, 3 (if none of those work, the webcam's not supported!)

    feed = 0
    for src in range(-1, 4):
        cam = cv2.VideoCapture(src)
        print(src)
        ret_val, testImg = cam.read()
        if testImg is None:
            continue
        else:
            feed = src

    #cam = cv2.VideoCapture(feed)
    cam = cv2.VideoCapture(0)

    # Reads in the image
    img1 = cv2.imread(image, 0)

    # Labels the image as the name passed in
    if args["label"] is not None:
        label = args["label"]
    else:
        # Takes the name of the image as the name
        if image[:2] == "./":
            label = image[2:-4]
        else:
            label = image[:-4]

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # Initiate SURF detector
    surf = cv2.xfeatures2d.SURF_create(200)

    # Find the keypoints and descriptors of the provided image with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    # Find the keypoints and descriptors of the provided image with SIFT
    kp2, des2 = sift.detectAndCompute(img1, None)
    # Find the keypoints and descriptors of the provided image with SURF
    kp3, des3 = surf.detectAndCompute(img1, None)

    FLANN_INDEX_KDTREE = 0
    # Option of changing 'trees' and 'checks' values for different levels of accuracy
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    # 5
    search_params = dict(checks = 50)                                 # 50

    # Fast Library for Approximate Nearest Neighbor Algorithm
    # creates FLANN object for use below
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Brute Force matcher
    # creates BFMatcher object for use below
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    # Setup for keypresses that allow the user to switch between the algorithms
    # the default algorithm on startup is ORB
    alg = 1

    # handles keypresses
    while True:
        c = cv2.waitKey(1)

        if c == ord('o'):
            alg = 1

        elif c == ord('i'):
            alg = 2

        elif c == ord('s'):
            alg = 3

        ret_val, img2 = cam.read()

        if img2 is None :               # did we get an image at all?
            #print ("No image")
            continue

        # The 'o' key starts the Orb algorithm, trades accuracy for speed
        if alg == 1:
            kp1_s, des1_s = orb.detectAndCompute(img2, None)

            # Uses the BFMatcher algorithm to search for similar elements of two images
            # uses simpler calculations than FLANN
            matches = bf.knnMatch(des1, des1_s, k = 2 )
            kp = kp1
            kp_L = kp1_s
            alg_type = "Orb"

        # The 'i' key starts the Sift algorithm, trades speed for accuracy
        elif alg == 2:
            kp2_s, des2_s = sift.detectAndCompute(img2, None)

            # Uses the FLANN algorithm to search for nearest neighbors between elements of two images
            # faster than the BFMatcher for larger datasets
            matches = flann.knnMatch(des2, des2_s, k = 2)
            kp = kp2
            kp_L = kp2_s
            alg_type = "Sift"

        # The 's' key starts the Surf algorithm, trades speed for accuracy
        # and is supposed to be faster than SIFT, but OpenCV's implementation is unoptimized
        elif alg == 3:
            kp3_s, des3_s = surf.detectAndCompute(img2, None)

            # Uses the FLANN algorithm to search for nearest neighbors between elements of two images
            # faster than the BFMatcher for larger datasets
            matches = flann.knnMatch(des3, des3_s, k = 2)
            kp = kp3
            kp_L = kp3_s
            alg_type = "Surf"

        if len(matches) == 0:
            continue

        # Store all the good matches (based off Lowe's ratio test)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # When there are enouugh matches, we convert the keypoints to floats in order to draw them later
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_L[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            # Homography adds a degree of rotation/translation invariance by mapping the transformation
            # of points between two images
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            if M is not None:
                dst = cv2.perspectiveTransform(pts,  M)

                intDst = np.int32(dst)

                # Polylines is an alternative bounding shape around the largest area of recognition
                #cv2.polylines(img2,[np.int32(dst)],True,(255, 0, 0), 2, cv2.LINE_AA)

                # Draws a bounding box around the area of most matched points
                cv2.rectangle(img2, (intDst[0][0][0], intDst[0][0][1]), (intDst[2][0][0], intDst[2][0][1]), (255, 0, 0), 1, cv2.LINE_AA, 0)
                cv2.putText(img2, label, (dst[0][0][0], dst[0][0][1]) , cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), lineType = cv2.LINE_AA )

            else:
                matchesMask = None

        else:
            #print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

        # Option of slicing the 'good' list to display a certain number of matches (ex. good[:6])
        img3 = cv2.drawMatches(img1, kp, img2, kp_L, good, None, **draw_params)
        # Makes the window 1.5x bigger
        img3 = cv2.resize(img3, (0,0), fx = 1.5, fy = 1.5)

        # Displays the keys for switching algorithms and the current algorithm
        height, width = img3.shape[:2]
        cv2.putText(img3, "Hold 'o' for ORB, 'i' for SIFT, and 's' for SURF", (width - 950, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.rectangle(img3, (width - 200, 0), (width, 40), (255,255,255), -1)
        cv2.putText(img3, "Detection Alg: "+ alg_type, (width - 190, 25), cv2.FONT_HERSHEY_SIMPLEX,
                     0.6, (0,0,0), 2, 9)

        if cv2.waitKey(1) == 27:
                    break  # esc to quit

        # Shows the current frame
        cv2.imshow("My Webcam", img3)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    feat_det(args["image"])
