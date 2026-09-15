import cv2
import numpy as np


# Harris Corner Detection Task
def harris_corner_detection(reference_img):
    # Read the image
    img = cv2.imread(reference_img)

    # Convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)  #dst contains information about How likely is each point in the image to be a corner

    # Expand the corners for better visibility
    dst = cv2.dilate(dst, None)

    # Mark the corners in red
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Save the image
    cv2.imwrite('harris.png', img)

    return img  #original picture with red corners

# Call the function
harris_corner_detection("reference_img.png")



# Feature-Based Image Alignment Task (Using SIFT Method)
def align_images(image_to_align, reference_image, max_features, good_match_precent):

    # Read the images
    image_to_align = cv2.imread(image_to_align)
    reference_image = cv2.imread(reference_image)

    # Convert to greyscale
    gray1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print("Keypoints image 1:", len(kp1))
    print("Keypoints image 2:", len(kp2))

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the two best matches for each descriptor
    matches = flann.knnMatch(des1, des2, k=2)

    # Select good matches using Lowe's ratio test
    good_matches = []

    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)

    print("Number of good matches:", len(good_matches))

    # Check if there are enough matches for homography
    if len(good_matches) < 4:
        print("Not enough good matches found.")
        return None, None

    # Get coordinates of matching points
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)


    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # Check if homography was found
    if M is None:
        print("Could not find homography.")
        return None, None

    # Get size of reference image
    height, width = reference_image.shape[:2]

    # Align the image
    aligned = cv2.warpPerspective(image_to_align, M,(width, height))

    # Draw matches
    matches_image = cv2.drawMatches(
        image_to_align,
        kp1,
        reference_image,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save the two results
    cv2.imwrite("aligned.png", aligned)
    cv2.imwrite("matches.png", matches_image)

    print("aligned.png and matches.png were saved.")

    return aligned, matches


# Run the function
aligned, matches = align_images(
    "align_this.jpg",
    "reference_img.png",
    10,
    0.7
)

