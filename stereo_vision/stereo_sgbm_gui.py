import cv2
import numpy as np

# Read left and right rectified images
img_l = cv2.imread('image_fixed_l.png', 0)
img_r = cv2.imread('image_fixed_r.png', 0)
if (img_l is None) or (img_r is None):
    raise Exception('Unable to read input images')

# GUI handler: No operation
def nop(x):
    pass

# Create GUI
window_name = 'Stereo Matcher'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 800)
cv2.createTrackbar('block_size', window_name, 2, 50, nop)
cv2.createTrackbar('min_disparity',window_name, 0, 25, nop)
cv2.createTrackbar('num_disparities', window_name, 0, 100, nop)
cv2.createTrackbar('speckle_range', window_name, 0, 100, nop)
cv2.createTrackbar('speckle_window_size', window_name, 0, 25, nop)
cv2.createTrackbar('uniqueness_ratio', window_name, 0, 100, nop)
cv2.createTrackbar('prefilter_cap', window_name, 5, 62, nop)
cv2.createTrackbar('disp12_max_diff', window_name, 5, 25, nop)
cv2.createTrackbar('p1', window_name, 5, 100, nop)
cv2.createTrackbar('p2', window_name, 25, 100, nop)

while True:
    # Read current values from GUI
    block_size = cv2.getTrackbarPos('block_size', window_name) * 2 + 1
    min_disparity = cv2.getTrackbarPos('min_disparity', window_name)
    num_disparities = (cv2.getTrackbarPos('num_disparities', window_name) + 1) * 16
    speckle_range = cv2.getTrackbarPos('speckle_range', window_name)
    speckle_window_size = cv2.getTrackbarPos('speckle_window_size', window_name)
    uniqueness_ratio = cv2.getTrackbarPos('uniqueness_ratio', window_name)
    prefilter_cap = cv2.getTrackbarPos('prefilter_cap', window_name) + 1
    disp12_max_diff = cv2.getTrackbarPos('disp12_max_diff', window_name)
    p1 = cv2.getTrackbarPos('p1', window_name)
    p2 = cv2.getTrackbarPos('p2', window_name)

    # Create, configure and run stereo block matcher
    stereo_matcher = cv2.StereoSGBM_create()
    stereo_matcher.setBlockSize(block_size)
    stereo_matcher.setMinDisparity(min_disparity)
    stereo_matcher.setNumDisparities(num_disparities)
    stereo_matcher.setSpeckleRange(speckle_range)
    stereo_matcher.setSpeckleWindowSize(speckle_window_size)
    stereo_matcher.setUniquenessRatio(uniqueness_ratio)
    stereo_matcher.setPreFilterCap(prefilter_cap)
    stereo_matcher.setDisp12MaxDiff(disp12_max_diff)
    stereo_matcher.setP1(p1)
    stereo_matcher.setP2(p2)
    disparity = stereo_matcher.compute(img_l, img_r) # Run!
    disparity = disparity.astype(np.float32)
    disparity = (disparity / 16.0 - min_disparity) / num_disparities

    # Display result
    cv2.imshow(window_name, disparity)

    # Keyboard handling
    key = cv2.waitKey(50) & 0xff
    if key == ord('p'):
        # Print parameters to console
        print('stereo_matcher = cv2.StereoBM_create()')
        print(f'stereo_matcher.setBlockSize({block_size})')
        print(f'stereo_matcher.setMinDisparity({min_disparity})')
        print(f'stereo_matcher.setNumDisparities({num_disparities})')
        print(f'stereo_matcher.setSpeckleRange({speckle_range})')
        print(f'stereo_matcher.setSpeckleWindowSize({speckle_window_size})')
        print(f'stereo_matcher.setUniquenessRatio({uniqueness_ratio})')
        print(f'stereo_matcher.setPreFilterCap({prefilter_cap})')
        print(f'stereo_matcher.setDisp12MaxDiff({disp12_max_diff})')
        print(f'stereo_matcher.setP1({p1})')
        print(f'stereo_matcher.setP2({p2})')
    elif key == ord('q'):
        # Quit program
        cv2.destroyAllWindows()
        break
