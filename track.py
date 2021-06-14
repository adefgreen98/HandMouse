
import itertools
import math
import os
import json

import numpy as np
import cv2

from collections import Counter


##### THRESHOLD LOADING #####

def load_thresh(filename='user_hand_thresh'):
    """Safely loads threshold. By default, it searches first for user-defined threhsolds, then for the default threshold.
    Finally, in case of failure, it uses a hard-coded threshold."""
    res = None
    with open(filename + '.json', mode='r') as fp:
        res = json.load(fp)
    for k in res.keys():
        for i in range(len(res[k])): 
            if len(res[k][i]) == 0:
                res[k][i] = [0,256]
    return res

### main dictionary for hand thresholding saved in external .json file (for storing and modifying) 
_hand_thresh_ = None

def save_thresh(_hdthresh_, filename='user_hand_thresh'):
    """Saves threshold with specified name."""
    with open(filename + '.json', mode='w') as fp:
        json.dump(_hdthresh_, fp, indent=2)

try:
    _hand_thresh_ = load_thresh()
except (FileNotFoundError, json.decoder.JSONDecodeError):
    try:
        _hand_thresh_ = load_thresh('default_hand_thresh')
    except FileNotFoundError:
        _hand_thresh_ = {
            "hsv": [[0,14],[66,165],[139,256]], 
            "rgb": [[], [], []], 
            "bgr": [[], [], []], 
            "ycr_cb": [[144,249],[147,160],[99,119]],
            "test": [[0, 255], [0, 0], [0, 0]]
        }
        save_thresh(_hand_thresh_, 'default_hand_thresh')


##### TRACKBARS #####
class Trackbars:
    """Contains sliders to define color-threshold. Allows for quick and clean updates when needed."""

    def __init__(self, winname, colorspace):
        self.winname = winname
        self.__thresh__ = _hand_thresh_
        self.colorspace = colorspace
        self.is_open = False
    

    def show_window(self):
        self.is_open = True
        def nothing(val):
            pass

        names = Trackbars.slider_names(self.colorspace)
        fn = nothing
        cv2.namedWindow(self.winname)
        for name in names:
            comps = Trackbars.components(self.colorspace)
            sw = {
                comps[0]: 0,
                comps[1]: 1,
                comps[2]: 2
            }

            cv2.createTrackbar(name, self.winname, 
            self.__thresh__[self.colorspace][sw[name.split()[0]]][0] if name.endswith('min') else self.__thresh__[self.colorspace][sw[name.split()[0]]][1], 
            256, fn)
    
    
    def close_window(self):
        self.is_open = False
        cv2.destroyWindow(self.winname)


    def read_sliders(self):
        if self.is_open:
            res = {name: [] for name in Trackbars.components(self.colorspace)}
            for name in Trackbars.slider_names(self.colorspace):
                v = cv2.getTrackbarPos(name, self.winname)
                res[name.split()[0]].append(cv2.getTrackbarPos(name, self.winname))
            return res
        else:
            return self.__thresh__


    def update(self):
        """Safely updates global threshold by reading current sliders' values."""
        if self.is_open:
            global _hand_thresh_
            v = self.read_sliders()
            self.update_threshold(self.colorspace, v)
            _hand_thresh_ = self.thresh()


    def update_threshold(self, colorspace, values):
        """Updates local threshold. Utility for the more used 'Trackbars.update()'."""
        for k in values.keys():
            idx = -1
            
            assert 0 <= len(values[k]) <= 2

            if len(values[k]) != 2: raise NotImplementedError

            if not values[k][0] <= values[k][1]:
                print("Warning: no changes performed because was set min value >> max value for channel {}".format(k))
            else:
                    
                if colorspace == 'ycr_cb':
                    if k == 'y': idx = 0
                    elif k == 'cr': idx = 1
                    elif k == 'cb': idx = 2
                    else: pass
                    if idx != -1:
                        self.__thresh__[colorspace][idx] = values[k]
                else:
                    idx = colorspace.find(k)
                    if idx != -1:
                        self.__thresh__[colorspace][idx] = values[k]
                    else:
                        print(
                            """Warning: performing no changes as the specified colorspace is {} but 
                            you are modifying {} parameter
                            """.format(colorspace, k)
                        )
    
    
    def thresh(self):
        """Returns current threshold."""
        return self.__thresh__
    

    def save(self):
        """Saves current local threshold."""
        print("Saving threshold...")
        save_thresh(self.__thresh__)
    

    @staticmethod
    def components(colorspace):
        """Returns color component names for a specified namespace."""
        if colorspace == 'ycr_cb':
            return ['y', 'cr', 'cb']
        else: return list(colorspace)

    
    @staticmethod
    def slider_names(colorspace):
        """Creates slider names for a specified namespace."""
        return [a + ' ' + b for a,b in itertools.product(Trackbars.components(colorspace), ['min', 'max'])]


## contains definitions of BLACK color in different colorspaces
_color_black_ = {
    'rgb': np.array([0,0,0], dtype=np.uint8),
    'bgr': np.array([0,0,0], dtype=np.uint8),
    'hsv': np.array([0,0,0], dtype=np.uint8),
    'ycr_cb': np.array([16, 128, 128], dtype=np.uint8)
} 


def put_text(_frame, _winname, camera): 
    """Puts window name on an OpenCV window."""
    text_color = (0,255,0)
    x_text, y_text =  int((3/5) * camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int((1/8) * camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.putText(_frame, _winname, (x_text, y_text), cv2.FONT_HERSHEY_PLAIN, 1.5, text_color, thickness=2)


##### COLOR THRESHOLDING UTILITIES #####

def cvt_color_space(img:np.ndarray, source, dest):
    """ Converts image from one colorspace to another. Allowed colorspaces are: HSV, 
    YCRCB, RGB, BGR, GRAY(scale)."""
    
    if source in ['ycrcb', 'ycbcr', 'YCrCb', 'YCRCB', 'YCBCR']:
        source = 'ycr_cb'
    if dest in ['ycrcb', 'ycbcr', 'YCrCb', 'YCRCB', 'YCBCR']:
        dest = 'ycr_cb'

    source = source.upper()
    dest = dest.upper()
    modes = ["BGR", "RGB", "GRAY", "HSV", "YCR_CB"]
    assert source in modes and dest in modes
    if source == "YCR_CB"  and dest == "GRAY": # needed bc inconsistency in OpenCV
        return np.stack([img[:,:,0]]*3, axis=2)
    if source == dest:
        # print("(warning): transforming into same color space won't cause any change")
        return img.copy()
    else:
        try:
            s = 'cv2.cvtColor(img, cv2.COLOR_{0}2{1})'.format(source, dest)
            return eval(s)
        except TypeError as e:
            raise ValueError("""Caught exception while evaluating command string in color conversion, with text:
                                    '{}'
            This is probably due to wrong usage of image as parameter passed to this function.
            """.format(e))


def thresh_mask(_img:np.ndarray, t1, t2, t3):
    """Returns binary mask obtained by applying color thresholding on an image."""
    return np.all(
            np.stack([
            (t1[0] <= _img[:,:,0]) & (_img[:,:,0] < t1[1]),
            (t2[0] <= _img[:,:,1]) & (_img[:,:,1] < t2[1]),
            (t3[0] <= _img[:,:,2]) & (_img[:,:,2] < t3[1])
            ], axis=2),
        axis=2)


def apply_2d_mask(image, mask):
    """Applies a binary mask to a 3-channel image."""
    for i in range(3):
        image[:,:,i][np.logical_not(mask)] = 0
    return image


def color_thresholding(img:np.ndarray, _t1=[0,256], _t2=[0,256], _t3=[0,256], source="bgr", dest="rgb"):
    """Applies specified thresholds to the 3 channels of the image. The 'source' and 'dest' parameters
    can be used when an image is from a colorspace but we want to compute the thresholding in another."""
    assert len(_t1) == len(_t2) == len(_t3) == 2
    res = cvt_color_space(img, source, dest)
    bmask = thresh_mask(res, _t1, _t2, _t3)
    res[np.logical_not(bmask)] = _color_black_[dest]
    return res


def perform_hand_thresholding(img:np.ndarray, mode="rgb", _dest=None):
    """
        Performs color thresholding to recognize hands. The 'dest' parameter can be used when an image belongs
        to a colorspace but the thresholding is wanted to be computed in anothr one.
    """
    if _dest is None:
        _dest = mode
    t1=_hand_thresh_[_dest][0]
    t2=_hand_thresh_[_dest][1]
    t3=_hand_thresh_[_dest][2]
    return color_thresholding(img, t1, t2, t3, source=mode, dest=_dest)


##### BACKGROUND UPDATES AND TRACKING #####

def simple_bg_update(image, bg, alpha=0.15):
    """Performs the simpler version of background update (not used in final project)."""
    bg = np.uint8(bg*(1-alpha) + alpha*image)
    return bg


def init_tracking_window(frame_width, frame_height):
    """Safely initializes where to show the tracking process."""
    w = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    # Cross
    cv2.line(w, (0, int(frame_height / 2)), (frame_width, int(frame_height / 2)), (255,255,255), 1)
    cv2.line(w, (int(frame_width / 2), 0), (int(frame_width / 2), frame_height), (255,255,255), 1)
    # Border
    cv2.line(w, (0,0), (0,frame_height-1), (255,255,255),1)
    cv2.line(w, (0,0), (frame_width-1,0), (255,255,255),1)
    cv2.line(w, (0,frame_height-1), (frame_width-1, frame_height-1), (255,255,255),1)
    cv2.line(w, (frame_width-1,0), (frame_width-1,frame_width-1), (255,255,255),1)
    return w


##### TRACKER METHODS #####

def compute_total_motion(corns, prevs, fw, fh):
    """Performs tracking by averaging all shift vectors. Returns ratio wrt. frame width and height, which is 
    always in range [0,1]x[0,1]."""
    motion = []

    corns_int = np.int0(corns)
    prevs_int = np.int0(prevs)

    corns_int = corns_int.reshape(corns_int.shape[0], 2)
    corns_mean = corns_int.mean(axis=0)

    prevs_int = prevs_int.reshape(prevs_int.shape[0], 2)
    prevs_mean = prevs_int.mean(axis=0)

    x_mot = (corns_mean[0]-prevs_mean[0])/fw
    y_mot = (corns_mean[1]-prevs_mean[1])/fh

    return [x_mot, y_mot]
    

##### MOUSE POSITION UPDATER #####

def update_mouse_position(window, curr_pos, prev_pos, motion_pred, acceleration=1.4):
    """Safely updates mouse position inside its window, preventing motion outside boundaries."""
    winsize = np.array([window.shape[1], window.shape[0]], dtype=np.int)
    tmp = curr_pos + motion_pred * winsize * acceleration
    tmp = tmp.astype(np.int32)
    if not np.all(np.logical_and(0 < tmp, tmp < winsize)): return curr_pos
    else: return tmp


def movingMask(_corners, _prev_corners, thresh):
    """Updates corners by masking the ones which are moving above a certain threshold."""
    moving = []
    moving_corns = []
    corns = []
    yes = 0
    no = 0
    _it = min(len(_corners), len(_prev_corners))
    for i in range(_it):
        if (np.abs(_corners[i][0][0] - _prev_corners[i][0][0]) < thresh):
            if (np.abs(_corners[i][0][1] - _prev_corners[i][0][1]) < thresh):
                moving.append("No")
                no += 1
                corns.append(_prev_corners[i])

            else:
                moving.append("Yes")
                yes += 1
                corns.append(_corners[i])
                moving_corns.append(_corners[i])
        else:
            moving.append("Yes")
            yes += 1
            corns.append(_corners[i])
            moving_corns.append(_corners[i])
            
    
    corns = np.array(corns, dtype=np.float32)
    corns = np.int0(corns)
    moving_corns = np.array(moving_corns, dtype=np.int32)
    
    return corns, yes, no, moving_corns


##### CONVEX HULL UTILITIES #####

def centeroidnp(arr):
    """Obtains the centroid from a NumPy array of shape (npoints, 1, 2)."""
    length = arr.shape[0]
    if length == 0:
        return 300, 300
    sum_x = np.sum(arr[:,0,0])
    sum_y = np.sum(arr[:,0,1])
    return sum_x/length, sum_y/length


def check_gesture(frame, shift_history, shift_thresh=0.0005, frame_thresh_percent=0.005):
    """Performs a double check to see if a gesture is going to be performed. 
    The performed checks are: 

        - current mean shift (computed over a certain history) is below a given threshold 
        - the percentage of nonzero pixels of the masked image is below a specified value
    """

    mean = np.sum(shift_history, axis=1)
    mean = [mean[0] / mean.shape[0], mean[1] / mean.shape[0]]

    mean_check = mean[0] < shift_thresh and mean[1] < shift_thresh
    frame_check = frame.nonzero()[0].shape[0] < int(frame_thresh_percent * frame.size)

    double_check = frame_check and mean_check

    return double_check


####################################################
def dist(p1,p2):
    """Shortage for L2 distance between 2 points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_max_contours(img, frame_prepared):
    """Gets the contour with maximal area. Used to detect hand's convex hull in a noisy image."""
    contours, hierarchy = cv2.findContours(frame_prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours is not None and len(contours)!=0:
        for c in contours:
            cv2.drawContours(img,c,-1,(0,255,0),1)
        c = max(contours, key = cv2.contourArea)
        if cv2.contourArea(c) < 8000:
            return None
        else:
            return c
    else: 
        return None

def get_hull(contour):
    """Computes the convex hull for a specified contour."""
    hull_no_points = cv2.convexHull(contour, returnPoints=False)
    hull_points = cv2.convexHull(contour, False)
    return hull_no_points, hull_points

def get_min_enclosing_circle(contour):
    """Gets the minimum enclosing circle for a contour. Used in convexity-defects detection (to get
    number of fingers)."""
    (x_c,y_c), radius = cv2.minEnclosingCircle(contour)
    center = (int(x_c) ,int(y_c))
    radius = int(radius)
    return center, radius

def centroid(hull_non_points):
    """Shortage for hull-centroid."""
    M = cv2.moments(hull_non_points)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid=(cX,cY)
    return centroid

def cos(a,b,c):
    """Shortage for cosine computation of an angle, specified as 3 points."""
    return -(a**2-b**2-c**2)/2*b*c     

def gesture(img):
    """Detects a gesture from input image. Returns the number of raised fingers (2,3,4 or 5)."""
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.dilate(frame_gray, np.ones((3,3), dtype=np.uint8), iterations=1)
    canny = cv2.Canny(frame_gray, 50, 150)
    frame_gray = cv2.dilate(canny, np.ones((3,3), dtype=np.uint8), iterations=1)   
    max_cont=get_max_contours(img, frame_gray)
    if max_cont is not None:
       cv2.drawContours(img,max_cont,-1,(0,255,0),3)
       hull_np,hull_p=get_hull(max_cont)
       cv2.drawContours(img,[hull_p],0,(255,0,0),1,8)
       circle_center,circle_radius=get_min_enclosing_circle(max_cont)
       cv2.circle(img,circle_center,circle_radius,(0,255,0),2)
       centr=centroid(hull_p)
       defects=cv2.convexityDefects(max_cont,hull_np)
       if defects is not None:
           count_def=0
           for i in range(defects.shape[0]): 
                s, e, f, d = defects[i][0]
                start = tuple(max_cont[s][0])
                end = tuple(max_cont[e][0])
                far = tuple(max_cont[f][0])
                depth=dist(end,far)
                conc=dist(end,start)
                zen=dist(far,start)
                if cos(conc,depth,zen) > 0 and dist(far, centr) < circle_radius/2 and (depth+zen) > 2/3 * circle_radius:
                   cv2.circle(img, far, 8, (255, 0, 0), -1)
                   count_def+=1
           if 0<count_def<=4:
               n=count_def + 1
               return n
           else:
               return -1


##### HELPERS (to keep main loop short) #####

def prepare_frame(_frame, gauss_bg, learning_rate=-1, median_blur_size=5, kernel_size=3, colorspace='ycr_cb'):
    """Prepares the frame for GoodFeaturesToTrack or LucasKanade (performed inside the main loop)."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    col_thresh = cvt_color_space(perform_hand_thresholding(_frame, mode='bgr', _dest=colorspace), colorspace, 'bgr')
    if gauss_bg is not None:
        mask = gauss_bg.apply(col_thresh, learning_rate)
    else: 
        # if is None, then we are detecting gesture --> no background adaptation is made
        mask = np.any(col_thresh != 0).astype(np.uint8)
    mask = cv2.medianBlur(mask, median_blur_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _gf_frame = apply_2d_mask(_frame, mask)

    _frame_gray = cv2.cvtColor(_gf_frame, cv2.COLOR_BGR2GRAY)

    return _gf_frame, col_thresh, _frame_gray


def gff(_frame, max_points):
    """Returns the GoodFeaturesToTrack points in the image, and their NumPy conversion."""
    _corners = cv2.goodFeaturesToTrack(_frame, maxCorners=max_points,
                    qualityLevel=0.01,
                    minDistance=10,
                    blockSize=7,
                    useHarrisDetector=True,
                    k=0.08)
    
    try: 
        _np_corners = np.int0(_corners)
    except TypeError:
        _np_corners = np.array([[[]]], dtype=np.int32)
    return _corners, _np_corners


def track_lk(_prev_frame, _frame, _prev_corners):
    """Performs LucasKanade motion approximation."""
    _corners, status, err = cv2.calcOpticalFlowPyrLK(_prev_frame, _frame, _prev_corners.astype(np.float32), None)
    _np_corners = np.int0(_corners)
    _np_prev_corners = np.int0(_prev_corners)
    _corns, _, _, moving_corners = movingMask(_np_corners, _np_prev_corners, 6)
    return _corns, _np_corners, _np_prev_corners, moving_corners


def check_total_motion(_prev_corns, _corns, fw, fh):
    """Checks if motion repects some criteria; if so, calculates the predicted mouse-motion."""

    if _prev_corns is not None:
        motion_pred_total = compute_total_motion(_corns, _prev_corns, fw, fh)
    else:
        motion_pred_total = [0,0]

    if np.isnan(motion_pred_total[0]) or np.isnan(motion_pred_total[1]):
        motion_pred_total = [0, 0] 
    else:
        if np.abs(motion_pred_total[0]) > 0.07:
            motion_pred_total[0] = 0
        if np.abs(motion_pred_total[1]) > 0.07:
            motion_pred_total[1] = 0
    return motion_pred_total


def frame_toshow(_gf_frame, _track_frame, _moving_corners, mouse_curr_pos, _ch_frame, winnames):
    """Stucks together different frames to be shown together in the main loop."""
    if _moving_corners is not None:
        for i,crn in enumerate(_moving_corners):
            x, y = crn.astype(int).ravel()
            cv2.circle(_gf_frame, (x, y), 3, np.array([i, 2*i, 255-i], float))
    center = (mouse_curr_pos[0].item(), mouse_curr_pos[1].item())
    cv2.circle(_track_frame, center, 10, (0, 255, 0), thickness=3)
    
    if _gf_frame is None:
        _gf_frame = np.zeros_like(_track_frame)
    if _ch_frame is None: 
        _ch_frame = np.zeros_like(_track_frame)
    put_text(_track_frame, winnames["tracking"], winnames["cap"])
    put_text(_gf_frame, winnames["features"], winnames["cap"])
    put_text(_ch_frame, winnames["gesture"], winnames["cap"])
    toshow = np.hstack([_gf_frame, _ch_frame, _track_frame])
    return toshow


##### Demo #####

def demo(colorspace):
    """Performs tracking using Gaussian background subtraction method."""

    print("Initializing Demo with Gaussian background subtraction.") 
    print("Press Q to end.")
    print("Press R to reset tracking position.")
    print("Press S to save current threshold.")
    print("Press C to visualize color thresholded image (press again to close the window).")
    print("Press G for emergency gesture deactivation.")

    framecnt = 0 
    # Frame width & height
    fw = 640 
    fh = 480
    
    # Window names
    winnames = {
        "main": "Track", 
        "features": "Features",
        "tbars": "Color Threshold Sliders",
        "tracking": "Tracking",
        "gesture": "Gesture",
        "color": "Thresholded Frame"
    }

    # Camera + Main Window
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
    cv2.namedWindow(winnames["main"], cv2.WINDOW_NORMAL)
    winnames["cap"] = cap

    # Trackbars
    trackbars = Trackbars(winnames["tbars"], colorspace)

    # Good Features + LK initialization
    prev_frame = prev_corns = None
    gf_secs_update = 1.0 # timestep to execute GoodFeatures
    gf_maximum_points = 50 # maximum feature points nr.

    # Tracking variables
    mouse_curr_pos = mouse_prev_pos = np.array([fw / 2, fh / 2], dtype=np.int)
    mouse_motion_vect = np.zeros_like(mouse_curr_pos)
    acceleration = 1.4 # Accelerates motion

    # Gaussian Background
    gbs_history = 300
    gauss = cv2.createBackgroundSubtractorMOG2(history=gbs_history)
    learning_rate = 0.01

    corns = prev_corns = None
    moving_corners = prev_moving_corners = None

    # Gesture utilities
    is_active_gesture = False
    nframes_check_gest = 30
    check_gest_history = []
    shift_thresh = 0.0001
    shift_history = []
    area_thresh = 0.007
    
    gest_max_size = 10
    gest_queue = []

    mouse_actions = {
        None: "None",
        -1: "None",
        0: "None",
        1: "None",
        2: "LeftClickDWN",
        4: "ClickUP",
        3: "RightClickDWN",
        5: "Close"
    }

    # Others
    visualize_color_threshold = False

    while True:
        # Read frame
        ret, frame = cap.read()
        # Update color threshold trackbars
        trackbars.update()
        # Frame To Show
        fts = None
        # Init track frame
        track_frame = init_tracking_window(fw, fh)
        
        if frame is not None:

            # Preparing frame
            cv2.flip(frame, 1, frame)

            gf_frame, color_mask_frame, frame_gray = prepare_frame(frame, gauss, learning_rate=learning_rate, colorspace=colorspace)

            np_corners = None
            pt1 = pt2 = None

            if framecnt % (30 * gf_secs_update) == 0:
                # Performs GoodFeatures detection
                framecnt = 0
                corns, np_corners = gff(frame_gray, gf_maximum_points)
            else:
                # Performs LucasKanade tracking
                corns, np_corners, np_prev_corners, moving_corners = track_lk(prev_frame, frame, prev_corns)

                if moving_corners.shape[0] == 0 and prev_moving_corners is not None:
                    moving_corners = prev_moving_corners

                motion_pred_total = check_total_motion(prev_corns, corns, fw, fh)
                mouse_curr_pos = update_mouse_position(track_frame, mouse_curr_pos, mouse_prev_pos, motion_pred_total, acceleration)
                mouse_curr_pos = mouse_curr_pos.astype(np.int32)

                if len(shift_history) == nframes_check_gest:
                    shift_history.pop(0)
                shift_history.append(motion_pred_total)
            
            final_ch_frame = None
            
            ##### DECOMMENT THIS TO ALWAYS VISUALIZE CH #####
            # is_active_gesture = True

            if not visualize_color_threshold:
                if is_active_gesture: 
                    # Performs gesture recog  
                    final_ch_frame = color_mask_frame.copy()   
                    if len(gest_queue) == gest_max_size:
                        gest_queue.pop(0)
                    gest_queue.append(gesture(final_ch_frame))

                    gest_detected = Counter(gest_queue).most_common(1)[0][0]

                    cv2.putText(final_ch_frame, mouse_actions[gest_detected], (int(0.1*fw), int(0.33*fh)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,0), thickness=2)
                
                    if gest_detected == 5:
                        is_active_gesture = False
                        gest_queue = []
                        shift_history = []
                        check_gest_history = []
                else:
                    # Prepares check for next loop
                    if len(check_gest_history) == nframes_check_gest:
                        # if majority of the queue is True then starts gesture recognition
                        mn = np.mean(np.array(check_gest_history, dtype=np.int))
                        is_active_gesture =  mn > 0.5
                        check_gest_history.pop(0)
                    if len(shift_history) == nframes_check_gest:
                        check_gest_history.append(check_gesture(frame_gray, shift_history, shift_thresh, frame_thresh_percent=area_thresh))

            fts = frame_toshow(gf_frame, track_frame, moving_corners, mouse_curr_pos, final_ch_frame, winnames)
            
            # Next step
            if corns is not None:
                prev_corns = corns
            prev_frame = frame
            prev_moving_corners = moving_corners

            # Showing
            cv2.imshow(winnames["main"], fts)
            if visualize_color_threshold:
                cv2.imshow(winnames["color"], color_mask_frame)

            
        user_input = cv2.waitKey(1) % 256
        if user_input == ord('r'): 
            mouse_curr_pos = np.array([fw / 2, fh / 2], dtype=np.int)
        elif user_input == ord('q'): 
            break
        elif user_input == ord('s'): 
            trackbars.save()
        elif user_input == ord('c'):
            if not visualize_color_threshold:
                visualize_color_threshold = True
                trackbars.show_window()
            else:
                visualize_color_threshold = False
                cv2.destroyWindow(winnames["color"])
                trackbars.close_window()
        elif user_input == ord('g'):
            is_active_gesture = False
            gest_queue = []
            shift_history = []
            check_gest_history = []
        
        framecnt += 1

    cap.release()
    cv2.destroyAllWindows()



##### Main #####

if __name__ == '__main__':
    demo('ycr_cb')