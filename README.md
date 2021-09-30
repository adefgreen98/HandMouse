
# Installation
Please use the command:
        'pip install -r requirements.txt'
to install required packages.


# Keymap

Q --> quit the program<br/>
S --> save current color threshold<br/>
R --> reset mouse position at (0,0)<br/>
C --> activate/deactivate threshold-choice mode<br/>
G --> emergency deactivation of gesture mode<br/>


# HandMouse

# Usage
The program can be launched with 'python track.py' from the correct environment.
It will detect hand motion and gestures, then show both of these onto 2 separate windows.

By pressing the 'C' key, one can activate the "threshold-setting mode": it allows to clearly
see the color-thresholded image, in order to have a more suitable threshold for the current
lighting and color conditions. 
Since the program's performance highly relies on threshold quality,
we recommend to set this up at the first usage.

A custom threshold can be set thanks to the sliders appearing together with the thresholded image,
and it can be saved for future use by pressing the 'S' button. Any saved custom threshold will 
be used by default on next uses.

To deactivate the color-threshold choice mode, press again the 'C' button.

To activate the 'gesture mode', the user should stay still for approximately 1-2 seconds. 
To deactivate the 'gesture mode', one can either use the 5-fingers (open hand) gesture or press the 'G' key.
Recognized gestures are:<br/>
2 fingers --> Left Click Down<br/>
3 fingers --> Right Click Down<br/>
4 fingers --> Click Up<br/>
5 fingers --> Close Gesture Mode<br/>

Once initialized, this program will open a  window with 3 frames:<br/>
- the first showing the GoodFeaturesToTrack (moving) points over the masked image;<br/>
- the second showing the result of the convex hull to detect gesture, only when gesture mode is active;<br/>
- the third showing predicted mouse motion.<br/>

To quit the program, press Q.

# Credits
Simone Caldarella (GoodFeaturesToTrack, tracking and refactoring)<br/>
Federico Pedeni (Color thresholding and background subtraction)<br/>
Gaia Trebucchi (Convex Hull and gesture detection)<br/>
