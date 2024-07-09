# Vehicle CutIn Detection Project

<p align="center">
 <a href="https://youtu.be/-GGcZB7PrDI"><img src="overview.gif" alt="Overview" width="50%" height="50%"></a>
 <br>Qualitative results. (click for full video)
</p>

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Abstract

The goal of the project was to develop a pipeline to reliably detect cars given a video from a roof-mounted camera: in this readme the reader will find a short summary of how I tackled the problem.

**Long story short**:
 - (baseline) HOG features + linear SVM to detect cars, temporal smoothing to discard false positive
 - (submission) [SSD deep network](https://arxiv.org/pdf/1512.02325.pdf) for detection, thresholds on detection confidence and label to discard false positive 
 
*That said, let's go into details!*

### Good old CV: Histogram of Oriented Gradients (HOG)

#### 1. Feature Extraction.

In the field of computer vision, a *features* is a compact representation that encodes information that is relevant for a given task. In our case, features must be informative enough to distinguish between *car* and *non-car* image patches as accurately as possible.

Here is an example of how the `vehicle` and `non-vehicle` classes look like in this dataset:

<p align="center">
  <img src="noncar_samples.png" alt="non_car_img">
  <br>Randomly-samples non-car patches.
</p>

<p align="center">
  <img src="car_samples.png" alt="car_img">
  <br>Randomly-samples car patches.
</p>

The most of the code that relates to feature extraction is contained in [`functions_feat_extraction.py`](functions_feat_extraction.py). Nonetheless, all parameters used in the phase of feature extraction are stored as dictionary in [`config.py`](config.py), in order to be able to access them from anywhere in the project.

Actual feature extraction is performed by the function `image_to_features`, which takes as input an image and the dictionary of parameters, and returns the features computed for that image. In order to perform batch feature extraction on the whole dataset (for training), `extract_features_from_file_list` takes as input a list of images and return a list of feature vectors, one for each input image.

For the task of car detection I used *color histograms* and *spatial features* to encode the object visual appearence and HOG features to encode the object's *shape*. While color the first two features are easy to understand and implement, HOG features can be a little bit trickier to master.

#### 2. Choosing HOG parameters.

HOG stands for *Histogram of Oriented Gradients* and refer to a powerful descriptor that has met with a wide success in the computer vision community, since its [introduction](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf) in 2005 with the main purpose of people detection. 

<p align="center">
  <img src="hog_car_vs_noncar.jpg" alt="hog" height="128">
  <br>Representation of HOG descriptors for a car patch (left) and a non-car patch (right).
</p>

The bad news is, HOG come along with a *lot* of parameters to tune in order to work properly. The main parameters are the size of the cell in which the gradients are accumulated, as well as the number of orientations used to discretize the histogram of gradients. Furthermore, one must specify the number of cells that compose a block, on which later a feature normalization will be performed. Finally, being the HOG computed on a single-channel image, arises the need of deciding which channel to use, eventually computing the feature on all channels then concatenating the result.

In order to select the right parameters, both the classifier accuracy and computational efficiency are to consider. After various attemps, I came up to the following parameters that are stored in [`config.py`](config.py):
```
# parameters used in the phase of feature extraction
feat_extraction_params = {'resize_h': 64,             # resize image height before feat extraction
                          'resize_w': 64,             # resize image height before feat extraction
                          'color_space': 'YCrCb',     # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                          'orient': 9,                # HOG orientations
                          'pix_per_cell': 8,          # HOG pixels per cell
                          'cell_per_block': 2,        # HOG cells per block
                          'hog_channel': "ALL",       # Can be 0, 1, 2, or "ALL"
                          'spatial_size': (32, 32),   # Spatial binning dimensions
                          'hist_bins': 16,            # Number of histogram bins
                          'spatial_feat': True,       # Spatial features on or off
                          'hist_feat': True,          # Histogram features on or off
                          'hog_feat': True}           # HOG features on or off
```

#### 3. Training the classifier

Once decided which features to used, we can train a classifier on these. In [`train.py`](train.py) I train a linear SVM for task of binary classification *car* vs *non-car*. First, training data are listed a feature vector is extracted for each image:
```
    cars = get_file_list_recursively(root_data_vehicle)
    notcars = get_file_list_recursively(root_data_non_vehicle)

    car_features = extract_features_from_file_list(cars, feat_extraction_params)
    notcar_features = extract_features_from_file_list(notcars, feat_extraction_params)
``` 
Then, the actual training set is composed as the set of all car and all non-car features (labels are given accordingly). Furthermore, feature vectors are standardize in order to have all the features in a similar range and ease training.
```
    feature_scaler = StandardScaler().fit(X)  # per-column scaler
    scaled_X = feature_scaler.transform(X)
```
Now, training the LinearSVM classifier is as easy as:
```
    svc = LinearSVC()  # svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
```
In order to have an idea of the classifier performance, we can make a prediction on the test set with `svc.score(X_test, y_test)`. Training the SVM with the features explained above took around 10 minutes on my laptop. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In a first phase, I implemented a naive sliding window approach in order to get windows at different scales for the purpose of classification. This is shown in function `compute_windows_multiscale` in [`functions_detection.py`](functions_detection.py). This turned out to be very slow. I utlimately implemented a function to jointly search the region of interest and to classify each window as suggested by the course instructor. The performance boost is due to the fact that HOG features are computed only once for the whole region of interest, then subsampled at different scales in order to have the same effect of a multiscale search, but in a more computationally efficient way. This function is called `find_cars` and implemented in [`functions_feat_extraction.py`](functions_feat_extraction.py). Of course the *tradeoff* is evident: the more the search scales and the more the overlap between adjacent windows, the less performing is the search from a computational point of view.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Whole classification pipelin using CV approach is implemented in [`main_hog.py`](main_hog.py). Each test image undergoes through the `process_pipeline` function, which is responsbile for all phases: feature extraction, classification and showing the results.

<p align="center">
  <img src="pipeline_hog.jpg" alt="hog" height="256">
  <br>Result of HOG pipeline on one of the test images.
</p>

In order to optimize the performance of the classifier, I started the training with different configuration of the parameters, and kept the best one. Performing detection at different scales also helped a lot, even if exceeding in this direction can lead to very long computational time for a single image. At the end of this pipeline, the whole processing, from image reading to writing the ouput blend, took about 0.5 second per frame.

### Computer Vision on Steroids, a.k.a. Deep Learning

#### 1. SSD (*Single Shot Multi-Box Detector*) network

In order to solve the aforementioned problems, I decided to use a deep network to perform the detection, thus replacing the HOG+SVM pipeline. For this task employed the recently proposed  [SSD deep network](https://arxiv.org/pdf/1512.02325.pdf) for detection. This paved the way for several huge advantages:
 - the network performs detection and classification in a single pass, and natively goes in GPU (*is fast*)
 - there is no more need to tune and validate hundreds of parameters related to the phase of feature extraction (*is robust*)
 - being the "car" class in very common, various pretrained models are available in different frameworks (Keras, Tensorflow etc.) that are already able to nicely distinguish this class of objects (*no need to retrain*)
 - the network outputs a confidence level along with the coordinates of the bounding box, so we can decide the tradeoff precision and recall just by tuning the confidence level we want (*less false positive*) 
 
The whole pipeline has been adapted to the make use of SSD network in file [`main_ssd.py`](main_ssd.py).

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/-GGcZB7PrDI)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In a first phase while I was still using HOG+SVM, I implemented a heatmap to average detection results from successive frames. The heatmap was thresholded to a minimum value before labeling regions, so to remove the major part of false positive. This process in shown in the thumbnails on the left of the previous figure.

When I turned to deep learning, as mentioned before I could rely on a *confidence score* to decide the tradeoff between precision and recall. The following figure shows the effect of thresholding SSD detection at different level of confidence. 

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="confidence_001.png" alt="low_confidence" height="256">
           <br>SSD Network result setting minimum confidence = 0.01
      </p>
    </th>
    <th>
      <p align="center">
           <img src="confidence_050.png" alt="high_confidence" height="256">
           <br>SSD Network result setting minimum confidence = 0.50
      </p>
    </th>
  </tr>
</table>

Actually, while using SSD network for detection for the project video I found that integrating detections over time was not only useless, but even detrimental for performance. Indeed, being detections very precide and false positive almost zero, there was no need anymore to carry on information from previous detections. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the first phase, the HOG+SVM approach turned out to be slightly frustrating, in that strongly relied on the parameter chosed to perform feature extraction, training and detection. Even if I found a set of parameters that more or less worked for the project video, I wasn't satisfied of the result, because parameters were so finely tuned on the project video that certainly were not robust to different situations. 

For this reason, I turned to deep learning, and I leveraged on an existing detection network (pretrained on Pascal VOC classes) to tackle the problem. From that moment, the sun shone again on this assignment!



---

## **Advanced Lane Finding**
<p align="center">
 <a href="https://youtu.be/BlHTbDip6oU"><img src="overview2.mp4" alt="Overview2" width="50%" height="50%"></a>
 <br>Qualitative results. (click for full video)
</p><br>
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img_overview]: overview.gif "Output Overview"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view)

### Camera Calibration

OpenCV provide some really helpful built-in functions for the task on camera calibration. First of all, to detect the calibration pattern in the [calibration images](./camera_cal/), we can use the function `cv2.findChessboardCorners(image, pattern_size)`. 

Once we have stored the correspondeces between 3D world and 2D image points for a bunch of images, we can proceed to actually calibrate the camera through `cv2.calibrateCamera()`. Among other things, this function returns both the *camera matrix* and the *distortion coefficients*, which we can use to undistort the frames.

The code for this steps can be found in [calibration_utils](calibration_utils.py).   

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following result (appreciating the effect of calibration is easier on the borders of the image): 

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="calibration_before.jpg" alt="calibration_before" width="60%" height="60%">
           <br>Chessboard image before calibration
      </p>
    </th>
    <th>
      <p align="center">
           <img src="calibration_after.jpg" alt="calibration_after" width="60%" height="60%">
           <br>Chessboard image after calibration
      </p>
    </th>
  </tr>
</table>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the camera is calibrated, we can use the camera matrix and distortion coefficients we found to undistort also the test images. Indeed, if we want to study the *geometry* of the road, we have to be sure that the images we're processing do not present distortions. Here's the result of distortion-correction on one of the test images:

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="test_calibration_before.jpg" alt="calibration_before" width="60%" height="60%">
           <br>Test image before calibration
      </p>
    </th>
    <th>
      <p align="center">
           <img src="test_calibration_after.jpg" alt="calibration_after" width="60%" height="60%">
           <br>Test image after calibration
      </p>
    </th>
  </tr>
</table>

In this case appreciating the result is slightly harder, but we can notice nonetheless some difference on both the very left and very right side of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Correctly creating the binary image from the input frame is the very first step of the whole pipeline that will lead us to detect the lane. For this reason, I found that is also one of the most important. If the binary image is bad, it's very difficult to recover and to obtain good results in the successive steps of the pipeline. The code related to this part can be found [here](./binarization_utils.py).

I used a combination of color and gradient thresholds to generate a binary image. In order to detect the white lines, I found that [equalizing the histogram](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) of the input frame before thresholding works really well to highlight the actual lane lines. For the yellow lines, I employed a threshold on V channel in [HSV](http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html) color space. Furthermore, I also convolve the input frame with Sobel kernel to get an estimate of the gradients of the lines. Finally, I make use of [morphological closure](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html) to *fill the gaps* in my binary image. Here I show every substep and the final output:
<p align="center">
  <img src="binarization.png" alt="binarization overview" width="90%" height="90%">
</p>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code relating to warping between the two perspective can be found [here](./perspective_utils.py). The function `calibration_utils.birdeye()` takes as input the frame (either color or binary) and returns the bird's-eye view of the scene. In order to perform the perspective warping, we need to map 4 points in the original space and 4 points in the warped space. For this purpose, both source and destination points are *hardcoded* (ok, I said it) as follows:

```
    h, w = img.shape[:2]

    src = np.float32([[w, h-10],    # br
                      [0, h-10],    # bl
                      [546, 460],   # tl
                      [732, 460]])  # tr
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<p align="center">
  <img src="perspective_output.png" alt="birdeye_view" width="90%" height="90%">
</p>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify which pixels of a given binary image belong to lane-lines, we have (at least) two possibilities. If we have a brand new frame, and we never identified where the lane-lines are, we must perform an exhaustive search on the frame. This search is implemented in `line_utils.get_fits_by_sliding_windows()`: starting from the bottom of the image, precisely from the peaks location of the histogram of the binary image, we slide two windows towards the upper side of the image, deciding which pixels belong to which lane-line.

On the other hand, if we're processing a video and we confidently identified lane-lines on the previous frame, we can limit our search in the neiborhood of the lane-lines we detected before: after all we're going at 30fps, so the lines won't be so far, right? This second approach is implemented in `line_utils.get_fits_by_previous_fits()`. In order to keep track of detected lines across successive frames, I employ a class defined in `line_utils.Line`, which helps in keeping the code cleaner.

```
class Line:

    def __init__(self, buffer_len=10):

        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None
    
    ... methods ...
```

The actual processing pipeline is implemented in function `process_pipeline()` in [`main.py`](./main.py). As it can be seen, when a detection of lane-lines is available for a previous frame, new lane-lines are searched through `line_utils.get_fits_by_previous_fits()`: otherwise, the more expensive sliding windows search is performed.

The qualitative result of this phase is shown here:

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="sliding_windows_before.png" alt="sliding_windows_before" width="60%" height="60%">
           <br>Bird's-eye view (binary)
      </p>
    </th>
    <th>
      <p align="center">
           <img src="sliding_windows_after.png" alt="sliding_windows_after" width="60%" height="60%">
           <br>Bird's-eye view (lane detected)
      </p>
    </th>
  </tr>
</table>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Offset from center of the lane is computed in `compute_offset_from_center()` as one of the step of the procecssing pipeline defined in [`main.py`](./main.py). The offset from the lane center can be computed under the hypothesis that the camera is fixed and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation from the lane center as the distance between the center of the image and the midpoint at the bottom of the image of the two lane-lines detected.  

During the previous lane-line detection phase, a 2nd order polynomial is fitted to each lane-line using `np.polyfit()`. This function returns the 3 coefficients that describe the curve, namely the coefficients of both the 2nd and 1st order terms plus the bias. From this coefficients, following [this](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) equation, we can compute the radius of curvature of the curve. From an implementation standpoint, I decided to move this methods as properties of `Line` class.

```
class Line:
  ... other stuff before ...
    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # radius of curvature of the line (averaged)
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The whole processing pipeline, which starts from input frame and comprises undistortion, binarization, lane detection and de-warping back onto the original image, is implemented in function `process_pipeline()` in [`main.py`](./main.py).

The qualitative result for one of the given test images follows:

<p align="center">
     <img src="test2.jpg" alt="output_example" width="60%" height="60%">
     <br>Qualitative result for test2.jpg
</p>



### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/BlHTbDip6oU).

---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I find that the more delicate aspect of the pipeline is the first step, namely the binarization of the input frame. Indeed, if that step fails, most of successive steps will lead to poor results. The bad news is that this part is implemented by thresholding the input frame, so we let the correct value of a threshold be our single-point of failure. This is *bad*! Being currently 2017, I think a CNN could be employed to successfully make this step more robust. Some datasets like [Synthia](http://synthia-dataset.net/) should hopefully  provide enough lane marking annotation to train a deep network. I must try this later :-)

### Acknowledgments

Implementation of [Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) was borrowed from [this repo](https://github.com/rykov8/ssd_keras) and then slightly modified for my purpose. Thank you [rykov8](https://github.com/rykov8) for porting this amazing network in Keras-Tensorflow!

### @Copyright-DHL Group-Vehicle Cut In Detection-Intel Unnati Industrial Internship Program(2024)
