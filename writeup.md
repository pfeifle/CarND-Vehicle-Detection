
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.JPG
[image2]: ./output_images/image2_1.JPG
[image3]: ./output_images/image2_2.JPG
[image4]: ./output_images/image2_3.JPG
[image5]: ./output_images/image3_1.JPG
[image6]: ./output_images/image3_2.JPG
[image7]: ./output_images/image3_3.JPG
[image8]: ./output_images/sliding_window_1.JPG
[image9]: ./output_images/sliding_window_2.JPG
[image10]: ./output_images/heat_map_1.JPG
[image11]: ./output_images/test_video_1.JPG
[image12]: ./output_images/test_video_2.JPG
[image13]: ./output_images/heat_map_video_1.JPG
[image14]: ./output_images/heat_map_video_2.JPG
[image15]: ./output_images/heat_map_video_3.JPG
[image16]: ./output_images/heat_map_video_4.JPG
[image17]: ./output_images/heat_map_video_5.JPG
[image18]: ./output_images/heat_map_video_6.JPG


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images, cf. code cell `#1`. I used the `KITT'` dataset for the vehicles and the `GTI` dataset for the non-vehicles, i.e. 5966  cars and 3900  non-cars of size:  (64, 64, 3)

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like (cf. code cell `#2`).
Below some of the resulting images (including their parameters) 

![alt text][image2]
![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and checked the classification accuracy. I used the combination of hog parameters which led to the highest classification accuracy. 

20% of the car and non-car images were used for validation purposes. I detected that the best result, i.e. `99,8` classification accuracy was achieved by using the following parameters:
    
    * colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    * orient = 9
    * pix_per_cell = 8
    * cell_per_block = 2
    * hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

The code for this can be found in code cell `#4` and `#5`  

Below are some of the results steaming from different parameter settings:

![alt text][image5]

![alt text][image6]

![alt text][image7]



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As mentioned in the last section, the classifier was already used to find the right parametrization of the hog parameters.

Selection of hog parameters and training of the classifier was done in code cell `#4` 

First the feature vectors was created by 

`X = np.vstack((car_features, notcar_features)).astype(np.float64)`                        

Next the vector values were scaled

`X_scaler = StandardScaler().fit(X)`
`scaled_X = X_scaler.transform(X)`

and the labeks car / notcar were created

`y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))`

Next, split up data into randomized training and test sets (20% was used for validation)

`rand_state = np.random.randint(0, 100)`
`X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)`

Based on these preparation steps we used a linear support vector machine and trained it

`svc = LinearSVC()`
`svc.fit(X_train, y_train)`


Finally, we checked the accuracy of the classifier by using
`svc.predict(X_test[0:n_predict])`

Results of these training epochs were depicted in the last section.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In `code cell 22` I played around with sliding windows. I tried several combinations of sliding windows to detect the cars. In general, the closer to the top of the image the smaller the sliding windows became.  The figure below depicts one of the tested combinations. 

![alt text][image8]



The figure below depicts the result for one of the sliding window combinations. 
 

![alt text][image9]


In the following code cell, the heatmap for the detected bounding boxes are depicted. The heatmap was originally created with a threshold value of only 1. The basic idea of the heatmap is to cluster the detected bounding  boxes and thus reduce the likelihood of false positives.

![alt text][image10]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tested various combinations to create the feature vectors for the linear SVM. In the code cell below you can find some examples:  

            test_features = X_scaler.transform(np.hstack(hog_features).reshape(1, -1))    
            test_features = X_scaler.transform(np.hstack((hog_features,spatial_features, hist_features)).reshape(1, -1))    
            test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    

Although the combination of HOG, spatial and color histogram was working quite nicely on the test images it produced a lot of false positives on the project video. Using only YCrCb 3-channel HOG features was reducing the number of false positives considerably. In order not to miss some cars, I used quite a lot of sliding window combinations:

            bbox_list = find_cars (image,  400, 650, 1.8) + find_cars (image,  380, 650, 1.6) + find_cars (image,  380, 650, 1.4) + find_cars (image,  380, 500, 1.2) + find_cars (image,  380, 500, 1)     

in combination with a rather small threshold value for the heatmap generation

    heat = apply_threshold(heat,1)
            

Noteworthy, is also that I limited the search space not only in y-dimension but also in x-dimension. Based on lane level positioning and a digital map, a real car knows that he is on the left-most lane and that he only needs to track the objects to the right of it. So I started searching for cars to be tracked only for x-values in the right half of the image. 

I stored the found bounding boxes in a pickle file so that it was available for the next frame. The bounding boxes for each of 6 consecutive frames was stored. The bounding boxes used in one frame was always the union of the bounding boxes of the current frame plus the last 5 frames (cf.code cell 27)

	def bb_pickle (bbox_list):
    	bb_pickle = pickle.load( open( "./bb_list.p", "rb" ))
    	box_list1 = bb_pickle["bb_list1"]
	    box_list2 = bb_pickle["bb_list2"]
	    box_list3 = bb_pickle["bb_list3"]
	    box_list4 = bb_pickle["bb_list4"]
	    box_list5 = bb_pickle["bb_list5"]
	    box_list6 = bb_pickle["bb_list6"]

	    bb_pickle["bb_list1"] = bbox_list
	    bb_pickle["bb_list2"] = box_list1
	    bb_pickle["bb_list3"] = box_list2
	    bb_pickle["bb_list4"] = box_list3
	    bb_pickle["bb_list5"] = box_list4
	    bb_pickle["bb_list6"] = box_list5
	    pickle.dump( bb_pickle,open( "./bb_list.p", "wb" ))
	    
	    return bbox_list + box_list1 + box_list2 + box_list3 + box_list4 + box_list5 


Here are some example images of my search based on the test video provided by Udacity:

![alt text][image11]

![alt text][image12]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to the project video result](./project_video.mp4)

Here's a [link to the test video result](./test_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video (cf. projcet video from 39.8 s) , the result of `scipy.ndimage.measurements.label()` and the bounding boxes are overlaid for each frame of the video. Note the first frame does not detect the white car as the threshold was set to 1:


![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]

 
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I struggled a lot with the combination of color histograms and histogram of oriented gradients. I think it is good to use a combination of them but for what reason ever it did not work for me. So I used HOG features only.
I also think that the straightforward heatmap approach could be improved by "projecting" the boxes found in former frames to a "new" position in the new frame. Basically, implementing something like a Kalman filter for the bounding boxes.

