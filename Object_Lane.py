import numpy as np
import cv2
from matplotlib import pyplot as plt

# In[2]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# In[3]:


# Model Loaded
# Use this model to detect the objects in a new image
model = cv2.dnn_DetectionModel(frozen_model, config_file)


# In[4]:

# The pre-trained model can detect a list of object classes
classLabels = []
file_name = "Labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())

# print(classLabels)


# In[5]:


# set configurariton of model
model.setInputSize(320, 320) # because in config file, we have defined as 320x320
model.setInputScale(1.0/127.5) # 255/2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
model.setInputSwapRB(True)


# ### Read Image

# In[6]:


# img = cv2.imread('wp6693787.jpeg')
# plt.imshow(img)


# # In[7]:


# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# # In[8]:


# ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5) # threshold => how much accurate you want
# #print(ClassIndex) # 3,1 => car, person


# # In[9]:


# font_scale = 3
# font = cv2.FONT_HERSHEY_PLAIN
# for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
#     #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     #cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)
#     cv2.rectangle(img, boxes, (255, 0, 0), 2)
#     cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
    


# # In[10]:


# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



def grayscale(img):
	return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


def show_image(img, img_name = "image"):
	cv2.imshow(img_name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def blur(img, kernel_size=5):
	return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

def edge_detector(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def detect_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    hough_lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    return hough_lines

def average_slope_intercepts(lines):
	left_lines = [] # (slope, intercept)
	left_weights = [] # (length of line)
	right_lines = [] # (slope, intercept)
	right_weights = [] # (length of line)

	for line in lines:
		for x1,y1,x2,y2 in line:
			if x2 == x1 or y2 == y1: # ignoring vertical line
				continue

			slope = (y2-y1) / float((x2-x1))
			intercept = y1 - slope*x1
			length_of_line = np.sqrt((y2-y1)**2 + (x2-x1)**2)

			if slope < 0:
				left_lines.append((slope, intercept))
				left_weights.append(length_of_line)
			else:
				right_lines.append((slope, intercept))
				right_weights.append(length_of_line)


	left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
	right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
	
	return left_lane, right_lane

def convert_line_SI_points(y1, y2, line):

	if line is None:
		return None

	slope, intercept = line

	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	return (x1, y1, x2, y2)
    
def lane_lines(img, lines):
	left_lines, right_lines = average_slope_intercepts(lines)

	y1 = img.shape[0] # bottom of the image
	y2 = y1*0.6# slightly lower than the middle

	left_lines = convert_line_SI_points(y1, y2, left_lines)
	right_lines = convert_line_SI_points(y1, y2, right_lines)

	return left_lines, right_lines


if __name__ == "__main__":
	
    cap = cv2.VideoCapture("video/solidWhiteRight.mp4")
    if not cap.isOpened():
        cap = cv2.VideoCapture(0);

    font_scale = 1.3
    font = cv2.FONT_HERSHEY_PLAIN

    while(cap.isOpened()):

        ret, img = cap.read()
        if(not ret):
            break

        # ClassIndex => Labeles indexes
        # confidence => if no object exists in the cell, confidence = 0 else 
        # 	confidence = IOU(intersection over union) b/w the predicted box and the ground truth
        # bbox(bounding box) => bounding box involving the x, y coordinate and 
        # 		the width and height and the confidence.
        ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.55)

        if(len(ClassIndex) != 0):
            # flatten() => if [[1,2,3], [4,5,6]] then [1,2,3,4,5,6]
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                if(ClassInd <= 80):
                    cv2.rectangle(img, boxes, (255, 0, 0), 2)
                    cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=1)
        
        # cv2.imshow("Object Detection", img)


        gray_image = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur_image = blur(gray_image, 5)
        edges = edge_detector(blur_image, 50, 150)
        mask = np.zeros_like(edges)
        ignore_mask_color = 255
        vertices = np.array([[(0,edges.shape[0]),(480, 310), (485, 310), (edges.shape[1],edges.shape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)


        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 15    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20 #minimum number of pixels making up a line
        max_line_gap = 15   # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        hough_lines = detect_hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

        left_line, right_line = lane_lines(blur_image, hough_lines)

        #left lane
        if left_line is not None:
            x1,y1,x2,y2 = left_line
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        #right lane
        if right_line is not None:
            x1,y1,x2,y2 = right_line
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
