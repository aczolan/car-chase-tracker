import imutils
import cv2
import math

from collections import deque

from drawing import *

class ColorTracker:
	color_lower_bound = (0, 0, 0)
	color_upper_bound = (0, 0, 0)
	basis_frame = None
	masked_frame = None
	tracked_points = deque(maxlen=32)
	contours = []
	current_circle_radius = 0
	current_vector = (0, 0)

	jump_detected = False
	should_draw_circle = False

	def processNewFrame(self, new_frame):
		#1. Get the contours from this frame
		#1a. Prepare this frame for examination
		modified_frame = new_frame.copy()
		self.basis_frame = new_frame.copy()
		# modified_frame = imutils.resize(modified_frame, width=600)
		# modified_frame = cv2.GaussianBlur(modified_frame, (11, 11), 0)
		# modified_frame = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2HSV)
		modified_frame = cv2.inRange(modified_frame, self.color_lower_bound, self.color_upper_bound)
		modified_frame = cv2.erode(modified_frame, None, iterations = 2)
		modified_frame = cv2.dilate(modified_frame, None, iterations = 2)
		self.masked_frame = modified_frame.copy()

		#1b. Get contours from this prepared frame object
		found_contours = cv2.findContours(modified_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		found_contours = imutils.grab_contours(found_contours)
		contours = list(found_contours)

		#2. Find the largest shape from these contours
		circle_min_radius = 10
		if len(found_contours) > 0:
			largestContour = max(found_contours, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(largestContour)
			M = cv2.moments(largestContour)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			#3. Track this shape's movement
			if radius < circle_min_radius:
				self.should_draw_circle = False
			else:
				self.should_draw_circle = True
				self.tracked_points.appendleft(center)
				self.current_circle_radius = radius
				#Draw the circle and centroid on the new frame
			#Draw the trail to the frame

		self.updateDirectionVector()
	#end processNewFrame

	def updateDirectionVector(self):
		if len(self.tracked_points) > 1:
			dX = self.tracked_points[-1][0] - self.tracked_points[1][0]
			dY = self.tracked_points[-1][1] - self.tracked_points[1][1]
			euclidean_distance_threshold = 200
			euclidean_distance = math.sqrt( math.pow(dX, 2) + math.pow(dY, 2) )
			print "Tracker {} : Euc dist: {}".format(id(self), euclidean_distance)
			if euclidean_distance > euclidean_distance_threshold:
				jump_detected = True
			else:
				jump_detected = False
				#Normalize dX and dY and make a vector
				if euclidean_distance != 0.0:
					vector_x = dX / euclidean_distance
					vector_y = dY / euclidean_distance
					self.current_vector = (vector_x, vector_y)
	#end updateDirectionVector

	def __init__(self, hsv_lower_bound, hsv_upper_bound):
		self.color_lower_bound = hsv_lower_bound
		self.color_upper_bound = hsv_upper_bound
	#end init
#end class

#Process the current frame. Apply blurs/conversions/masks
#Return a modified frame and list of contours in the frame. Tuple: (frame, contours)
def getContours(frame, show_masks):

	num_masks = 10

	greenLower = (29, 86, 6) #Lower boundary for green detection in HSV color space
	greenUpper = (64, 255, 255) #Upper boundary for green detection

	hsv_color_ranges = []
	#create bounds here
	mask_hue_length = round(360 / num_masks)
	for i in range(num_masks):
		hue_lower_bound = mask_hue_length * i
		hue_upper_bound = mask_hue_length * i - 1
		this_lower_bound = (hue_lower_bound, 86, 6)
		this_upper_bound = (hue_upper_bound, 255, 255)
		new_color_range = (this_lower_bound, this_upper_bound)
		hsv_color_ranges.add(new_color_range)

	modified_frame = frame

	#Resize the frame, blur it, and convert it to the HSV color space
	modified_frame = imutils.resize(modified_frame, width=600)
	modified_frame = cv2.GaussianBlur(modified_frame, (11, 11), 0)
	hsv_frame = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2HSV)

	#example entry:
	#key is color range tuple, value is  actual frame
	#((0, 0, 0), (255, 255, 255)), frameobject
	all_masks = {}

	for color_range in hsv_color_ranges:
		#Create a mask for pixels that are in this color range
		this_mask = cv2.inRange(hsv_frame.copy(), color_range[0], color_range[1])
		this_mask = cv2.erode(this_mask, None, iterations = 2)
		this_mask = cv2.dilate(this_mask, None, iterations = 2)
		all_masks.add(color_range, this_mask)

	if show_masks:
		for mask in all_masks.items():
			cv2.imshow("Mask {}".format(mask[0][0]), mask[1])

	#Find contours in each mask


	# #Construct a mask for the color "green"
	# #Then, perform a series of dilations and erosions to remove small blobs left in the mask
	# mask = cv2.inRange(hsv, greenLower, greenUpper)
	# mask = cv2.erode(mask, None, iterations=2)
	# mask = cv2.dilate(mask, None, iterations=2)


	#Find contours in the mask
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	contours = imutils.grab_contours(contours)

	return (modifiedFrame, contours)
#end getContours

#Process the contours found on this frame
#Return the new frame with the circle, centroid, and trail drawn on it
def processContours(frame, contours, tracked_points, jump_detected):

	circleMinRadius = 10
	newFrame = frame

	#Only proceed if at least one contour was found
	if len(contours) == 0:
		return None

	#Find the largest contour in the mask
	#Then, use it to compute the minimum enclosing circle and its centroid

	largestContour = max(contours, key=cv2.contourArea)
	((x, y), radius) = cv2.minEnclosingCircle(largestContour)
	M = cv2.moments(largestContour) #M is a "Moment"
	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

	if radius > circleMinRadius:

		#Draw the circle and centroid on the frame
		newFrame = drawCircleToFrame(newFrame, (int (x), int(y)), int(radius), center)

		#Then, update the list of tracked points
		tracked_points.appendleft(center)

	newFrame = drawTrailToFrame(newFrame, tracked_points, jump_detected)

	return newFrame
#end processContours
