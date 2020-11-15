import imutils
import cv2
import math

from collections import deque

from drawing import *
from utils import *

class ColorTracker:
	color_lower_bound = (0, 0, 0)
	color_upper_bound = (0, 0, 0)
	basis_frame = None #Unedited frame
	masked_frame = None #Mask frame containing black and white pixels
	arrow_img = None #Arrow indicator showing current direction
	tracked_points = deque(maxlen=32)
	contours = []
	current_circle_center = (0, 0)
	current_circle_radius = 0
	current_vector = (0, 0)
	summed_vector = (0, 0)

	euclidean_distance_threshold = 999
	jump_detected = False
	should_draw_circle = False
	show_mask_window = False

	#Percentage of the frame near the edges to deny tracking
	border_x_percent = 0.01
	border_y_percent = 0.01

	#Whether or not this tracker's direction vector is ready
	dirvector_ready = False

	#Number of usable tracked points currently found
	num_usable_datapoints = 0

	def processNewFrame(self, new_frame):
		#1. Get the contours from this frame
		#1a. Prepare this frame for examination
		modified_frame = new_frame.copy()
		self.basis_frame = new_frame.copy()

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
			#Check if circle meets radius threshold
			if radius < circle_min_radius:
				self.should_draw_circle = False
			else:
				self.should_draw_circle = True
				self.tracked_points.appendleft(center)
				self.current_circle_center = (int(round(x)), int(round(y)))
				self.current_circle_radius = radius
		else:
			self.resetVectors()

		self.updateDirectionVector()

	#end processNewFrame

	def updateDirectionVector(self):
		if len(self.tracked_points) < 2:
			return

		#Get most recent vector
		newest_x, newest_y = self.tracked_points[0]
		recent_dX = self.tracked_points[0][0] - self.tracked_points[1][0]
		recent_dY = self.tracked_points[0][1] - self.tracked_points[1][1]
		self.current_vector = (-recent_dX, -recent_dY)

		#1. Check for euclidean distance jump and/or if point is near border
		euclidean_distance = getEuclideanDistance(self.current_vector)

		frame_height, frame_width, frame_channels = self.basis_frame.shape
		left_border = self.border_x_percent * frame_width
		right_border = (1.0 - self.border_x_percent) * frame_width
		top_border = self.border_y_percent * frame_height
		bottom_border = (1.0 - self.border_y_percent) * frame_height

		if euclidean_distance > self.euclidean_distance_threshold:
			jump_detected = True
			self.tracked_points.clear()
			self.dirvector_ready = False
			self.resetVectors()
			return
		else:
			jump_detected = False

		#2. Check for border point
		if ( newest_x < left_border 
		  or newest_x > right_border 
		  or newest_y < top_border 
		  or newest_y > bottom_border ):
			self.tracked_points.clear()
			self.dirvector_ready = False
			self.resetVectors()
			return

		#3. If these points are valid, then create a direction vector.
		this_vector_history = []
		for i in range(1, len(self.tracked_points)):
			#Find the vector between the newest point and all the recorded points
			this_dX = self.tracked_points[0][0] - self.tracked_points[i][0]
			this_dY = self.tracked_points[0][1] - self.tracked_points[i][1]
			this_vector_history.append((-this_dX, -this_dY))
			self.num_usable_datapoints += 1

		#Add all the vectors in the vector history
		self.summed_vector = addVectors(this_vector_history)

		self.arrow_img = createArrowImg(self.summed_vector)

		self.dirvector_ready = True

	#end updateDirectionVector

	def resetVectors(self):
		self.tracked_points.clear()
		self.current_vector = (0, 0)
		self.summed_vector = (0, 0)
		self.dirvector_ready = False
		self.num_usable_datapoints = 0

	def __init__(self, hsv_lower_bound, hsv_upper_bound, jump_distance_threshold):
		self.color_lower_bound = hsv_lower_bound
		self.color_upper_bound = hsv_upper_bound
		self.euclidean_distance_threshold = jump_distance_threshold
	#end init
#end class
