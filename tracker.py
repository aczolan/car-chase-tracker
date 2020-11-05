from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import pandas as pd
import sys
import math

from contours import *
from drawing import *
from utils import *

#Global vars
g_frameCounter = 0 #Number of frames that have passed
g_trackedPoints = deque(maxlen=32) #List of tracked points
g_directionChangeArray = [] #List of direction changes (direction, timestamp)
g_currentDirection = "" #Current direction

g_euclideanThreshold = 9999999 #Euclidean distance threshold
g_jumpDetected = False #Current tracked object has moved past the movement threshold

g_numTrackers = 5 #Max is 10 for now
g_showMasks = False

#Parse the command line arguments
#Return video file path
def parseCommandLineArgs(args):
	if len(args) >= 2:
		videoFilePath = args[1]
		print "Video path: {}".format(videoFilePath)
		return videoFilePath
		if any("-showMasks" in arg for arg in args):
			showMasks = True

	print "Error: Must specify video file path."
	print "Usage: python object_movement.py object_tracking_example.mp4"
	return None
#end parseCommandLineArgs

#Get the changes in direction (dX and dY)
#Return (dX, dY) tuple
def getDirectionChanges():

	global g_trackedPoints
	global g_currentDirection
	global g_frameCounter
	global g_directionChangeArray
	global g_jumpDetected

	thisDirection = "" #Current direction of the most recent x-y movement

	#Thresholds for significant movement
	xDirectionThreshold = 20 
	yDirectionThreshold = 20

	#Compute the difference between the x and y coordnates.
	dX = g_trackedPoints[-1][0] - g_trackedPoints[1][0]
	dY = g_trackedPoints[-1][0] - g_trackedPoints[1][1]

	euclideanDistance = math.sqrt( math.pow(dX, 2) + math.pow(dY, 2) )
	print "Euc dist: {}".format(euclideanDistance)

	if euclideanDistance <= g_euclideanThreshold:
		#Assess the current travel direction
		g_jumpDetected = False

		#Then, re-initialize the direction text variables
		(dirX, dirY) = ("", "")

		#Ensure there is significant movement in the x-direction
		if np.abs(dX) > xDirectionThreshold:
			#Set x-direction text
			dirX = "East" if np.sign(dX) == 1 else "West"

		#Ensure there is significant movement in the y-direction
		if np.abs(dX) > yDirectionThreshold:
			#Set the y-direction text
			dirY = "North" if np.sign(dY) == 1 else "South"

		#Check if both directions are non-empty
		if dirX != "" and dirY != "":
			thisDirection = "{}-{}".format(dirY, dirX)
		#Otherwise, only one direction is non-empty
		else:
			thisDirection = dirX if dirX != "" else dirY

		#Update global direction vars
		if not ( thisDirection in g_currentDirection ) :
			print "New direction: {}".format(thisDirection)
			g_currentDirection = thisDirection
			g_directionChangeArray.append({
				"direction": g_currentDirection,
				"timestamp": g_frameCounter
				})
	else:
		#Do not update the 
		g_jumpDetected = True

	return (dX, dY)
#end getDirectionChanges

#main body
if __name__ == "__main__":

	#Parse command line args and get the video file path
	videoFilePath = parseCommandLineArgs(sys.argv)
	if videoFilePath == None or videoFilePath == "":
		sys.exit(1)

	videoStream = cv2.VideoCapture(videoFilePath)

	ok, frame = videoStream.read()
	if not ok:
		#Error reading the first frame
		print "Error reading video stream"
		sys.exit(1)

	#Skip a few seconds into the video
	framesToSkip = 180
	while framesToSkip > 0:
		cv2.imshow("Frame", frame)
		ok, frame = videoStream.read()
		if not ok:
			break
		framesToSkip -= 1
	#end while

	#Allow the camera or video file to warm up
	time.sleep(2.0)

	#Initialize all trackers
	all_trackers = []
	#mask_hue_length = round(360 / g_numTrackers)
	mask_hue_length = 36
	mask_hue_start_val = 30 #Start at green
	for i in range(g_numTrackers):
		hue_lower_bound = (mask_hue_start_val + mask_hue_length * i) % 360
		hue_upper_bound = (mask_hue_start_val + (mask_hue_length * (i + 1)) - 1) % 360

		this_lower_bound = (hue_lower_bound, 86, 6)
		this_upper_bound = (hue_upper_bound, 255, 255)
		new_tracker = ColorTracker(this_lower_bound, this_upper_bound)
		all_trackers.append(new_tracker)

	#Main while loop. On every iteration, get the next frame from the video stream and process it.
	g_frameCounter = 0
	while True:

		#Grab the current frame
		current_frame = videoStream.read()

		#"Handle the frame from VideoCapture or VideoStream"
		current_frame = current_frame[1]

		#If we didn't grab a frame, then we have reached the end of the video
		if current_frame is None:
			break
		prepped_frame = current_frame.copy()

		#Do initial frame prep
		prepped_frame = imutils.resize(prepped_frame, width=600)
		working_frame = prepped_frame.copy()

		prepped_frame = cv2.GaussianBlur(prepped_frame, (11, 11), 0)
		prepped_frame = cv2.cvtColor(prepped_frame, cv2.COLOR_BGR2HSV)

		#Iterate over all trackers and give them this frame
		#Get the direction vectors of each tracker also
		all_direction_vectors = []

		for tracker in all_trackers:
			#Update tracker with this frame
			tracker.processNewFrame(prepped_frame)
			#all_direction_vectors.append(tracker.current_vector)
			all_direction_vectors.append(tracker.summed_vector)

			#Draw shit on the screen
			if tracker.should_draw_circle and len(tracker.tracked_points) > 1:
				current_point = tracker.tracked_points[0]
				#Draw the circle
				working_frame = drawCircleToFrame(working_frame, current_point, int(round(tracker.current_circle_radius)), current_point)
				#Draw the arrow
				working_frame = drawArrowToFrame(working_frame, current_point, tracker.current_vector)

		overall_direction_vector = addVectorsAndNormalize(all_direction_vectors)

		#Determine direction from this vector
		direction_string = determineDirection(overall_direction_vector)
		#Draw this on the screen
		working_frame = drawDirectionText(working_frame, direction_string, overall_direction_vector[0], overall_direction_vector[1], g_frameCounter)

		#We're ready to draw the modified frame now.
		cv2.imshow("Frame", working_frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		g_frameCounter += 1

	#end big while loop

	outputFileName = "tracker_output.csv"
	linesToWrite = []
	for entry in g_directionChangeArray:
		nextLine = ""
		for val in entry.values():
			nextLine += str(val)
			if entry.values()[-1] != val:
				nextLine += ","

		nextLine += "\n"
		linesToWrite.append(nextLine)

	myCsvFile = open(outputFileName, 'w')
	myCsvFile.writelines(linesToWrite)
	print "Wrote to output file {}".format(outputFileName)
	myCsvFile.close()

	#dataFrame stuff

	# dataFrame = pd.DataFrame(g_directionChangeArray)

	# print "dataFrame is {}".format(type(dataFrame))

	# dataFrame = dataFrame[dataFrame['direction'] != ""]

	# jsonFile = open('output.json', 'w')
	# jsonFile.write(dataFrame.to_json(orient='records'))
	# jsonFile.close()

	# csvFile = open('output.csv', 'w')
	# csvFile.write(dataFrame.to_csv(index=False, line_terminator='\n'))
	# csvFile.close()

	#Release the camera and close all windows
	videoStream.release()
	cv2.destroyAllWindows()

#end main body
