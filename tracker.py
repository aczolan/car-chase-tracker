from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import pandas as pd
import sys

#Global vars
g_frameCounter = 0 #Number of frames that have passed
g_trackedPoints = deque(maxlen=32) #List of tracked points
g_directionChangeArray = [] #List of direction changes (direction, timestamp)
g_currentDirection = "" #Current direction

#Parse the command line arguments
#Return video file path
def parseCommandLineArgs(args):
	if len(args) >= 2:
		videoFilePath = args[1]
		print "Video path: {}".format(videoFilePath)
		return videoFilePath

	print "Error: Must specify video file path."
	print "Usage: python object_movement.py object_tracking_example.mp4"
	return None
#end parseCommandLineArgs

#Process the current frame. Apply blurs/conversions/masks
#Return a modified frame and list of contours in the frame. Tuple: (frame, contours)
def getContours(frame):

	greenLower = (29, 86, 6) #Lower boundary for green detection in HSV color space
	greenUpper = (64, 255, 255) #Upper boundary for green detection

	modifiedFrame = frame

	print "--getContours, modifiedFrame is {}".format(type(modifiedFrame))

	#Resize the frame, blur it, and convert it to the HSV color space
	modifiedFrame = imutils.resize(modifiedFrame, width=600)
	blurred = cv2.GaussianBlur(modifiedFrame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	print "--getContours, hsv is {}".format(type(hsv))

	#Construct a mask for the color "green"
	#Then, perform a series of dilations and erosions to remove small blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	print "--getContours, mask is {}".format(type(mask))

	#Find contours in the mask
	contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	print "--getContours, (1) contours is {}".format(type(contours))
	print "--getContours, (1) contours[0] is {}".format(type(contours[0]))
	print "--getContours, (1) contours[1] is {}".format(type(contours[1]))

	contours = imutils.grab_contours(contours)

	print "--getContours, (2) contours is {}, length {}".format( type(contours), len(contours) )

	return (modifiedFrame, contours)
#end getContours

#Process the contours found on this frame
#Return the new frame with the circle, centroid, and trail drawn on it
def processContours(frame, contours):

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
		g_trackedPoints.appendleft(center)

	print "What is newFrame? {}".format(type(newFrame))

	newFrame = drawTrailToFrame(newFrame, g_trackedPoints)

	print "What is newFrame? {}".format(type(newFrame))

	return newFrame
#end processContours

#Draw the circle and centroid (for the most recent tracked point) on the frame
#Return the new frame with the circle and centroid drawn on it
def drawCircleToFrame(frame, circleCenter, circleRadius, centroidCenter):
	circleColor = (0, 255, 255)
	centroidColor = (0, 0, 255)
	centroidRadius = 5

	newFrame = frame

	cv2.circle(newFrame, circleCenter, circleRadius, circleColor, 2)
	cv2.circle(newFrame, centroidCenter, centroidRadius, centroidColor, -1)

	return newFrame
#end drawCircleToFrame

#Draw connected lines that trail behind the tracked point
#Return the new frame with the trail drawn on it
def drawTrailToFrame(frame, points):

	newFrame = frame

	#Iterate over all points and draw line segments between them
	for pointNum in np.arange(1, len(points)):

		if points[pointNum - 1] is None or points[pointNum] is None:
			continue

		thickness = int(np.sqrt(32 / float(pointNum + 1)) * 2.5) #thickness
		startPoint = points[pointNum - 1] #start of line segment
		endPoint = points[pointNum] #end of line segment
		color = (0, 0, 255) #color

		#add this line segment to the frame
		cv2.line(newFrame, startPoint, endPoint, color, thickness)

	return newFrame
#end drawTrailToFrame

#Get the changes in direction (dX and dY)
#Return (dX, dY) tuple
def getDirectionChanges():

	global g_trackedPoints
	global g_currentDirection
	global g_frameCounter

	thisDirection = "" #Current direction of the most recent x-y movement

	#Thresholds for significant movement
	xDirectionThreshold = 20 
	yDirectionThreshold = 20

	#Compute the difference between the x and y coordnates.
	dX = g_trackedPoints[-1][0] - g_trackedPoints[1][0]
	dY = g_trackedPoints[-1][0] - g_trackedPoints[1][1]

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
	if ( g_currentDirection != "" ) and not ( thisDirection in g_currentDirection ) :
		g_currentDirection = thisDirection
		g_directionChangeArray.append({
			"direction": g_currentDirection,
			"timestamp": g_frameCounter
			})

	return (dX, dY)
#end getDirectionChanges

#Draw the text that displays the current direction and dX/dY values
#Return the new frame with the text drawn on it
def drawDirectionText(frame, direction, dX, dY):

	global g_frameCounter

	directionText = direction
	directionTextPosition = (10, 30)
	directionTextFont = cv2.FONT_HERSHEY_SIMPLEX
	directionTextSize = 0.65
	directionTextColor = (0, 0, 255)

	dxdyText = "frame: {}, dx: {}, dy: {}".format(g_frameCounter, dX, dY)
	dxdyTextPosition = (10, frame.shape[0] - 10)
	dxdyTextFont = cv2.FONT_HERSHEY_SIMPLEX
	dxdyTextSize = 0.35
	dxdyTextColor = (0, 0, 255)

	newFrame = frame

	cv2.putText(newFrame, direction, directionTextPosition, directionTextFont, directionTextSize, directionTextColor, 3)
	cv2.putText(newFrame, dxdyText, dxdyTextPosition, dxdyTextFont, dxdyTextSize, dxdyTextColor, 1)

	return newFrame
#end getDirectionText


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
	framesToSkip = 300
	while framesToSkip > 0:
		cv2.imshow("Frame", frame)
		ok, frame = videoStream.read()
		if not ok:
			break
		framesToSkip -= 1
	#end while

	#Allow the camera or video file to warm up
	time.sleep(2.0)

	#Main while loop. On every iteration, get the next frame from the video stream and process it.
	g_frameCounter = 0
	while True:
		print "Frame Counter: {}".format(g_frameCounter)

		#Grab the current frame
		currentFrame = videoStream.read()

		#"Handle the frame from VideoCapture or VideoStream"
		currentFrame = currentFrame[1]

		#If we didn't grab a frame, then we have reached the end of the video
		if currentFrame is None:
			break
		print "currentFrame is {}".format(type(currentFrame))
		workingFrame = currentFrame

		#Get contours of this frame
		resizedFrame, contours = getContours(workingFrame)

		print "resizedFrame is {}".format(type(resizedFrame))
		print "contours is {}".format(type(contours))
		print "is contours empty? {}".format(len(contours) == 0)

		if resizedFrame is None:
			#Do not change workingFrame
			pass
		else:
			workingFrame = resizedFrame


		#Process the contours and get a frame with stuff drawn on it
		frameWithTrackerStuff = processContours(workingFrame, contours)

		if frameWithTrackerStuff is None:
			#Do not change workingFrame
			pass
		else:
			workingFrame = frameWithTrackerStuff

		# frameToDisplay = None
		# if frameWithTrackerStuff is None:
		# 	frameToDisplay = currentFrame
		# 	print "frameToDisplay takes currentFrame"
		# 	print "currentFrame is {}".format(type(currentFrame))
		# else:
		# 	frameToDisplay = frameWithTrackerStuff
		# 	print "frameToDisplay takes frameWithTrackerStuff"

		# print "frameWithTrackerStuff is {}".format(type(frameWithTrackerStuff))
		# print "frameToDisplay is {}".format(type(frameToDisplay))

		#Get the current traveling direction of this frame (compared to the last frame)
		frameWithDirectionText = None
		if len(g_trackedPoints) > 2:

			frameWithDirectionText = workingFrame
			(thisFrameDX, thisFrameDY) = getDirectionChanges()
			frameWithDirectionText = drawDirectionText(frameWithDirectionText, g_currentDirection, thisFrameDX, thisFrameDY)

		if frameWithDirectionText is None:
			#Do not change workingFrame
			pass
		else:
			workingFrame = frameWithDirectionText


		#We're ready to draw the modified frame now.
		cv2.imshow("Frame", workingFrame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		g_frameCounter += 1

	#end big while loop

#end main body
