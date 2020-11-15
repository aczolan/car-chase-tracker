import cv2
import numpy as np

def drawArrowToFrame(frame, originPoint, arrowVector):
	arrow_color = (0, 255, 0)
	arrow_end_point_x = originPoint[0] + arrowVector[0]
	arrow_end_point_y = originPoint[1] + arrowVector[1]
	arrow_end_point = (int(arrow_end_point_x), int(arrow_end_point_y))

	newFrame = frame
	cv2.arrowedLine(newFrame, originPoint, arrow_end_point, arrow_color, 2)
	return newFrame
#end drawArrowToFrame

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

def drawDotToFrame(frame, dotCenter):
	dotColor = (255, 0, 0)
	dotRadius = 5

	newFrame = frame

	cv2.circle(newFrame, dotCenter, dotRadius, dotColor, -1)

	return newFrame

#Draw connected lines that trail behind the tracked point
#Return the new frame with the trail drawn on it
def drawTrailToFrame(frame, points, jump_detected):

	newFrame = frame

	#Iterate over all points and draw line segments between them
	for pointNum in numpy.arange(1, len(points)):

		if points[pointNum - 1] is None or points[pointNum] is None:
			continue

		thickness = int(numpy.sqrt(32 / float(pointNum + 1)) * 2.5) #thickness
		startPoint = points[pointNum - 1] #start of line segment
		endPoint = points[pointNum] #end of line segment
		color = (0, 0, 255) #color

		if jump_detected:
			color = (255, 0, 0)

		#add this line segment to the frame
		cv2.line(newFrame, startPoint, endPoint, color, thickness)

	return newFrame
#end drawTrailToFrame

#Draw the text that displays the current direction and dX/dY values
#Return the new frame with the text drawn on it
def drawDirectionText(frame, direction, dX, dY, frame_counter):

	directionText = direction
	directionTextPosition = (10, 30)
	directionTextFont = cv2.FONT_HERSHEY_SIMPLEX
	directionTextSize = 0.65
	directionTextColor = (0, 0, 255)

	dxdyText = "frame: {}, dx: {}, dy: {}".format(frame_counter, dX, dY)
	dxdyTextPosition = (10, frame.shape[0] - 10)
	dxdyTextFont = cv2.FONT_HERSHEY_SIMPLEX
	dxdyTextSize = 0.35
	dxdyTextColor = (0, 0, 255)

	newFrame = frame

	cv2.putText(newFrame, direction, directionTextPosition, directionTextFont, directionTextSize, directionTextColor, 3)
	cv2.putText(newFrame, dxdyText, dxdyTextPosition, dxdyTextFont, dxdyTextSize, dxdyTextColor, 1)

	return newFrame
#end getDirectionText

def createArrowImg(arrow_vector):
	width = 200
	height = 200
	img = np.zeros(shape=[width, height, 3], dtype=np.uint8)

	#Draw center point
	red = (0, 0, 255)
	center_point = ((width / 2), (height / 2))
	img = cv2.circle(img, center_point, 5, red, -1)

	#Draw arrow
	end_point = (center_point[0] + int(round(arrow_vector[0])), center_point[1] + int(round(arrow_vector[1])))
	color = (255, 255, 255)
	thickness = 5

	img = cv2.arrowedLine(img, center_point, end_point, color, thickness)

	return img
#end createArrowImg