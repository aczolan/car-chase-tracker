import cv2
import numpy as np

def drawArrowToFrame(frame, origin_point, arrow_vector, arrow_color):
	arrow_end_point_x = origin_point[0] + arrow_vector[0]
	arrow_end_point_y = origin_point[1] + arrow_vector[1]
	arrow_end_point = (int(arrow_end_point_x), int(arrow_end_point_y))

	new_frame = frame
	cv2.arrowedLine(new_frame, origin_point, arrow_end_point, arrow_color, 2)
	return new_frame
#end drawArrowToFrame

#Draw the circle and centroid (for the most recent tracked point) on the frame
#Return the new frame with the circle and centroid drawn on it
def drawCircleToFrame(frame, circle_center, circle_radius, centroid_center):
	circle_color = (0, 255, 255)
	centroid_color = (0, 0, 255)
	centroid_radius = 5

	new_frame = frame

	cv2.circle(new_frame, circle_center, circle_radius, circle_color, 2)
	cv2.circle(new_frame, centroid_center, centroid_radius, centroid_color, -1)

	return new_frame
#end drawCircleToFrame

def drawDotToFrame(frame, dot_center, dot_color):
	dot_radius = 5

	newFrame = frame

	cv2.circle(newFrame, dot_center, dot_radius, dot_color, -1)

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

	#Draw outer circle
	outer_circle_radius = 50
	img = cv2.circle(img, center_point, outer_circle_radius, red, 2)

	#Draw cardinal directions
	text_font = cv2.FONT_HERSHEY_SIMPLEX
	text_size = 0.5
	text_color = red
	cv2.putText(img, "N", (center_point[0] - 5, center_point[1] - outer_circle_radius - 6), text_font, text_size, red, 2)
	cv2.putText(img, "S", (center_point[0] - 5, center_point[1] + outer_circle_radius + 15), text_font, text_size, red, 2)
	cv2.putText(img, "W", (center_point[0] - outer_circle_radius - 16, center_point[1] + 6), text_font, text_size, red, 2)
	cv2.putText(img, "E", (center_point[0] + outer_circle_radius + 4, center_point[1] + 6), text_font, text_size, red, 2)

	#Draw arrow
	end_point = (center_point[0] + int(round(arrow_vector[0])), center_point[1] + int(round(arrow_vector[1])))
	color = (255, 255, 255)
	thickness = 3

	img = cv2.arrowedLine(img, center_point, end_point, color, thickness)

	return img
#end createArrowImg