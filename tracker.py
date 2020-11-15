from imutils.video import VideoStream
import argparse
import cv2
import imutils
import time
import pandas as pd
import sys

from contours import *
from drawing import *
from utils import *

#Global vars
g_frameCounter = 0 #Number of frames that have passed
g_directionChangeArray = [] #List of direction changes (direction, timestamp), gets written to output

g_numTrackers = 10 #Max is 10 for now
g_showMasks = False
g_showArrows = True

#Parse the command line arguments
#Return video file path
def parseCommandLineArgs(args):
	global g_showMasks

	if len(args) >= 2:
		print "Args: {}".format(args)
		videoFilePath = args[1]
		print "Video path: {}".format(videoFilePath)
		return videoFilePath
		if "-showMasks" in args:
			g_showMasks = True

	print "Error: Must specify video file path."
	print "Usage: python object_movement.py object_tracking_example.mp4"
	return None
#end parseCommandLineArgs

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
	mask_hue_length = 18
	mask_hue_start_val = 30 #Start at green
	for i in range(g_numTrackers):
		hue_lower_bound = (mask_hue_start_val + mask_hue_length * i) % 180
		hue_upper_bound = (mask_hue_start_val + (mask_hue_length * (i + 1)) - 1) % 180

		this_lower_bound = (hue_lower_bound, 86, 6)
		this_upper_bound = (hue_upper_bound, 255, 255)
		new_tracker = ColorTracker(this_lower_bound, this_upper_bound)
		new_tracker.show_mask_window = g_showMasks
		all_trackers.append(new_tracker)
	white_tracker = ColorTracker((0, 0, 125), (179, 10, 255))
	all_trackers.append(white_tracker)

	#Main while loop. On every iteration, get the next frame from the video stream and process it.
	g_frameCounter = 0

	#Record direction changes and place them in a "bucket" to be periodically processed.
	frame_bucket_counter = 1
	direction_change_bucket = []
	print "Show Masks: {}".format(g_showMasks)

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
		prepped_frame = prepped_frame[0:-65, 0:600]
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

			#Draw things on the screen now
			if tracker.should_draw_circle and len(tracker.tracked_points) > 1:
				current_point = tracker.tracked_points[0]
				#Draw the circle
				working_frame = drawCircleToFrame(working_frame, tracker.current_circle_center, int(round(tracker.current_circle_radius)), current_point)
				#Draw the arrow
				working_frame = drawArrowToFrame(working_frame, current_point, multiplyVectorByScalar(normalizeVector(tracker.summed_vector), 40))

		overall_direction_vector = addVectors(all_direction_vectors)
		direction_change_bucket.append(overall_direction_vector)

		#Determine direction from this vector
		direction_string = determineDirection(normalizeVector(overall_direction_vector))
		#Draw this on the screen
		working_frame = drawDirectionText(working_frame, direction_string, overall_direction_vector[0], overall_direction_vector[1], g_frameCounter)


		g_frameCounter += 1
		if frame_bucket_counter % 50 == 0:
			output_direction = determineDirection(addVectorsAndNormalize(direction_change_bucket))
			output_entry = {
				"direction": output_direction,
				"timestamp": g_frameCounter
				}
			g_directionChangeArray.append(output_entry)
			print output_entry
			frame_bucket_counter = 0
			del direction_change_bucket[:]
		frame_bucket_counter += 1


		# We're ready to draw the modified frame now.
		cv2.imshow("Frame", working_frame)

		# Display the masks if specified
		if g_showMasks:
			for tracker in all_trackers:
				window_title = "Tracker {}".format(tracker.color_lower_bound)
				cv2.imshow(window_title, tracker.masked_frame)

		if g_showArrows:
			for tracker in all_trackers:
				window_title = "Tracker {} Vector".format(tracker.color_lower_bound)
				arrow_image = createArrowImg(tracker.summed_vector)
				cv2.imshow(window_title, arrow_image)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break


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
