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

#Global variables and Settings

#Main vector acquisition modes (choose one)
g_vectorAdditionMode = True #Combine vectors every using their angles and magnitudes
g_vectorAngleAvgMode = False #Experimental. Just retrieve vector angles in degrees every frame

#Direction vector calculation method (choose one)
g_useSumVectors = True #Combine results from all previous valid tracked points
g_useRecentVectors = False #Only use the most current tracked point

#Optional Views
g_showMasks = False #Show color mask window for each tracker
g_showArrows = False #Show current direction vector window for each tracker
g_showDirVectorsPerFrame = False #Show a window for the cumulative direction vector across all trackers on each frame
g_showBucketVectors = False #Show a window for the combined direction vector at each bucket interval
g_showTrackedPoints = True #Show current valid tracked points in the main window

#Tracker and bucket settings
g_numberOfTrackers = 10 #Max is 10 for now
g_trackerEuclidThreshold = 200 #Euclidean jump threshold in pixels-distance
g_bucketInterval = 100 #Frame interval for combining datapoints

#Internal use
g_frameCounter = 0 #Number of frames that have passed
g_directionChangeArray = [] #List of direction changes (direction, timestamp), gets written to output

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

#Draws a tracker's enclosing circle, direction vectors, and tracked points onto the frame
def drawTrackerInfoOnFrame(working_frame, tracker):
	#Draw things for this tracker
	if len(tracker.tracked_points) > 1:
		current_point = tracker.tracked_points[0]
		if tracker.should_draw_circle:
			working_frame = drawCircleToFrame(working_frame, tracker.current_circle_center, int(round(tracker.current_circle_radius)), current_point)
		#Draw arrow for sum vector
		summed_vector_color = (0, 255, 0)
		working_frame = drawArrowToFrame(working_frame, current_point, tracker.summed_vector, summed_vector_color)
		#Draw arrow for the most recent vector
		current_vector_color = (150, 150, 0)
		working_frame = drawArrowToFrame(working_frame, current_point, tracker.current_vector, current_vector_color)
		#Draw tracked points
		if g_showTrackedPoints:
			tracked_points_dot_color = (255, 0, 0) #cv2.cvtColor(tracker.color_lower_bound, cv2.COLOR_HSV2BGR)
			for p in tracker.tracked_points:
				working_frame = drawDotToFrame(working_frame, p, tracked_points_dot_color)
	return working_frame

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

	#Initialize all trackers
	all_trackers = []
	#mask_hue_length = round(360 / g_numberOfTrackers)
	mask_hue_length = 18
	mask_hue_start_val = 30 #Start at green
	for i in range(g_numberOfTrackers):
		hue_lower_bound = (mask_hue_start_val + mask_hue_length * i) % 180
		hue_upper_bound = (mask_hue_start_val + (mask_hue_length * (i + 1)) - 1) % 180

		this_lower_bound = (hue_lower_bound, 86, 6)
		this_upper_bound = (hue_upper_bound, 255, 255)
		new_tracker = ColorTracker(this_lower_bound, this_upper_bound, g_trackerEuclidThreshold)
		new_tracker.show_mask_window = g_showMasks
		all_trackers.append(new_tracker)
	white_tracker = ColorTracker((0, 0, 125), (179, 10, 255), g_trackerEuclidThreshold)
	all_trackers.append(white_tracker)

	#Main while loop. On every iteration, get the next frame from the video stream and process it.
	g_frameCounter = 0

	#Record direction changes and place them in a "bucket" to be periodically processed.
	frame_bucket_counter = 1
	direction_change_bucket = []
	angles_bucket = []
	most_recent_bucket_vector = (0, 0)
	bucket_datapoint_count = 0

	while True:

		#time.sleep(0.05)

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
		all_direction_vectors = [] #Used in vector addition mode
		all_direction_angles = [] #Used in angle averaging mode
		this_frame_direction_vector = (0, 0) #Overall direction vector determined from this frame
		this_frame_num_datapoints = 0 #Number of vectors/angles found on this frame

		# --- Vector Addition Mode ---

		if g_vectorAdditionMode:
			for tracker in all_trackers:
				tracker.processNewFrame(prepped_frame)
				if tracker.dirvector_ready:

					#Get direction vector for this frame
					if g_useRecentVectors:
						all_direction_vectors.append(normalizeVector(tracker.current_vector))
					elif g_useSumVectors:
						all_direction_vectors.append(normalizeVector(tracker.summed_vector))

					working_frame = drawTrackerInfoOnFrame(working_frame, tracker)
					this_frame_num_datapoints += tracker.num_usable_datapoints

			this_frame_direction_vector = addVectors(all_direction_vectors)
			#Multiply this vector by -1 since its inverse is the camera pan direction (what we're looking for)
			#this_frame_direction_vector = multiplyVectorByScalar(this_frame_direction_vector, -1)
			direction_change_bucket.append(this_frame_direction_vector)
			bucket_datapoint_count += this_frame_num_datapoints

			#Determine direction for just this frame
			direction_string = determineDirectionFromVector(normalizeVector(convertFromRDtoRUVector(this_frame_direction_vector)))
			#Draw this direction on the screen
			working_frame = drawDirectionText(working_frame, direction_string, this_frame_direction_vector[0], this_frame_direction_vector[1], g_frameCounter)

			#Increment frame counter and handle bucket
			g_frameCounter += 1
			if frame_bucket_counter % g_bucketInterval == 0:
				most_recent_bucket_vector = addVectorsAndNormalize(direction_change_bucket)
				#Multiply this vector by -1 since its inverse is the camera pan direction (what we're looking for)
				most_recent_bucket_vector = multiplyVectorByScalar(most_recent_bucket_vector, -1)
				output_direction_string = determineDirectionFromVector(convertFromRDtoRUVector(most_recent_bucket_vector))
				output_entry = createOutputEntry(output_direction_string, g_frameCounter, bucket_datapoint_count)
				g_directionChangeArray.append(output_entry)
				print(output_entry)
				#Reset the bucket
				frame_bucket_counter = 0
				bucket_datapoint_count = 0
				del direction_change_bucket[:]
			frame_bucket_counter += 1

		# --- Vector Angle Averaging Mode ---

		elif g_vectorAngleAvgMode:
			for tracker in all_trackers:
				tracker.processNewFrame(prepped_frame)
				if tracker.dirvector_ready:

					#Get direction angle for this frame
					if g_useRecentVectors:
						all_direction_angles.append(getAngleFromVector(tracker.current_vector))
					elif g_useSumVectors:
						all_direction_angles.append(getAngleFromVector(tracker.summed_vector))

					working_frame = drawTrackerInfoOnFrame(working_frame, tracker)
					this_frame_num_datapoints += tracker.num_usable_datapoints

			overall_direction_angle = averageFloatsInList(all_direction_angles)
			angles_bucket.append(overall_direction_angle)
			bucket_datapoint_count += this_frame_num_datapoints

			#Determine direction for just this frame
			this_frame_direction_vector = getNormalizedRUVectorFromAngle(overall_direction_angle)
			direction_string = determineDirectionFromAngle(overall_direction_angle)
			#Draw this direction on the screen
			working_frame = drawDirectionText(working_frame, direction_string, this_frame_direction_vector[0], this_frame_direction_vector[1], g_frameCounter)
			
			#Increment frame counter and handle bucket
			g_frameCounter += 1
			if frame_bucket_counter % g_bucketInterval == 0:
				bucket_average_angle = averageFloatsInList(angles_bucket)
				most_recent_bucket_vector = getNormalizedRUVectorFromAngle(bucket_average_angle)
				#Convert back to RD coordinates for OpenCV
				most_recent_bucket_vector = (most_recent_bucket_vector[0], -most_recent_bucket_vector[1])
				output_direction_string = determineDirectionFromAngle(bucket_average_angle)
				output_entry = createOutputEntry(output_direction_string, g_frameCounter, bucket_datapoint_count)
				g_directionChangeArray.append(output_entry)
				print(output_entry)
				frame_bucket_counter = 0
				bucket_datapoint_count = 0
				del angles_bucket[:]
			frame_bucket_counter += 1

		# We're ready to draw the modified frame now.
		cv2.imshow("Frame", working_frame)

		# Display optional views if specified
		if g_showMasks or g_showArrows or g_showDirVectorsPerFrame or g_showBucketVectors:
			for tracker in all_trackers:

				if g_showMasks:
					mask_window_title = "Tracker {}".format(tracker.color_lower_bound)
					cv2.imshow(mask_window_title, tracker.masked_frame)
				if g_showArrows:
					arrow_window_title = "Tracker {} Vector".format(tracker.color_lower_bound)
					arrow_image = createArrowImg(tracker.summed_vector)
					cv2.imshow(arrow_window_title, arrow_image)

			if g_showDirVectorsPerFrame:
				dirvector_window_title = "Direction Vector (this frame)"
				#Multiply this vector by 40 to show it better
				dirvector_image = createArrowImg(multiplyVectorByScalar(this_frame_direction_vector, 40))
				cv2.imshow(dirvector_window_title, dirvector_image)
			if g_showBucketVectors:
				bucketvector_window_title = "Last Bucket Vector"
				#Multiply this vector by 40 to show it better
				bucketvector_image = createArrowImg(multiplyVectorByScalar(most_recent_bucket_vector, 40))
				cv2.imshow(bucketvector_window_title, bucketvector_image)

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

	output_csv_file = open(outputFileName, 'w')
	output_csv_file.writelines(linesToWrite)
	print "Wrote to output file {}".format(outputFileName)
	output_csv_file.close()

	#dataFrame stuff

	# dataFrame = pd.DataFrame(g_directionChangeArray)

	# print "dataFrame is {}".format(type(dataFrame))

	# dataFrame = dataFrame[dataFrame['direction'] != ""]

	# jsonFile = open('output.json', 'w')
	# jsonFile.write(dataFrame.to_json(orient='records'))
	# jsonFile.close()

	#Release the camera and close all windows
	videoStream.release()
	cv2.destroyAllWindows()

#end main body
