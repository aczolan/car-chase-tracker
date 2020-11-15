import math
import numpy as np

def getDirectionInSemicircle(vector, north_or_south_string):
	thresh_val_1 = 0.38268 # cos(3pi/8)
	thresh_val_2 = 0.92388 # cos(pi/8)
	direction = ""

	if vector[0] < (-1 * thresh_val_2):
		direction = "West"
	elif vector[0] < (-1 * thresh_val_1):
		direction = "{}-West".format(north_or_south_string)
	elif vector[0] < thresh_val_1:
		direction = "North"
	elif vector[0] < thresh_val_2:
		direction = "{}-East".format(north_or_south_string)
	else:
		direction = "East"

	return direction
#end function

def determineDirectionFromVector(normalized_vector):
	if normalized_vector[1] >= 0:
		#Quadrant 1 or 2
		return getDirectionInSemicircle(normalized_vector, "North")
	else:
		#Quadrant 3 or 4
		return getDirectionInSemicircle(normalized_vector, "South")
#end determineDirection

def determineDirectionFromAngle(angleDeg):
	directions = ["East", "North-East", "North", "North-West", "West", "South-West", "South", "South-East", "East"]
	num_rotations = int(angleDeg / 45.0) - 1
	return directions[num_rotations]

def getEuclideanDistance(vector):
	return math.sqrt( math.pow(vector[0], 2) + math.pow(vector[1], 2) )

def normalizeVector(vector):
	mag = getEuclideanDistance(vector)
	if mag != 0.0:
		return ( (vector[0] / mag), (vector[1] / mag) )
	else:
		return (0, 0)

def addVectors(vectors_list):
	sum_vector = []
	sum_vector.append(0)
	sum_vector.append(0)

	for vec in vectors_list:
		sum_vector[0] = sum_vector[0] + vec[0]
		sum_vector[1] = sum_vector[1] + vec[1]

	return sum_vector

def addVectorsAndNormalize(vectors_list):
	return normalizeVector(addVectors(vectors_list))

def multiplyVectorByScalar(vector, scalar):
	new_vector = []
	for i in range(len(vector)):
		new_vector.append(vector[i] * scalar)
	return new_vector

#Convert a right-down x,y vector (used in cv)to right,up x,y vector (normal cartesian coordinates)
def convertFromRDtoRUVector(rd_vector):
	new_vector = (rd_vector[0], -rd_vector[1])
	return new_vector

#Get the angle from zero (in degrees) from a given vector
def getAngleFromVector(vector):
	if vector[0] == 0.0:
		if vector[1] >= 0.0:
			return 90.0
		elif vector[1] < 0.0:
			return 270.0
		else:
			return 0.0
	return np.arctan(vector[1] / vector[0])

def averageFloatsInList(values):
	if len(values) == 0:
		return 0.0
	this_sum = 0
	for f in values:
		this_sum += f
	return this_sum / len(values)

def getNormalizedRUVectorFromAngle(angleDeg):
	x_component = np.cos(angleDeg)
	y_component = np.sin(angleDeg)
	#This angle is already normalized
	return (x_component, y_component)

def createOutputEntry(direction_text, timestamp, num_data_points):
	return {
		"direction": direction_text,
		"timestamp": timestamp,
		"num. data points": num_data_points
		}
