import math

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

def determineDirection(normalized_vector):
	if normalized_vector[1] >= 0:
		#Quadrant 1 or 2
		return getDirectionInSemicircle(normalized_vector, "North")
	else:
		#Quadrant 3 or 4
		return getDirectionInSemicircle(normalized_vector, "South")
#end determineDirection

def getEuclideanDistance(vector):
	return math.sqrt( math.pow(vector[0], 2) + math.pow(vector[1], 2) )

def normalizeVector(vector):
	mag = getEuclideanDistance(vector)
	if mag != 0.0:
		return ( (vector[0] / mag), (vector[1] / mag) )
	else:
		return (0, 0)

def addVectorsAndNormalize(vector_list):
	#Add all the vectors in the list, then normalize the sum
	sum_vector = []
	sum_vector.append(0)
	sum_vector.append(0)

	for vec in vector_list:
		sum_vector[0] = sum_vector[0] + vec[0]
		sum_vector[1] = sum_vector[1] + vec[1]

	#Normalize this sum if possible
	return normalizeVector(sum_vector)