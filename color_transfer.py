import sys, os
import cv2
import numpy as np

def read_file(sn,tn):
	s = cv2.imread(sn)
	s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
	t = cv2.imread(tn)
	t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
	return s, t
    
def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std
    
def color_transfer(sourcesN, targetsN):
	print("Converting picture ...")   
	s, t = read_file(sourcesN,targetsN)
	s_mean, s_std = get_mean_and_std(s)
	t_mean, t_std = get_mean_and_std(t)
	height, width, channel = s.shape
	for i in range(0,height):
		for j in range(0,width):
			for k in range(0,channel):
				x = s[i,j,k]
				x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
				x = round(x)
                # boundary check
				x = 0 if x<0 else x
				x = 255 if x>255 else x
				s[i,j,k] = x

	s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
	return s


right = color_transfer(sys.argv[1], sys.argv[2])
cv2.imwrite(sys.argv[3], right)
