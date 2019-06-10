import cv2
import numpy as np
from scipy import ndimage

gray_image = cv2.imread("dangit.png",0)

color_image = cv2.cvtColor(gray_image,cv2.COLOR_GRAY2RGB)

lut = np.zeros((1,256,3))
lut[:,0:128,:] = np.concatenate(( np.ones((1,128,1))*17, np.ones((1,128,1))*143, np.ones((1,128,1))*0 ),2)
lut[:,128:,:] = np.ones((1,128,3))*0

new_image = cv2.LUT(color_image,lut)

cv2.imwrite('test_matrix.png',new_image)

new_image = cv2.imread('test_matrix.png',1)

cv2.imshow('image',new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
