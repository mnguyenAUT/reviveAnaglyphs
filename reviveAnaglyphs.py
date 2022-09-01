import os, sys, cv2
import numpy as np
from wand.image import Image

if len(sys.argv) != 2:
    print("Usage:")
    print("python reviveAnaglyphs.py [input image]")
    print("For example: 'python reviveAnaglyphs.py download.jpg'")
    exit(0)

imgQR = cv2.imread(sys.argv[1])

b, g, r = cv2.split(imgQR)
r = cv2.GaussianBlur(r,(3,3),0)
g = cv2.GaussianBlur(g,(3,3),0)
b = cv2.GaussianBlur(b,(3,3),0)
gb = cv2.addWeighted(g,0.5,b,0.5,0)

cv2.imwrite("r.png", r)
cv2.imwrite("gb.png", gb)

mean_r = (np.mean(r))
mean_gb = (np.mean(gb))



#to raise some brightness
r = cv2.add(r, (mean_gb-mean_r) )
#gb = cv2.add(gb)

#r = cv2.equalizeHist(r)
#gb = cv2.equalizeHist(gb)

    
cv2.imwrite("left.png", r)
cv2.imwrite("right.png", gb)

os.chdir("./colorization")

os.system("python demo_release.py -i ../left.png")
gb = cv2.addWeighted(g,0.5,b,0.5,0)

one = cv2.imread("saved_eccv16.png")
two = cv2.imread("saved_siggraph17.png")

leftImage = cv2.addWeighted(one,0.6,two,0.6,0)
cv2.imwrite("../left_colour.png", leftImage)

os.system("python demo_release.py -i ../right.png")
one = cv2.imread("saved_eccv16.png")
two = cv2.imread("saved_siggraph17.png")

rightImage = cv2.addWeighted(one,0.6,two,0.6,0)
cv2.imwrite("../right_colour.png", rightImage)

os.chdir("../")
os.system("python color_transfer.py right_colour.png left_colour.png right_colour_fixed.png")

os.chdir("./GFPGAN")
os.system("python inference_gfpgan.py -i ../left_colour.png -v 1.3")
os.system("python inference_gfpgan.py -i ../right_colour_fixed.png -v 1.3")

os.system("cp results/restored_imgs/left_colour.png ../final_left.png")
os.system("cp results/restored_imgs/right_colour_fixed.png ../final_right.png")

os.chdir("../")
os.system("python color_transfer.py final_left.png final_right.png final_left.png")

with Image(filename='final_left.png') as img:    
    img.virtual_pixel = 'black'
    img.distort('barrel', (0.1, 0.0, 0.0, 1.0))
    img.save(filename='left_barrel.png')

with Image(filename='final_right.png') as img:    
    img.virtual_pixel = 'black'
    img.distort('barrel', (0.1, 0.0, 0.0, 1.0))
    img.save(filename='right_barrel.png')
    
leftImage = cv2.imread("left_barrel.png")
rightImage = cv2.imread("right_barrel.png")
vrEye = np.hstack((leftImage, rightImage))
cv2.imwrite("VR-stereo.jpg", vrEye)
vrEyeX = np.hstack((rightImage,leftImage))
cv2.imwrite("VR-stereoX.jpg", vrEyeX)

#clean up
if os.path.exists('final_left.png'):
    os.remove('final_left.png')
if os.path.exists('final_right.png'):
    os.remove('final_right.png')
#if os.path.exists('left.png'):
#    os.remove('left.png')
#if os.path.exists('right.png'):
#    os.remove('right.png')
if os.path.exists('left_barrel.png'):
    os.remove('left_barrel.png')
if os.path.exists('right_barrel.png'):
    os.remove('right_barrel.png')
