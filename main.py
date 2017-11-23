from __future__ import print_function
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


img = None
thresh = None
blurKernelSize = 3
morphoKernelSize = 5
cannyMinVal = 100
cannyMaxVal = 200
pltOriginal = None
pltComputed = None


def computeCV():
    print("Computing... ", cannyMinVal, " ", cannyMaxVal)
    blur = cv2.blur(img, (blurKernelSize, blurKernelSize))
    pltOriginal.set_data(blur)
    edges = cv2.Canny(blur, cannyMinVal, cannyMaxVal)

    kernel = np.ones((morphoKernelSize, morphoKernelSize), np.uint8)

    opening = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    ret, thresh = cv2.threshold(opening, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(thresh, [box], -1, (255, 255, 255), 3)

    pltComputed.set_data(thresh)
    print("Computing: ok")


def updateCannyMinVal(val):
    global cannyMinVal
    cannyMinVal = int(val)
    computeCV()

def updateCannyMaxVal(val):
    global cannyMaxVal
    cannyMaxVal = int(val)
    computeCV()

def updateBlur(val):
    global blurKernelSize
    blurKernelSize = int(val)
    computeCV()

def updateMorphoKernel(val):
    global morphoKernelSize
    morphoKernelSize = int(val)
    computeCV()

if __name__ == '__main__':
    img = cv2.imread('C:/Users/Philippe/Pictures/contrastDePhase-stack_axon.jpg', 0)
    height, width = img.shape
    plt.subplot(211)
    pltOriginal = plt.imshow(np.zeros((height, width)), cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(212)
    pltComputed = plt.imshow(np.zeros((height, width)), cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title('opening Image'), plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(bottom=0.2, right=0.9, top=0.9)
    axCannyMinVal = plt.axes([0.2, 0.01, 0.7, 0.02])
    sCannyMinVal = Slider(axCannyMinVal, 'CannyMinVal', 1, 255, valinit=100, valfmt='%d')
    sCannyMinVal.on_changed(updateCannyMinVal)

    axCannyMaxVal = plt.axes([0.2, 0.04, 0.7, 0.02])
    sCannyMaxVal = Slider(axCannyMaxVal, 'CannyMaxVal', 1, 500, valinit=200, valfmt='%d')
    sCannyMaxVal.on_changed(updateCannyMaxVal)

    axBlur = plt.axes([0.2, 0.07, 0.7, 0.02])
    sBlur = Slider(axBlur, 'Blur', 1, 100, valinit=3, valfmt='%d')
    sBlur.on_changed(updateBlur)

    axMorphoKS = plt.axes([0.2, 0.1, 0.7, 0.02])
    sMorphoKS = Slider(axMorphoKS, 'MorphoKS', 1, 100, valinit=5, valfmt='%d')
    sMorphoKS.on_changed(updateMorphoKernel)

    computeCV()

    plt.show()





