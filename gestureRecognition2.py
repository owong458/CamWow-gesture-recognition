from __future__ import print_function
import cv2
import math

#This code uses color recognition to create a binary mask of a hand, finds the contour of the hand,
#and then uses convexity defects to determine the number of fingers being held up.
#Will still run if objects other than hand are in stream, but may not count as intended if those objects are close
#enough to skin color.

#Note: May occasionally throw "error: (-5:Bad argument) The convex hull indices are not monotonous, which can be the
#cs sometimes when
#tase when the input contour contains self-intersections in function 'convexityDefects'". This happenhe conditions are poor, leading to a really wacky contour. Under good conditions, this won't happen.

#Tuning may be required. Optimized for performance on a uniformly lit white background.

#argument for cv.VideoCapture should be the ID of the camera supplying input, add this before running
capture = cv2.VideoCapture(0)
if not capture.isOpened:
    print('Unable to open')
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    #creates copy of original frame for later use in display
    fgContour = frame.copy()
    #IMPORTANT: Depending on lighting conditions and skin color, these values may require adjustments to ensure that
    #the hand but not the background is caught within the range.
    upperHue = (40, 255, 0.95 * 255)
    lowerHue = (0, 0.15 * 255, 0)
    #Converts input image to HSL, then creates a binary mask.
    #Blurring is used to reduce noise and create a smoother contour.
    frameHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frameHLSBlur = cv2.blur(frameHLS, (2, 2), 0)
    fgMask = cv2.inRange(frameHLSBlur, lowerHue, upperHue)
    blur = cv2.blur(fgMask, (5, 5), 0)
    ret, fgThresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # "0.15 * (frame.shape[0] * frame.shape[1])" is 15% of the total number of pixels on the frame
    if cv2.countNonZero(fgMask) > 0.15 * frame.shape[0] * frame.shape[1]:
        print("Something is in front of the camera")
        #Finds all contours of the image
        contours = cv2.findContours(fgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        #Finds contour with largest area, stores index of this largest contour
        maxArea = 0
        maxIndex = 0
        shapeDet=0;
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > maxArea:
                maxArea = area
                maxIndex = i
            #Checks area to only check for shapes in contours large enough to be something
            if area > 5000:
                #Approximates the contour as a shape
                approx = cv2.approxPolyDP(contours[i],80,True)
                #Counts corners, determines if it is a triangle or rectangle
                objCor = len(approx)
                #print(objCor)
                if objCor == 3:
                    print("Triangle")
                    shapeDet = 1
                    maxIndex = i
                    break
                elif objCor == 4:
                    print("Rectangle")
                    shapeDet = 1
                    maxIndex = i
                    break
                #An attempt to recognize circles made by index finger and thumb, currently nonfunctional
                #elif objCor > 4:
                    #conHull = cv2.convexHull(contours[i], returnPoints=False)
                    #conDef = cv2.convexityDefects(contours[i], conHull)
                    #isCircle = True
                    #for j in range(len(conDef)):
                        #if conDef[j][0][3] > 100:
                            #isCircle = False
                    #if isCircle:
                        #print("Circle")
                        #shapeDet = 1
                        #maxIndex = i
                        #break
        #Checks if a shape was found in the contours, if yes, then draws the shape and skips convex hulling
        if shapeDet == 1:
            cv2.drawContours(fgContour, contours, maxIndex, (10, 100, 50), 20)
            cv2.imshow('juicy beans', fgContour)
            cv2.waitKey(30)
            continue
        #Finds convex hull as both points and indices on the original contour, draws the largest contour
        hull = None
        if len(contours) > 0:
            hull = cv2.convexHull(contours[maxIndex])
            hullIndex = cv2.convexHull(contours[maxIndex], returnPoints=False)
            cv2.drawContours(fgContour, contours, maxIndex, (10, 100, 50), 20)
            cv2.drawContours(fgContour, hull, -1, (255, 0, 0), 20)
            prevPoint = hull[0]
            #creates a list of lists to associate each hull point to the central point of any cluster it's in
            pointIndices = []
            currentPointMinIdx=0
            currentPointMaxIdx=0
            for i in range(len(hull)):
                if i == 0:
                    continue
                else:
                    difference = hull[i]-hull[i-1]
                    distanceSq = difference[0][0]*difference[0][0]+difference[0][1]*difference[0][1]
                    if distanceSq<4000:
                        currentPointMaxIdx=i
                    else:
                        j = currentPointMinIdx
                        while j <= currentPointMaxIdx:
                            idx = (int)((currentPointMaxIdx + currentPointMinIdx) / 2)
                            hullIdxJ = hullIndex[j]
                            hullIndexIdx = hullIndex[idx]
                            pointIndices.append([hullIdxJ, hullIndexIdx])
                            j += 1
                        currentPointMinIdx = i
                        currentPointMaxIdx = i
            #generates convexity defects, uses those convexivity defects to determine the angle generated by such defects
            #and the central points of the 2 neighboring clusters
            fingerCount=0
            if len(hull)>0:
                defects = cv2.convexityDefects(contours[maxIndex], hullIndex)
                for i in range(len(defects)):
                    #ignores defects if smaller than a certain amount
                    if defects[i][0][3]/256 < 100:
                        #print(defects[i][0][3])
                        continue
                    else:
                        #finds previous and next central hull point, as well as point of maximum defect, as coordinates
                        prevHullIdx = [0]
                        nextHullIdx = [0]
                        for j in range(len(pointIndices)):
                            if hullIndex[j][0] == defects[i][0][0]:
                                prevHullIdx = pointIndices[j][1]
                            if hullIndex[j][0] == defects[i][0][1]:
                                nextHullIdx = pointIndices[j][1]
                        currentDefectIdx = defects[i][0][2]
                        prevHullPoint = contours[maxIndex][prevHullIdx]
                        nextHullPoint = contours[maxIndex][nextHullIdx]
                        defectPoint = contours[maxIndex][currentDefectIdx]
                        #draws a circle around defect point
                        cv2.circle(fgContour, (defectPoint[0][0], defectPoint[0][1]), 40, (0, 0, 255), 10)
                        #uses law of cosine to determine angle of defect
                        difference = prevHullPoint-nextHullPoint
                        distanceBetweenSq = difference[0][0][0]*difference[0][0][0]+difference[0][0][1]*difference[0][0][1]
                        aDiff = defectPoint-prevHullPoint
                        bDiff = defectPoint-nextHullPoint
                        aDiffSq = aDiff[0][0][0]*aDiff[0][0][0]+aDiff[0][0][1]*aDiff[0][0][1]
                        bDiffSq = bDiff[0][0][0] * bDiff[0][0][0] + bDiff[0][0][1] * bDiff[0][0][1]
                        cosAlpha = (distanceBetweenSq-aDiffSq-bDiffSq)/(-2*math.sqrt(aDiffSq)*math.sqrt(bDiffSq))
                        if cosAlpha > 1:
                            #print("domain issue")
                            alpha = 1
                        else:
                            alpha = math.acos(cosAlpha)
                        #draws circles around previous and next central hull points
                        cv2.circle(fgContour, (nextHullPoint[0][0][0], nextHullPoint[0][0][1]), 40, (0, 255, 0), 10)
                        cv2.circle(fgContour, (prevHullPoint[0][0][0], prevHullPoint[0][0][1]), 40, (0, 255, 255), 10)
                        #print(alpha)
                        #if distanceSq<1000:
                        #    continue
                        if math.fabs(alpha) < 1.5:
                            fingerCount += 1
                #accounts for previous code counting number of spaces between fingers, not the fingers themselves
                if fingerCount >= 1:
                    fingerCount += 1
                #handles the case of holding up one finger
                else:
                    for k in range(len(pointIndices)):
                        #for each hull point aside from the first and the last, finds the two neighboring hull points
                        #and uses law of cosines to determine if the angle is small enough to be a finger
                        if k == 0:
                            continue
                        if k == len(pointIndices)-1:
                            continue
                        m = k-1
                        n = k+1
                        while pointIndices[m][1] == pointIndices[k][1]:
                            if m == 0:
                                break
                            m -= 1
                        while pointIndices[n][1] == pointIndices[k][1]:
                            if n == len(pointIndices)-1:
                                break
                            n += 1
                        prevIdx = pointIndices[m][1]
                        nextIdx = pointIndices[n][1]
                        currentPointIdx = pointIndices[k][1]
                        currentPoint = contours[maxIndex][currentPointIdx]
                        prevCentralPoint = contours[maxIndex][prevIdx]
                        nextCentralPoint = contours[maxIndex][nextIdx]
                        difference = nextCentralPoint-prevCentralPoint
                        distanceBetweenSq = difference[0][0][0] * difference[0][0][0] + difference[0][0][1] * difference[0][0][1]
                        aDiff = currentPoint-prevCentralPoint
                        bDiff = nextCentralPoint-currentPoint
                        aDiffSq = aDiff[0][0][0] * aDiff[0][0][0] + aDiff[0][0][1] * aDiff[0][0][1]
                        bDiffSq = bDiff[0][0][0] * bDiff[0][0][0] + bDiff[0][0][1] * bDiff[0][0][1]
                        cosBeta = (distanceBetweenSq - aDiffSq - bDiffSq) / (-2 * math.sqrt(aDiffSq) * math.sqrt(bDiffSq))
                        if cosBeta > 1:
                            #print("domain issue")
                            beta = 1
                        else:
                            beta = math.acos(cosBeta)
                        #print(beta)
                        if beta < 1.5:
                            fingerCount += 1
                            break



            print("Number of fingers held up:")
            print(fingerCount)
            cv2.imshow('juicy beans', fgContour)
            cv2.waitKey(30)
        #havent seen this actually be used yet, but I'll keep it here just in case this section is somehow accessed
        else:
            cv2.imshow('shady beans', fgThresh)
            cv2.waitKey(30)
    else:
        print("Nothing is in front of the camera")
        cv2.imshow('juicy beans', fgContour)
        cv2.waitKey(30)