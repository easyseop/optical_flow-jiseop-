#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import time

lk_params = dict(winSize = (15,15),
                maxLevel = 10,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))


feature_params = dict(maxCorners = 10,
                     qualityLevel = 0.3,
                     minDistance = 20,
                     blockSize = 7)

trajectory_len = 10
detect_interval = 1
trajectories = []
frame_idx = 0

# cap = cv2.VideoCapture('/Users/seop/파이썬/open cv/광주-아파트-붕괴.mp4')
cap = cv2.VideoCapture(0)

while True:
    
    #start time to calculate FPS
    start = time.time()
    
    suc,frame = cap.read()
#     print(suc,frame)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    
    #Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method

    if len(trajectories) > 0:
        img0,img1 = prev_gray,frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1,1,2)
        p1,_st,_err = cv2.calcOpticalFlowPyrLK(img0,img1,p0,None,**lk_params)
        p0r,_st,_err = cv2.calcOpticalFlowPyrLK(img1,img0,p1,None,**lk_params)
        d = abs(p0-p0r).reshape(-1,2).max(-1)
        good = d<1
        
        new_trajectories = []
        
        #Get all the trajectories
        for trajectory,(x,y), good_flag in zip(trajectories,p1.reshape(-1,2),good):
            if not good_flag:
                continue
            trajectory.append((x,y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)
            #Newest detected point
            cv2.circle(img,(int(x),int(y)),2,(0,0,255),-1)
            
        trajectories = new_trajectories
        
        # Draw all the trajectories
        cv2.polylines(img,[np.int32(trajectory) for trajectory in trajectories],False,(0,255,0))
        cv2.putText(img,'track count : %d'%len(trajectories),(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        
    # Update interval - When to update and detect new features
    if frame_idx % detect_interval ==0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest trajectory
        for x,y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
            cv2.circle(mask,(x,y),5,0,-1)

        # Detect the good features to track
        p = cv2.goodFeaturesToTrack(frame_gray,mask = mask,**feature_params)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x,y in np.float32(p).reshape(-1,2):
                trajectories.append([(x,y)])
                    
    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    # Show Results
    cv2.putText(img,f"{fps:.2f} FPS",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
    cv2.imshow('Optical Flow',img)
    cv2.imshow('Mask',mask)

   
    if cv2.waitKey(10) & 0xFF== ord('q'):
        break

            
cap.release() #자원 해제
cv2.destroyAllWindows() # 모든 창 닫기
cv2.waitKey(1)


# In[3]:


import numpy as np
import cv2
import time



def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr




cap = cv2.VideoCapture(0)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()


    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    prevgray = gray


    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)qq

    print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))
    cv2.imshow('flow HSV', draw_hsv(flow))


    key = cv2.waitKey(5)
    if key == ord('q'):
        break


cap.release() #자원 해제
cv2.destroyAllWindows() # 모든 창 닫기
cv2.waitKey(1)


# In[ ]:




