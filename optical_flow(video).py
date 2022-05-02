import numpy as np
import cv2
import time
from scipy import signal

lk_params = dict(winSize = (15,15),
                maxLevel = 10,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

def draw_flow(img, flow, step=100):
    global arr
    I1 = np.array(img)

    S = np.shape(I1) # 사진의 크기 또는 사이즈 ex) 640*480

    I1_smooth = cv2.GaussianBlur(I1,(3,3),0)
    
    
  
    # First Derivative in X direction
    Ix = signal.convolve2d(I1_smooth,[[-0.25,0.25],[-0.25,0.25]],'same') 
    Iy = signal.convolve2d(I1_smooth,[[-0.25,-0.25],[0.25,0.25]],'same') 
    # First Derivative in XY direction
    It = signal.convolve2d(I1_smooth,[[0.25,0.25],[0.25,0.25]],'same')
   
    # finding the good features
    features = cv2.goodFeaturesToTrack(I1_smooth,10000,0.01,10)

    feature = np.int0(features)
   
   
    
    u = v = np.nan*np.ones(S)
    
    # Calculating the u and v arrays for the good features obtained n the previous step.

    x = []
    y= []
    for i in feature:

        x.append(i[0][0])
        y.append(i[0][1])
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(-1,)
    y = y.reshape(-1,)
    
    fx, fy = flow[y,x].T

    
    
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
#     print('y : ',y , 'x : ',x)
#     fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:

        cv2.circle(img_bgr, (x1, y1), 1, (255, 255, 0), -1)


    return img_bgr





#
# cap = cv2.VideoCapture('/Users/seop/Desktop/video.mov')
cap = cv2.VideoCapture(0)

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
trajectories = []

while True:

    suc, img = cap.read()
    print(suc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()


    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     flow = cv2.calcOpticalFlowPyrLK(prevgray, gray, None,**lk_params)
  
    prevgray = gray


    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

    print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))



    key = cv2.waitKey(10)
    if key == ord('q'):
        break


cap.release() #자원 해제
cv2.destroyAllWindows() # 모든 창 닫기
cv2.waitKey(1)