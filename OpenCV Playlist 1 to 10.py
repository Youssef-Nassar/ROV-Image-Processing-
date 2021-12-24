#Video 3
import cv2 

img = cv2.imread("test.jpg",1)

cv2.imshow('image',img)
k = cv2.waitKey(0)

if k == 27:
  cv2.destroyAllWindows()
elif k == ord('s'):
  cv2.imwrite("test_copy.jpg",img)
#________________________________________________
#________________________________________________

#Video 4 & 6
import cv2 
camera = cv2.VideoCapture(0)
camera.set(3,700)
camera.set(4,700)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc , 20.0 , (640,480))

while(camera.isOpened()):
  ret, frame = camera.read()

  if ret == True:
    print(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out.write(frame)
    cv2.imshow('farme', O_frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break
camera.release()
cv2.destroyAllWindows()

#________________________________________________
#________________________________________________

#Video 5
import cv2
import numpy 

img = numpy.zeros([512,512,3],numpy.uint8)
img = cv2.line(img , (0,0) , (255,255) , (147,96,44), 10)
img = cv2.arrowedLine(img , (0,255) , (255,255) , (255,0,0), 10)
img = cv2.rectangle(img , (384,0) , (510,128) , (0,0,255), 10)
img = cv2.circle(img , (447,63) , 63,(0,255,0),-1)
img = cv2.putText(img ,'TEST', (10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,255),10,cv2.LINE_AA)


cv2.imshow('image',img)
k = cv2.waitKey(0)

if k == 27:
  cv2.destroyAllWindows()
elif k == ord('s'):
  cv2.imwrite("test_copy.jpg",img)

#________________________________________________
#________________________________________________

#Video 7

import cv2
import datetime

img = cv2.VideoCapture(0)

print(img.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(img.get(cv2.CAP_PROP_FRAME_WIDTH))

while(img.isOpened()):
    ret, frame = img.read()
    if ret == True:

       font = cv2.FONT_HERSHEY_SIMPLEX
       text = 'Width: '+ str(img.get(3)) + ' Height:' + str(img.get(4))
       date = str(datetime.datetime.now())
       frame = cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
       frame = cv2.putText(frame, date, (10, 100), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
       cv2.imshow('frame', frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
         break
    else:
        break

img.release()
cv2.destroyAllWindows()

#________________________________________________
#________________________________________________

#Video 8 & 9

import numpy as np
import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,', ' ,y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', '+ str(y)
        cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        cv2.imshow('image', img)
            
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ', '+ str(green)+ ', '+ str(red)
        cv2.putText(img, strBGR, (x, y), font, .5, (0, 255, 255), 2)
        cv2.imshow('image', img)

img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread('lena.jpg')
cv2.imshow('image', img)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

#________________________________________________
#________________________________________________

#Video 10
