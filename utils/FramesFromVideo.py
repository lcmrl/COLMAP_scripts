import cv2

cap = cv2.VideoCapture(r"C:\Users\Luscias\Desktop\provvisorio\provvisorio4\calibration\cut_for_calibration.mp4")
out_dir = r"C:\Users\Luscias\Desktop\provvisorio\provvisorio4\calibration\imgs"
frames = [k for k in range(0,10000,1)]
print(frames)

### MAIN ###
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    ## Display the resulting frame
    #cv2.imshow('Fram', frame)
#
    ## Press Q on keyboard to  exit
    #if cv2.waitKey(100) & 0xFF == ord('q'):
    #  break

    if i in frames:
        print(i)
        cv2.imwrite(f"{out_dir}/parte1_{i}"+".jpg", frame)
    i += 1

cap.release()
cv2.destroyAllWindows()