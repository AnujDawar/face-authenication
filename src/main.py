import classify
import sys
import cv2
import preprocess
import time

classifier = classify.Classify()
preprocessor = preprocess.PreProcessor()

camera = cv2.VideoCapture(0)
i = 0;
start = time.time()

prediction_out_dir = "prediction_output_images/"

print("start_time ", start)

while True:
    return_value, image = camera.read()
    # print("time new capture:", time.asctime( time.localtime(time.time()) ))
    # cv2.imwrite('opencv'+str(i)+'.png', image)			commented by anuj
    cv2.imwrite(prediction_out_dir + 'opencv.png', image)
    # print("time image written:", time.asctime( time.localtime(time.time()) ))
    # bb = (preprocessor.align('opencv'+str(i)+'.png'))		commented by anuj
    bb = (preprocessor.align(prediction_out_dir + 'opencv.png'))
    # print("time alignment done: ", time.asctime( time.localtime(time.time()) ))

    if bb.any() == False:
    	pass
    else:
    	cv2.rectangle(image, (bb[0],bb[1]), (bb[2],bb[3]), (0, 255, 0), 5)
    	# print("time rectange drawn", time.asctime( time.localtime(time.time()) ))
    	name = classifier.predict(prediction_out_dir + 'temp.png', threshold = 0.8)
    	print("prdecited name: ", name)
    	# print("time prediction done", time.asctime( time.localtime(time.time()) ))
    	font = cv2.FONT_HERSHEY_SIMPLEX 
    	cv2.putText(image, name, (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    	# print("time putText", time.asctime( time.localtime(time.time()) ))
    	cv2.imshow('frame',image)
    	# print("time image show", time.asctime( time.localtime(time.time()) ))
    	cv2.imwrite(prediction_out_dir + 'opencv.png', image)
    	# print("time new image written", time.asctime( time.localtime(time.time()) ))
	    # cv2.imshow('frame',image)			commented by anuj

    cv2.imshow('frame', image)
    # print("time image shown", time.asctime( time.localtime(time.time()) ))
    # print("\n\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i+=1

end_time = time.time()
print("end_time ", end_time)
print(i/(end_time-start))

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
# print(classifier.predict('temp.png'))		commented by anuj
