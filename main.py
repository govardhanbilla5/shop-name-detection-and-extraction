from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2, datetime, csv
import argparse

def ocr_process(ocr, frame):
	ocr_result = ocr.ocr(frame, cls=True)
	if ocr_result[0] is not None:
		text_result = [elements[1][0] for elements in ocr_result[0]]
		text = " ".join(text_result)
		return text
		
def detection(frame, model, ocr):
	results = model.predict(frame, conf=0.30)
	shop_data = []
	for r in results:
		
		boxes = r.boxes
		for box in boxes:
			b = (box.xyxy[0]).tolist() # get box coordinates in (top, left, bottom, right) format
			c = box.cls
			d = (box.conf).tolist()
			d = "{:.2f}".format(d[0])
			ocr_frame = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
			text = ocr_process(ocr, ocr_frame)
			shop_data.append(text)
			cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
			cv2.putText(frame, str(model.names[int(c)])+", conf:"+d, (int(b[0]), int(b[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	return list(set(shop_data))

def video_process(source, model):
	#Load model
	model = YOLO(model)
	#paddleocr
	ocr = PaddleOCR(use_angle_cls=True, lang='en')
	
	#source
	cap = cv2.VideoCapture(source)
	
	#File
	save_filename = datetime.datetime.now()
	save_filename = save_filename.strftime("%Y-%m-%d_%H:%M:%S")
	f = open(save_filename+'.csv', 'w')
	column_names = ["Shop name", "start_time", "end_time"]
	writer = csv.writer(f)
	writer.writerow(column_names)
	
	data = {}
	
	while True:
		ret, frame = cap.read()
		if not ret:	break
		
		result_data = shop_name_frame = detection(frame, model, ocr)
		for i in result_data:
			timestamp = str((cap.get(cv2.CAP_PROP_POS_MSEC)/1000)/60)
			if i in data:
				data[i] = [i, data[i][1], timestamp]
			else:
				data[i] = [i, timestamp, timestamp]
		   	
	for i in data:
		writer.writerow(data[i])
	f.close()
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--intput_source', type=str, default='/home/redon/POC/Proj/Dataset/videos/1.mp4', help='video source file')
	parser.add_argument('--weights', type=str, default='best.pt', help='model file')
	args = parser.parse_args()
	
	video_process(args.intput_source, args.weights)
