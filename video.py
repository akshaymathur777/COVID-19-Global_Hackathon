import sys
"""You need to have the Tensorflow Object Detection API installed for this code to work
This file needs to be stored in the Object Detection folder of Object Detection API
"""
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import argparse
import math

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", help = "Path of the input images directory")
	parser.add_argument("--frozen_graph", help = "Path of the frozen graph model")
	parser.add_argument("--label_map", help = "Path of the label map file")
	parser.add_argument("--output_dir", help = "Path of the output directory")
	parser.add_argument("--num_output_classes", help="Defines the number of output classes", type=int)
	parser.add_argument("--n_jobs", help="Number of GPU jobs in parallel", type=int)
	parser.add_argument("--delay", help="Delay for queue in seconds", type=int, default=0)
	args = parser.parse_args()

	threshold = 100
	
	sys.path.append("..")
	cap = cv2.VideoCapture("input.avi")
	fps = 30
	capSize = (640, 480) 
	#out = cv2.VideoWriter('output_overpass.avi', -1, fps, capSize,True)
	VIDEO_TYPE = {
	'avi': cv2.VideoWriter_fourcc(*'XVID'),
	'mp4': cv2.VideoWriter_fourcc(*'XVID'),
	}
	filename = 'filename.avi'
	def get_video_type(filename):
		filename, ext = os.path.splitext(filename)
		if ext in VIDEO_TYPE:
			return  VIDEO_TYPE[ext]
		return VIDEO_TYPE['avi']
	out = cv2.VideoWriter('input_output.avi', get_video_type(filename), fps, (int(cap.get(3)),int(cap.get(4))))
	PATH_TO_CKPT = args.frozen_graph
	PATH_TO_LABELS = args.label_map
	NUM_CLASSES = args.num_output_classes
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	num_violations = []

	def load_image_into_numpy_array(image):
		(im_width, im_height) = image.size
		return res

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			while cap.isOpened():
				box_points = []
				box_mid = []
				social_distance_violations = []
				ret, image_np = cap.read()
				if ret == True:
					image_np_expanded = np.expand_dims(image_np, axis=0)
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					scores = detection_graph.get_tensor_by_name('detection_scores:0')
					classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
					(boxes, scores, classes, num_detections) = sess.run(
						[boxes, scores, classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})
					print("Num boxes", num_detections[0])
					#print(type(num_detections[0]))
					#print(boxes)
					vis_util.visualize_boxes_and_labels_on_image_array(
						image_np,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						use_normalized_coordinates=True,
						line_thickness=8)
					for i in range (int(num_detections[0])):
						height = len(image_np)
						width = len(image_np[0])
						row = boxes[0][i]
						#print(row)
						#print(type(row))
						#print(len(row))
						ymin = int((row[0]*height))
						xmin = int((row[1]*width))
						ymax = int((row[2]*height))
						xmax = int((row[3]*width))
						x_val = (xmin,xmax)
						y_val = (ymin,ymax)
						one_set = (xmin, xmax, ymin, ymax)
						box_points.append(one_set)
						x_mid = int((xmax + xmin)/2)
						y_mid = int((ymax + ymin)/2)
						box_mid.append((x_mid,y_mid))
					print(box_mid)
					print(box_points)


					for i in range(len(box_mid)):
						for j in range(1, len(box_mid)):
							d = (box_mid[i][0]-box_mid[j][0])**2 + (box_mid[i][1]-box_mid[j][1])**2
							dis = float(math.sqrt(d))

							pixels_per_metric = (box_points[i][1] - box_points[i][0]) / 0.45
							#print("ppm", pixels_per_metric)
							#threshold = pixels_per_metric
							if dis < threshold and dis > 0:
								print("distance", dis)
								social_distance_violations.append(box_points[i])
								social_distance_violations.append(box_points[j])
					social_distance_violations = set(social_distance_violations)
					print("social_distance_violations", len(social_distance_violations))
					print(social_distance_violations)

					social_distance_violations = list(social_distance_violations)
					num_violations.append(len(social_distance_violations))

					for i in range(len(social_distance_violations)):
						start_point = (social_distance_violations[i][0], social_distance_violations[i][2])
						end_point = (social_distance_violations[i][1], social_distance_violations[i][3]) 
						color = (0, 0, 255)
						thickness = 5
						image = cv2.rectangle(image_np, start_point, end_point, color, thickness)
					

					out.write(image_np)
					cv2.imshow('Output',image_np)

					if cv2.waitKey(1) & 0xFF == ord('q'):
						cv2.destroyAllWindows()
						break
				else:
					break
cap.release()
out.release()
cv2.destroyAllWindows()
