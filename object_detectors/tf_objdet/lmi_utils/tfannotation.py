# import the necessary packages
from tf_objdet.models.research.object_detection.utils.dataset_util import bytes_list_feature
from tf_objdet.models.research.object_detection.utils.dataset_util import float_list_feature
from tf_objdet.models.research.object_detection.utils.dataset_util import int64_list_feature
from tf_objdet.models.research.object_detection.utils.dataset_util import int64_feature
from tf_objdet.models.research.object_detection.utils.dataset_util import bytes_feature

class TFAnnotation:
	def __init__(self):
		# initialize additional variables, including the the image
		# itself, spatial dimensions, encoding, and filename
		self.image = None
		self.width = None
		self.height = None
		self.encoding = None
		self.filename = None
		
		# initialize the bounding box + label lists
		self.xMins = []
		self.xMaxs = []
		self.yMins = []
		self.yMaxs = []
		self.textLabels = []
		self.classes = []
		
		# initalize masks
		self.is_mask=False
		self.masks = []
		
		# initialize keypoints (required)
		self.is_keypoint=False
		self.keypoints_x = []
		self.keypoints_y = []
		self.keypoints_visibility = []
		self.keypoints_name = []
		self.num_keypoints = []
		
		# initialize keypoints (optional)
		# self.include_keypoint = keypoint_annotations_dict is not None
		# self.num_annotations_skipped = 0
		# self.num_keypoint_annotation_used = 0
		# self.num_keypoint_annotation_skipped = 0
	
		
	def build(self):
		# encode the attributes using their respective TensorFlow
		# encoding function
		w = int64_feature(self.width)
		h = int64_feature(self.height)
		filename = bytes_feature(self.filename.encode("utf8"))
		encoding = bytes_feature(self.encoding.encode("utf8"))
		image = bytes_feature(self.image)
		xMins = float_list_feature(self.xMins)
		xMaxs = float_list_feature(self.xMaxs)
		yMins = float_list_feature(self.yMins)
		yMaxs = float_list_feature(self.yMaxs)		
		textLabels = bytes_list_feature(self.textLabels)
		classes = int64_list_feature(self.classes)
		# difficult = int64_list_feature(self.difficult)

		# construct the TensorFlow-compatible data dictionary
		data = {
			"image/height": h,
			"image/width": w,
			"image/filename": filename,
			'image/source_id': filename,
			"image/encoded": image,
			"image/format": encoding,
			"image/object/bbox/xmin": xMins,
			"image/object/bbox/xmax": xMaxs,
			"image/object/bbox/ymin": yMins,
			"image/object/bbox/ymax": yMaxs,
			"image/object/class/text": textLabels,
			"image/object/class/label": classes,
			# "image/object/difficult": difficult,
		}

		if self.is_mask:
			masks = bytes_list_feature(self.masks)
			data["image/object/mask"]=masks
		
		if self.is_keypoint:
			keypoints_x=float_list_feature(self.keypoints_x)
			keypoints_y=float_list_feature(self.keypoints_y)
			keypoints_visibility=int64_list_feature(self.keypoints_visibility)
			keypoints_name = bytes_list_feature(self.keypoints_name)
			keypoints_num=int64_feature(self.num_keypoints)
			data["image/object/keypoint/x"]=keypoints_x
			data["image/object/keypoint/y"]=keypoints_y
			data["image/object/keypoint/num"]=keypoints_num
			data["image/object/keypoint/visibility"]=keypoints_visibility
			data["image/object/keypoint/text"]=keypoints_name
			#TODO: add keyponts used/skipped? 


		# return the data dictionary
		return data