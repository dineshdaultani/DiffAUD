import torch
import numpy as np
import random
from PIL import Image, ImageStat, ImageChops
import imagedegrade.im as degrade # https://github.com/mastnk/imagedegrade
import imagedegrade.np as np_degrade # https://github.com/mastnk/imagedegrade

random.seed(0)

def jpegcompresswithclean(img, jpeg_quality):
	"""
	Apply JPEG distortion to clean images
	If JPEG quality factor is in [1, 100], JPEG distortion is applied.
	If it is not in [1, 100], clean image will be returned.
	Args:
		img (PIL image) : clean image
		jpeg_quality (int) : JPEG quality factor
	Returns:
		img (PIL image) : JPEG image or clean image
	"""
	if (jpeg_quality >= 1) and (jpeg_quality <= 100):
		ret_img = degrade.jpeg(img, jpeg_quality)
	else:
		ret_img = img

	return ret_img

def degradation_function(deg_type):
	"""
	Get the pointer of a degradation function from imagedegrade.im
	Args:
		deg_type (string) : degradtion type
	Returns:
		ret_func (pinter) : the poiter of your selected degradation function
		ret_adj (folat) : the adjsutment of degradation level
	"""
	if deg_type == 'jpeg':
		ret_func = jpegcompresswithclean
		ret_adj = 1.0
	elif deg_type == 'noise':
		ret_func = np_degrade.noise
		ret_adj = 1.0
	elif deg_type == 'blur':
		ret_func = np_degrade.blur
		ret_adj = 10.0
	elif deg_type == 'saltpepper':
		ret_func = np_degrade.saltpepper
		ret_adj = 100.0
	else:
		msg = 'This degradation is not supported.'
		raise LookupError(msg)

	return ret_func, ret_adj

def normalize_level(deg_type, level):
	"""
	Normaliza degradation levels
	Args:
		deg_type (string) : degradation type
		level (int or float) : degradation level
	Returns:
		normalized degradation level (float)
	"""
	if deg_type == 'jpeg':
		ret_adj = 100.0
	elif deg_type == 'noise':
		ret_adj = 255.0
	elif deg_type == 'blur':
		ret_adj = 10.0
	elif deg_type == 'saltpepper':
		ret_adj = 1.0
	else:
		ret_adj = 1.0
	ret = np.array([float(level)/ret_adj])
	ret.astype(np.float32)
	return ret

def calc_deglev_rescaledimg(deg_type, img_org, resized_img_org, inp_resized_img_org, inp_resized_img_deg, deg_param):
	"""
	True degration level for rescaled degraded images
	Args:
		deg_type (string) : degradation type
		img_org (PIL images) : clean image
		resized_img_org (PIL images) : resized clean image
		inp_resized_img_org (PIL images) : resized clean image input into CNN
		inp_resized_img_deg (PIL images) : resized degraded image input into CNN
		deg_param (float) : degradation parameter used by degradation operator
	Returns:
		true degradation level (JPEG and Gaussian, S&P noise: RMSE, Gaussina blur: rescaled std)
	"""
	if deg_type == 'blur':
		true_deglev = deg_param / float(img_org.width) * float(resized_img_org.width) # Rescaling the standard deviation
	else:
		img_diff = ImageChops.difference(inp_resized_img_org, inp_resized_img_deg)
		stat_diff = ImageStat.Stat(img_diff)
		mse = sum(stat_diff.sum2) / float(len(stat_diff.count)) / float(stat_diff.count[0])
		true_deglev = np.sqrt(mse) # RMSE based on the maximum intensity 255.0
		if deg_type == 'jpeg':	# This operation is necessary for the normalize_level function.
			true_deglev *= (100.0 / 255.0)
		elif deg_type == 'saltpepper':
			true_deglev /= 255.0

	return true_deglev

class DegApplyWithLevel(torch.nn.Module):
	"""
	Data augmentation of degradations
	This transform returns not only a degraded image but also a degradation level.
	"""
	def __init__(self, deg_type, deg_range, deg_list):
		"""
		deg_range or deg_list are not input at the same time.
		Args:
			deg_type (string) : degradtion type
			deg_range (int, int) : range of degradation levels
			deg_list (list) : list of degradation levels
		"""
		super().__init__()
		self.deg_type = deg_type
		if deg_range is None and deg_list is None:
			msg = 'Both deg_range and deg_list do not have values.'
			raise TypeError(msg)
		elif (deg_range is not None) and (deg_list is not None):
			msg = 'deg_range or deg_list have values.'
			raise TypeError(msg)
		else:
			self.deg_range = deg_range
			self.deg_list = deg_list

		self.deg_func, self.deg_adj = degradation_function(deg_type)

	def forward(self, img):
		"""
		Get a degraded image and a degradation level
		Args:
			img : clean image (PIL image)
		Returns:
			degraded image (PIL image) : degraded image
			deg_lev (float) : degradation level
		"""
		if self.deg_range is not None:
			deg_lev = random.randint(self.deg_range[0], self.deg_range[1])
			if self.deg_adj > 1.0:
				deg_lev = deg_lev / self.deg_adj
		else:
			deg_lev = random.choice(self.deg_list)
			if self.deg_adj > 1.0:
				deg_lev = deg_lev / self.deg_adj
  
		# If deg_type is other than jpeg than clip the degraded images between 0 and 255. 
		if self.deg_type != 'jpeg':
			img = np.array(img)
		deg_img = self.deg_func(img, deg_lev)
		# Clip and convert to PIL image
		if self.deg_type != 'jpeg':
			deg_img = np.uint8(deg_img.clip(0, 255))
			deg_img = Image.fromarray(deg_img)
		return deg_img, deg_lev

	def __repr__(self):
		return self.__class__.__name__ + '(deg_type={}, deg_range=({},{}))'.format(self.deg_type, self.deg_range[0], self.deg_range[1])

class DegradationApply(DegApplyWithLevel):
	"""
	Data augmentation of degradations
	"""
	def forward(self, img):
		"""
		Get a degraded image
		Args:
			img : clean image (PIL image)
		Returns:
			degraded image (PIL image) : degraded image
		"""
		deg_img, deg_lev = super().forward(img)
		return deg_img

def apply_sequence_degs(clean_img, deg_seq = 'weak', dataset = 'imagenet'):
    """
    This function could be utilized to prepare the sequence of degradation images
    with the below configurations:
        Weak degradation: Blur (P*: 2) -> Noise (SD*: 10) -> JPEG (QF*: 80) 
        Medium degradation: Blur (P*: 4) -> Noise (SD*: 20) -> JPEG (QF*: 60) 
        Strong degradation: Blur (P*: 8) -> Noise (SD*: 30) -> JPEG (QF*: 40) 
        P*: Standard deviation of the Gaussian kernel wrt percentage of image size
        SD*: Standard deviation
        QF*: Quality factor
       
    Args: 
        clean_img: PIL image
        deg_seq: Either of the following values weak, medium, and strong.
        dataset: Support Cifar-10 and Imagenet datasets. 
    Returns:
        deg_img: Returns a PIL image with the applied degradations.
    """
    # Resize and crop in case of imagenet dataset
    if dataset == 'imagenet':
        image_size = 256
        resize_crop_t = [CenterCrop(image_size)]
    elif dataset == 'cifar_10':
        image_size = 32
        resize_crop_t = []
    else:
        raise NotImplementedError(f'{dataset} dataset is not supported!')
    
    if deg_seq == 'weak':
        deg_transform = transforms.Compose([
                                            *resize_crop_t,
                                            DegradationApply('blur', deg_range=[round(image_size*0.02), round(image_size*0.02)], deg_list=None),
                                            DegradationApply('noise', deg_range=[10, 10], deg_list=None),
                                            DegradationApply('jpeg', deg_range=[80, 80], deg_list=None),
                                            ])
    elif deg_seq == 'medium':
        deg_transform = transforms.Compose([
                                            *resize_crop_t,
                                            DegradationApply('blur', deg_range=[round(image_size*0.04), round(image_size*0.04)], deg_list=None),
                                            DegradationApply('noise', deg_range=[20, 20], deg_list=None),
                                            DegradationApply('jpeg', deg_range=[60, 60], deg_list=None),
                                            ])
    elif deg_seq == 'strong':
        deg_transform = transforms.Compose([
                                            *resize_crop_t,
                                            DegradationApply('blur', deg_range=[round(image_size*0.08), round(image_size*0.08)], deg_list=None),
                                            DegradationApply('noise', deg_range=[30, 30], deg_list=None),
                                            DegradationApply('jpeg', deg_range=[40, 40], deg_list=None),
                                            ])
    else:
        return NotImplementedError(f'{deg_seq} sequence of degradation is not supported!')
    
    return deg_transform(clean_img)