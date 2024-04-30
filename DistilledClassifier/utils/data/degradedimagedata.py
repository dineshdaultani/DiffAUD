import numpy as np

def get_type_range(degtype):
	"""
	Get degradation type and range
	Args:
		degtype (string) : jpeg, noise, blur, saltpepper
	"""
	if degtype == "jpeg":
		deg_range = [1, 101]
	elif degtype == "noise":
		deg_range = [0, 50]
	elif degtype == "blur":
		deg_range = [0, 50]
	elif degtype == 'saltpepper':
		deg_range = [0, 25]
	else:
		raise NotImplementedError
	return deg_range

def get_minmax_normalizedlevel(deg_type):
	"""
	Min and Max of normalized degradation levels
	Args:
		deg_type (string) : degradation type
	Returns:
		normalized degradation level (tuple)
	"""
	if deg_type == 'jpeg':
		ret_adj, max_l, min_l = 100.0, 101.0, 1.0
	elif deg_type == 'noise':
		ret_adj, max_l, min_l = 255.0, 50.0, 0.0
	elif deg_type == 'blur':
		ret_adj, max_l, min_l = 100.0, 50.0, 0.0
	elif deg_type == 'saltpepper':
		ret_adj, max_l, min_l = 100.0, 5.0, 0.0
	else:
		ret_adj, max_l, min_l = 1.0, 1.0, 0.0

	return min_l/ret_adj, max_l/ret_adj

def fix_seed_noise_sl(is_fixed):
	"""
	Fix the seed of Gaussian and Binomial distributions
	This is only used for evalution purpose.
	If you fix the seed, please do not forget to unfix the seed.
	Args:
		is_fixed (bool) : True if the seed is fixed
	"""
	if is_fixed:
		np.random.seed(seed=301)
	else:
		np.random.seed(seed=None)

def get_type_list(degtype):
	"""
	Get degradation type and range
	Args:
		degtype (string) : jpeg, noise, blur, saltpepper
	"""
	if degtype == "jpeg":
		deg_type = "jpeg"
		deg_list = [10, 30, 50, 70, 90]
	elif degtype == "noise":
		deg_type = "noise"
		deg_list = [np.sqrt(0.05)*255, np.sqrt(0.1)*255, 
					np.sqrt(0.15)*255, np.sqrt(0.2)*255, np.sqrt(0.25)*255]
	elif degtype == "blur":
		deg_type = "blur"
		deg_list = [10., 20., 30., 40., 50.]
	else:
		deg_type = "saltpepper"
		deg_list = [5., 10., 15., 20., 25.]

	return deg_type, deg_list