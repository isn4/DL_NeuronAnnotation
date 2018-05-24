# Authors: Ishtar Nyawira, Kristi Bushman

import os
import numpy as np
from scipy import misc
from sklearn.metrics import confusion_matrix

def confusion_mat(test_label_dir, predicted_img_dir):
	overall_tn = 0
	overall_tp = 0
	overall_fn = 0
	overall_fp = 0

	img_rows = 352 # not 360
	img_cols = 480
	number_test_imgs = 50
	total_pixels = float(img_rows * img_cols * number_test_imgs)

	for file in os.listdir(test_label_dir):
	    y_true = 1 - np.asarray(misc.imread(test_label_dir + file))[:352,:]
	    y_pred = np.asarray(misc.imread(predicted_img_dir + file[:-4] + '_pred.png')).astype(bool).astype(int)
	    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
	    overall_fp += fp
	    overall_fn += fn
	    overall_tp += tp
	    overall_tn += tn

	overall_tp /= total_pixels
	overall_fp /= total_pixels
	overall_fn /= total_pixels
	overall_tn /= total_pixels

	return overall_tp, overall_fp, overall_fn, overall_tn


def kap(TP, FP, FN, TN):
	fa = TP + TN
	N = TP + FP + FN + TN
	fc = ( (TN+FN)*(TN+FP) + (FP+TP)*(FN+TP) ) / N


	kap_score = (fa - fc) / (N - fc)
	return kap_score


def auc(TP, FP, FN, TN):
	fpr = FP / (FP+TN)	# order of mag. lower for unet (flood & rg)
	fnr = FN / (FN+TP) 	# order of mag. higher for unet (rg & man)
	p3 = .5 * (fpr+fnr)

	print("\tfnr = %f" %fnr)
	print("\tfpr = %f" %fpr)
	auc_score = 1 - p3
	return auc_score


def main():
	print('-------- UNET --------')
	# FLOODFILL
	# don't forget final '/'!
	test_label_dir = "/path/to/your/floodfill/test_labels/"
	predicted_img_dir = "/path/to/your/floodfill/predicted_labels/"
	flood_TP, flood_FP, flood_FN, flood_TN = confusion_mat(test_label_dir, predicted_img_dir)
	f_kap = kap(flood_TP, flood_FP, flood_FN, flood_TN)
	f_auc = auc(flood_TP, flood_FP, flood_FN, flood_TN)
	print('flood kap = %f' %f_kap)
	print('flood auc = %f' %f_auc)
	print

	# REGION GROWING
	# don't forget final '/'!
	test_label_dir = "/path/to/your/region_growing/test_labels/"
	predicted_img_dir = "/path/to/your/region_growing/predicted_labels/"
	rg_TP, rg_FP, rg_FN, rg_TN = confusion_mat(test_label_dir, predicted_img_dir)
	rg_kap = kap(rg_TP, rg_FP, rg_FN, rg_TN)
	rg_auc = auc(rg_TP, rg_FP, rg_FN, rg_TN)
	print('rg kap = %f' %rg_kap)
	print('rg auc = %f' %rg_auc)
	print

	# MANUAL
	# don't forget final '/'!
	test_label_dir = "/path/to/your/manual/test_labels/"
	predicted_img_dir = "/path/to/your/manual/predicted_labels/"
	man_TP, man_FP, man_FN, man_TN = confusion_mat(test_label_dir, predicted_img_dir)
	man_kap = kap(man_TP, man_FP, man_FN, man_TN)
	man_auc = auc(man_TP, man_FP, man_FN, man_TN)
	print('man kap = %f' %man_kap)
	print('man auc = %f' %man_auc)
	print

	print('-------- SEGNET --------')
	# FLOODFILL
	# don't forget final '/'!
	test_label_dir = "/path/to/your/floodfill/test_labels/"
	predicted_img_dir = "/path/to/your/floodfill/predicted_labels/"
	flood_TP, flood_FP, flood_FN, flood_TN = confusion_mat(test_label_dir, predicted_img_dir)
	f_kap = kap(flood_TP, flood_FP, flood_FN, flood_TN)
	f_auc = auc(flood_TP, flood_FP, flood_FN, flood_TN)
	print('flood kap = %f' %f_kap)
	print('flood auc = %f' %f_auc)
	print

	# REGION GROWING
	# don't forget final '/'!
	test_label_dir = "/path/to/your/region_growing/test_labels/"
	predicted_img_dir = "/path/to/your/region_growing/predicted_labels/"
	rg_TP, rg_FP, rg_FN, rg_TN = confusion_mat(test_label_dir, predicted_img_dir)
	rg_kap = kap(rg_TP, rg_FP, rg_FN, rg_TN)
	rg_auc = auc(rg_TP, rg_FP, rg_FN, rg_TN)
	print('rg kap = %f' %rg_kap)
	print('rg auc = %f' %rg_auc)
	print

	# MANUAL
	# don't forget final '/'!
	test_label_dir = "/path/to/your/manual/test_labels/"
	predicted_img_dir = "/path/to/your/manual/predicted_labels/"
	flood_TP, flood_FP, flood_FN, flood_TN = confusion_mat(test_label_dir, predicted_img_dir)
	man_kap = kap(man_TP, man_FP, man_FN, man_TN)
	man_auc = auc(man_TP, man_FP, man_FN, man_TN)
	print('man kap = %f' %man_kap)
	print('man auc = %f' %man_auc)




main()