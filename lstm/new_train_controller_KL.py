#new_train_controller_KL.py



import argparse
import datetime
import logging
import os
import sys
import warnings
import random
from math import log
from typing import List, Dict,Tuple
import glob
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append('./')

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression


from utils.datacollection.logger import *  
from lstm.new_dataset_KL import DenoiseDataModule
from lstm.new_controller_KL import GAFS
from lstm.utils_meter import (
    AvgrageMeter,
    pairwise_accuracy,
    hamming_distance,
    count_parameters_in_MB,
)
from new_feature_env_KL import *  
from new_Record_KL import SelectionRecord, TransformationRecord



TASK_DICT = {'airfoil': 'reg', 'airfoil_1': 'reg', 'airfoil_2': 'reg', 'airfoil_3': 'reg', 'airfoil_4': 'reg', 'airfoil_5': 'reg', 
             'amazon_employee': 'cls', 
            'amazon_employee_1': 'cls', 'amazon_employee_2': 'cls', 'amazon_employee_3': 'cls', 'amazon_employee_4': 'cls', 
            'amazon_employee_5': 'cls', 
            'amazon_employee_6': 'cls', 'amazon_employee_7': 'cls', 'amazon_employee_8': 'cls', 'amazon_employee_9': 'cls', 
            'amazon_employee_10': 'cls', 
            'amazon_employee_11': 'cls', 'amazon_employee_12': 'cls', 'amazon_employee_13': 'cls', 'amazon_employee_14': 'cls', 
            'amazon_employee_15': 'cls', 'amazon_employee_16': 'cls', 'amazon_employee_17': 'cls', 'amazon_employee_18': 'cls', 
            'amazon_employee_19': 'cls',
            'amazon_employee_20': 'cls', 'amazon_employee_21': 'cls', 'amazon_employee_22': 'cls', 'amazon_employee_23': 'cls', 
            'amazon_employee_24': 'cls', 'amazon_employee_25': 'cls', 'amazon_employee_26': 'cls', 'amazon_employee_27': 'cls', 
            'amazon_employee_28': 'cls', 'amazon_employee_29': 'cls', 'amazon_employee_30': 'cls', 'amazon_employee_31': 'cls', 
            'amazon_employee_32': 'cls', 'amazon_employee_33': 'cls', 'amazon_employee_34': 'cls', 'amazon_employee_35': 'cls', 
            'amazon_employee_36': 'cls', 'amazon_employee_37': 'cls', 'amazon_employee_38': 'cls', 'amazon_employee_39': 'cls', 
            'amazon_employee_40': 'cls', 'amazon_employee_41': 'cls', 'amazon_employee_42': 'cls', 'amazon_employee_43': 'cls', 
            'amazon_employee_44': 'cls', 'amazon_employee_45': 'cls', 'amazon_employee_46': 'cls', 'amazon_employee_47': 'cls', 
            'amazon_employee_48': 'cls', 'amazon_employee_49': 'cls', 'amazon_employee_50': 'cls', 'amazon_employee_51': 'cls', 
            'amazon_employee_52': 'cls', 'amazon_employee_53': 'cls', 'amazon_employee_54': 'cls', 'amazon_employee_55': 'cls', 
            'amazon_employee_56': 'cls', 'amazon_employee_57': 'cls', 'amazon_employee_58': 'cls', 'amazon_employee_59': 'cls', 
            'amazon_employee_60': 'cls', 'amazon_employee_61': 'cls', 'amazon_employee_62': 'cls', 'amazon_employee_63': 'cls', 
            'amazon_employee_64': 'cls', 'amazon_employee_65': 'cls', 'amazon_employee_66': 'cls', 'amazon_employee_67': 'cls', 
            'amazon_employee_68': 'cls', 'amazon_employee_69': 'cls', 'amazon_employee_70': 'cls', 'amazon_employee_71': 'cls', 
            'amazon_employee_72': 'cls', 'amazon_employee_73': 'cls', 'amazon_employee_74': 'cls', 'amazon_employee_75': 'cls', 
            'amazon_employee_76': 'cls', 'amazon_employee_77': 'cls', 'amazon_employee_78': 'cls', 'amazon_employee_79': 'cls', 
            'amazon_employee_80': 'cls', 'amazon_employee_81': 'cls', 'amazon_employee_82': 'cls', 'amazon_employee_83': 'cls', 
            'amazon_employee_84': 'cls', 'amazon_employee_85': 'cls', 'amazon_employee_86': 'cls', 'amazon_employee_87': 'cls', 
            'amazon_employee_88': 'cls', 'amazon_employee_89': 'cls', 'amazon_employee_90': 'cls', 'amazon_employee_91': 'cls', 
            'amazon_employee_92': 'cls', 'amazon_employee_93': 'cls', 'amazon_employee_94': 'cls', 'amazon_employee_95': 'cls', 
            'amazon_employee_96': 'cls', 'amazon_employee_97': 'cls', 'amazon_employee_98': 'cls', 'amazon_employee_99': 'cls', 
            'amazon_employee_100': 'cls', 'amazon_employee_101': 'cls',
             'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 
			 'german_credit': 'cls', 
			 'german_credit_a1': 'cls', 
             'german_credit_a1_1': 'cls','german_credit_a1_2': 'cls','german_credit_a1_3': 'cls','german_credit_a1_4': 'cls','german_credit_a1_5': 'cls',
             'higgs': 'cls','higgs_1': 'cls', 'higgs_2': 'cls', 'higgs_3': 'cls', 'higgs_4': 'cls', 'higgs_5': 'cls', 'higgs_6': 'cls',
             'higgs_7': 'cls', 'higgs_8': 'cls', 'higgs_9': 'cls', 'higgs_10': 'cls', 'higgs_11': 'cls', 'higgs_12': 'cls', 
             'higgs_13': 'cls', 'higgs_14': 'cls', 'higgs_15': 'cls', 'higgs_16': 'cls', 'higgs_17': 'cls', 'higgs_18': 'cls', 
             'higgs_19': 'cls', 'higgs_20': 'cls',
             'housing_boston': 'reg', 'housing_boston_1': 'reg','housing_boston_2': 'reg','housing_boston_3': 'reg','housing_boston_4': 'reg','housing_boston_5': 'reg',
             'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 
             'openml_620_1': 'reg','openml_620_2': 'reg','openml_620_3': 'reg','openml_620_4': 'reg',
             'pima_indian': 'cls','pima_indian_1': 'cls','pima_indian_2': 'cls','pima_indian_3': 'cls','pima_indian_4': 'cls','pima_indian_5': 'cls',
             'spam_base': 'cls',
			 'spam_base_1': 'cls','spam_base_2': 'cls','spam_base_3': 'cls','spam_base_4': 'cls','spam_base_5': 'cls',
             'spam_base_6': 'cls','spam_base_7': 'cls','spam_base_8': 'cls','spam_base_9': 'cls','spam_base_10': 'cls', 
             'spectf': 'cls', 'spectf_1': 'cls', 'spectf_2': 'cls', 'spectf_3': 'cls', 
             'spectf_a1': 'cls','spectf_a1_1': 'cls','spectf_a1_2': 'cls',
			 'spectf_a0p5': 'cls','spectf_a0p5_1': 'cls','spectf_a0p5_2': 'cls', 'spectf_a0p5_3': 'cls',
			 'svmguide3': 'cls','svmguide3_1': 'cls','svmguide3_2': 'cls','svmguide3_3': 'cls','svmguide3_4': 'cls','svmguide3_5': 'cls',
             'svmguide3_a1': 'cls',
             'svmguide3_a1_1': 'cls','svmguide3_a1_2': 'cls','svmguide3_a1_3': 'cls','svmguide3_a1_4': 'cls',
			 'svmguide3_a0p5': 'cls',
			 'svmguide3_a0p5_1': 'cls','svmguide3_a0p5_2': 'cls','svmguide3_a0p5_3': 'cls','svmguide3_a0p5_4': 'cls',
			 'uci_credit_card': 'cls', 
	         'credit_default_a1': 'cls',
             'credit_default_a1_1': 'cls','credit_default_a1_2': 'cls','credit_default_a1_3': 'cls','credit_default_a1_4': 'cls','credit_default_a1_5': 'cls', 
             'wine_red': 'cls', 'wine_red_1': 'cls', 'wine_red_2': 'cls', 'wine_red_3': 'cls', 'wine_red_4': 'cls', 'wine_red_5': 'cls', 
             'wine_red_a0p5': 'cls',
             'wine_red_a0p5_1': 'cls','wine_red_a0p5_2': 'cls','wine_red_a0p5_3': 'cls','wine_red_a0p5_4': 'cls',
             'wine_red_a1': 'cls',
             'wine_red_a1_1': 'cls','wine_red_a1_2': 'cls','wine_red_a1_3': 'cls','wine_red_a1_4': 'cls',
             'wine_red_a2': 'cls',
             'wine_red_a2_1': 'cls','wine_red_a2_2': 'cls','wine_red_a2_3': 'cls','wine_red_a2_4': 'cls',
			 'wine_white': 'cls','wine_white_1': 'cls','wine_white_2': 'cls','wine_white_3': 'cls','wine_white_4': 'cls','wine_white_5': 'cls',
             'wine_white_6': 'cls','wine_white_7': 'cls','wine_white_8': 'cls','wine_white_9': 'cls','wine_white_10': 'cls',
             'openml_586': 'reg', 'openml_586_1': 'reg', 'openml_586_2': 'reg', 'openml_586_3': 'reg', 'openml_586_4': 'reg',
             'openml_586_a0p5': 'reg','openml_586_a0p5_1': 'reg','openml_586_a0p5_2': 'reg','openml_586_a0p5_3': 'reg','openml_586_a0p5_4': 'reg',
			 'openml_586_a1': 'reg',
             'openml_586_a1_1': 'reg','openml_586_a1_2': 'reg','openml_586_a1_3': 'reg','openml_586_a1_4': 'reg',
             'openml_586_a2': 'reg',
             'openml_586_a2_1': 'reg','openml_586_a2_2': 'reg','openml_586_a2_3': 'reg','openml_586_a2_4': 'reg',
			 'openml_589': 'reg', 'openml_589_1': 'reg','openml_589_2': 'reg','openml_589_3': 'reg','openml_589_4': 'reg','openml_589_5': 'reg',
             'openml_607': 'reg', 'openml_607_1': 'reg','openml_607_2': 'reg','openml_607_3': 'reg','openml_607_4': 'reg','openml_607_5': 'reg',
             'openml_616': 'reg', 'openml_616_1': 'reg','openml_616_2': 'reg','openml_616_3': 'reg','openml_616_4': 'reg','openml_616_5': 'reg',
             'openml_618': 'reg', 'openml_618_1': 'reg','openml_618_2': 'reg', 'openml_618_3': 'reg', 'openml_618_4': 'reg', 'openml_618_5': 'reg', 
             'openml_637': 'reg','openml_637_1': 'reg','openml_637_2': 'reg','openml_637_3': 'reg','openml_637_4': 'reg','openml_637_5': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det', 'arrhythmia': 'cls',
             'nomao': 'cls', 'megawatt1': 'cls', 'activity': 'mcls', 'mice_protein': 'mcls', 'coil-20': 'mcls',
             'isolet': 'mcls', 'minist': 'mcls',
             'minist_fashion': 'mcls'
             }

baseline_name = [
	'kbest',
	'mrmr',
	'lasso',
	'rfe',
	# 'gfs',
	'lassonet',
	'sarlfs',
	'marlfs',
]



def gafs_train(train_queue, model: GAFS, optimizer):
	objs = AvgrageMeter()
	mse = AvgrageMeter()
	nll = AvgrageMeter()
	kl = AvgrageMeter()  
	model.train()
	for step, sample in enumerate(train_queue):
		encoder_input = sample['encoder_input']
		encoder_target = sample['encoder_target']
		decoder_input = sample['decoder_input']
		decoder_target = sample['decoder_target']
		
		encoder_input = encoder_input.cuda(model.gpu)
		encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
		decoder_input = decoder_input.cuda(model.gpu)
		decoder_target = decoder_target.cuda(model.gpu)
		
		optimizer.zero_grad()
		predict_value, log_prob, arch, mu, logvar = model.forward(encoder_input, decoder_input)
		loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())  # mse loss
		loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))  # ce loss
		loss_kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) #KL loss

		loss = (1 - args.beta) * loss_1 + args.beta * loss_2 + args.kl_weight * loss_kld 
		#loss = (1 - args.beta) * loss_1 + (args.beta) * loss_2
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
		optimizer.step()
		
		n = encoder_input.size(0)
		objs.update(loss.data, n)
		mse.update(loss_1.data, n)
		nll.update(loss_2.data, n)
		kl.update(loss_kld.data, n)
	
	return objs.avg, mse.avg, nll.avg



def gafs_valid(queue, model: GAFS):
	pa = AvgrageMeter()
	hs = AvgrageMeter()
	mse = AvgrageMeter()
	with torch.no_grad():
		model.eval()
		for step, sample in enumerate(queue):
			encoder_input = sample['encoder_input']
			encoder_target = sample['encoder_target']
			decoder_target = sample['decoder_target']
			
			encoder_input = encoder_input.cuda(model.gpu)
			encoder_target = encoder_target.cuda(model.gpu)
			decoder_target = decoder_target.cuda(model.gpu)
			
			predict_value, logits, arch, mu, logvar = model.forward(encoder_input)
			n = encoder_input.size(0)
			pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
			                                 predict_value.data.squeeze().tolist())
			hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
			mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
			pa.update(pairwise_acc, n)
			hs.update(hamming_dis, n)
	return mse.avg, pa.avg, hs.avg



def gafs_infer(queue, model, step, direction='+', beams=None):
	new_gen_list = []
	original_transformation = []
	model.eval()
	for i, sample in enumerate(queue):
		encoder_input = sample['encoder_input']
		encoder_input = encoder_input.cuda(model.gpu)
		model.zero_grad()
		new_gen = model.generate_new_feature(encoder_input, predict_lambda=step, direction=direction, beams=beams)
		new_gen_list.append(new_gen.data)
		original_transformation.append(encoder_input)
	return torch.cat(new_gen_list, 0), torch.cat(original_transformation, 0)


def select_top_k(choice: Tensor, labels: Tensor, k: int) -> Tuple[Tensor, Tensor]:
	values, indices = torch.topk(labels, k, dim=0)
	return choice[indices.squeeze()], labels[indices.squeeze()]



def report_performance(
	global_fe: GlobalFeatureEvaluator,
	fe_instances: Dict[str, TransformationFeatureEvaluator],
	best_sequence: TransformationRecord,
	best_record_avg: TransformationRecord,
	best_record_base: TransformationRecord,
	args: argparse.Namespace
	) -> pd.DataFrame:

	results = {
		'Sequence Type': [],
		'Global Performance': [],
		**{f'Local_{subset}': [] for subset in fe_instances.keys()}
	}
	info(f"[Check Init] fe_instances.keys(): {list(fe_instances.keys())}")

	def evaluate_sequence(seq_name, sequence, global_perf):

		results['Sequence Type'].append(seq_name)
		results['Global Performance'].append(global_perf)
		
		for subset_name, fe in fe_instances.items():
			try:
				local_data = sequence.op(fe.original.copy(), args.add_origin)
				info(f"local_data {subset_name} length: {len(local_data)}")
				if args.add_transformed:
					original_trans = sequence.original_record.op(fe.original.copy())
					local_data = pd.concat([local_data, original_trans], axis=1).T.drop_duplicates().T
				perf = fe.get_performance(local_data)
			except Exception as e:
				perf = np.nan
			results[f'Local_{subset_name}'].append(perf)


	def add_supplement_features(base_name, base_sequence, global_perf, supplement_type='importance'):

		for supplement_n in [1, 2, 3, 4, 5, 6]:
			if supplement_type == 'importance':
				row_name = f'{base_name} (with Top {supplement_n})'
			elif supplement_type == 'mi':
				row_name = f'{base_name} (with MI Top {supplement_n})'
			else:  # combined_mi
				row_name = f'{base_name} (with Combined MI Top {supplement_n})'
			
			results['Sequence Type'].append(row_name)
			results['Global Performance'].append(global_perf)
			
			for subset_name, fe in fe_instances.items():
				subset_records = fe.records.r_list
				if not subset_records:
					results[f'Local_{subset_name}'].append(np.nan)
					continue
					
				subset_best = max(subset_records, key=lambda x: x.performance)
				try:
	
					local_data_global = base_sequence.op(fe.original.copy(), args.add_origin)
					local_data = subset_best.op(fe.original.copy(), args.add_origin)
					if args.add_transformed:
						original_trans = subset_best.original_record.op(fe.original.copy())
						local_data = pd.concat([local_data, original_trans], axis=1).T.drop_duplicates().T
					
					
					perf, sorted_features = fe.get_performance(local_data, return_sorted_features=True)

					
					missing_features = [feat for feat in sorted_features if feat not in local_data_global.columns]

					

					if supplement_type == 'importance':
						features_to_add = missing_features[:supplement_n]
					elif supplement_type == 'mi':
						features_to_add = select_by_mi(local_data, missing_features, supplement_n)
					else:  # combined_mi
						features_to_add = select_by_combined_mi(local_data, local_data_global, 
																	missing_features, supplement_n)
					
					if features_to_add:
						additional_features = local_data[features_to_add]
						augmented_data = pd.concat([additional_features, local_data_global], axis=1)

					else:
						augmented_data = local_data_global
					
					augmented_perf = fe.get_performance(augmented_data)
				except Exception as e:
					info(f"error adding feature for  {subset_name} : {e}")
					augmented_perf = np.nan
				results[f'Local_{subset_name}'].append(augmented_perf)


	def select_by_mi(local_data, missing_features, supplement_n):
		selected_features = []
		remaining_features = missing_features.copy()
		
		for _ in range(supplement_n):
			if not remaining_features:
				break
			mi_scores = {}
			for feat in remaining_features:
				if not selected_features:
					mi_score = 0
				else:
					mi_score = 0
					for sf in selected_features:
						mi = mutual_info_regression(local_data[[sf]], local_data[feat], random_state=0)[0]
						mi_score += mi
				mi_scores[feat] = mi_score
			
			if selected_features:
				sorted_by_mi = sorted(remaining_features, key=lambda x: mi_scores[x])
				best_feat = sorted_by_mi[0]
			else:
				best_feat = remaining_features[0]
			
			selected_features.append(best_feat)
			remaining_features.remove(best_feat)
		
		return selected_features[:supplement_n]


	def select_by_combined_mi(local_data, local_data_global, missing_features, supplement_n):

		selected_features = []
		remaining_features = missing_features.copy()
		
		for _ in range(supplement_n):
			if not remaining_features:
				break
			mi_scores = {}
			for feat in remaining_features:
				if not selected_features:
					intra_score = 0
				else:
					intra_score = 0
					for sf in selected_features:
						intra_score += mutual_info_regression(local_data[[sf]], local_data[feat], random_state=0)[0]
				
				global_score = 0
				for gf in local_data_global.columns:
					global_score += mutual_info_regression(local_data_global[[gf]], local_data[feat], random_state=0)[0]
				global_score = global_score / len(local_data_global.columns)
				
				mi_scores[feat] = 0.8 * intra_score + 0.2 * global_score
			
			best_feat = min(remaining_features, key=lambda x: mi_scores[x])
			selected_features.append(best_feat)
			remaining_features.remove(best_feat)
		
		return selected_features[:supplement_n]

	evaluate_sequence('Global Optimal', best_sequence, best_sequence.performance)


	evaluate_sequence('Global Avg Optimal', best_record_avg, best_record_avg.performance)


	evaluate_sequence('Global Base Optimal (Federated Weight)', best_record_base, best_record_base.performance)


	merged_records_all = global_fe.records.r_list
	merged_subset_best = max(merged_records_all, key=lambda x: x.performance)
	global_data_merged = merged_subset_best.op(global_fe.original.copy(), args.add_origin)
	merged_avg_perf = global_fe.get_performance(global_data_merged)
	evaluate_sequence('Local Avg Optimal', merged_subset_best, merged_avg_perf)




	metric_collect = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
	for subset_name, fe in fe_instances.items():
		subset_records = fe.records.r_list
		if not subset_records:
			continue
			
		subset_best = max(subset_records, key=lambda x: x.performance)

		
		try:
			Dg = subset_best.op(global_fe.original.copy(), args.add_origin)
			p, r, f, a = global_fe.get_performance_allmetric(Dg)
			metric_collect['precision'].append(p)
			metric_collect['recall'].append(r)
			metric_collect['f1'].append(f)
			metric_collect['auc'].append(a)
		except Exception as e:
			info(f'error getting metrics for  {subset_name} best-seq : {e}')
			for k in metric_collect:
				metric_collect[k].append(np.nan)
		

		try:
			global_data = subset_best.op(global_fe.original.copy(), args.add_origin)
			info(f"global_data length: {len(global_data)}")
			if args.add_transformed:
				original_trans_global = subset_best.original_record.op(global_fe.original.copy())
				global_data = pd.concat([global_data, original_trans_global], axis=1).T.drop_duplicates().T
			global_perf_subset = global_fe.get_performance(global_data)
		except Exception as e:
			info(f"error eval {subset_name} : {e}")
			global_perf_subset = np.nan


		results['Sequence Type'].append(f'Local Optimal ({subset_name})')
		results['Global Performance'].append(global_perf_subset)
		
		for eval_subset_name, eval_fe in fe_instances.items():
			try:
				local_data = subset_best.op(eval_fe.original.copy(), args.add_origin)
				if args.add_transformed:
					original_trans_local = subset_best.original_record.op(eval_fe.original.copy())
					local_data = pd.concat([local_data, original_trans_local], axis=1).T.drop_duplicates().T
				perf = eval_fe.get_performance(local_data)
			except Exception as e:
				info(f"error eval {eval_subset_name} : {e}")
				perf = np.nan
			results[f'Local_{eval_subset_name}'].append(perf)


	add_supplement_features('Global Optimal', best_sequence, best_sequence.performance, 'importance')
	add_supplement_features('Global Optimal', best_sequence, best_sequence.performance, 'mi')
	add_supplement_features('Global Optimal', best_sequence, best_sequence.performance, 'combined_mi')


	add_supplement_features('Global Avg Optimal', best_record_avg, best_record_avg.performance, 'importance')
	add_supplement_features('Global Avg Optimal', best_record_avg, best_record_avg.performance, 'mi')
	add_supplement_features('Global Avg Optimal', best_record_avg, best_record_avg.performance, 'combined_mi')


	add_supplement_features('Global Base Optimal (base Weight)', best_record_base, best_record_base.performance, 'importance')
	add_supplement_features('Global Base Optimal (base Weight)', best_record_base, best_record_base.performance, 'mi')
	add_supplement_features('Global Base Optimal (base Weight)', best_record_base, best_record_base.performance, 'combined_mi')


	df = pd.DataFrame(results)
	df.set_index('Sequence Type', inplace=True)

	local_columns = df.columns[df.columns.str.startswith('Local_')]
	df['Mean_LP'] = df[local_columns].mean(axis=1)

	task_type = TASK_DICT[args.task_name]
	Dg = best_record_avg.op(global_fe.original.copy(), args.add_origin)
	precision, recall, f1, auc = global_fe.get_performance_allmetric(Dg)


	if task_type == 'cls':
		info(f"test_task_new on Global Avg Optimal → "
				f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
	elif task_type == 'reg':
		info(f"test_task_new on Global Avg Optimal → "
				f"1-MAE={precision:.4f}, 1-MSE={recall:.4f}, 1-RAE={f1:.4f}, 1-RMSE={auc:.4f}")
	info('--'*50)

	Dg_local_merged = merged_subset_best.op(global_fe.original.copy(), args.add_origin)
	precision, recall, f1, auc = global_fe.get_performance_allmetric(Dg_local_merged)

	if task_type == 'cls':
		info(f"test_task_new on merged_subset_best → "
				f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
	elif task_type == 'reg':
		info(f"test_task_new on merged_subset_best → "
				f"1-MAE={precision:.4f}, 1-MSE={recall:.4f}, 1-RAE={f1:.4f}, 1-RMSE={auc:.4f}")
	info('--'*50)


	task_type = TASK_DICT[args.task_name]
	Dg_base = best_record_base.op(global_fe.original.copy(), args.add_origin)
	precision_base, recall_base, f1_base, auc_base = global_fe.get_performance_allmetric(Dg_base)

	if task_type == 'cls':
		info(f"test_task_new on Global Base Optimal → "
				f"Precision={precision_base:.4f}, Recall={recall_base:.4f}, F1={f1_base:.4f}, AUC={auc_base:.4f}")
	elif task_type == 'reg':
		info(f"test_task_new on Global Base Optimal → "
				f"1-MAE={precision_base:.4f}, 1-MSE={recall_base:.4f}, 1-RAE={f1_base:.4f}, 1-RMSE={auc_base:.4f}")
	info('--'*50)

	if any(len(v) for v in metric_collect.values()):
		mean_p = np.nanmean(metric_collect['precision'])
		mean_r = np.nanmean(metric_collect['recall'])
		mean_f = np.nanmean(metric_collect['f1'])
		mean_a = np.nanmean(metric_collect['auc'])
		
		if task_type == 'cls':
			info(f"Mean of subset-best metrics → "
					f"Precision={mean_p:.4f}, Recall={mean_r:.4f}, "
					f"F1={mean_f:.4f}, AUC={mean_a:.4f}")
		else:
			info(f"Mean of subset-best metrics → "
					f"1-MAE={mean_p:.4f}, 1-MSE={mean_r:.4f}, "
					f"1-RAE={mean_f:.4f}, 1-RMSE={mean_a:.4f}")
		info('--' * 50)

	global_df = best_record_avg.op(global_fe.original.copy(), args.add_origin)
	perf, sorted_feats = global_fe.get_performance_score(global_df, return_sorted_features=True)

	for feat, score in sorted_feats[:10]:
		info(f"  {feat:30s}  {score:30.4f}")

	return df.round(4)



def main(args):
	if not torch.cuda.is_available():
		info('No GPU found!')
		sys.exit(1)
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	cudnn.enabled = True
	cudnn.benchmark = False
	cudnn.deterministic = True
	device = int(args.gpu)
	info(f"Args = {args}")
	
	task_type = TASK_DICT.get(args.task_name, 'cls')
	#eval_methods = 'XGB'
	

	dm = DenoiseDataModule(args) 
	fe = dm.global_fe
	fe_instances = dm.fe_instances



	model = GAFS(fe, args, dm.tokenizer)
	train_queue = dm.train_dataloader()
	valid_queue = dm.val_dataloader()
	infer_queue = dm.infer_dataloader()
	




	maybe_load_from = os.path.join(f'{base_path}', 'history', f'{dm.fe.task_name}', f'model_dmp{args.keyword}',
	                               f'{dm.fe.task_name}_{args.load_epoch}.encoder.pt')
	info(f'we load model from {maybe_load_from}:{os.path.exists(maybe_load_from)}')
	if args.load_epoch > 0 and os.path.exists(maybe_load_from):
		base_load_path = os.path.join(f'{base_path}', 'history')
		start_epoch = args.load_epoch
		model = model.from_pretrain(base_load_path, fe, args, dm.tokenizer, start_epoch, keyword=args.keyword)
		model = model.cuda(device)
		mse, pa, hs = gafs_valid(valid_queue, model)
		info("Evaluation on valid data")
		info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(start_epoch, mse, pa,
		                                                                                       hs))
	else:
		start_epoch = 0
		model = model.cuda(device)
	
	info(f"param size = {count_parameters_in_MB(model)}MB")
	info('Training Encoder-Predictor-Decoder')

	
	best_composite = float('inf')
	patience = 10        
	patience_counter = 0
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
	best_epoch = start_epoch
	best_composite = float('inf')

	for epoch in range(start_epoch + 1, args.epochs + 1):
		nao_loss, nao_mse, nao_ce = gafs_train(train_queue, model, optimizer)
		info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, nao_loss, nao_mse, nao_ce))
		if epoch % 5 == 0 or epoch == 1:
			model.save_to(f'{base_path}/history', epoch, keyword=args.keyword)
			info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, nao_loss, nao_mse, nao_ce))
		if epoch % 5 == 0 or epoch == 1:
			mse, pa, hs = gafs_valid(valid_queue, model)
			info("Evaluation on valid data")
			info('epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
			                                                                                       hs))
			composite = mse + (1 - pa)
			
			if composite < best_composite:
				best_composite = composite
				best_epoch = epoch 
				model.save_to(f'{base_path}/history', epoch, keyword=f'{args.keyword}_best') # Save best model separately
				patience_counter = 0   
			else:
				patience_counter += 1
				
			info("Current composite score: {:.6f}, best composite score: {:.6f}, patience: {}".format(
				composite, best_composite, patience_counter))
				
			if patience_counter >= patience:
				info("Early stopping triggered at epoch {:04d}".format(epoch))
				break
		




	new_selection = []
	new_choice = []
	predict_step_size = 0
	original_transformation = []
	total_num = 0
	valid_num = 0
	while len(new_selection) < args.new_gen:
		predict_step_size += 1
		info('Generate new architectures with step size {:d}'.format(predict_step_size))
		new_record, original_record = gafs_infer(infer_queue, model, direction='+', step=predict_step_size,
		                                         beams=args.beams)
		new_choice.append((new_record, original_record))
		for choice, original_choice in zip(new_record, original_record):
			record_ = TransformationRecord.from_tensor(choice, dm.tokenizer)
			original_record_ = TransformationRecord.from_tensor(original_choice, dm.tokenizer)
			if record_ not in new_selection:  # record_ not in fe.records.r_list and
				new_selection.append((record_, original_record_))
				valid_num += len(record_.ops)
				total_num += len(record_.input_ops)
				info(f'gen {record_.valid}: {len(record_.ops)}/{len(record_.input_ops)}')
				info(f'{len(new_selection)} new choice generated now', )
		if predict_step_size > args.max_step_size:
			break
	info(f'build {len(new_selection)} new choice with valid rate {(valid_num / total_num) * 100}% !!!')
	

	best_selection = None
	best_optimal = -10
	previous_optimal = dm.fe.best_grfg
	info(f'the best performance for this task is {previous_optimal}')
	count = 0
	for record, original_record_ in new_selection:
		if not record.valid:
			count += 1
			info(f'invalid percentage as : {count}/{len(new_selection)}')
			continue		
		test_data = record.op(fe.original.copy(), args.add_origin)
		original_trans = original_record_.op(fe.original.copy())


		cols_gen = list(test_data.columns[:-1])
		cols_ori = list(original_trans.columns[:-1])
		final_cols = []
		data = []

		for index, i in enumerate(cols_gen):
			if final_cols.__contains__(i):
				continue
			else:
				final_cols.append(i)
				data.append(test_data.iloc[:, index])
	
		if args.add_transformed:
			for index, i in enumerate(cols_ori):
				if final_cols.__contains__(i):
					continue
				else:
					final_cols.append(i)
					data.append(original_trans.iloc[:, index])
		data.append(test_data.iloc[:, -1])
		final_cols.append('label')
		final_ds = pd.concat(data, axis=1)
		final_ds.columns = final_cols
		try:
			result = fe.get_performance(final_ds)
			record.performance = result  
			if result > best_optimal:
				best_selection = final_ds
				best_optimal = result
				
				info(f'found best performance on {dm.fe.task_name} : {best_optimal}')
				info(f'the column is {final_ds.columns}')
		except:
			error('something wrong with this feature set, e.g., Nan or Inf')
	
	best_str = '{:.4f}'.format(best_optimal * 100)
	#best_selection.to_hdf(f'{base_path}/history/1000-xm-{dm.fe.task_name}-{best_str}-{args.keyword}.hdf', key='xm')
	best_selection.to_hdf(f'{base_path}/history/1000-xm-{dm.fe.task_name}-{best_str}-{args.keyword}-base.hdf', key='xm')
	info(f'build {len(new_selection)} new choice with valid rate {(valid_num / total_num) * 100}% !!!')
	info(f'the original performance is : {fe.get_performance(fe.original)}')
	info(f'found best performance on {dm.fe.task_name} : {best_optimal}')




	sample_sizes = {}
	for subset_name, local_fe in fe_instances.items():
		if local_fe.original is None or local_fe.original.shape[0] == 0:
			raise ValueError(f" {subset_name} empty")
		sample_sizes[subset_name] = local_fe.original.shape[0]

	total_samples = sum(sample_sizes.values())
	weights = {subset: size / total_samples for subset, size in sample_sizes.items()}

	candidate_scores = {}  
	for record, original_record_ in new_selection:
		if not record.valid:
			continue

		local_perf_avg = {}  
		for subset_name, local_fe in fe_instances.items():
			try:
				test_data_avg = record.op(local_fe.original.copy(), args.add_origin)
				perf = local_fe.get_performance(test_data_avg)
				local_perf_avg[subset_name] = perf
			except Exception as e:
				info(f"error eval {subset_name} : {e}")
				local_perf_avg[subset_name] = np.nan

		valid_perf = [local_perf_avg[s] for s in local_perf_avg if not np.isnan(local_perf_avg[s])]
		if len(valid_perf) == 0:
			weighted_perf = np.nan
		else:
			valid_weights = [weights[s] for s in local_perf_avg if not np.isnan(local_perf_avg[s])]
			weight_sum = sum(valid_weights)
			weighted_perf = sum(local_perf_avg[s] * weights[s] for s in local_perf_avg if not np.isnan(local_perf_avg[s])) / weight_sum

		candidate_scores[record] = weighted_perf

	if len(candidate_scores) == 0:
		raise ValueError("no valid candidates")
	best_record_avg = max(candidate_scores, key=candidate_scores.get)
	
	
	candidate_scores_base = {}  
	for record, original_record_ in new_selection:
		if not record.valid:
			continue
		local_perf_base = {}  
		for subset_name, local_fe in fe_instances.items():
			try:
				test_data_base = record.op(local_fe.original.copy(), args.add_origin)
				perf = local_fe.get_performance(test_data_base)
				local_perf_base[subset_name] = perf
			except Exception as e:
				info(f"error eval {subset_name} : {e}")
				local_perf_base[subset_name] = np.nan

		valid_perf = [local_perf_base[s] for s in local_perf_base if not np.isnan(local_perf_base[s])]
		if len(valid_perf) == 0:
			weighted_perf = np.nan
		else:
			valid_weights = [fe.base_weights[s] for s in local_perf_base if not np.isnan(local_perf_base[s])]
			weight_sum = sum(valid_weights)
			weighted_perf = sum(local_perf_base[s] * fe.base_weights[s] for s in local_perf_base if not np.isnan(local_perf_base[s])) / weight_sum

		candidate_scores_base[record] = weighted_perf


	if len(candidate_scores_base) == 0:
		raise ValueError("no valid candidates")
	best_record_base = max(candidate_scores_base, key=candidate_scores_base.get)




	fe_instances = dm.fe_instances
	perf_df_1 = None 
	if best_selection is not None:
		best_record = next((r for r, _ in new_selection if r.performance == best_optimal),None)
		if best_record:
			info(f'start to report FL performance')
			perf_df = report_performance(
				global_fe=fe,
				fe_instances=fe_instances,
				best_sequence=best_record,
				best_record_avg=best_record_avg,
				best_record_base=best_record_base,
				args=args
			)
			perf_df_1 = perf_df
			report_path = os.path.join(
				base_path, 'history', fe.task_name, 
				f'bias_performance_report_base_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
			perf_df_1.to_csv(report_path)
			info(f"Performance(base) report saved to {report_path}")

			info("\nPerformance(base) Summary:\n" + str(perf_df_1))

	

	fe_robust = dm.global_fe_robust
	fe_instances_robust = dm.fe_instances_robust
	


	if args.robust_load_epoch > 0:

		load_epoch = args.robust_load_epoch
		load_keyword = f'{args.keyword}-robust'
		start_epoch_robust = args.robust_load_epoch  
		info(f"[ROBUST] load robust model, epoch {load_epoch}")
	else:
		if args.robust_base_epoch > 0:
			load_epoch = args.robust_base_epoch
		else:
			load_epoch = max([int(re.search(r'_(\d+)\.encoder\.pt$', f).group(1)) 
							for f in glob.glob(os.path.join(f'{base_path}', 'history', 
											f'{dm.fe.task_name}', 
											f'model_dmp{args.keyword}_best', '*.encoder.pt'))
							if re.search(r'_(\d+)\.encoder\.pt$', f)] or [best_epoch])
		load_keyword = args.keyword
		start_epoch_robust = 0  
		info(f"[ROBUST] load base model epoch {load_epoch}")






	base_load_path = os.path.join(f'{base_path}', 'history')
	model_robust = GAFS(fe_robust, args, dm.tokenizer).cuda(device)


	for suffix in ['', '_best']:
		try:

			model_robust = model_robust.from_pretrain(
				base_load_path, fe_robust, args, dm.tokenizer,
				epoch=load_epoch, 
				keyword=f'{load_keyword}{suffix}'  
			)
			model_robust = model_robust.cuda(device)
			info(f"[ROBUST] loaded: {load_keyword}{suffix} epoch {load_epoch}")
			break  
		except:
			continue  
	else:

		raise FileNotFoundError(f"didn't fine model : epoch {load_epoch}")



	robust_epochs = args.robust_epochs if args.robust_epochs > 0 else args.epochs
	
	should_train = True  
	if args.robust_load_epoch > 0 and not args.robust_continue_train:
		should_train = False
		info("[ROBUST] load robust model,inference only")
	elif args.robust_load_epoch > 0 and args.robust_continue_train:
		info(f"[ROBUST] load robust model epoch {args.robust_load_epoch}, continue training")
	else:
		info("[ROBUST] train from base model")
	



	train_queue_robust = dm.train_dataloader_robust()
	valid_queue_robust = dm.val_dataloader_robust()
	infer_queue_robust = dm.infer_dataloader_robust()


	if should_train:
		best_composite_robust = float('inf')
		patience = 10        
		patience_counter = 0
		best_epoch_robust = 0

		optimizer_robust = torch.optim.Adam(model_robust.parameters(), lr=args.lr, weight_decay=args.l2_reg)
		for epoch in range( start_epoch_robust + 1, robust_epochs + 1):
			nao_loss, nao_mse, nao_ce = gafs_train(train_queue_robust, model_robust, optimizer_robust)
			info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, nao_loss, nao_mse, nao_ce))
			if epoch % 10 == 0 or epoch == 1:
				model_robust.save_to(f'{base_path}/history', epoch, keyword=f'{args.keyword}-robust')
				info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f}".format(epoch, nao_loss, nao_mse, nao_ce))
			if epoch % 10 == 0 or epoch == 1:
				mse, pa, hs = gafs_valid(valid_queue_robust, model_robust)
				info("[rb] Evaluation on valid data")
				info('[rb] epoch {:04d} mse {:.6f} pairwise accuracy {:.6f} hamming distance {:.6f}'.format(epoch, mse, pa,
																									hs))

				composite = mse + (1 - pa)
				
				if composite < best_composite_robust:
					best_composite_robust = composite
					best_epoch_robust = epoch
					model_robust.save_to(f'{base_path}/history', epoch, keyword=f'{args.keyword}-robust_best')
					patience_counter = 0   
				else:
					patience_counter += 1
					
				info("[rb] Current composite score: {:.6f}, best composite score: {:.6f}, patience: {}".format(
					composite, best_composite_robust, patience_counter))
					

				if patience_counter >= patience:
					info("[rb] Early stopping triggered at epoch {:04d}".format(epoch))
					break
	else:
		info("[ROBUST] no training, start inference")		
		
	new_selection = []
	new_choice = []
	predict_step_size = 0
	original_transformation = []
	total_num = 0
	valid_num = 0
	while len(new_selection) < args.new_gen:
		predict_step_size += 1
		info('Generate new architectures with step size {:d}'.format(predict_step_size))
		new_record, original_record = gafs_infer(infer_queue_robust, model_robust, direction='+', step=predict_step_size,
		                                         beams=args.beams)
		new_choice.append((new_record, original_record))
		for choice, original_choice in zip(new_record, original_record):
			record_ = TransformationRecord.from_tensor(choice, dm.tokenizer)
			original_record_ = TransformationRecord.from_tensor(original_choice, dm.tokenizer)
			if record_ not in new_selection:  
				new_selection.append((record_, original_record_))
				valid_num += len(record_.ops)
				total_num += len(record_.input_ops)
				info(f'gen {record_.valid}: {len(record_.ops)}/{len(record_.input_ops)}')
				info(f'{len(new_selection)} new choice generated now', )
		if predict_step_size > args.max_step_size:
			break
	info(f'[rb] build {len(new_selection)} new choice with valid rate {(valid_num / total_num) * 100}% !!!')

	best_selection = None
	best_optimal = -10
	previous_optimal = dm.fe.best_grfg
	info(f'the best performance for this task is {previous_optimal}')
	count = 0
	for record, original_record_ in new_selection:
		if not record.valid:
			count += 1
			info(f'invalid percentage as : {count}/{len(new_selection)}')
			continue		
		test_data = record.op(fe_robust.original.copy(), args.add_origin)
		original_trans = original_record_.op(fe_robust.original.copy())

		cols_gen = list(test_data.columns[:-1])
		cols_ori = list(original_trans.columns[:-1])
		final_cols = []
		data = []

		for index, i in enumerate(cols_gen):
			if final_cols.__contains__(i):
				continue
			else:
				final_cols.append(i)
				data.append(test_data.iloc[:, index])
	
		if args.add_transformed:
			for index, i in enumerate(cols_ori):
				if final_cols.__contains__(i):
					continue
				else:
					final_cols.append(i)
					data.append(original_trans.iloc[:, index])
		data.append(test_data.iloc[:, -1])
		final_cols.append('label')
		final_ds = pd.concat(data, axis=1)
		final_ds.columns = final_cols
		try:
			result = fe_robust.get_performance(final_ds)
			record.performance = result  
			if result > best_optimal:
				best_selection = final_ds
				best_optimal = result
				
				info(f'found best performance on {dm.fe.task_name} : {best_optimal}')
				info(f'the column is {final_ds.columns}')
		except:
			error('something wrong with this feature set, e.g., Nan or Inf')
	
	best_str_robust = '{:.4f}'.format(best_optimal * 100)
	best_selection.to_hdf(f'{base_path}/history/1000-xm-{fe_robust.task_name}-{best_str_robust}-{args.keyword}-robust.hdf', key='xm')
	info(f'build {len(new_selection)} new choice with valid rate {(valid_num / total_num) * 100}% !!!')
	info(f'the original performance is : {fe_robust.get_performance(fe_robust.original)}')
	info(f'found best performance on {fe_robust.task_name} : {best_optimal}')





	sample_sizes = {}
	for subset_name, local_fe in fe_instances_robust.items():
		if local_fe.original is None or local_fe.original.shape[0] == 0:
			raise ValueError(f" {subset_name} empty")
		sample_sizes[subset_name] = local_fe.original.shape[0]

	total_samples = sum(sample_sizes.values())
	weights = {subset: size / total_samples for subset, size in sample_sizes.items()}
	candidate_scores = {}  
	for record, original_record_ in new_selection:
		if not record.valid:
			continue
		local_perf_avg = {}  
		for subset_name, local_fe in fe_instances_robust.items():
			try:
				test_data_avg = record.op(local_fe.original.copy(), args.add_origin)
				perf = local_fe.get_performance(test_data_avg)
				local_perf_avg[subset_name] = perf
			except Exception as e:
				info(f"error eval {subset_name} : {e}")
				local_perf_avg[subset_name] = np.nan

		valid_perf = [local_perf_avg[s] for s in local_perf_avg if not np.isnan(local_perf_avg[s])]
		if len(valid_perf) == 0:
			weighted_perf = np.nan
		else:
			valid_weights = [weights[s] for s in local_perf_avg if not np.isnan(local_perf_avg[s])]
			weight_sum = sum(valid_weights)
			weighted_perf = sum(local_perf_avg[s] * weights[s] for s in local_perf_avg if not np.isnan(local_perf_avg[s])) / weight_sum

		candidate_scores[record] = weighted_perf


	if len(candidate_scores) == 0:
		raise ValueError("no valid candidates")
	best_record_avg = max(candidate_scores, key=candidate_scores.get)
		

	candidate_scores_base = {}  
	for record, original_record_ in new_selection:
		if not record.valid:
			continue
		local_perf_base = {}  
		for subset_name, local_fe in fe_instances.items():
			try:
				test_data_base = record.op(local_fe.original.copy(), args.add_origin)
				perf = local_fe.get_performance(test_data_base)
				local_perf_base[subset_name] = perf
			except Exception as e:
				info(f"error eval {subset_name} : {e}")
				local_perf_base[subset_name] = np.nan

		valid_perf = [local_perf_base[s] for s in local_perf_base if not np.isnan(local_perf_base[s])]
		if len(valid_perf) == 0:
			weighted_perf = np.nan
		else:
			valid_weights = [fe.base_weights[s] for s in local_perf_base if not np.isnan(local_perf_base[s])]
			weight_sum = sum(valid_weights)
			weighted_perf = sum(local_perf_base[s] * fe.base_weights[s] for s in local_perf_base if not np.isnan(local_perf_base[s])) / weight_sum

		candidate_scores_base[record] = weighted_perf

	if len(candidate_scores_base) == 0:
		raise ValueError("no valid candidates")
	best_record_base = max(candidate_scores_base, key=candidate_scores_base.get)



	fe_instances_robust = dm.fe_instances_robust
	perf_df_2 = None 

	if best_selection is not None:

		best_record = next((r for r, _ in new_selection if r.performance == best_optimal),None)
		if best_record:

			info(f'start to report FL performance')
			perf_df = report_performance(
				global_fe=fe_robust,
				fe_instances=fe_instances_robust,
				best_sequence=best_record,
				best_record_avg=best_record_avg,
				best_record_base=best_record_base,
				args=args
			)
			perf_df_2 = perf_df

			report_path = os.path.join(
				base_path, 'history', fe_robust.task_name, 
				f'bias_performance_report_robust_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
			perf_df_2.to_csv(report_path)
			info(f"Performance(rb) report saved to {report_path}")


			info("\nPerformance(rb) Summary:\n" + str(perf_df_2))
	
		
		
	combined_path = os.path.join(
		base_path, 'history', fe.task_name,
		f'bias_performance_report_COMBINED_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
	)
	if perf_df_1 is not None:
		perf_df_1['stage'] = 'base'
	if perf_df_2 is not None:
		perf_df_2['stage'] = 'robust'
	if perf_df_1 is not None or perf_df_2 is not None:
		combined = pd.concat([df for df in [perf_df_1, perf_df_2] if df is not None], axis=0)
		os.makedirs(os.path.dirname(combined_path), exist_ok=True)
		combined.to_csv(combined_path, index=False)
		info("\n=== Unified Performance Summary ===\n" + str(combined))



	'''check paramter size'''
	'''
	def print_model_size(model):
		total = sum(p.numel() for p in model.parameters())
		trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total parameters: {total:,}")
		print(f"Trainable parameters: {trainable:,}")

	#  model = GAFS(...) 
	print_model_size(model)
	'''


	'''
	#save embeddings for visualization
	embeds, perf = [], []

	model.eval()
	with torch.no_grad():
		for batch in infer_queue:                       # dm.infer_dataloader()
			ids  = batch['encoder_input'].to(device)    # [B, L]
			vec  = model.embed(ids)                     # [B, hid_dim]
			embeds.append(vec.cpu())

			if 'encoder_target' in batch:               
				p = batch['encoder_target'].squeeze(-1) # [B]
				perf.extend(p.cpu().tolist())
			else:
				
				perf.extend([float('nan')]*ids.size(0))

	embeds = torch.cat(embeds, 0).numpy()   # (N, hid_dim)
	perf   = np.array(perf)                 # (N,)
	print(f"embeds shape: {embeds.shape}, perf shape: {perf.shape}")
	print('--'*50)
	print(f'embeds: {embeds[:2]}')
	print(f'perf: {perf[:2]}')

	N, D = embeds.shape
	df = pd.DataFrame(embeds, columns=[f'dim{i}' for i in range(D)])
	df.insert(0, 'id', np.arange(N))
	df['perf'] = perf

	parts = []
	if args.strategy_appendix:
		parts.append(f"strat-{args.strategy_appendix}")
	if args.keyword:
		parts.append(f"kw-{args.keyword}")
	suffix = "_".join(parts) if parts else "default"

	out_path = os.path.join(
		base_path,
		"history",
		args.task_name,
		f"seed_embedding_perf_{suffix}.csv"
	)
	df.to_csv(out_path, index=False)
	print(f"CSV saved → {out_path}")

	'''







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task_name', type=str, choices=['airfoil', 'amazon_employee',
	                                                      'ap_omentum_ovary', 'german_credit','german_credit_a1',
	                                                      'higgs', 'housing_boston', 'ionosphere',
	                                                      'lymphography', 'messidor_features', 'openml_620',
	                                                      'pima_indian', 'spam_base', 'spectf','spectf_a1','spectf_a0p5','svmguide3','svmguide3_a1','svmguide3_a0p5',
	                                                      'uci_credit_card','credit_default_a1','credit_default_a0p5','wine_red','wine_red_a0p5','wine_red_a1','wine_red_a2','wine_white', 
														  'openml_586','openml_586_a0p5','openml_586_a1','openml_586_a2',
	                                                      'openml_589', 'openml_607', 'openml_616', 'openml_618',
	                                                      'openml_637'], default='lymphography')
	parser.add_argument('--mask_whole_op_p', type=float, default=0.0)
	parser.add_argument('--mask_op_p', type=float, default=0.0)
	parser.add_argument('--disorder_p', type=float, default=0.0)
	parser.add_argument('--num', type=int, default=12)
	
	parser.add_argument('--method_name', type=str, choices=['rnn'], default='rnn')
	
	parser.add_argument('--encoder_layers', type=int, default=1)
	parser.add_argument('--encoder_hidden_size', type=int, default=64)
	parser.add_argument('--encoder_emb_size', type=int, default=32)
	parser.add_argument('--mlp_layers', type=int, default=2)
	parser.add_argument('--mlp_hidden_size', type=int, default=200)
	parser.add_argument('--decoder_layers', type=int, default=1)
	parser.add_argument('--decoder_hidden_size', type=int, default=64)
	
	parser.add_argument('--encoder_dropout', type=float, default=0)
	parser.add_argument('--mlp_dropout', type=float, default=0)
	parser.add_argument('--decoder_dropout', type=float, default=0)
	
	parser.add_argument('--new_gen', type=int, default=200)
	
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--batch_size', type=int, default=2048)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--beta', type=float, default=0.95)
	parser.add_argument('--grad_bound', type=float, default=5.0)
	parser.add_argument('--l2_reg', type=float, default=0.0)
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--top_k', type=int, default=20)
	
	parser.add_argument('--load_epoch', type=int, default=0)
	parser.add_argument('--train_top_k', type=int, default=512)
	parser.add_argument('--epochs', type=int, default=2000)
	parser.add_argument('--eval', type=bool, default=False)
	parser.add_argument('--max_step_size', type=int, default=5)
	
	parser.add_argument('--beams', type=int, default=5)
	parser.add_argument('--add_origin', type=bool, default=True)
	parser.add_argument('--add_transformed', type=bool, default=False)
	parser.add_argument('--gpu', type=int, default=0)
	
	parser.add_argument('--keyword', type=str, default='')
	parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
	parser.add_argument('--from_scratch', action='store_true', help='start from scratch')
	parser.add_argument('--strategy_appendix', type=str, default='4-sigma', help='feducated strategy appendix')
	parser.add_argument('--align_top_k', type=int, default=1500, help='align top k')
	parser.add_argument('--kl_weight', type=float, default=0.0001, 
						help='KL divergence loss')
	parser.add_argument('--robust_base_epoch', type=int, default=0, 
						help='Robust epoch ')
	parser.add_argument('--robust_load_epoch', type=int, default=0,
						help='load robust epoch')
	parser.add_argument('--robust_epochs', type=int, default=60,
						help='Robust training epochs')
	parser.add_argument('--robust_continue_train', action='store_true',
						help='training robust model')

	args = parser.parse_args()



	log_dir = os.path.join(base_path, 'history', args.task_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	log_file = os.path.join(log_dir, f'training_{timestamp}.log')
	logger_instance = Logger(args, log_file)
	
	main(args)
	
