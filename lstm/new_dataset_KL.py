#new_dataset_KL.py



import math
import os
from collections import defaultdict
from math import log
from typing import List
import pickle
import json
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer
from utils.datacollection.logger import info, error
from new_Record_KL import TransformationRecordList
from new_Record_KL import *
from new_feature_env_KL import TransformationFeatureEvaluator, GlobalFeatureEvaluator
from lstm.gen_config import init_config


pad_idx = 1
eos_idx = 2

base_path = '../data'


local_range_map = {
    "openml_620": range(1, 5),
    "openml_586": range(1, 5),
    "openml_586_a1": range(1, 5),
    "openml_586_a0p5": range(1, 5),
    "openml_589": range(1, 6),
    "openml_637": range(1, 6),
    "openml_607": range(1, 6),
    "openml_616": range(1, 6),
    "openml_618": range(1, 6),
    "airfoil": range(1, 6),
    "housing_boston": range(1, 6),
    "pima_indian": range(1, 5),
    "spectf": range(1, 4),
    "spectf_a1": range(1, 4),
    "spectf_a0p5": range(1, 4),
    "wine_red": range(1, 6),
    "wine_white": range(1, 11),
    "german_credit": range(1, 6),
    "german_credit_a1": range(1, 6),
    "svmguide3": range(1, 6),
    "svmguide3_a1": range(1, 5),
    "svmguide3_a0p5": range(1, 5),
    "higgs": range(1, 11),
    "spam_base": range(1, 11),
    "messidor_features":range(1, 6),
    "german_credit": range(1, 6),
    "ionosphere": range(1, 4),
    "spectf_a1": range(1, 3),
    "amazon_employee": range(1, 6),
    "credit_default": range(1, 6),
    "credit_default_a1": range(1, 6),
    "credit_default_a0p5": range(1, 6),
}


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
             'german_credit_1': 'cls','german_credit_2': 'cls','german_credit_3': 'cls','german_credit_4': 'cls','german_credit_5': 'cls',
             'german_credit_a1': 'cls', 
             'german_credit_a1_1': 'cls','german_credit_a1_2': 'cls','german_credit_a1_3': 'cls','german_credit_a1_4': 'cls','german_credit_a1_5': 'cls',
             'higgs': 'cls','higgs_1': 'cls', 'higgs_2': 'cls', 'higgs_3': 'cls', 'higgs_4': 'cls', 'higgs_5': 'cls', 'higgs_6': 'cls',
             'higgs_7': 'cls', 'higgs_8': 'cls', 'higgs_9': 'cls', 'higgs_10': 'cls', 'higgs_11': 'cls', 'higgs_12': 'cls', 
             'higgs_13': 'cls', 'higgs_14': 'cls', 'higgs_15': 'cls', 'higgs_16': 'cls', 'higgs_17': 'cls', 'higgs_18': 'cls', 
             'higgs_19': 'cls', 'higgs_20': 'cls',
             'housing_boston': 'reg', 'housing_boston_1': 'reg','housing_boston_2': 'reg','housing_boston_3': 'reg','housing_boston_4': 'reg','housing_boston_5': 'reg',
             'ionosphere': 'cls',
             'ionosphere_1': 'cls',  'ionosphere_2': 'cls', 'ionosphere_3': 'cls', 
             'lymphography': 'cls',
             'messidor_features': 'cls', 
             'messidor_features_1': 'cls','messidor_features_2': 'cls','messidor_features_3': 'cls','messidor_features_4': 'cls','messidor_features_5': 'cls',
             'openml_620': 'reg', 
             'openml_620_1': 'reg','openml_620_2': 'reg','openml_620_3': 'reg','openml_620_4': 'reg',
             'pima_indian': 'cls','pima_indian_1': 'cls','pima_indian_2': 'cls','pima_indian_3': 'cls','pima_indian_4': 'cls','pima_indian_5': 'cls',
             'spam_base': 'cls', 
             'spam_base_1': 'cls','spam_base_2': 'cls','spam_base_3': 'cls','spam_base_4': 'cls','spam_base_5': 'cls',
             'spam_base_6': 'cls','spam_base_7': 'cls','spam_base_8': 'cls','spam_base_9': 'cls','spam_base_10': 'cls',
             'spectf': 'cls', 'spectf_1': 'cls', 'spectf_2': 'cls', 'spectf_3': 'cls', 
             'spectf_a1': 'cls','spectf_a1_1': 'cls','spectf_a1_2': 'cls',
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



def adjust_weights_for_sequence(base_w, valid_mask,    
                                mode='drop',           
                                drop_k=1,              
                                min_share=None,        
                                shrink_ratio=0.2):    

    w = np.array(base_w, dtype=float) * valid_mask
    s = w.sum()
    if s <= 0:
        return np.array(base_w, dtype=float) * 0.0  
    w = w / s  

    idx = np.argsort(w)  
    if drop_k and drop_k > 0:
        tgt = idx[:drop_k]
        if mode == 'drop':
            w[tgt] = 0.0
        else:
            w[tgt] *= shrink_ratio

    if min_share is not None:
        small = (w < float(min_share))
        if mode == 'drop':
            w[small] = 0.0
        else:
            w[small] *= shrink_ratio

    if w.sum() == 0:         
        w = np.array(base_w, dtype=float) * valid_mask

    w = w / (w.sum() + 1e-12)
    return w




class DenoiseDataModule:

    def __init__(self, params):
        
        task_name = params.task_name
        VOCABS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vocabs')        
        vocab_filename = f"{task_name}_vocab.json"
        vocab_file = os.path.join(VOCABS_DIR, vocab_filename)
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
    
        #suppose_vocab = f'{base_path}/history/{params.task_name}/vocab.json'
        #suppose_merge_file = f'{base_path}/history/{params.task_name}/merge.txt'
        HERE = os.path.dirname(os.path.abspath(__file__))
        suppose_merge_file = os.path.join(HERE, "merge.txt")
        #self.tokenizer = BartTokenizer(suppose_vocab, suppose_merge_file)
        #self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer(vocab_file, suppose_merge_file)
        self.task_type = TASK_DICT.get(params.task_name, 'cls')
        info(f"[DataModule] task {params.task_name} detected as {self.task_type}")





        def collect_global_seqs_with_fe_instances(subset_names):

             
            global_seqs = set()
            fe_instances = {}  
            
            for name in subset_names:
                fe = TransformationFeatureEvaluator(name, from_scratch=False) 
                fe_instances[name] = fe  
                

                top_k = params.align_top_k         
                records = fe.records.r_list
                if top_k and top_k > 0:             
                    records = sorted(records, key=lambda r: r.performance, reverse=True)[:top_k]

                for rec in records:                
                    global_seqs.add(tuple(rec.operation))

            return [list(seq) for seq in global_seqs], fe_instances

        
        local_range = local_range_map.get(params.task_name, range(1, 5))
        task_names_local = [f"{params.task_name}_{i}" for i in local_range]
    

        cache_file_base = f"{base_path}/history/{params.task_name}/aligned_records_KL_base.pkl" 
        cache_file_robust = f"{base_path}/history/{params.task_name}/aligned_records_KL_robust.pkl" 


        if params.strategy_appendix:
            base, ext = os.path.splitext(cache_file_base)
            cache_file = f"{base}.{params.strategy_appendix}{ext}"
            cache_file_robust = f"{base}.{params.strategy_appendix}_robust{ext}"
        else:
            cache_file = cache_file_base
            cache_file_robust = cache_file_base.replace('.pkl', '_robust.pkl')
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        global_seqs, fe_instances = collect_global_seqs_with_fe_instances(task_names_local)
        self.fe_instances = fe_instances
        self.fe_instances_robust = self.fe_instances 
        

        did_expand = False
        if (not params.from_scratch) and os.path.exists(cache_file):
            info(f"[DenoiseDataModule] detected aligned file {cache_file}, skip  recalculation.")
            expanded_fes = list(fe_instances.values()) 

        else:
            expanded_fes = []
            for name in task_names_local:
                fe = fe_instances[name]
                existing_seqs = {tuple(record.operation) for record in fe.records.r_list}
                
                info(f"Expanding {name}...")
                pbar = tqdm.tqdm(total=len(global_seqs), desc=f'Processing {name}')
                

                for seq in global_seqs:
                    seq_tuple = tuple(seq)
                    if seq_tuple not in existing_seqs:
                        new_record = TransformationRecord(seq, performance=None)
                        try:
                            generated_data = new_record.op(fe.original,with_original=True)
                            per = fe.get_performance(generated_data)
                            new_record.performance = per
                            fe.records.append_record(new_record)
                        except Exception as e:
                            info(f"Failed to generate performance for {seq_tuple}: {e}")
                            pass
                    pbar.update(1)
                pbar.close()
                expanded_fes.append(fe)
            did_expand = True


        if did_expand:
            for name, fe in fe_instances.items():
                out_path = os.path.join(
                    base_path, "history", params.task_name,
                    f"aligned_local_{name}.pkl"
                )
                with open(out_path, "wb") as f:
                    pickle.dump(fe.records.r_list, f)
                info(f"[DataModule] saved unweighted aligned records for {name} to {out_path}")




        def federated_aggregation(subset_fes, weights=None,cache_file=None,from_scratch=False,strategy_appendix='4-sigma',weight_strategy='multi_factor',
                          cache_file_robust=None, 
                          build_robust_variant=True,
                          robust_mode='drop',      # 'drop' or 'shrink'
                          robust_drop_k=1,
                          robust_min_share=None,   
                          robust_shrink_ratio=0.2,
                          return_weights = True):
                                  

            def compute_feature_entropy(fe):
                try:
                    X = fe.original.iloc[:, :-1].values
                    n_samples, n_features = X.shape
                    
                    feature_entropies = []
                    for i in range(n_features):
                        feature = X[:, i]
                        
                
                        n_unique = len(np.unique(feature))
                        
                        if n_unique <= 10:
                            
                            unique, counts = np.unique(feature, return_counts=True)
                            probs = counts / counts.sum()
                            entropy = -np.sum(probs * np.log(probs + 1e-10))
                            max_entropy = np.log(n_unique) if n_unique > 1 else 1
                        else:
                            n_bins = 10 
                            hist, _ = np.histogram(feature, bins=n_bins)
                            hist = hist[hist > 0]
                            probs = hist / hist.sum()
                            entropy = -np.sum(probs * np.log(probs + 1e-10))
                            max_entropy = np.log(n_bins)
                        
                        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
                        feature_entropies.append(norm_entropy)
                    
                    return np.mean(feature_entropies)
                    
                except Exception as e:
                    info(f"failed calculating entropy: {e}")
                    return 0.5  
            
            def compute_condition_number(fe):
                try:
                    X = fe.original.iloc[:, :-1].values
                    

                    from sklearn.preprocessing import StandardScaler
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    corr_matrix = np.corrcoef(X_scaled.T)
                    
                    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)
                    
                    eigenvalues = np.linalg.eigvalsh(corr_matrix)
                    eigenvalues = eigenvalues[eigenvalues > 1e-10]  
                    
                    if len(eigenvalues) > 0:
                        cond_number = np.max(eigenvalues) / np.min(eigenvalues)
                    else:
                        cond_number = 1.0
                    
                    stability_score = np.exp(-0.1 * (cond_number - 1))
                    stability_score = np.clip(stability_score, 0, 1)
                    
                    return stability_score
                    
                except Exception as e:
                    info(f"failed calculationg conditional number: {e}")
                    return 0.5  

            sample_sizes = []
            feature_entropies = []
            stability_scores = []

            for fe in subset_fes:

                if fe.original is None or fe.original.shape[0] == 0:
                    raise ValueError(f"Subset {fe.task_name} is empty")
                n_samples = fe.original.shape[0]
                sample_sizes.append(n_samples)
                

                f_entropy = compute_feature_entropy(fe)
                feature_entropies.append(f_entropy)
                

                stability = compute_condition_number(fe)
                stability_scores.append(stability)
                



            if weight_strategy == 'multi_factor':

                sample_sizes = np.array(sample_sizes, dtype=float)
                feature_entropies = np.array(feature_entropies)
                stability_scores = np.array(stability_scores)
                

                sample_weights = sample_sizes / sample_sizes.sum()
                

                adjustment_scores = 0.6 * feature_entropies + 0.4 * stability_scores

                if adjustment_scores.std() > 1e-6:

                    adjustment_scores = (adjustment_scores - adjustment_scores.min()) / \
                                    (adjustment_scores.max() - adjustment_scores.min() + 1e-10)
                else:

                    adjustment_scores = np.ones_like(adjustment_scores) / len(adjustment_scores)  
    

                adjustment_weights = adjustment_scores / adjustment_scores.sum()


                weights = 0.9 * sample_weights + 0.1 * adjustment_weights


                weights = weights / weights.sum()


            elif weight_strategy == 'sample_only':

                sample_sizes = np.array(sample_sizes, dtype=float)
                weights = sample_sizes / sample_sizes.sum()
                info("pure sample size based")
                
            else:

                if weights is None:
                    sample_sizes = np.array(sample_sizes, dtype=float)
                    weights = sample_sizes / sample_sizes.sum()
                else:
                    weights = np.array(weights) / np.sum(weights)
                info("self-defined weighting")


            subset_names = [fe.task_name for fe in subset_fes]  
            self.base_weights_map = {name: float(w) for name, w in zip(subset_names, weights)}


            if not from_scratch and cache_file and os.path.exists(cache_file):
                info(f"[federated_aggregation] detected {cache_file},load from local...")
                with open(cache_file, 'rb') as f:
                    merged_records = pickle.load(f)

                merged_records_robust = None 
                 
                if build_robust_variant and cache_file_robust and os.path.exists(cache_file_robust):
                    info(f"detect robust file {cache_file_robust},load from local...")
                    with open(cache_file_robust, 'rb') as f:
                        merged_records_robust = pickle.load(f)

                if return_weights:
                    return merged_records, merged_records_robust, weights
                elif build_robust_variant:
                    return merged_records, merged_records_robust
                else:
                    return merged_records

            info('=='*40)
            info("[federated_aggregation] no existing file found, start aligning... ")
            info('=='*40)



            aligned_perfs = defaultdict(list)
            for fe in subset_fes:
                for record in fe.records.r_list:
                    seq_tuple = tuple(record.operation)
                    aligned_perfs[seq_tuple].append(record.performance)


            merged_records = []
            merged_records_robust = []   
            
            
            
            base_w = np.asarray(weights, dtype=float) 

            for seq, perfs in aligned_perfs.items():
                if len(perfs) != len(subset_fes):
                    continue

                weighted_perf = sum(p * w for p, w in zip(perfs, weights))
                merged_records.append(TransformationRecord(list(seq), weighted_perf))
                
                perf_vec = np.array(perfs, dtype=float)

                valid_mask = ~np.isnan(perf_vec) 


                if build_robust_variant:
                    w_adj = adjust_weights_for_sequence(
                        base_w, valid_mask,
                        mode=robust_mode,
                        drop_k=robust_drop_k,
                        min_share=robust_min_share,
                        shrink_ratio=robust_shrink_ratio
                    )
                    perf_robust = float(np.nansum(perf_vec * w_adj))
                    merged_records_robust.append(TransformationRecord(list(seq), perf_robust))



            # If Cache_file is specified, save the result to the cache file
            if cache_file:
                info("=="*40)
                info(f"[federated_aggregation] finished, Writing file to {cache_file}...")
                info("=="*40)
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(merged_records, f)
                except Exception as e:
                    info("Error during dump to pickle:", e)
                    raise
                info("File writing finished,return merged_records")
            #return merged_records
            if build_robust_variant and cache_file_robust:
                with open(cache_file_robust, 'wb') as f:
                    pickle.dump(merged_records_robust, f)

            if return_weights and build_robust_variant:
                return merged_records, merged_records_robust, weights
            elif build_robust_variant:
                return merged_records, merged_records_robust
            else:
                return merged_records


        
        merged_records, merged_records_robust, weights = federated_aggregation(
            subset_fes=expanded_fes,
            cache_file=cache_file,
            from_scratch=params.from_scratch,
            weight_strategy='multi_factor',
            build_robust_variant=True,
            robust_mode='drop',            
            robust_drop_k=1,               
            robust_min_share=None,        
            robust_shrink_ratio=0.2,
            return_weights=True,
            cache_file_robust=cache_file_robust)
        

        self.base_weights = weights
        


        info(f"base weights are:{weights}")
        self.global_fe = GlobalFeatureEvaluator(task_name = params.task_name, merged_records = merged_records)
        self.fe = self.global_fe
        

        self.global_fe_robust = GlobalFeatureEvaluator(task_name = params.task_name, merged_records = merged_records_robust)
        self.fe_robust = self.global_fe_robust
        
        
        self.global_fe.base_weights = self.base_weights_map
        self.global_fe_robust.base_weights = self.base_weights_map


        r_list = list(self.fe.records.r_list)
        info("=="*40)
        info(f"self.fe.records.r_list len: {len(self.fe.records.r_list)}")
        info(f'the length of r_list is {len(r_list)}')
        info(f"params.top_k : {params.top_k}")
        info(f"params.train_top_k : {params.train_top_k}")
        info("=="*40)


        r_list_robust = list(self.fe_robust.records.r_list)
        info("=="*40)
        info(f"self.fe.records.r_list len: {len(self.fe_robust.records.r_list)}")
        info(f'the length of r_list is {len(r_list_robust)}')
        info(f"params.top_k : {params.top_k}")
        info(f"params.train_top_k: {params.train_top_k}")
        info("=="*40)




        self.train_dataset = DenoiseDataset(r_list=r_list, num=params.num, ds_size=self.fe.ds_size,
                                            tokenizer=self.tokenizer,
                                            mask_whole_op_p=params.mask_whole_op_p, mask_op_p=params.mask_op_p,
                                            disorder_p=params.disorder_p, top=params.train_top_k)
        
        self.val_dataset = DenoiseDataset(r_list=r_list, num=params.num, ds_size=self.fe.ds_size,
                                            tokenizer=self.tokenizer,
                                            mask_whole_op_p=params.mask_whole_op_p, mask_op_p=params.mask_op_p,
                                            disorder_p=params.disorder_p, top=params.top_k+80,
                                            train=False)
        
        self.infer_dataset = DenoiseDataset(r_list=r_list, num=0, ds_size=self.fe.ds_size, tokenizer=self.tokenizer,
                                            mask_whole_op_p=0, mask_op_p=0, disorder_p=0, minmax=False, top=params.top_k,
                                            train=False)
        
        self.batch_size = params.batch_size


        #robust version/weight adjusted
        self.train_dataset_robust = DenoiseDataset(r_list=r_list_robust, num=params.num, ds_size=self.fe_robust.ds_size,
                                            tokenizer=self.tokenizer,
                                            mask_whole_op_p=params.mask_whole_op_p, mask_op_p=params.mask_op_p,
                                            disorder_p=params.disorder_p, top=params.train_top_k)
        
        self.val_dataset_robust = DenoiseDataset(r_list=r_list_robust, num=params.num, ds_size=self.fe_robust.ds_size,
                                            tokenizer=self.tokenizer,
                                            mask_whole_op_p=params.mask_whole_op_p, mask_op_p=params.mask_op_p,
                                            disorder_p=params.disorder_p, top=params.top_k+80,
                                            train=False)
        
        self.infer_dataset_robust = DenoiseDataset(r_list=r_list_robust, num=0, ds_size=self.fe_robust.ds_size, tokenizer=self.tokenizer,
                                            mask_whole_op_p=0, mask_op_p=0, disorder_p=0, minmax=False, top=params.top_k,
                                            train=False)




    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=128)
        return loader
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=128)
        return loader
    def infer_dataloader(self):
        loader = DataLoader(self.infer_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=128)
        return loader


    def train_dataloader_robust(self):
        loader = DataLoader(self.train_dataset_robust, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=128)
        return loader
    def val_dataloader_robust(self):
        loader = DataLoader(self.val_dataset_robust, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=128)
        return loader
    def infer_dataloader_robust(self):
        loader = DataLoader(self.infer_dataset_robust, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=128)
        return loader

class DenoiseDataset(Dataset):
    
    def __init__(self, r_list, ds_size, tokenizer: BartTokenizer, num=12
                 , mask_whole_op_p=0.1, mask_op_p=0.3, disorder_p=0.1, minmax=True, top=None, train=True):
 
        
        
        #Initial parameters
        self.num = num
        self.max_seq_length = ds_size + 2  # with <s> and <\s>
        self.vocab = tokenizer.get_vocab()
        self.tokenizer = tokenizer
        self.original_records = list(r_list)
        self.operation_list = []  # for normal mask
        self.ops_list = []  # store the split ops [for permuatation and specific mask]
        self.performance_list = []  # store the truth label
        self.length_op_list = []
        self.train = train
        performance_list = []
        for record in self.original_records:
            performance_list.append(record.performance)
        
        self.mask_whole_op_p = mask_whole_op_p
        self.disorder_p = disorder_p 
        self.mask_op_p = mask_op_p
        self.item_list = []



        info("=="*40)
        print(f'Length of performance list is: {len(performance_list)}')
        info("=="*40)
        #Decide the seq number you want to use with top, if top is None, use all/And later permutation will only use these index
        if top is not None:
            _, indices = torch.topk(torch.tensor(performance_list), top, dim=0) 
        else:
            indices = range(len(performance_list))
        performance_list = []



        #Three parallel list is expanded/permutated at the same time: 
        #self.ops_list, self.operation_list, self.performance_list
        #   TF seg sequence permutation: to improve order-invariance/increase sample number
        for indice in tqdm.tqdm(indices, desc='denoise data gen'):
            record = self.original_records[indice]
            self.ops_list.append(record.ops)
            performance_list.append(record.performance)
            self.operation_list.append(record.operation)
            tmp = []
            for op_ in record.ops:
                tmp.append(len(op_))
            self.length_op_list.append(tmp)
            if num > 1:
                all, p = record.get_permutated_ops(num)
                for ops_, p_ in zip(all, p):
                    self.ops_list.append(ops_)
                    performance_list.append(p_)
                    source_seq = []
                    for i in ops_:
                        source_seq.extend([str(j) for j in i])
                        source_seq.append('4')
                    source_seq = source_seq[:-1]
                    self.operation_list.append(source_seq)
                    tmp = []
                    for op_ in ops_[:-1]:
                        tmp.append(len(op_))
                    self.length_op_list.append(tmp)

        if minmax:
            min_val = min(performance_list)
            max_val = max(performance_list)
            self.original_performance = performance_list.copy()
            if min_val == max_val:
                self.performance_list = [[1] * len(performance_list)]
            self.performance_list = [[(i - min_val) / (max_val - min_val)] for i in performance_list]
        else:
            self.original_performance = [[i] for i in performance_list]
            self.performance_list = self.original_performance.copy()
        
        

        for i in range(len(self)):
            self.item_list.append(self.process_item(i))

    def __getitem__(self, item):
        #for fast item retrieval through index
        return self.item_list[item]


    def select_top_k(self, k):
        p = torch.tensor(self.original_performance)
        values, indices = torch.topk(p, k, dim=0)
        op_list = []
        performance_list = values.tolist()
        for i in indices:
            op_list.append(self.operation_list[i.numpy()[0]])
        TRL = TransformationRecordList(self.max_seq_length - 2)
        for op, per in zip(op_list, performance_list):
            TRL.append(op, per)

        return DenoiseDataset(TRL.r_list, self.max_seq_length - 2, self.tokenizer, num=self.num
                              , mask_whole_op_p=0, mask_op_p=0, disorder_p=0)


    def process_item(self, item):
        original_seq = [str(i) for i in self.operation_list[item]]
        original_ops = self.ops_list[item]
        original_p = self.performance_list[item]
        

        ops = original_ops.copy()
        if self.mask_whole_op_p > 0:
            ops = self.add_whole_op_mask(ops, self.mask_whole_op_p)
        if self.disorder_p > 0:
            ops = self.disorder_ops(ops, self.disorder_p)
        

        source_seq = []
        for i in ops:
            source_seq.extend([str(j) for j in i])
            source_seq.append('4')
        source_seq = source_seq[:-1]
        if self.mask_op_p > 0:
            source_seq = self.add_op_mask(source_seq, self.mask_op_p)
        assert len(original_seq) == len(source_seq)
        
        
        

        decoder_seq, decoder_mask = self.padding_seq(original_seq, for_encoder=False)
        original_deocer_target, _ = self.padding_seq(original_seq)
        source_seq, source_mask = self.padding_seq(source_seq)

        if self.train:
            sample = {
                'encoder_input': torch.LongTensor(source_seq), 
                'decoder_input': torch.LongTensor(decoder_seq), 
                'encoder_target': torch.FloatTensor(original_p),  
                'decoder_target': torch.LongTensor(original_deocer_target) 
            }
        else:
            sample = {
                'encoder_input': torch.LongTensor(source_seq),
                'decoder_target': torch.LongTensor(original_deocer_target)
            }
            if original_p is not None:
                sample['encoder_target'] = torch.FloatTensor(original_p)
        return sample



    
    def __len__(self):
        return len(self.operation_list)

    def padding_seq(self, seq: List[str], for_encoder=True) -> object:
        if for_encoder:
            code = self.encode_build(seq) 
        else:
            code = self.decode_build(seq) 
        t_padding = [self.tokenizer.pad_token_id for _ in range(self.max_seq_length - len(code))]
        padding_mask = torch.tensor([1 for _ in range(self.max_seq_length)])
        padding_mask[len(code):] = 0
        code.extend(t_padding)
        return code, padding_mask

    def encode_build(self, seq_list):
        return [self.vocab[i] for i in seq_list]

    def decode_build(self, seq_list):
        return [2] + [self.vocab[i] for i in seq_list]

    def add_whole_op_mask(self, ops, p):
        if p <= 0:
            return ops
        ops_length = len(ops)
        num_to_mask_whole = math.ceil(np.sum(ops_length) * p)
        mask_whole_indice = np.random.choice(ops_length, num_to_mask_whole, replace=False)
        ops_ = []
        for indice, op in enumerate(ops):
            if indice in mask_whole_indice:
                op = ['<mask>'] * len(op)
            ops_.append(op)
        return ops_

    def disorder_ops(self, ops, p):
        if p <= 0:
            return ops
        ops_length = len(ops)
        num_to_disorder = math.ceil(ops_length * p)
        disorder_indice = np.random.choice(ops_length, num_to_disorder, replace=False)
        ops_ = []
        for indice, op in enumerate(ops):
            if indice in disorder_indice:
                op = np.random.permutation(op)
            ops_.append(op)
        return ops_

    def add_op_mask(self, ops_seq, p):
        if p <= 0:
            return ops_seq
        ops_length = len(ops_seq)
        num_to_mask = math.ceil(ops_length * p)
        mask_indice = np.random.choice(ops_length, num_to_mask, replace=False)
        for i in mask_indice:
            ops_seq[i] = '<mask>'
        return ops_seq



if __name__ == '__main__':
    task_dict = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary':
        'cls', 'german_credit': 'cls', 'higgs': 'cls',
                 'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
                 'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
                 'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
                 'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
                 'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
                 'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg'
                 }
    task_names = task_dict.keys()
    for name in task_names:
        args = init_config()
        args.task_name = name
        dl = DenoiseDataModule(args)
        for i in dl.train_dataloader():
            print(i)
            break
        break
