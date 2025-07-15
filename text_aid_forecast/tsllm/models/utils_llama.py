from tsllm.models.llama_2 import LLAMAmodel
from tqdm import tqdm

import numpy as np 
import pandas as pd

import os , sys
from  tsllm.serialize import  deserialize_str , ori_scale_deserialize
import tsllm.models.fourier_transforms as ft

'''
    Note that : 
    This script references https://github.com/ngruver/llmtime and https://arxiv.org/pdf/2310.07820.pdf. Thank you for your work.
'''
STEP_MULTIPLIER = 1.2 
    
def load_model_by_name(config):
    print_debug(my_print, "utils_llama: load_model_by_name: model name", config.model.name, config.debug_mode)
    if config.model.name in ['llama-7b', 'llama-13b', 'llama-70b', 'llama-7b-chat', 'llama-13b-chat', 'llama-70b-chat']:
        return LLAMAmodel(config=config)
    return None


def get_output_format(preds , test , results_list , model_name , input_strs ):
    
    samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
    medians = [sample.median(axis=0) for sample in samples]
    samples = samples if len(samples) > 1 else samples[0]
    medians = medians if len(medians) > 1 else medians[0]
    out_dict = {
        'samples': samples,
        'median':  medians,
        'info': {
            'Method': model_name,
        },
        'completions_list': results_list,
        'input_strs': input_strs,
    }
    return out_dict

def get_predict_results(model , input_strs, test, description, config, batch_size, num_samples, scalers=None ):
    print_debug(my_print, "utils_llama:get_predict_results:model:", model, True)
    print_debug(my_print, "utils_llama:get_predict_results:input_strs:", input_strs, True)
    print_debug(my_print, "utils_llama:get_predict_results:test:", test, True)
    print_debug(my_print, "utils_llama:get_predict_results:description:", description, True)
    print_debug(my_print, "utils_llama:get_predict_results:config:", config, True)
    print_debug(my_print, "utils_llama:get_predict_results:batch_size:", batch_size, True)
    print_debug(my_print, "utils_llama:get_predict_results:num_samples:", num_samples, True)
    print_debug(my_print, "utils_llama:get_predict_results:scalers:", scalers, True)
    debug_node = config.debug_mode
    results_list = []
    batch_preds = []
    for input_str in tqdm(input_strs):
        res = model.run(input_str , description , config.model.test_len*STEP_MULTIPLIER , config, batch_size ,num_samples , config.model.temp )
        print_debug(my_print, "utils: get_predict_results: run: res", res[:3], debug_node)
        results_list.append(res)

    print_debug(my_print, "utils: get_predict_results: results_list length",len(results_list), debug_node)
    print_debug(my_print, "utils: get_predict_results: results_list first three completions",results_list[0][:3], debug_node)
    for completions, scaler in zip(results_list, scalers):
        preds = []
        for completion in completions:
            print_debug(my_print,"utils: get_predict_results: completion",completion, debug_node)
            # Convert the output string to a numpy array. 
            if scaler is not None :
                deserialized_pred = deserialize_str(completion, config.model.settings, ignore_last=False, steps=config.model.test_len) # explain this line code
            else :
                deserialized_pred = ori_scale_deserialize(completion)
            print_debug(my_print,"utils: get_predict_results: deserialized_pred",deserialized_pred, debug_node)
            #  Ensure the forecasting output length matches the ground-truth length.
            pred = handle_prediction(deserialized_pred , expected_length=config.model.test_len, strict=False)
            print_debug(my_print, "utils: get_predict_results: handle_prediction: pred", pred[:3], debug_node)
            # If there is a rescaling operation, restore the scale.
            if (pred is not None) and (scaler is not None)  :
                pred = scaler.inv_transform(pred)
                print_debug(my_print, "utils: get_predict_results: inv_transform: pred", pred[:3], debug_node)
                if config.is_fourier:
                    # If the model is trained with Fourier transform, we need to inverse the Fourier transform.
                    pred = ft.inverse_fourier_transform(pred)
                    print_debug(my_print,"utils:get_predict_results:inverse_fourier_transform:pred",pred, debug_node)
                preds.append(pred)
            else :
                print_debug(my_print,"utils: get_predict_results: inv_transform: error pred", pred, debug_node)
                preds.append(pred)

        # The batch_size here is 1, preds contain 20 predicted results
        batch_preds.append(preds)
        
    # Package the results
    out_dict = get_output_format(batch_preds, test , results_list , config.model.name , input_strs )
    return out_dict 

def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization, which may be too long or too short, or None if deserialization failed on the first prediction step.

    Args:
        pred (array-like or None): The predicted values. None indicates deserialization failed.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    else:
        if len(pred) < expected_length:
            if strict:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
                return None
            else:
                print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
                return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
        else:
            return pred[:expected_length]


def print_debug(f, header, value, debug_mode = False):
    if debug_mode : f(header, value)

def my_print(header, value):
    print(f"{header}: {value}")