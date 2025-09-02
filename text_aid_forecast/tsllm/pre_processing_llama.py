from tsllm.models.utils_llama import print_debug, my_print
from tsllm.token_utils import get_scaler , truncate_input
from tsllm.serialize import serialize_arr , ori_scale_serialize
from omegaconf import open_dict
import numpy as np
import tsllm.models.fourier_transforms as ft
import tsllm.models.decompose_series as ds
    
def pre_processing(train, test, describtion  , config , tokenizer, debug_node=False  ):
    if 'rescale' in config.experiment.preprocess : 
        return rescale_pre_processing(train, test,  describtion , config , tokenizer, debug_node )
    if 'ori_scale' in  config.experiment.preprocess :
        return ori_scale_pre_processing(train, test , config  )

def ori_scale_pre_processing(train, test , config ):
    if not isinstance(train, list):
        train = [train]
        test = [test]
        
    # Add a temporary key value to represent the length of the prediction target.
    with open_dict(config):
        config.model.test_len = len(test[0])
        
    assert all(len(t)==config.model.test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'
    input_arrs = [train[i].values for i in range(len(train))]
    input_strs = ori_scale_serialize(input_arrs) 
    return None, input_strs , [None]*len(input_strs) , test
    
def rescale_pre_processing(train, test, describtions , config , tokenizer, debug_node=False ):
    '''
        Note that : 
        This script references https://github.com/ngruver/llmtime and https://arxiv.org/pdf/2310.07820.pdf. Thank you for your work.
        
        tokenizer is to help input fit the maximum context length (model`s)
    '''
    if not isinstance(train, list):
        train = [train]
        test = [test]
        describtions = [describtions]
    
    with open_dict(config):
        config.model.test_len = len(test[0])
        
    assert all(len(t)==config.model.test_len for t in test), f'All test series must have same length, got {[len(t) for t in test]}'
    # train is a list of pd.Series, each representing a time series.
    print_debug(my_print, "rescale_pre_processing: train len:", len(train), debug_node)
    print_debug(my_print, "rescale_pre_processing: train first three values of the first row", train[0][:3], debug_node)
    scalers = [get_scaler(train[i].values, alpha=config.model.alpha, beta=config.model.beta, basic=config.model.basic) for i in range(len(train))]
    print_debug(my_print, "pre_processing_llama: rescale_pre_processing: scaler:", scalers, debug_node)
    input_arrs = [train[i].values for i in range(len(train))] # convert pd.Series to np.array
    print_debug(my_print,"pre_processing_llama: rescale_pre_processing: input_arrs:",input_arrs, debug_node)
    input_trend_arrs, input_season_arrs, input_resid_arrs = [], [], []
    for i in range(len(train)) :
      trend, seasonal, resid = ds.decompose_series(train[i].values,period=config.model.decompose.period,robust=config.model.decompose.robust)
      input_trend_arrs.append(trend)
      input_season_arrs.append(seasonal)
      input_resid_arrs.append(resid)
    print_debug(my_print,"pre_processing_llama: rescale_pre_processing: input_trend_arrs:", input_trend_arrs,debug_node)
    print_debug(my_print,"pre_processing_llama: rescale_pre_processing: input_season_arrs:",input_season_arrs, debug_node)
    print_debug(my_print,"pre_processing_llama: rescale_pre_processing: input_resid_arrs:", input_resid_arrs,debug_node)
    '''
        Normailize time series, to make rescaled result locate in certain range 
        
        Normalize example : 
            112 -> 0.25917881 
            118 -> 0.27170962 
            .... 

        q= 478.82 ; min_ = -12.099  (q is not the max_value)
        transform     : (x - min_) / q
        inv_transform : x * q + min_ 
    '''
    if config.is_fourier:
        input_arrs = ft.fourier_transform(input_arrs)
        print_debug(my_print, "pre_processing_llama: rescale_pre_processing: fourier_transform:input_ft_arrs:", input_arrs, debug_node)

    print_debug(my_print,"pre_processing_llama: rescale_pre_processing: default input_arrs:",input_arrs, debug_node)
    transformed_input_arrs = np.array([scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)])
    # rtss = []
    # ita_scalers = zip(input_trend_arrs, scalers)
    # print_debug(my_print, "pre_processing_llama: rescale_pre_processing: ita_scalers:",ita_scalers, debug_node)
    # for trend_arr, scaler in ita_scalers:
    #   rtr = scaler.transform(trend_arr)
    #   print_debug(my_print,"pre_processing_llama: rescale_pre_processing: rtr:",rtr, debug_node)
    #   rtss.append(rtr)
    # print_debug(my_print,"pre_processing_llama: rescale_pre_processing: rtss:", rtss, debug_node)
    # transformed_trend_arrs = np.array(rtss)
    transformed_trend_arrs = np.array([scaler.transform(trend_arr) for trend_arr, scaler in zip(input_trend_arrs, scalers)])
    transformed_season_arrs = np.array([scaler.transform(season_arr) for season_arr, scaler in zip(input_season_arrs, scalers)])
    transformed_resid_arrs = np.array([scaler.transform(resid_arr) for resid_arr, scaler in zip(input_resid_arrs, scalers)])
    print_debug(my_print, "pre_processing_llama:rescale_pre_processing: after transformed_trend_arrs:", transformed_trend_arrs, debug_node)

    '''
        Shift the decimal point to ensure that values after rescaling fall within the 0-2000 range as much as possible.
         
        example : 
            0.25917881  -> [0 0 0 ...0 2 5 9] -> 259
            1.05070799  -> [0 0 0 ...1 0 5 0] -> 1050
        input_strs: ['627, 661, 739, 723,....']
    '''

    input_strs = [
      serialize_arr(scaled_input_arr, config.model.settings) for scaled_input_arr in transformed_input_arrs
    ] # convert np.array to str
    input_trend_strs = [
      serialize_arr(scaled_input_arr, config.model.settings) for scaled_input_arr in transformed_trend_arrs
    ]
    print_debug(my_print, "pre_processing_llama:rescale_pre_processing: serialize_arr:input_trend_strs:", input_trend_strs, debug_node)
    input_season_strs = [serialize_arr(scaled_input_arr, config.model.settings) for scaled_input_arr in transformed_season_arrs]
    input_resid_strs = [serialize_arr(scaled_input_arr, config.model.settings) for scaled_input_arr in transformed_resid_arrs]

    truncated_input_arrs, truncated_input_strs = zip(*[
      truncate_input(input_array, input_str, description, config , tokenizer ) for input_array, input_str ,description in zip(input_arrs, input_strs , describtions )
    ]) # truncate input to fit the model's maximum context length
    truncated_trend_arrs, truncated_trend_strs = zip(*[
      truncate_input(input_array, input_str, description, config , tokenizer ) for input_array, input_str ,description in zip(input_trend_arrs, input_trend_strs , describtions )
    ])
    truncated_season_arrs, truncated_season_strs = zip(*[truncate_input(input_array, input_str, description, config , tokenizer ) for input_array, input_str ,description in zip(input_season_arrs, input_season_strs , describtions )])
    truncated_resid_arrs, truncated_resid_strs = zip(*[truncate_input(input_array, input_str, description, config , tokenizer ) for input_array, input_str ,description in zip(input_resid_arrs, input_resid_strs , describtions )])
    print_debug(my_print, "pre_processing_llama:rescale_pre_processing: truncated_trend_arr:", truncated_trend_arrs, debug_node)
    print_debug(my_print, "pre_processing_llama:rescale_pre_processing: truncated_season_arr:", truncated_season_arrs, debug_node)
    print_debug(my_print, "pre_processing_llama:rescale_pre_processing: truncated_resid_arr:", truncated_resid_arrs, debug_node)
    return truncated_input_arrs, truncated_input_strs , scalers , test, truncated_trend_arrs, truncated_trend_strs, truncated_season_arrs, truncated_season_strs, truncated_resid_arrs, truncated_resid_strs