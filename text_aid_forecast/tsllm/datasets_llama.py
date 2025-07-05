import pandas as pd
import os 
from tsllm.models.utils_llama import print_debug, my_print

def get_datasets(config):
    if "forecast" in config.experiment.task :
        get_datasets(config,testfrac=0.2 )
        
def get_datasets(config,testfrac=0.2 ):
    # v2.jsonl is the dataset you download. It may have other names. 
    # Please put your dataset under target Path : config.experiment.data_path
    debug_mode = config.debug_mode
    is_test_mode = config.is_test_mode
    ts_length = config.ts_length
    print_debug(my_print, "config.experiment.data_path", config.experiment.data_path, debug_mode)
    data_list = pd.read_json(os.path.join(config.experiment.data_path ,'TS_Dataset.jsonl' ), lines=True)
    datas = []
    data_indexs =[]
    for _, row in data_list.iterrows():    
        try:
            series = pd.Series(row['series'])
            if(is_test_mode):
                series = series.head(ts_length)
            print_debug(my_print, "datasets_llama: get_datasets: series length", len(series), debug_mode)
            splitpoint = int(len(series)*(1-testfrac))
            train = series.iloc[:splitpoint]
            test  = series.iloc[splitpoint:]
            print_debug(my_print, "datasets_llama: get_datasets: series train length",
                        len(train), debug_mode)
            print_debug(my_print, "datasets_llama: get_datasets: series test length",
                        len(test), debug_mode)
            ts_info =''
            if config.experiment.description_type !='':
                if 'description' in config.experiment.description_type : 
                    # Prepend Description
                    ts_info = row['description']+ ' '
                if 'characteristics' in config.experiment.description_type : 
                    # Prepend Characteristics
                    ts_info += row['characteristics']+ ' '
                if 'metadata' in config.experiment.description_type :
                    # Prepend Metadata
                    meta_info = row['metadata']
                    ts_info +=  "The time series was collected between {} and {} with a collection frequency of {}, and the data Unit is \"{}\".  You will predict the next {} data points.".format(meta_info['start'] , meta_info['end'] , meta_info['frequency']  , meta_info['units'] , len(test)  )
            datas.append((train, test , ts_info ))
            data_indexs.append(row['uuid'])
            if len(data_indexs) >= (config.experiment.num_of_sampels) :  break # no need loading the whole set
        except: 
            continue
    datasets = dict(zip(data_indexs,datas))
    return datasets