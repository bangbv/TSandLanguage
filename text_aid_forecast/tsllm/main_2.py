import hydra
from omegaconf import DictConfig
import os , sys
from tsllm.models.utils_2 import load_model_by_name , get_predict_results
from tsllm.datasets_2 import get_datasets
from tsllm.pre_processing_2 import pre_processing
import os , pickle , time

from tsllm.token_utils import build_save_path  , is_completion
from tsllm.models.utils_2 import print_debug, my_print


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    debug_mode = config.debug_mode
    is_test_mode = config.is_test_mode
    print_debug(my_print, "Running with config", config, debug_mode)
    datasets = get_datasets(config)
    print_debug(my_print, "Length of datasets:", len(datasets), debug_mode)
    model = load_model_by_name(config)

    num_samples = 20 if 'gpt' in config.model.name else 96
    batch_size =  0  if 'gpt' in config.model.name else 6

    scalers = None
    save_dir = build_save_path(config)
    print_debug(my_print, "Save dir:", len(datasets), debug_mode)
    for dsname,data in datasets.items():
        if is_completion(save_dir , dsname, is_test_mode) : continue
        outs_dict = {}
        train, test , description= data
        # print(train , ' -- tes tLen:' , len(test) )
        _, input_strs ,  scalers , test  = pre_processing(train, test , description , config , model.tokenizer, debug_mode )
        # print(input_strs)
        try:
            out = get_predict_results(model , input_strs  , test , description  , config, batch_size, num_samples, scalers = scalers )
            print(f"main:run: the result {out}")
            outs_dict[config.model.name] = out
        except Exception as e:
            print(f"Failed {dsname} {config.model.name}" + str(e) )
            continue
        with open(f'{save_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(outs_dict,f)

if __name__ == "__main__":
    run()