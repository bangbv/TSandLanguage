import hydra
from omegaconf import DictConfig
import os , sys
from tsllm.models.utils_llama import load_model_by_name , get_predict_results
from tsllm.datasets_llama import get_datasets
from tsllm.pre_processing_llama import pre_processing
import os , pickle , time

from tsllm.token_utils import build_save_path  , is_completion
import logging


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    debug_mode = config.debug_mode
    logger = logging.getLogger("tsllm:main_llama")
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.debug({"tsllm:main_llama:config": config})

    is_test_mode = config.is_test_mode
    datasets = get_datasets(config)
    logger.debug({"tsllm:main_llama:Length of datasets": len(datasets)})
    model = load_model_by_name(config)
    num_samples = 20 if 'gpt' in config.model.name else 96
    batch_size =  0  if 'gpt' in config.model.name else 6

    scalers = None
    save_dir = build_save_path(config)
    logger.debug({"tsllm:main_llama:save_dir": save_dir})
    for dsname,data in datasets.items():
        if is_completion(save_dir , dsname, is_test_mode) : continue
        outs_dict = {}
        train, test , description= data
        logger.debug({"tsllm:main_llama:Processing dataset": dsname})
        _, input_strs ,  scalers , test, truncated_trend_arrs, trend_strs, season_arrs, season_strs, resid_arrs, resid_strs  = pre_processing(train, test , description , config , model.tokenizer, debug_mode )
        logger.debug({"tsllm:main_llama:run:pre_processing:input_strs": input_strs})
        try:
            out = get_predict_results(model , input_strs, trend_strs, season_strs, resid_strs , test , description  , config, batch_size, num_samples, scalers = scalers )
            logger.debug({"tsllm:main_llama:run:get_predict_results:out": out})
            outs_dict[config.model.name] = out
        except Exception as e:
            print(f"Failed {dsname} {config.model.name}" + str(e) )
            continue
        with open(f'{save_dir}/{dsname}.pkl','wb') as f:
            pickle.dump(outs_dict,f)

if __name__ == "__main__":
    run()