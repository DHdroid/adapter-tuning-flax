import os

from flax import traverse_util
import pickle


def save_adapter_params(params, adapter_prefix, save_path=None):
    params_dict = traverse_util.flatten_dict(params)
    adapter_name = f"{adapter_prefix}_adapter"
    adapter_params = dict()
    for path in params_dict:
        if adapter_name in path:
            adapter_params[path] = params_dict[path]
    save_path = save_path or f"{adapter_name}.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(adapter_params, f)
    return


def load_adapter_params(params, config):
    params_dict = traverse_util.flatten_dict(params)
    for adapter in config.adapters:
        if adapter["pretrained_weights"]:
            adapter_name = f"{adapter['name_prefix']}_adapter"
            weights_path = adapter["pretrained_weights"]

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"File named {weights_path} does not exist")
            
            with open(weights_path, "rb") as f:
                loaded_weights = pickle.load(f)

            for path in params_dict:
                if adapter_name in path:
                    params_dict[path] = loaded_weights[path]
    
    params = traverse_util.unflatten_dict(params_dict)
    return params
