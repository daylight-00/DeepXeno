# test.py

import sys, os
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import argparse
from dataprovider import DataProvider
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

import re
def normalize_hla_name(hla_name):
    hla_name = re.sub(r'\*|:|-', '', hla_name)
    return hla_name
def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def test_model(model, dataloader, device, target_layer):
    model.eval()
    print(model)
    all_features = []
    all_preds = []
    
    # Dictionary to store the features
    features = {}

    # Define the hook function
    def hook_fn(module, input, output):
        # Detach and move to CPU, then convert to numpy
        features['feature'] = output.detach().cpu().numpy()

    # Register the hook
    target_layer = model.self_attn
    hook = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            X = batch[:-1]  # 입력 데이터
            X = [x.to(device) for x in X]
            y_pred = model(*X)
            
            # Hook을 통해 캡처된 feature 가져오기
            feature = features.get('feature')
            if feature is not None:
                # sqeeze
                all_features.append(np.squeeze(feature))
            
            # 예측값 수집 (필요 시)
            all_preds.append(y_pred.cpu().numpy())

    # 훅 제거
    hook.remove()
    print (f'Length of all_features: {len(all_features)}')
    print (f'Length of all_preds: {len(all_preds)}')
    # 모든 feature와 예측값을 하나로 합치기
    print([arr.shape for arr in all_features])

    all_features = np.concatenate(all_features, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    return all_preds, all_features

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)

    # Data loading
    DATA_PROVIDER_ARGS = {
        "epi_path": config['Data']['test_path'],
        "epi_args": config['Data']['test_args'],
        "hla_path": config['Data']['hla_path'],
        "hla_args": config['Data']['hla_args'],
    }

    # Model loading
    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config["chkp_path"], config["chkp_name"] + '-' + config["Test"]["chkp_prefix"] + '.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f'Model loaded on {device}')

    data_provider = DataProvider(**DATA_PROVIDER_ARGS)
    dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Test"]["batch_size"] if "batch_size" in config["Test"] else len(dataset)
    num_workers = config["Data"]["num_workers"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y = np.array([sample[-1] for sample in data_provider.samples])  # Optimized data access

    target_layer_name = config["Test"]["target_layer"]
    target_layer = dict([*model.named_modules()])[target_layer_name]
    
    # Test the model and extract features
    y_pred, features  = test_model(model, dataloader, device, target_layer)

    # Save features and predictions using h5py

    # epi_list와 hla_list 생성
    epi_list = [sample[1] for sample in data_provider.samples]
    hla_list = [sample[0] for sample in data_provider.samples]
    print(f'Length of epi_list: {len(epi_list)}')
    print(f'Length of hla_list: {len(hla_list)}')
    # epi_hla_list 생성 및 정규화
    epi_hla_list = [f'{epi_list[i]}_{hla_list[i]}' for i in range(len(epi_list))]
    epi_hla_list = [normalize_hla_name(hla) for hla in epi_hla_list]

    # 고유성 확인
    unique_epi_hla = set(epi_hla_list)
    print(f'Unique epi_hla_list count: {len(unique_epi_hla)}')
    print(f'Total epi_hla_list count: {len(epi_hla_list)}')
    # print(features.shape)
    # features = features[0]
    # features 리스트 길이 확인
    print(f'Length of epi_hla_list: {len(epi_hla_list)}')
    print(f'Length of features: {len(features)}')

    # HDF5 파일 열기
    save_path = config["Test"]["feat_path"]

    with h5py.File(save_path, 'a') as f:
        for i, epi in enumerate(epi_hla_list):
            try:
                if epi in f:
                    print(f'Dataset {epi} already exists. Skipping.')
                    continue
                f.create_dataset(epi, data=features[i])
            except Exception as e:
                print(f'Error: {e}')
                print(f'Failed to save dataset for {epi}')
                continue
    
    print(f'Features and predictions saved to {save_path}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()

    main(args.config_path)
