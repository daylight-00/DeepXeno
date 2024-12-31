# train.py

import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # 추가
from torch.utils.data import DataLoader, Subset
import logging
import time
import importlib.util
import argparse
from dataprovider import DataProvider
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def train_model(model, train_loader, val_loader, criterion, optimizer, device, log_file, num_epochs, patience, model_path, regularize=False, scheduler=None):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - [Training process]: %(message)s'
    )
    for epoch in range(num_epochs):
        if early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break

        start_time = time.time()
        epoch_loss = 0
        val_loss = 0
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0

        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                outputs = model(*X)
                loss = criterion(outputs, Y)
                if regularize:
                    loss = model.regularize(loss, device)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_train += (predicted == Y).sum().item()
                total_train += Y.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                
                outputs = model(*X)
                loss = criterion(outputs, Y)
                val_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == Y).sum().item()
                total_val += Y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        epoch_time = int(time.time() - start_time)
        logging.info(f'\
[{model_path}]-[Epoch {epoch+1:03d}/{num_epochs:03d}] - \
Time: {epoch_time} s, \
Train Acc: {train_acc:.5f}, \
Val Acc: {val_acc:.5f}, \
Train Loss: {avg_train_loss:.5f}, \
Val Loss: {avg_val_loss:.5f}'\
)
        if (epoch+1 >= 30 ) and (epoch+1 % 5 == 0):
            torch.save(model.state_dict(), f"{model_path}-epoch_{epoch+1}.pt")
        
        # Early stopping and best model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), f"{model_path}-best.pt")
            print(f"Best model updated at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and epoch+1 > 12:
                early_stop = True

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    set_seed(config["seed"])
    
    # Create DataProvider instance
    data_provider = DataProvider(
        epi_path=config["Data"]["epi_path"],
        epi_args=config["Data"]["epi_args"],
        hla_path=config["Data"]["hla_path"],
        hla_args=config["Data"]["hla_args"],
    )
    print(f"Total samples: {len(data_provider)}")

    # Extract targets for stratified splitting
    y = np.array([data_provider.samples[i][2] for i in range(len(data_provider))])
    additional_column = np.array([data_provider.samples[i][0] for i in range(len(data_provider))])  # 예: 특정 컬럼

    # Create a binary or multi-label array for additional stratification
    # 예를 들어, additional_column이 범주형 변수라고 가정
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    additional_encoded = le.fit_transform(additional_column)
    
    # Create multi-label stratification
    # 각 샘플에 대해 [타겟 클래스, 추가 컬럼]
    stratify_labels = np.vstack((y, additional_encoded)).T  # Shape: (n_samples, 2)

    # Initialize MultilabelStratifiedKFold
    num_folds = config["CrossValidation"]["num_folds"]
    mskf = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config["seed"])

    # Create EncodedDataset instance once
    full_dataset = config["encoder"](data_provider, **config["encoder_args"])
    batch_size = config["Train"]["batch_size"]
    num_workers = config["Data"]["num_workers"]

    # To store metrics for all folds
    fold_metrics = []

    for fold, (train_indices, val_indices) in enumerate(mskf.split(np.arange(len(full_dataset)), stratify_labels)):
        print(f"\n=== Fold {fold+1}/{num_folds} ===")

        # Create subsets for this fold
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        print(f"Samples in train set: {len(train_subset)}")
        print(f"Samples in validation set: {len(val_subset)}")

        # Create DataLoaders for this fold
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Initialize model, criterion, optimizer, etc. for this fold
        model = config["model"](**config["model_args"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f'Model loaded on {device}')

        criterion = config["Train"]["criterion"]()
        optimizer = config["Train"]["optimizer"](
            model.parameters(),
            **config["Train"]["optimizer_args"]
            )
        num_epochs = config["Train"]["num_epochs"]

        if config["Train"]["use_scheduler"]:
            from utils.scheduler import CosineAnnealingWarmUpRestarts
            optimizer = torch.optim.Adam(model.parameters(), lr=0)  # Reset optimizer
            total_training_steps    = num_epochs * len(train_loader)
            cycle_len               = total_training_steps // 10
            warmup_steps            = cycle_len // 10
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer,
                T_0     = cycle_len,    # cycle length
                T_mult  = 1,            # cycle length multiplier after each cycle.
                T_up    = warmup_steps, # warmup length
                eta_max = 0.1,          # max learning rate
                gamma   = 0.8           # max learning rate decay
                )
            print(f"Scheduling with cycle length {cycle_len} and warmup steps {warmup_steps}")
            print("Optimizer is changed to Adam with lr=0 due to scheduler.")
        else:
            scheduler = None

        # Define model path and log file for this fold
        fold_model_path = os.path.join(config["chkp_path"], f"{config['chkp_name']}_fold{fold+1}")
        fold_log_file = config["log_file"].replace(".log", f"_fold{fold+1}.log")

        os.makedirs(config["chkp_path"], exist_ok=True)

        # Start training for this fold
        train_model(
            model           = model,
            train_loader    = train_loader,
            val_loader      = val_loader,
            criterion       = criterion,
            optimizer       = optimizer,
            device          = device,
            log_file        = fold_log_file,
            num_epochs      = num_epochs,
            patience        = config["Train"]["patience"],
            regularize      = config["Train"]["regularize"],
            model_path      = fold_model_path,
            scheduler       = scheduler
            )
        
        # After training, evaluate the best model on validation set
        model.load_state_dict(torch.load(f"{fold_model_path}-best.pt", weights_only=True))
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                
                outputs = model(*X)
                loss = criterion(outputs, Y)
                val_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == Y).sum().item()
                total_val += Y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        print(f"Fold {fold+1} - Val Loss: {avg_val_loss:.5f}, Val Acc: {val_acc:.5f}")
        fold_metrics.append({'fold': fold+1, 'val_loss': avg_val_loss, 'val_acc': val_acc})

    # Aggregate metrics
    avg_loss = np.mean([m['val_loss'] for m in fold_metrics])
    std_loss = np.std([m['val_loss'] for m in fold_metrics])
    avg_acc = np.mean([m['val_acc'] for m in fold_metrics])
    std_acc = np.std([m['val_acc'] for m in fold_metrics])

    print("\n=== Cross-Validation Results ===")
    for m in fold_metrics:
        print(f"Fold {m['fold']} - Val Loss: {m['val_loss']:.5f}, Val Acc: {m['val_acc']:.5f}")
    print(f"Average Val Loss: {avg_loss:.5f} ± {std_loss:.5f}")
    print(f"Average Val Acc: {avg_acc:.5f} ± {std_acc:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with cross-validation using specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()
    main(args.config_path)
