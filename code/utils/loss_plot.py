# %%
import sys, os
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt
import importlib.util
import argparse
import re
import os

def load_config(config_path):
    """동적으로 config 파일을 import하는 함수"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config  # config.py 내부의 config 딕셔너리 반환

# 로그 파일에서 값 읽어오기
def read_log_file(log_file, model_name):
    epochs = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    pattern = re.compile(
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \[Training process\]: '
        r'\[(.*?)\]-\[Epoch (\d+)/(\d+)\] - '
        r'Time: (\d+) s, Train Acc: ([\d\.]+), Val Acc: ([\d\.]+), '
        r'Train Loss: ([\d\.]+), Val Loss: ([\d\.]+)'
    )
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match and model_name in match.group(1):
                epoch = int(match.group(2))         # Epoch number
                train_acc = float(match.group(5))   # Train Accuracy
                val_acc = float(match.group(6))     # Validation Accuracy
                train_loss = float(match.group(7))  # Train Loss
                val_loss = float(match.group(8))    # Validation Loss
                
                epochs.append(epoch)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    return epochs, train_losses, val_losses, train_accuracies, val_accuracies

# %%
def main(config_path):
    # 로그 파일 경로
    config = load_config(config_path)
    log_file = config['log_file']
    model_name = config['chkp_name']
    plot_path = config['plot_path']
    os.makedirs(plot_path, exist_ok=True)
    save_path = f'{plot_path}/loss_plot-{model_name}.png'
    # 로그 파일에서 값 읽기
    epochs, train_losses, val_losses, train_accuracies, val_accuracies = read_log_file(log_file, model_name)

    # 그래프 그리기
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red')
    ax1.plot(epochs, val_losses, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='tab:blue')
    ax2.plot(epochs, val_accuracies, label='Validation Accuracy', color='tab:cyan')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Train and Validation Loss and Accuracy Over Epochs')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Loss plot saved as {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot train and validation loss and accuracy over epochs')
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()

    main(args.config_path)