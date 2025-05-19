import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def make_parser():
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--log_file', required=True, type=str, help='Path to log file')
    parser.add_argument('--save_path', type=str, default='./train_plots', help='Directory to save plots')
    parser.add_argument('--save_name', type=str, default='training_progress', help='Name of saved plot')
    parser.add_argument('--n_last_epochs', type=int, default=50, help='Number of final epochs to plot')
    return parser

def plot_training_progress(train_loss, test_loss, train_acc1, test_acc1, train_acc5, test_acc5, train_lr, epoch, n_epoch, save_path, save_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # First subplot: Loss with learning rate on secondary axis
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()  # Create a secondary y-axis
    
    l1, = ax1.plot(epoch, train_loss, 'b-', label='Train loss')
    l2, = ax1.plot(epoch, test_loss, 'r-', label='Test loss')
    l3, = ax1_twin.plot(epoch, train_lr, 'g--', label='Learning rate', alpha=0.5)
    
    ax1.set_title('Loss and Learning Rate', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1_twin.set_ylabel('Learning Rate', fontsize=10, color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends from both axes
    lines = [l1, l2, l3]
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper right')
    
    # Second subplot: Top-1 Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epoch, train_acc1, 'b-', label='Train top-1 accuracy')
    ax2.plot(epoch, test_acc1, 'r-', label='Test top-1 accuracy')
    ax2.set_title('Top-1 Accuracy', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    # Third subplot: Top-5 Accuracy
    ax3 = axes[1, 0]
    ax3.plot(epoch, train_acc5, 'b-', label='Train top-5 accuracy')
    ax3.plot(epoch, test_acc5, 'r-', label='Test top-5 accuracy')
    ax3.set_title('Top-5 Accuracy', fontsize=12)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Accuracy (%)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='lower right')
    
    # Fourth subplot: Recent Loss Progress
    ax4 = axes[1, 1]
    last_epochs = epoch[-n_epoch:]
    ax4.plot(last_epochs, train_loss[-n_epoch:], 'b-', label='Train loss')
    ax4.plot(last_epochs, test_loss[-n_epoch:], 'r-', label='Test loss')
    ax4.set_title(f'Loss (Last {n_epoch} epochs)', fontsize=12)
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('Loss', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper right')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(save_path, save_name + '.png'), dpi=300)
    plt.show()

def main(args):
    # Load and parse log file
    try:
        with open(args.log_file, 'r') as f:
            lines = f.readlines()
        
        data_dict = {
            'train_loss': [], 'test_loss': [], 'train_train_acc1': [], 'test_acc1': [],
            'train_train_acc5': [], 'test_acc5': [], 'train_lr': [], 'epoch': []
        }
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                for key in data_dict.keys():
                    if key in data:
                        data_dict[key].append(data[key])
            except json.JSONDecodeError:
                try:
                    data = eval(line.strip())
                    for key in data_dict.keys():
                        if key in data:
                            data_dict[key].append(data[key])
                except:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue
        
        # Convert to numpy arrays
        train_loss = np.array(data_dict['train_loss'])
        test_loss = np.array(data_dict['test_loss'])
        train_acc1 = np.array(data_dict['train_train_acc1'])
        test_acc1 = np.array(data_dict['test_acc1'])
        train_acc5 = np.array(data_dict['train_train_acc5'])
        test_acc5 = np.array(data_dict['test_acc5'])
        train_lr = np.array(data_dict['train_lr'])
        epoch = np.array(data_dict['epoch'])
        
        # Sort all data by epoch to ensure smooth plots
        if len(epoch) > 0:
            sort_idx = np.argsort(epoch)
            train_loss = train_loss[sort_idx]
            test_loss = test_loss[sort_idx]
            train_acc1 = train_acc1[sort_idx]
            test_acc1 = test_acc1[sort_idx]
            train_acc5 = train_acc5[sort_idx]
            test_acc5 = test_acc5[sort_idx]
            train_lr = train_lr[sort_idx]
            epoch = epoch[sort_idx]
            
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
                
            plot_training_progress(
                train_loss, test_loss, train_acc1, test_acc1, train_acc5, 
                test_acc5, train_lr, epoch, args.n_last_epochs, 
                args.save_path, args.save_name
            )
        else:
            print("No valid data found in the log file.")
    except Exception as e:
        print(f"Error processing log file: {e}")

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)
