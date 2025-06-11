import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import json

def make_parser():
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--log_files', nargs='+', required=True, type=str, help='Paths to log files')
    parser.add_argument('--labels', nargs='+', type=str, help='Labels for each log file (for the legend)')
    parser.add_argument('--save_path', type=str, default='./train_plots', help='Directory to save plots')
    parser.add_argument('--save_name', type=str, default='training_progress', help='Name of saved plot')
    parser.add_argument('--n_last_epochs', type=int, default=50, help='Number of final epochs to plot')
    return parser

def plot_training_progress(data_list, labels, n_epoch, save_path, save_name):
    """
    Plot training progress for multiple runs
    
    Parameters:
    data_list: List of dictionaries containing training data for multiple runs
    labels: List of labels for each run
    n_epoch: Number of final epochs to plot in the fourth subplot
             (if 0, only loss and top-1 accuracy plots are shown)
    save_path: Directory to save the plot
    save_name: Name of the saved plot
    """
    # Determine layout based on n_epoch
    if n_epoch == 0:
        # Only show loss and top-1 accuracy
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1, ax2 = axes
    else:
        # Show all four plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]
    
    # Define colors and line styles for different runs
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    train_styles = ['-', '-', '-', '-', '-']
    test_styles = ['--', '--', '--', '--', '--']
    
    # First subplot: Loss
    ax1_twin = ax1.twinx()  # Create a secondary y-axis
    lines = []
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        color_idx = i % len(colors)
        
        # Extract data
        train_loss = data['train_loss']
        test_loss = data['test_loss']
        train_lr = data['train_lr']
        epoch = data['epoch']
        
        l1, = ax1.plot(epoch, train_loss, color=colors[color_idx], linestyle=train_styles[0], 
                     label=f'{label} train loss')
        l2, = ax1.plot(epoch, test_loss, color=colors[color_idx], linestyle=test_styles[0], 
                     label=f'{label} test loss')
        
        # Only plot learning rate for the first run to avoid cluttering
        if i == 0:
            l3, = ax1_twin.plot(epoch, train_lr, color='gray', linestyle='--', 
                             label='Learning rate', alpha=0.5)
            lines.extend([l1, l2, l3])
        else:
            lines.extend([l1, l2])
    
    ax1.set_title('Loss and Learning Rate', fontsize=12)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1_twin.set_ylabel('Learning Rate', fontsize=10, color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(lines, [line.get_label() for line in lines], loc='upper right', fontsize='small')
    
    # Second subplot: Top-1 Accuracy
    for i, (data, label) in enumerate(zip(data_list, labels)):
        color_idx = i % len(colors)
        
        train_acc1 = data['train_acc1']
        test_acc1 = data['test_acc1']
        epoch = data['epoch']
        
        ax2.plot(epoch, train_acc1, color=colors[color_idx], linestyle=train_styles[0], 
               label=f'{label} train')
        ax2.plot(epoch, test_acc1, color=colors[color_idx], linestyle=test_styles[0], 
               label=f'{label} test')
    
    ax2.set_title('Top-1 Accuracy', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right', fontsize='small')
    
    # Only create the remaining plots if n_epoch > 0
    if n_epoch > 0:
        # Third subplot: Top-5 Accuracy
        for i, (data, label) in enumerate(zip(data_list, labels)):
            color_idx = i % len(colors)
            
            train_acc5 = data['train_acc5']
            test_acc5 = data['test_acc5']
            epoch = data['epoch']
            
            ax3.plot(epoch, train_acc5, color=colors[color_idx], linestyle=train_styles[0], 
                   label=f'{label} train')
            ax3.plot(epoch, test_acc5, color=colors[color_idx], linestyle=test_styles[0], 
                   label=f'{label} test')
        
        ax3.set_title('Top-5 Accuracy', fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('Accuracy (%)', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='lower right', fontsize='small')
        
        # Fourth subplot: Recent Loss Progress
        for i, (data, label) in enumerate(zip(data_list, labels)):
            color_idx = i % len(colors)
            
            train_loss = data['train_loss']
            test_loss = data['test_loss']
            epoch = data['epoch']
            
            # Get the last n_epoch epochs or all if fewer are available
            last_n = min(n_epoch, len(epoch))
            last_epochs = epoch[-last_n:]
            
            ax4.plot(last_epochs, train_loss[-last_n:], color=colors[color_idx], linestyle=train_styles[0], 
                   label=f'{label} train')
            ax4.plot(last_epochs, test_loss[-last_n:], color=colors[color_idx], linestyle=test_styles[0], 
                   label=f'{label} test')
        
        ax4.set_title(f'Loss (Last {n_epoch} epochs)', fontsize=12)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('Loss', fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(save_path, save_name + '.png'), dpi=300)
    plt.show()

def parse_log_file(log_file):
    """Parse a single log file and return the data dictionary"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        data_dict = {
            'train_loss': [], 'test_loss': [], 'train_acc1': [], 'test_acc1': [],
            'train_acc5': [], 'test_acc5': [], 'train_lr': [], 'epoch': []
        }
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                for key in data_dict.keys():
                    if key in data:
                        data_dict[key].append(data[key])
                # Handle the specific case of train_train_acc1 and train_train_acc5 keys
                if 'train_train_acc1' in data and 'train_acc1' not in data:
                    data_dict['train_acc1'].append(data['train_train_acc1'])
                if 'train_train_acc5' in data and 'train_acc5' not in data:
                    data_dict['train_acc5'].append(data['train_train_acc5'])
            except json.JSONDecodeError:
                try:
                    data = eval(line.strip())
                    for key in data_dict.keys():
                        if key in data:
                            data_dict[key].append(data[key])
                    # Handle the specific case of train_train_acc1 and train_train_acc5 keys
                    if 'train_train_acc1' in data and 'train_acc1' not in data:
                        data_dict['train_acc1'].append(data['train_train_acc1'])
                    if 'train_train_acc5' in data and 'train_acc5' not in data:
                        data_dict['train_acc5'].append(data['train_train_acc5'])
                except:
                    print(f"Warning: Could not parse line: {line.strip()}")
                    continue
        
        # Convert to numpy arrays
        for key in data_dict:
            if data_dict[key]:  # Check if the list is not empty
                data_dict[key] = np.array(data_dict[key])
            else:
                data_dict[key] = np.array([])
                
        # Sort all data by epoch to ensure smooth plots
        if len(data_dict['epoch']) > 0:
            sort_idx = np.argsort(data_dict['epoch'])
            for key in data_dict:
                if len(data_dict[key]) > 0:
                    data_dict[key] = data_dict[key][sort_idx]
            
            return data_dict
        else:
            print(f"No valid data found in the log file: {log_file}")
            return None
    except Exception as e:
        print(f"Error processing log file {log_file}: {e}")
        return None

def main(args):
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Parse all log files
    data_list = []
    valid_data_count = 0
    
    for log_file in args.log_files:
        data_dict = parse_log_file(log_file)
        if data_dict is not None:
            data_list.append(data_dict)
            valid_data_count += 1
    
    # Check if labels were provided
    if args.labels and len(args.labels) >= valid_data_count:
        labels = args.labels[:valid_data_count]
    else:
        # If no labels were provided or not enough, generate default ones
        labels = [f"Run {i+1}" for i in range(valid_data_count)]
    
    if valid_data_count > 0:
        plot_training_progress(
            data_list, labels, args.n_last_epochs, 
            args.save_path, args.save_name
        )
    else:
        print("No valid data found in any of the provided log files.")

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)
