import os
from pathlib import Path
import argparse
from collections import defaultdict
import time


def find_all_files(directory, extensions=None):
    """Find all files with specified extensions in directory and subdirectories.
    
    Args:
        directory (str or Path): Directory to search
        extensions (list, optional): List of file extensions to include. If None, include all files.
        
    Returns:
        dict: Dictionary mapping filenames to their full paths
    """
    directory = Path(directory)
    file_dict = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.lower().endswith(ext.lower()) for ext in extensions):
                file_dict[file] = os.path.join(root, file)
                
    return file_dict


def check_data_leakage(train_dir, val_dir, extensions=None):
    """Check for filename duplicates between training and validation sets.
    
    Args:
        train_dir (str or Path): Directory containing training data
        val_dir (str or Path): Directory containing validation data
        extensions (list, optional): File extensions to consider (e.g., ['.jpg', '.jpeg', '.png'])
        
    Returns:
        tuple: (duplicates_found, duplicate_files)
    """
    print(f"Checking for data leakage between:\n- Training: {train_dir}\n- Validation: {val_dir}")
    
    # Default to common image extensions if none provided
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.JPEG']
        
    print(f"Scanning for files with extensions: {', '.join(extensions)}")
    start_time = time.time()
    
    # Find all files in both directories
    train_files = find_all_files(train_dir, extensions)
    val_files = find_all_files(val_dir, extensions)
    
    # Print statistics
    print(f"\nFound {len(train_files)} training files")
    print(f"Found {len(val_files)} validation files")
    
    # Find duplicates (same filename in both sets)
    duplicates = set(train_files.keys()).intersection(set(val_files.keys()))
    
    # Group duplicates by extension for analysis
    extensions_count = defaultdict(int)
    for file in duplicates:
        ext = os.path.splitext(file)[1].lower()
        extensions_count[ext] += 1
    
    # Prepare results
    duplicates_info = []
    for filename in duplicates:
        duplicates_info.append({
            'filename': filename,
            'train_path': train_files[filename],
            'val_path': val_files[filename],
        })
    
    elapsed_time = time.time() - start_time
    
    # Report results
    print(f"\nCheck completed in {elapsed_time:.2f} seconds")
    
    if duplicates:
        print(f"\n⚠️  POTENTIAL DATA LEAKAGE DETECTED: {len(duplicates)} duplicate filenames found!")
        print("\nBreakdown by extension:")
        for ext, count in sorted(extensions_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ext}: {count} files")
        
        print("\nFirst 10 duplicates:")
        for i, info in enumerate(duplicates_info[:10]):
            print(f"{i+1}. {info['filename']}")
            print(f"   Training: {info['train_path']}")
            print(f"   Validation: {info['val_path']}")
    else:
        print("\n✅ No data leakage detected! Training and validation sets have unique filenames.")
    
    return bool(duplicates), duplicates_info


def main():
    parser = argparse.ArgumentParser(description='Check for data leakage between training and validation sets')
    parser.add_argument('--train', required=True, help='Path to training data directory')
    parser.add_argument('--val', required=True, help='Path to validation data directory')
    parser.add_argument('--extensions', nargs='+', default=None, 
                        help='File extensions to check (e.g., .jpg .png)')
    parser.add_argument('--report', help='Optional path to save detailed report')
    
    args = parser.parse_args()
    
    has_duplicates, duplicates_info = check_data_leakage(args.train, args.val, args.extensions)
    
    # Save detailed report if requested
    if args.report and has_duplicates:
        import json
        with open(args.report, 'w') as f:
            json.dump({
                'num_duplicates': len(duplicates_info),
                'duplicates': duplicates_info
            }, f, indent=2)
        print(f"\nDetailed report saved to {args.report}")
    
    # Return exit code based on whether duplicates were found
    return 1 if has_duplicates else 0


if __name__ == "__main__":
    exit(main())

