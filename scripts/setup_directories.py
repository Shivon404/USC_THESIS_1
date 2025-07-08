# setup_directories.py
# Run this first to create all necessary directories
# Place this file in the same folder as reddit_miner.py

import os

def create_project_structure():
    """
    Create the complete directory structure for the USC thesis project
    """
    print("Setting up USC Thesis project directory structure...")
    
    # Get current directory (where this script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If we're in the scripts folder, go up one level to project root
    if os.path.basename(current_dir) == 'scripts':
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir
    
    # Define directory structure
    directories = [
        'data',
        'data/raw',
        'data/processed', 
        'data/exports',
        'results',
        'scripts',
        'analysis'
    ]
    
    print(f"Project root: {project_root}")
    
    # Create directories
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created: {dir_path}")
        except Exception as e:
            print(f"✗ Failed to create {dir_path}: {e}")
    
    # Create a test file in each directory to verify write permissions
    test_files = {
        'data/raw/test.txt': 'Raw data will be saved here',
        'data/processed/test.txt': 'Processed data will be saved here',
        'data/exports/test.txt': 'Final exports will be saved here',
        'results/test.txt': 'Visualizations and reports will be saved here'
    }
    
    print("\nTesting write permissions...")
    for file_path, content in test_files.items():
        full_path = os.path.join(project_root, file_path)
        try:
            with open(full_path, 'w') as f:
                f.write(content)
            print(f"✓ Write test successful: {full_path}")
            # Remove test file
            os.remove(full_path)
        except Exception as e:
            print(f"✗ Write test failed: {full_path} - {e}")
    
    print("\nDirectory structure setup complete!")
    print(f"Project structure:")
    for root, dirs, files in os.walk(project_root):
        level = root.replace(project_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")

if __name__ == "__main__":
    create_project_structure()