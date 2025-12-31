import os
import json
import argparse

def unique_categories(path_to_categories, output_path):
    unique_categories = set()

    for root, dirs, files in os.walk(path_to_categories):
        for file in files:
            if file.endswith('.json') and 'categories' in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for category in data:
                        unique_categories.add(category)
                        
    with open(output_path, 'w') as out_file:
        for category in sorted(unique_categories):
            out_file.write(f"{category}\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_categories', type=str, required=True, help='Path to the concepts file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save unique categories')
    args = parser.parse_args()
    unique_categories(args.path_to_categories, args.output_path)