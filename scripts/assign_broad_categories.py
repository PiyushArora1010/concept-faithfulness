import os
import json
import argparse

def assign_categories(path_to_categories, mapping_dic):

    for root, dirs, files in os.walk(path_to_categories):
        for file in files:
            mapped_categories = {}
            if file.endswith('.json') and 'categories' in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for category in data:
                        if category in mapping_dic:
                            mapped_categories[category] = mapping_dic[category]
                        else:
                            mapped_categories[category] = "Uncategorized"
                mapped_file_path = file_path.replace('categories', 'broad_categories')
                with open(mapped_file_path, 'w') as out_file:
                    json.dump(mapped_categories, out_file, indent=4)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_categories', type=str, required=True, help='Path to the concepts file')
    parser.add_argument('--path_to_mapping', type=str, required=True, help='Path to the category mapping file')
    args = parser.parse_args()
    
    with open(args.path_to_mapping, 'r') as f:
        mapping_dic = json.load(f)
    assign_categories(args.path_to_categories, mapping_dic)