import splitfolders
import os

input_folder = '../data/raw' 
output_folder = '../data/processed'

def split_data(input_path, output_path, ratio=(.7,.1, .2)):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Splitting data...")
    
    splitfolders.ratio(input_path, 
                       output=output_path, 
                       seed=1337, 
                       ratio=ratio, 
                       group_prefix=None, 
                       move=False)

    print("Successfully split data.")
    
if __name__ == "__main__":
    split_data(input_folder, output_folder)