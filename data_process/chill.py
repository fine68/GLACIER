import pandas as pd
import numpy as np

def process_csv(file_path, output_path):
    
    df = pd.read_csv(file_path)

    
    for i in range(1, 36, 2):
        col1 = df.columns[i]  
        col2 = df.columns[i + 1]  

        
        abs_diff = (df[col1] - df[col2]).abs()

        
        mask = abs_diff > 3
        df.loc[~mask, [col1, col2]] = 0

    
    df.to_csv(output_path, index=False)


file_path = 'data_chillfix.csv'  
output_path = 'chill1.csv'  
process_csv(file_path, output_path)