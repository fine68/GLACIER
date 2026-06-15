import pandas as pd
import numpy as np

def process_csv(file_path, output_path):

    df = pd.read_csv(file_path)

    
    relevant_cols = [col for col in df.columns if "Cooling Water Inlet Temperature" in col or "Condenser Inlet Temperature" in col]

    
    def calculate_average(row):
        non_zero_values = row[row != 0]
        if len(non_zero_values) > 0:
            return non_zero_values.sum() / len(non_zero_values)
        return 0

    
    df['Cooling Water Inlet Average Temperature'] = df[relevant_cols].apply(calculate_average, axis=1)

    
    df.to_csv(output_path, index=False)


file_path = 'chill4.csv'  
output_path = 'chill5.csv'  
process_csv(file_path, output_path)