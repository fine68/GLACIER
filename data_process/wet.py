import pandas as pd
import math


df = pd.read_csv('data3.csv')


def calculate_wetbulb_temp(temp, rh):
    return (temp * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + 
            math.atan(temp + rh) - 
            math.atan(rh - 1.676331) + 
            0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 
            4.686035)


df['shiqiu2'] = df.apply(lambda row: calculate_wetbulb_temp(row['Communication Building_Outdoor Temperature 1'], row['Communication Building_Outdoor Humidity 1']), axis=1)


df.to_csv('data6.csv', index=False)

print(df.head())
