import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

data_total = {
    'NOC': ['USA', 'CHN', 'GBR', 'GER', 'AUS', 'JPN', 'ITA', 'FRA', 'CUB', 'HUN', 'CAN', 'KOR', 'ROU', 'NED', 'BRA'],
    'Pred_Total_Medals': [118.367, 91.150, 63.359, 57.660, 53.202, 46.022, 38.276, 36.244, 28.471, 25.694, 24.094, 23.385, 22.318, 20.855, 20.832],
    'Total_Medals_LowerCI': [117.950, 91.148, 63.064, 57.626, 52.199, 45.911, 35.991, 31.277, 27.921, 25.324, 23.924, 22.946, 21.939, 20.305, 20.831],
    'Total_Medals_UpperCI': [118.784, 91.152, 63.655, 57.694, 54.204, 46.133, 40.561, 41.211, 29.021, 26.064, 24.263, 23.824, 22.697, 21.406, 20.832],
    '2024_Total_Medals': [126, 91, 65, 33, 53, 45, 40, 64, 9, 19, 27, 32, 9, 34, 20]
}

data_gold = {
    'NOC': ['USA', 'CHN', 'GER', 'JPN', 'GBR', 'AUS', 'ITA', 'ROU', 'NED', 'FRA', 'CUB', 'CAN', 'HUN', 'KOR', 'NZL'],
    'Pred_Gold_Medals': [46.467, 42.116, 28.546, 21.606, 19.375, 17.139, 10.737, 9.296, 8.818, 8.292, 8.058, 7.420, 6.635, 6.472, 6.373],
    'Gold_Medals_LowerCI': [45.033, 42.116, 28.049, 21.021, 19.247, 17.121, 9.621, 8.331, 7.484, 8.119, 7.759, 6.466, 5.628, 6.328, 6.071],
    'Gold_Medals_UpperCI': [47.901, 42.116, 29.043, 22.192, 19.502, 17.157, 11.853, 10.261, 10.152, 8.464, 8.356, 8.373, 7.642, 6.617, 6.674],
    '2024_Gold_Medals': [40, 40, 12, 20, 14, 18, 12, 3, 15, 16, 2, 9, 6, 13, 10]
}

df_total = pd.DataFrame(data_total)
df_gold = pd.DataFrame(data_gold)

plt.figure(figsize=(12, 8))
bar_width = 0.35
index = range(len(df_total))

plt.bar(index, df_total['Pred_Total_Medals'], bar_width, label='Predicted Total Medals', alpha=0.7, color='#4C72B0')

plt.errorbar(index, df_total['Pred_Total_Medals'], yerr=[df_total['Pred_Total_Medals'] - df_total['Total_Medals_LowerCI'],
                                                  df_total['Total_Medals_UpperCI'] - df_total['Pred_Total_Medals']],
             fmt='none', ecolor='#C44E52', capsize=5, label='Confidence Interval')

plt.plot(index, df_total['2024_Total_Medals'], 'o-', color='#55A868', label='2024 Total Medals')

plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Medals', fontsize=12)
plt.title('Predicted Total Medals for 2028 with Confidence Intervals', fontsize=14)
plt.xticks(index, df_total['NOC'], rotation=45, ha='right')  # 设置x轴标签为国家代码
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
bar_width = 0.35
index = range(len(df_gold))

plt.bar(index, df_gold['Pred_Gold_Medals'], bar_width, label='Predicted Gold Medals', alpha=0.7, color='#4C72B0')

plt.errorbar(index, df_gold['Pred_Gold_Medals'], yerr=[df_gold['Pred_Gold_Medals'] - df_gold['Gold_Medals_LowerCI'],
                                                  df_gold['Gold_Medals_UpperCI'] - df_gold['Pred_Gold_Medals']],
             fmt='none', ecolor='#C44E52', capsize=5, label='Confidence Interval')

plt.plot(index, df_gold['2024_Gold_Medals'], 'o-', color='#55A868', label='2024 Gold Medals')

plt.xlabel('Country', fontsize=12)
plt.ylabel('Gold Medals', fontsize=12)
plt.title('Predicted Gold Medals for 2028 with Confidence Intervals', fontsize=14)
plt.xticks(index, df_gold['NOC'], rotation=45, ha='right')  # 设置x轴标签为国家代码
plt.legend()

plt.tight_layout()
plt.show()