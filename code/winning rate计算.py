import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

data_2020_2024 = data[data['Year'].isin([2004, 2008]) & data['NOC'].isin(['USA', 'CHN'])].copy()

valid_medals = {'Gold', 'Silver', 'Bronze'}
data_2020_2024['Medal'] = data_2020_2024['Medal'].apply(lambda x: 'No Medal' if x not in valid_medals else x)
data_2020_2024['Medal'] = data_2020_2024['Medal'].fillna('No Medal')

medal_points_map = {'Gold': 3, 'Silver': 2, 'Bronze': 1, 'No Medal': 0}
data_2020_2024['Medal_Points'] = data_2020_2024['Medal'].map(medal_points_map)

# 获取2020年和2024年都参与的项目
data_2020 = data_2020_2024[data_2020_2024['Year'] == 2020]
data_2024 = data_2020_2024[data_2020_2024['Year'] == 2024]

# 计算2020年和2024年都有的项目
common_events = set(data_2020['Event']).intersection(set(data_2024['Event']))

#筛选出2020年和2024年都参与的项目数据
common_events_data_2020 = data_2020[data_2020['Event'].isin(common_events)]
common_events_data_2024 = data_2024[data_2024['Event'].isin(common_events)]

# 获取2024年新增的项目（2024年有，2020年没有）
new_events_2024 = set(data_2024['Event']) - set(data_2020['Event'])

# 筛选出2024年新增的项目数据
new_events_data_2024 = data_2024[data_2024['Event'].isin(new_events_2024)]

# 计算2020年和2024年每个项目的奖牌得分与参与人数
# 计算2020年和2024年每个项目和国家的参与人数
participant_count_2020 = data_2020.groupby(['Event', 'NOC']).size().reset_index(name='Participant_Count_2020')
participant_count_2024 = data_2024.groupby(['Event', 'NOC']).size().reset_index(name='Participant_Count_2024')

# 计算2020年和2024年每个项目的奖牌得分
participant_data_2020 = data_2020.groupby(['Event', 'NOC']).agg({
    'Medal_Points': 'sum',  # 奖牌得分
    'Medal': lambda x: (x != 'No Medal').sum()  # 奖牌数量
}).reset_index()

participant_data_2024 = data_2024.groupby(['Event', 'NOC']).agg({
    'Medal_Points': 'sum',  # 奖牌得分
    'Medal': lambda x: (x != 'No Medal').sum()  # 奖牌数量
}).reset_index()

# 合并奖牌得分和参与人数数据
merged_data_2020 = pd.merge(participant_data_2020, participant_count_2020, on=['Event', 'NOC'], how='left')
merged_data_2024 = pd.merge(participant_data_2024, participant_count_2024, on=['Event', 'NOC'], how='left')

# 计算Winning Rate（项目奖牌得分 / 参与人数）
merged_data_2020['Winning_Rate_2020'] = merged_data_2020['Medal_Points'] / merged_data_2020['Participant_Count_2020']
merged_data_2024['Winning_Rate_2024'] = merged_data_2024['Medal_Points'] / merged_data_2024['Participant_Count_2024']
merged_data_2020_non_zero = merged_data_2020[merged_data_2020['Winning_Rate_2020'] > 0]
merged_data_2024_non_zero = merged_data_2024[merged_data_2024['Winning_Rate_2024'] > 0]
common_events_data_2020_merged = merged_data_2020_non_zero[merged_data_2020_non_zero['Event'].isin(common_events)]
pivot_common_2020 = common_events_data_2020_merged.pivot_table(index='Event', columns='NOC', values='Winning_Rate_2020', aggfunc='mean').fillna(0)

plt.figure(figsize=(16, 12))
sns.heatmap(pivot_common_2020, annot=True, fmt=".2f", cmap='YlGnBu', cbar=True, annot_kws={'size':6})  # 调整字号
plt.title('Winning Rate for Common Events (2020) between CHN and FRA', fontsize=16)  # 调整标题字体
plt.xlabel('NOC', fontsize=14)
plt.ylabel('Event', fontsize=14)
plt.tight_layout()
plt.show()

common_events_data_2024_merged = merged_data_2024_non_zero[merged_data_2024_non_zero['Event'].isin(common_events)]
pivot_common_2024 = common_events_data_2024_merged.pivot_table(index='Event', columns='NOC', values='Winning_Rate_2024', aggfunc='mean').fillna(0)

plt.figure(figsize=(16, 12))
sns.heatmap(pivot_common_2024, annot=True, fmt=".2f", cmap='YlGnBu', cbar=True, annot_kws={'size':6})  # 调整字号
plt.title('Winning Rate for Common Events (2024) between CHN and FRA', fontsize=16)  # 调整标题字体
plt.xlabel('NOC', fontsize=14)  # 调整x轴标签字体
plt.ylabel('Event', fontsize=14)  # 调整y轴标签字体
plt.tight_layout()
plt.show()

# 视化新增项目
new_events_data_2024_merged = merged_data_2024_non_zero[merged_data_2024_non_zero['Event'].isin(new_events_2024)]
pivot_new = new_events_data_2024_merged.pivot_table(index='Event', columns='NOC', values='Winning_Rate_2024', aggfunc='mean').fillna(0)

plt.figure(figsize=(16, 12))
sns.heatmap(pivot_new, annot=True, fmt=".2f", cmap='YlGnBu', cbar=True)
plt.title('Winning Rate for New Events in 2024 between China and USA')
plt.xlabel('NOC')
plt.ylabel('Event')
plt.tight_layout()
plt.show()

common_events_data_2020_merged.to_excel('common_events_2020_winning_rate.xlsx', index=False)
common_events_data_2024_merged.to_excel('common_events_2024_winning_rate.xlsx', index=False)
new_events_data_2024_merged.to_excel('new_events_2024_winning_rate.xlsx', index=False)

print("Winning Rate data for common events in 2020 saved to 'common_events_2020_winning_rate.xlsx'.")
print("Winning Rate data for common events in 2024 saved to 'common_events_2024_winning_rate.xlsx'.")
print("Winning Rate data for new events in 2024 saved to 'new_events_2024_winning_rate.xlsx'.")