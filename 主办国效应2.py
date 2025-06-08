import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
data = pd.read_csv('merged_olympic_data.csv')

total_participants = data.groupby(['Year', 'NOC', 'Event'])['Name'].nunique().reset_index(name='Total_Participants')

data = data[data['Medal'] != 'No medal']
data['Medal_Points'] = data['Medal'].map({'Gold': 3, 'Silver': 2, 'Bronze': 1})  # 计算奖牌得分
medal_points = data.groupby(['Year', 'NOC', 'Event'])['Medal_Points'].sum().reset_index(name='Medal_Points')
event_data = pd.merge(medal_points, total_participants, on=['Year', 'NOC', 'Event'], how='left')
event_data['Winning_Rate'] = event_data['Medal_Points'] / event_data['Total_Participants']

# 筛选2000年以后的数据
event_data = event_data[event_data['Year'] >= 2000]
# 计算每年每个国家的新增项目
yearly_events = data.groupby(['Year', 'NOC'])['Event'].unique().reset_index(name='Events')
yearly_events['Previous_Year_Events'] = yearly_events.groupby('NOC')['Events'].shift(1)
yearly_events['New_Events'] = yearly_events.apply(lambda row: np.setdiff1d(row['Events'], row['Previous_Year_Events']), axis=1)
# 筛选出新增项目的年份和国家
new_events_data = yearly_events[yearly_events['New_Events'].apply(lambda x: len(x) > 0)]
# 筛选2000年以后的新增项目数据
new_events_data = new_events_data[new_events_data['Year'] >= 2000]

noc_list = ["USA", "CHN", "GBR", "GER", "AUS", "JPN", "ITA", "FRA", "CUB", "HUN", "CAN", "KOR", "ROU", "NED", "BRA", "AZE", "ESP", "UKR", "NZL", "UZB"]
new_events_data = new_events_data[new_events_data['NOC'].isin(noc_list)]

# 筛选出主办国数据
host_data = data[data['Is_Host_Country'] == 1]
# 计算主办国在新增项目中的获胜率
new_events_data['Host_Country_Winning_Rate'] = new_events_data.apply(
    lambda row: event_data[(event_data['NOC'] == row['NOC']) & (event_data['Year'] == row['Year'])]['Winning_Rate'].mean(), axis=1)
# 计算所有其他国家在新增项目中的平均获胜率
all_other_countries_data = event_data[~event_data['NOC'].isin(new_events_data['NOC'])]
new_events_data['Other_Countries_Avg_Winning_Rate'] = new_events_data.apply(
    lambda row: all_other_countries_data[(all_other_countries_data['Year'] == row['Year'])]['Winning_Rate'].mean(), axis=1)
# 用0填充可能出现的NaN值
new_events_data['Host_Country_Winning_Rate'] = new_events_data['Host_Country_Winning_Rate'].fillna(0)
new_events_data['Other_Countries_Avg_Winning_Rate'] = new_events_data['Other_Countries_Avg_Winning_Rate'].fillna(0)

# 比较
new_events_data['Winning_Rate_Difference'] = new_events_data['Host_Country_Winning_Rate'] - new_events_data['Other_Countries_Avg_Winning_Rate']
years_to_plot = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
# 可视化
plt.figure(figsize=(16, 8))
plt.bar(new_events_data['Year'], new_events_data['Winning_Rate_Difference'], color='skyblue')
plt.title('Host Country Effect: Winning Rate Difference in New Events (2000 and Later)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Winning Rate Difference (Host Country - Other Countries)', fontsize=14)
plt.xticks(years_to_plot, rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('host_country_effect_2000_and_later_diff.png')

new_events_data.to_excel('host_country_effect_analysis_2000_and_later_diff.xlsx', index=False)
print("Host country effect analysis saved to 'host_country_effect_analysis_2000_and_later_diff.xlsx'.")
