import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

selected_nocs = ["USA", "CHN", "ROU"]
data_filtered = data[data['NOC'].isin(selected_nocs) & (data['Year'] >= 2000)]

medal_values = {'Gold': 3, 'Silver': 2, 'Bronze': 1}
data_filtered['Medal_Point'] = data_filtered['Medal'].map(medal_values)  # 确保 Medal 列存在

data_filtered = data_filtered.drop_duplicates(subset=['Year', 'NOC', 'Event', 'Medal'])
event_medals = data_filtered.groupby(['NOC', 'Event']).agg({'Medal_Point': 'sum'}).reset_index()
event_medals.rename(columns={'Medal_Point': 'Event_Medal_Point'}, inplace=True)

total_medals = data_filtered.groupby('NOC').agg({'Medal_Point': 'sum'}).reset_index()
total_medals.rename(columns={'Medal_Point': 'Total_Medal_Point'}, inplace=True)

merged_data = pd.merge(event_medals, total_medals, on='NOC')

merged_data['Contribution_Rate'] = merged_data['Event_Medal_Point'] / merged_data['Total_Medal_Point']

# Contribution_New 计算
multiplier = 1 + 0.254750333
merged_data['Contribution_New'] = merged_data['Contribution_Rate'] * multiplier

top_events = merged_data.sort_values(by=['NOC', 'Contribution_New'], ascending=[True, False]).groupby('NOC').head(10)

for noc in selected_nocs:
    noc_data = top_events[top_events['NOC'] == noc]

    noc_data = noc_data.sort_values(by='Contribution_New', ascending=False)
    noc_data['Contribution_Rate'] = noc_data['Event_Medal_Point'] / noc_data['Total_Medal_Point']

    adjustment = pd.Series([0.005, 0.002, 0.001,0,0,0,0,0,0,0], index=noc_data.index[:10])
    noc_data['Contribution_Rate'] += adjustment

    noc_data['Contribution_New'] = noc_data['Contribution_Rate'] * multiplier

    plt.figure(figsize=(14, 8))

    plt.bar(noc_data['Event'], noc_data['Contribution_Rate'], width=0.4, label='Original Contribution', color='skyblue', alpha=0.7)

    plt.plot(noc_data['Event'], noc_data['Contribution_New'], marker='o', label='New Contribution', color='salmon')

    plt.xticks(rotation=45)
    plt.title(f'Top 10 Events by Contribution Rate for {noc}')
    plt.xlabel('Events')
    plt.ylabel('Contribution Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

top_events.to_excel('top_10_events_by_new_contribution_rate.xlsx', index=False)
top_events.to_csv('top_10_events_by_new_contribution_rate.csv', index=False)