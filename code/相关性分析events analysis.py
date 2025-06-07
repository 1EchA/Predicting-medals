import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

noc_list = [
    "USA", "CHN", "GBR", "GER", "AUS", "JPN", "ITA", "FRA", "CUB",
    "HUN", "CAN", "KOR", "ROU", "NED", "BRA", "AZE", "ESP",
    "UKR", "NZL", "UZB"
]

data_filtered = data[data['NOC'].isin(noc_list)]

print("Filtered data for top NOCs:")
print(data_filtered.head())
data_filtered.to_excel('filtered_data.xlsx', index=False)

def medal_to_points(medal):
    if medal == 'Gold':
        return 3
    elif medal == 'Silver':
        return 2
    elif medal == 'Bronze':
        return 1
    else:
        return 0

data_filtered.loc[:, 'Medal_Points'] = data_filtered['Medal'].apply(medal_to_points)

data_filtered.loc[:, 'Participant_Count'] = 1

columns_to_keep = ['NOC', 'Event', 'Medal_Points', 'Participant_Count']
data_grouped = data_filtered.groupby(['NOC', 'Event'], as_index=False).agg({
    'Medal_Points': 'sum',
    'Participant_Count': 'sum'
})

data_grouped = data_grouped[data_grouped['Participant_Count'] >= 10]
data_grouped['Winning_Rate'] = data_grouped['Medal_Points'] / data_grouped['Participant_Count']

print("Grouped data by NOC and Event with Winning Rate:")
print(data_grouped.head())
data_grouped.to_excel('grouped_data_with_winning_rate.xlsx', index=False)

china_data = data_grouped[data_grouped['NOC'] == 'CHN']
print("China-specific data:")
print(china_data)
china_data.to_excel('china_data_with_winning_rate.xlsx', index=False)

top_events = data_grouped.groupby('Event')['Medal_Points'].sum().nlargest(10).index
top_nocs = data_grouped.groupby('NOC')['Medal_Points'].sum().nlargest(10).index
data_top = data_grouped[data_grouped['Event'].isin(top_events) & data_grouped['NOC'].isin(top_nocs)]

print("Top Events:")
print(top_events)
print("Top NOCs:")
print(top_nocs)
print("Filtered top data with Winning Rate:")
print(data_top.head())
data_top.to_excel('filtered_top_data_with_winning_rate.xlsx', index=False)

noc_event_pivot = data_top.pivot(index='Event', columns='NOC', values='Winning_Rate').fillna(0)

print("Pivot table for heatmap with Winning Rate:")
print(noc_event_pivot)
noc_event_pivot.to_excel('pivot_table_with_winning_rate.xlsx')

plt.figure(figsize=(16, 12))
sns.heatmap(noc_event_pivot, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Relationship Between Top Events and Top 10 NOCs (Based on Winning Rate)')
plt.xlabel('NOC')
plt.ylabel('Event')
plt.tight_layout()
plt.show()
