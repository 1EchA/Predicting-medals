import pandas as pd

# 加载数据
data = pd.read_csv('merged_athletes_medals.csv')
import pandas as pd

data['Total'] = data['Total'].fillna(0)
medals_min = data[['Year','NOC','Total','Gold']].drop_duplicates(subset=['Year','NOC']).copy()
medals_min = medals_min.sort_values(by=['NOC','Year'])
hist_series = (
    medals_min.groupby('NOC')['Total']
    .apply(lambda x: x.shift().cumsum())
    .fillna(0)
)

medals_min['Historical_Medals'] = hist_series.reset_index(drop=True).values
unique_sport_event = data.drop_duplicates(subset=['Year','NOC','Sport','Event'])
total_events_df = (
    unique_sport_event.groupby(['Year','NOC'])['Event']
    .count()
    .reset_index(name='Total_Events')
)
unique_names = data.drop_duplicates(subset=['Year','NOC','Name'])
participants_df = (
    unique_names.groupby(['Year','NOC'])['Name']
    .count()
    .reset_index(name='Total_Participants')
)
gender_counts = (
    unique_names.groupby(['Year','NOC','Sex'])['Name']
    .count()
    .reset_index()
)
gc_pivot = gender_counts.pivot(index=['Year','NOC'], columns='Sex', values='Name').fillna(0)
gc_pivot['Gender_Ratio'] = gc_pivot['Male'] / (gc_pivot['Male'] + gc_pivot['Female'])
gc_pivot.reset_index(inplace=True)

final_df = medals_min.copy()  # 包含 [Year, NOC, Total, Gold, Historical_Medals]

final_df = pd.merge(final_df, total_events_df, on=['Year','NOC'], how='left')

final_df = pd.merge(final_df, participants_df, on=['Year','NOC'], how='left')

final_df = pd.merge(final_df, gc_pivot[['Year','NOC','Gender_Ratio']], on=['Year','NOC'], how='left')

data['Is_Host'] = (data['NOC'] == data['Host_NOC']).astype(int)
host_df = data[['Year','NOC','Is_Host']].drop_duplicates(subset=['Year','NOC'])
final_df = pd.merge(final_df, host_df, on=['Year','NOC'], how='left')

final_df.fillna(0, inplace=True)

print(final_df.head(10))

final_df.to_csv('constructed_dataset_with_gold.csv', index=False)
