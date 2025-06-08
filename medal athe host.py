import pandas as pd
from difflib import get_close_matches

athletes_data = pd.read_csv('merged_athletes_with_hosts.csv')
medals_data = pd.read_csv('cleaned_medals.csv')

print("Athletes 表预览：")
print(athletes_data.head())
print("\nMedals 表预览：")
print(medals_data.head())

athletes_data['NOC'] = athletes_data['NOC'].str.strip().str.upper()
medals_data['NOC'] = medals_data['NOC'].str.strip().str.upper()

# 扩展映射表
additional_mapping = {
    "UAR": "EGY", "YMD": "YEM", "YAR": "YEM",
    "CRT": "ROT", "BWI": "MIX", "EOR": "KOR",
    "FRY": "YUG", "SCG": "SRB", "RU1": "RUS"
}
athletes_data['NOC'] = athletes_data['NOC'].replace(additional_mapping)

# 合并数据
merged_data = athletes_data.merge(
    medals_data,
    on=['NOC', 'Year'],
    how='left',
    suffixes=('_athletes', '_medals')
)

unmatched_records = merged_data[merged_data[['Gold', 'Silver', 'Bronze', 'Total']].isnull().any(axis=1)]
print("\n未匹配的记录数量：", unmatched_records.shape[0])
print("未匹配的记录示例：")
print(unmatched_records[['Name', 'NOC', 'Year', 'Team', 'Gold', 'Silver', 'Bronze']].head(10))

unmatched_noc = unmatched_records['NOC'].unique()
new_rows = []

for noc in unmatched_noc:
    if noc not in medals_data['NOC'].values:
        new_rows.append({
            'NOC': noc, 'Gold': 0, 'Silver': 0, 'Bronze': 0, 'Total': 0,
            'Year': None, 'Medal_Score': 0, 'Medal_Share': 0
        })

new_rows_df = pd.DataFrame(new_rows)
medals_data = pd.concat([medals_data, new_rows_df], ignore_index=True)

merged_data = athletes_data.merge(
    medals_data,
    on=['NOC', 'Year'],
    how='left',
    suffixes=('_athletes', '_medals')
)


merged_data.to_csv('merged_athletes_medals.csv', index=False)
print("\n修正后的数据已保存为 'merged_athletes_medals.csv'")
print("\n合并后的数据预览：")
print(merged_data.head())
print("\n合并后的数据基本信息：")
print(merged_data.info())

