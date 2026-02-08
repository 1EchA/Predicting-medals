import pandas as pd

# 'df_events' 是第一个表格，包含年份和事件总数
# 'df_athletes' 是第二个表格，包含参赛者的详细信息

# 读取数据
df_events = pd.read_csv('olympic_events.csv')
df_athletes = pd.read_csv('merged_athletes_medals(1).csv')

# 数据合并
df_merged = pd.merge(df_athletes, df_events, on='Year', how='left')

# 显示合并后的数据
df_merged.to_csv('merged_olympic_data.csv', index=False)

print("CSV文件已成功保存为 'merged_olympic_data.csv'")
print(df_merged)
