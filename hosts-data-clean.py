import pandas as pd

hosts = pd.read_csv('data/summerOly_hosts.csv')

hosts['Host'] = hosts['Host'].str.replace(r'\s+', ' ', regex=True).str.strip()

hosts['Year'] = hosts['Year'].astype(int)

host_to_noc = {
    "Athens, Greece": "GRE", "Paris, France": "FRA", "St. Louis, United States": "USA",
    "London, United Kingdom": "GBR", "Berlin, Germany": "GER", "Tokyo, Japan": "JPN",
    "Los Angeles, United States": "USA", "Helsinki, Finland": "FIN",
    "Melbourne, Australia": "AUS", "Rome, Italy": "ITA", "Mexico City, Mexico": "MEX",
    "Munich, West Germany": "GER", "Montreal, Canada": "CAN", "Moscow, Soviet Union": "URS",
    "Seoul, South Korea": "KOR", "Barcelona, Spain": "ESP", "Atlanta, United States": "USA",
    "Sydney, Australia": "AUS", "Beijing, China": "CHN", "Rio De Janeiro, Brazil": "BRA",
    "Antwerp, Belgium": "BEL", "Amsterdam, Netherlands": "NED", "Brisbane, Australia": "AUS",
    "Stockholm, Sweden": "SWE", "Helsinki, Finland": "FIN"
}

hosts['Host_NOC'] = hosts['Host'].map(host_to_noc)

# 特殊处理 2020 年东京奥运会延期到 2021 年的情况
hosts.loc[hosts['Year'] == 2020, 'Host_NOC'] = 'JPN'  # 映射为日本
hosts.loc[hosts['Year'] == 2020, 'Postponed_To'] = 2021  # 添加延期年份

unmapped_hosts = hosts[hosts['Host_NOC'].isnull()]
print("\n未映射的主办国：")
print(unmapped_hosts)

hosts.to_csv('cleaned_summerOly_hosts.csv', index=False, encoding='utf-8')
print("\n清洗后的主办国数据已保存为 'cleaned_summerOly_hosts.csv'")

df_cleaned = pd.read_csv('cleaned_summerOly_athletes.csv')

df_merged = df_cleaned.merge(hosts, on='Year', how='left')

df_merged['Is_Host_Country'] = (df_merged['NOC'] == df_merged['Host_NOC']).astype(int)

print("\n合并后的运动员数据预览：")
print(df_merged.head())

df_merged.to_csv('merged_athletes_with_hosts.csv', index=False, encoding='utf-8')
print("\n合并后的运动员数据已保存为 'merged_athletes_with_hosts.csv'")
