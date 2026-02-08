import pandas as pd

# 加载数据
df = pd.read_csv('data/summerOly_athletes.csv')

print("数据预览：")
print(df.head())
print("\n数据概览：")
print(df.info())

# 删除重复
print("\n重复记录数量：", df.duplicated().sum())
df = df.drop_duplicates()

print("\n删除后的重复记录数量：", df.duplicated().sum())

print("\n每列缺失值数量：")
print(df.isnull().sum())

df['Medal'] = df['Medal'].fillna('No medal')

df_cleaned = df.dropna(subset=['Medal', 'Year', 'Sport', 'Event'])

print("\n删除缺失值后的数据概览：")
print(df_cleaned.isnull().sum())
print("\n处理前数据行数:", len(df), "处理后数据行数:", len(df_cleaned))

# 标准化字段
df_cleaned['Name'] = df_cleaned['Name'].str.replace(r'[^\w\s]', '', regex=True).str.title()

df_cleaned['Sex'] = df_cleaned['Sex'].replace({'M': 'Male', 'F': 'Female'})

df_cleaned['Medal'] = df_cleaned['Medal'].replace({
    'Gold Medal': 'Gold',
    'Silver Medal': 'Silver',
    'Bronze Medal': 'Bronze',
    'No Medal': 'No medal'
})

df_cleaned['Team'] = df_cleaned['Team'].str.replace('/', '-', regex=True)

# 异常值处理
current_year = pd.to_datetime('today').year
df_cleaned = df_cleaned[(df_cleaned['Year'] >= 1896) & (df_cleaned['Year'] <= current_year)]

df_cleaned = df_cleaned[df_cleaned['Sex'].isin(['Male', 'Female'])]
df_cleaned = df_cleaned[df_cleaned['Medal'].isin(['Gold', 'Silver', 'Bronze', 'No medal'])]

df_cleaned_duplicates = df_cleaned[df_cleaned.duplicated(subset=['Name', 'Year', 'Event', 'Medal'], keep=False)]
print("\n重复的奖牌记录：")
print(df_cleaned_duplicates)

df_cleaned = df_cleaned.drop_duplicates(subset=['Name', 'Year', 'Event', 'Medal'], keep='first')

df_cleaned.to_csv('cleaned_summerOly_athletes.csv', index=False)
print("\n清洗后的数据已保存为 'cleaned_summerOly_athletes.csv'")
