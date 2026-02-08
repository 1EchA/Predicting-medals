import pandas as pd
import re
programs = pd.read_csv('data/summerOly_programs.csv')
print("数据预览：")
print(programs.head())
print("\n数据基本信息：")
print(programs.info())
print("\n重复记录数量：", programs.duplicated().sum())

def clean_value(value):
    if isinstance(value, str):
        # 去掉数据中 `?` 并提取数字部分
        cleaned_value = re.sub(r'\?', '', value).strip()
        if cleaned_value.isdigit():
            return int(cleaned_value)
        else:
            return None  # 如果只有 `?` 或无法解析为数字直接视作NA
    return value

# 对所有年份列应用清洗函数
for col in programs.columns[4:]:
    programs[col] = programs[col].apply(clean_value)

programs['Discipline'].fillna(programs['Sport'], inplace=True)

print("\n清洗后的数据预览：")
print(programs.head())

# 保存清洗后的表格
programs.to_csv('cleaned_summerOly_programs.csv', index=False)
print("\n清洗后的数据已保存为 'cleaned_summerOly_programs.csv'")
