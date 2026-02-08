import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('merged_athletes_medals.csv')

country_noc = 'GBR'

country_data = data[data['NOC'] == country_noc]

country_data['Year'] = pd.to_numeric(country_data['Year'], errors='coerce')
country_data = country_data.dropna(subset=['Year', 'Total'])
medals_by_year = country_data[['Year', 'Total']].dropna()
medals_by_year = medals_by_year.sort_values(by='Year')
host_years = country_data[country_data['Is_Host_Country'] == 1]['Year'].unique()

plt.figure(figsize=(12, 6))
plt.plot(
    medals_by_year['Year'],
    medals_by_year['Total'],
    label=f'{country_noc} Medal Count',
    marker='o',
    color='blue'
)

for year in host_years:
    if year in medals_by_year['Year'].values:
        plt.axvline(x=year, color='red', linestyle='--', alpha=0.6)
        plt.text(
            year,
            medals_by_year[medals_by_year['Year'] == year]['Total'].values[0] * 1.05,  # 调整y位置
            f'Host: {int(year)}',
            color='red',
            rotation=45,
            ha='right',
            va='bottom'
        )

plt.title(f'Medal Trend for {country_noc} (Host Years Highlighted)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Total Medals', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
