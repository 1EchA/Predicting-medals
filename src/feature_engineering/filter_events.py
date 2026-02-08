import pandas as pd

file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

events_of_interest = [
    'B-Boys', 'B-Girls', 'Duet', 'Group All-Around', 'Men -58kg', 'Men\'s +92kg', 'Men\'s 102kg',
    'Men\'s 4 x 100m Medley Relay', 'Men\'s 4 x 100m Medley Relay', 'Men\'s 51kg', 'Men\'s 63.5kg',
    'Men\'s Canoe Double 500m', 'Men\'s Canoe Single', 'Men\'s Doubles', 'Men\'s Greco-Roman 130kg',
    'Men\'s Kayak Single', 'Men\'s Sabre Team', 'Men\'s Speed', 'Men\'s Synchronised 10m Platform',
    'Men\'s Synchronised 3m Springboard', 'Mixed 4 x 100m Medley Relay', 'Mixed Doubles', 'Skeet Mixed Team',
    'Women\'s +81kg', 'Women\'s 4 x 100m Freestyle Relay', 'Women\'s 4 x 100m Medley Relay',
    'Women\'s 4 x 200m Freestyle Relay', 'Women\'s 50kg', 'Women\'s 54kg', 'Women\'s 60kg', 'Women\'s 66kg',
    'Women\'s 75kg', 'Women\'s Canoe Double 500m', 'Women\'s Doubles', 'Women\'s Kayak Cross',
    'Women\'s Kite', 'Women\'s Skiff', 'Women\'s Speed', 'Women\'s Synchronised 10m Platform',
    'Women\'s Synchronised 3m Springboard'
]

data_2020_2024 = data[data['Year'].isin([2020, 2024]) & data['NOC'].isin(['FRA', 'CHN'])]

filtered_data = data_2020_2024[data_2020_2024['Event'].isin(events_of_interest)]

filtered_data.to_excel('filtered_data_FRA_CHN_2020_2024_new_events.xlsx', index=False)
print("Filtered data for specified events in 2020 and 2024 for FRA and CHN saved to 'filtered_data_FRA_CHN_2020_2024_new_events.xlsx'.")
