import pandas as pd


medals = pd.read_csv('data/summerOly_medal_counts.csv')

print("数据预览：")
print(medals.head())

print("\n数据基本信息：")
print(medals.info())

print("\n重复记录数量：", medals.duplicated().sum())

print("\n每列缺失值数量：")
print(medals.isnull().sum())

medals = medals.drop_duplicates(subset=['Year', 'NOC'])

medals[['Gold', 'Silver', 'Bronze', 'Total']] = medals[['Gold', 'Silver', 'Bronze', 'Total']].fillna(0)

medals['NOC'] = medals['NOC'].fillna('Unknown')

country_to_noc = {
    "United States": "USA",
    "Great Britain": "GBR",
    "Germany": "GER",
    "France": "FRA",
    "China": "CHN",
    "Japan": "JPN",
    "Italy": "ITA",
    "Australia": "AUS",
    "Australasia": "ANZ",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Soviet Union": "URS",
    "Unified Team": "EUN",
    "Czechoslovakia": "TCH",
    "Yugoslavia": "YUG",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Netherlands": "NED",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Hungary": "HUN",
    "Denmark": "DEN",
    "Poland": "POL",
    "Switzerland": "SUI",
    "Canada": "CAN",
    "Russia": "RUS",
    "Romania": "ROU",
    "India": "IND",
    "Argentina": "ARG",
    "Portugal": "POR",
    "Cuba": "CUB",
    "Iran": "IRI",
    "Turkey": "TUR",
    "Mexico": "MEX",
    "Kenya": "KEN",
    "Jamaica": "JAM",
    "Norway": "NOR",
    "Greece": "GRE",
    "South Africa": "RSA",
    "Egypt": "EGY",
    "Israel": "ISR",
    "Singapore": "SGP",
    "Hong Kong": "HKG",
    "Philippines": "PHI",
    "Thailand": "THA",
    "New Zealand": "NZL",
    "Austria": "AUT",
    "Ukraine": "UKR",
    "Kazakhstan": "KAZ",
    "Bulgaria": "BUL",
    "Uzbekistan": "UZB",
    "Belarus": "BLR",
    "Croatia": "CRO",
    "Slovakia": "SVK",
    "Slovenia": "SLO",
    "Estonia": "EST",
    "Latvia": "LAT",
    "Lithuania": "LTU",
    "Azerbaijan": "AZE",
    "Armenia": "ARM",
    "Georgia": "GEO",
    "Moldova": "MDA",
    "Namibia": "NAM",
    "Zambia": "ZAM",
    "Zimbabwe": "ZIM",
    "Ethiopia": "ETH",
    "Morocco": "MAR",
    "Algeria": "ALG",
    "Tunisia": "TUN",
    "Sudan": "SUD",
    "Botswana": "BOT",
    "Cote d'Ivoire": "CIV",
    "Nigeria": "NGR",
    "Burundi": "BDI",
    "Tanzania": "TAN",
    "Independent Olympic Participants": "IOP",
    "Refugee Olympic Team": "ROT"
}

medals['NOC'] = medals['NOC'].map(country_to_noc).fillna(medals['NOC'])

unmapped_nocs = medals[~medals['NOC'].isin(country_to_noc.values())]['NOC'].unique()
print("\n未标准化的国家/地区标签：", unmapped_nocs)

additional_mapping = {
    "Mixed team": "MIX",
    "Luxembourg": "LUX",
    "Bohemia": "BOH",
    "Russian Empire": "RU1",
    "Finland": "FIN",
    "Uruguay": "URU",
    "Haiti": "HAI",
    "Ireland": "IRL",
    "Chile": "CHI",
    "United States ": "USA",
    "Italy ": "ITA",
    "France ": "FRA",
    "Sweden ": "SWE",
    "Japan ": "JPN",
    "Hungary ": "HUN",
    "Germany ": "GER",
    "Finland ": "FIN",
    "Great Britain ": "GBR",
    "Poland ": "POL",
    "Australia ": "AUS",
    "Argentina ": "ARG",
    "Canada ": "CAN",
    "Netherlands ": "NED",
    "South Africa ": "RSA",
    "Ireland ": "IRL",
    "Czechoslovakia ": "TCH",
    "Austria ": "AUT",
    "India ": "IND",
    "Denmark ": "DEN",
    "Mexico ": "MEX",
    "Latvia ": "LAT",
    "New Zealand ": "NZL",
    "Switzerland ": "SUI",
    "Philippines ": "PHI",
    "Belgium ": "BEL",
    "Spain ": "ESP",
    "Uruguay ": "URU",
    "Peru": "PER",
    "Ceylon": "CEY",
    "Trinidad and Tobago": "TTO",
    "Panama": "PAN",
    "Puerto Rico": "PUR",
    "Lebanon": "LIB",
    "Venezuela": "VEN",
    "United Team of Germany": "EUN",
    "Iceland": "ISL",
    "Pakistan": "PAK",
    "Bahamas": "BAH",
    "Soviet Union ": "URS",
    "East Germany": "GDR",
    "West Germany": "FRG",
    "Mongolia": "MGL",
    "Uganda": "UGA",
    "Cameroon": "CMR",
    "Taiwan": "TPE",
    "Colombia": "COL",
    "Niger": "NIG",
    "Bermuda": "BER",
    "Guyana": "GUY",
    "Ivory Coast": "CIV",
    "Syria": "SYR",
    "Chinese Taipei": "TPE",
    "Dominican Republic": "DOM",
    "Suriname": "SUR",
    "Costa Rica": "CRC",
    "Indonesia": "INA",
    "Netherlands Antilles": "AHO",
    "Senegal": "SEN",
    "Virgin Islands": "ISV",
    "Djibouti": "DJI",
    "Malaysia": "MAS",
    "Qatar": "QAT",
    "Czech Republic": "CZE",
    "FR Yugoslavia": "FRY",
    "Ecuador": "ECU",
    "Tonga": "TGA",
    "Mozambique": "MOZ",
    "Saudi Arabia": "KSA",
    "Sri Lanka": "SRI",
    "Vietnam": "VIE",
    "Barbados": "BAR",
    "Kuwait": "KUW",
    "Kyrgyzstan": "KGZ",
    "Macedonia": "MKD",
    "United Arab Emirates": "UAE",
    "Serbia and Montenegro": "SCG",
    "Paraguay": "PAR",
    "Eritrea": "ERI",
    "Serbia": "SRB",
    "Tajikistan": "TJK",
    "Samoa": "SAM",
    "Afghanistan": "AFG",
    "Mauritius": "MRI",
    "Togo": "TOG",
    "Bahrain": "BRN",
    "Grenada": "GRN",
    "Cyprus": "CYP",
    "Gabon": "GAB",
    "Guatemala": "GUA",
    "Montenegro": "MNE",
    "Independent Olympic Athletes": "IOA",
    "Fiji": "FIJ",
    "Jordan": "JOR",
    "Kosovo": "KOS",
    "ROC": "ROC",
    "San Marino": "SMR",
    "North Macedonia": "MKD",
    "Turkmenistan": "TKM",
    "Burkina Faso": "BUR",
    "Saint Lucia": "LCA",
    "Dominica": "DMA",
    "Albania": "ALB",
    "Cabo Verde": "CPV",
    'United Team of Germany ': 'EUN',
    'Turkey ': 'TUR',
    'Romania ': 'ROU',
    'Bulgaria ': 'BUL',
    'Yugoslavia ': 'YUG',
    'Pakistan ': 'PAK',
    'Ethiopia ': 'ETH',
    'Greece ': 'GRE',
    'Norway ': 'NOR',
    'Iran ': 'IRI',
    'Egypt ': 'EGY',
    'Formosa ': 'TPE',
    'Ghana ': 'GHA',
    'Morocco ': 'MAR',
    'Portugal ': 'POR',
    'Singapore ': 'SGP',
    'Brazil ': 'BRA',
    'British West Indies ': 'BWI',
    'Iraq ': 'IRQ',
    'Venezuela ': 'VEN'
}

country_to_noc.update(additional_mapping)

medals['NOC'] = medals['NOC'].map(country_to_noc).fillna(medals['NOC'])

unmapped_nocs = medals[~medals['NOC'].isin(country_to_noc.values())]['NOC'].unique()
print("\n剩余未标准化的国家标签：", unmapped_nocs)

medals['Total_Check'] = medals['Gold'] + medals['Silver'] + medals['Bronze']
medals['Logic_Error'] = medals['Total'] != medals['Total_Check']

logic_errors = medals[medals['Logic_Error']]
if not logic_errors.empty:
    print("\n逻辑错误记录：")
    print(logic_errors)

medals.loc[medals['Logic_Error'], 'Total'] = medals['Total_Check']

medals.drop(columns=['Total_Check', 'Logic_Error'], inplace=True)

medals['Year'] = medals['Year'].astype(int)
medals[['Gold', 'Silver', 'Bronze', 'Total']] = medals[['Gold', 'Silver', 'Bronze', 'Total']].astype(int)

medals['Medal_Score'] = medals['Gold'] * 3 + medals['Silver'] * 2 + medals['Bronze']

total_medals_by_year = medals.groupby('Year')['Total'].transform('sum')
medals['Medal_Share'] = medals['Total'] / total_medals_by_year

medals.to_csv('cleaned_medals.csv', index=False)
print("\n清洗后的数据已保存为 'cleaned_medals.csv'")

print("\n清洗后的数据预览：")
print(medals.head())
print("\n清洗后的数据基本信息：")
print(medals.info())