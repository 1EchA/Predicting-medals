import pandas as pd
import statsmodels.api as sm

# 读取数据
df = pd.read_csv('2Screened_Data.CSV')


# 事件研究
def event_study_model(data, coach_var, time_var):
    if data.empty:
        print("数据为空，无法进行回归分析")
        return None
    X = sm.add_constant(data[[coach_var, time_var]])  # 添加常数项
    y = data['Contribution']  # 目标变量
    try:
        model = sm.OLS(y, X).fit()  # 进行回归
        return model
    except Exception as e:
        print(f"回归分析失败：{e}")
        return None


# 定义不同项目的事件窗口
event_windows = {
    'CHN': {
        'Volleyball Women\'s Volleyball': [
            (2016, 2024, 2008, 2015)  # 中国女子排球项目
        ]
    },
    'USA': {
        'Volleyball Women\'s Volleyball': [
            (2008, 2016, 2000, 2007)  # 美国女子排球项目
        ],
        'Women\'s Gymnastics': [
            (1984, 1992, 1976, 1983)  # 美国女子体操项目
        ]
    },
    'ROU': {
        'Women\'s Gymnastics': [
            (1976, 1984, 1972, 1975)  # 罗马尼亚女子体操项目
        ]
    }
}

results = []

# 按照不同的项目和事件窗口进行回归分析
for country, sports in event_windows.items():
    for event, windows in sports.items():
        for (event_window_start, event_window_end, pre_event_window_start, pre_event_window_end) in windows:
            print(f"正在处理国家：{country}，项目：{event}")

            pre_event_data = df[(df['NOC'] == country) &
                                (df['Year'] >= pre_event_window_start) &
                                (df['Year'] <= pre_event_window_end) &
                                (df['Event'] == event)]

            event_window_data = df[(df['NOC'] == country) &
                                   (df['Year'] >= event_window_start) &
                                   (df['Year'] <= event_window_end) &
                                   (df['Event'] == event)]

            country_data = pd.concat([pre_event_data, event_window_data], axis=0)

            if country_data.empty:
                print(f"国家 {country} 的 {event} 在事件前窗口和事件窗口内没有数据")
                continue

            model = event_study_model(country_data, 'Coach', 'Year')

            if model:
                p_values = model.pvalues
                results.append({
                    'Country': country,
                    'Event': event,
                    'Event_Window': (event_window_start, event_window_end),
                    'Pre_Event_Window': (pre_event_window_start, pre_event_window_end),
                    'Coefficients': model.params,
                    'R2': model.rsquared,
                    'P_Values': p_values
                })

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv('Event_Study_Results_with_pvalues.csv', index=False)
    print(results_df)
else:
    print("没有有效的回归结果")
