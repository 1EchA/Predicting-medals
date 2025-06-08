import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
##这段GBRT代码是打包好的，从数据构建到训练建模一条龙解决

matplotlib.use('Agg')

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

from matplotlib import rcParams

# 设置支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


# 构建特征表

def build_feature_dataset(data_file):
    data = pd.read_csv(data_file)
    data['Total'] = data['Total'].fillna(0)
    data['Gold'] = data['Gold'].fillna(0)
    # 去掉一些过时的NOC
    data = data[~data['NOC'].isin(['URS', 'RUS', 'YUG', 'EUN', 'BOH', 'TCH'])]
    # 将东德（GDR）和西德（FRG）映射到德国（GER）
    data['NOC'] = data['NOC'].replace({'GDR': 'GER', 'FRG': 'GER'})

    # 去重
    medals_min = data[['Year','NOC','Total','Gold']].drop_duplicates(['Year','NOC']).copy()
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
    if 'Male' not in gc_pivot.columns:
        gc_pivot['Male'] = 0
    if 'Female' not in gc_pivot.columns:
        gc_pivot['Female'] = 0
    gc_pivot['Gender_Ratio'] = gc_pivot['Male'] / (gc_pivot['Male'] + gc_pivot['Female'])
    gc_pivot.reset_index(inplace=True)

    data['Is_Host'] = (data['NOC'] == data['Host_NOC']).astype(int)
    host_df = data[['Year','NOC','Is_Host']].drop_duplicates(['Year','NOC'])

    olympic_events_df = data[['Year', 'NOC', 'Olympic_events']].drop_duplicates(['Year', 'NOC'])

    final_df = medals_min.copy()  # [Year, NOC, Total, Gold, Historical_Medals]
    final_df = pd.merge(final_df, total_events_df, on=['Year','NOC'], how='left')
    final_df = pd.merge(final_df, participants_df, on=['Year','NOC'], how='left')
    final_df = pd.merge(final_df, gc_pivot[['Year','NOC','Gender_Ratio']], on=['Year','NOC'], how='left')
    final_df = pd.merge(final_df, host_df, on=['Year','NOC'], how='left')
    final_df = pd.merge(final_df, olympic_events_df, on=['Year','NOC'], how='left')

    final_df.fillna(0, inplace=True)

    final_df.rename(columns={'Total':'Total_Medals','Gold':'Gold_Medals'}, inplace=True)

    return final_df

# 外推函数

def simple_feature_extrapolation(grp, feature, future_year=2028):

    if grp['Year'].nunique() < 2:
        # 如果只有一届或无数据，就直接返回最后一条数据，不做线性回归
        return grp[feature].iloc[-1]
    X = grp[['Year']].values
    y = grp[feature].values
    lr = LinearRegression()
    lr.fit(X, y)
    return lr.predict([[future_year]])[0]



# 残差图绘制

def plot_residuals(y_true, y_pred, title="残差图"):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_pred, residuals, color='#84c3b7', alpha=0.8, edgecolor='#236255', linewidth=0.5
    )
    plt.axhline(0, linestyle='--', color='#4875b3', linewidth=1)
    plt.title(title, fontsize=14, color='#374151')
    plt.xlabel("Predicted Values", fontsize=12, color='#4B5563')
    plt.ylabel("Residuals", fontsize=12, color='#4B5563')
    plt.grid(alpha=0.2, color='#E5E7EB')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")  # 保存图像
    print(f"Plot saved as {title.replace(' ', '_')}.png")



# 预测2028奖牌数量

def train_overall_model(df, target_col='Total_Medals'):
    feature_cols = ['Total_Events', 'Gender_Ratio', 'Is_Host', 'Historical_Medals', 'Total_Participants', 'Olympic_events']

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 4],
        'min_samples_leaf': [1, 2],
        'subsample': [1.0, 0.8]
    }

    gbrt = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=gbrt,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_s, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    mean_mse_cv = -grid_search.best_score_
    mean_rmse_cv = sqrt(mean_mse_cv)

    best_model.fit(X_s, y)
    y_pred_all = best_model.predict(X_s)
    mse_all = mean_squared_error(y, y_pred_all)
    rmse_all = sqrt(mse_all)
    r2_all = r2_score(y, y_pred_all)

    print(f"\n[整体模型评估结果 - {target_col}]")
    print(f"最优参数: {best_params}")
    print(f"K折平均 MSE={mean_mse_cv:.3f}, RMSE={mean_rmse_cv:.3f}")
    print(f"整体模型 MSE={mse_all:.3f}, RMSE={rmse_all:.3f}, R²={r2_all:.3f}")

    feature_importances = best_model.feature_importances_
    print("\n特征重要性:")
    for feature, importance in zip(feature_cols, feature_importances):
        print(f"{feature}: {importance:.4f}")
    plot_residuals(y, y_pred_all, title=f"residuals plot（{target_col}）")

    return best_model, scaler
def train_and_predict_for_each_country(df, target_col='Total_Medals',
                                       future_year=2028,
                                       host_noc='USA',
                                       output_metrics_file='metrics_per_country.csv',
                                       output_pred_file='predict_2028_per_country.csv'):

    # 参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 4],
        'min_samples_leaf': [1, 2],
        'subsample': [1.0, 0.8]
    }

    feature_cols = ['Total_Events', 'Gender_Ratio', 'Is_Host', 'Historical_Medals', 'Total_Participants', 'Olympic_events']

    metrics_records = []

    predictions_2028 = []

    grouped = df.groupby('NOC')

    overall_model, overall_scaler = train_overall_model(df, target_col)

    for noc, group in grouped:
        country_data = group.dropna(subset=[target_col]).copy()
        if len(country_data) < 5:
            use_overall_model = True
        else:
            use_overall_model = False

        if use_overall_model:
            model = overall_model
            scaler = overall_scaler
        else:
            X = country_data[feature_cols].values
            y = country_data[target_col].values

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            gbrt = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                estimator=gbrt,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=5,
                n_jobs=-1
            )
            grid_search.fit(X_s, y)

            best_model = grid_search.best_estimator_

            mean_mse_cv = -grid_search.best_score_
            mean_rmse_cv = sqrt(mean_mse_cv)

            # 在全部数据上重训
            best_model.fit(X_s, y)
            y_pred_all = best_model.predict(X_s)
            mse_all = mean_squared_error(y, y_pred_all)
            rmse_all = sqrt(mse_all)
            r2_all = r2_score(y, y_pred_all)

            model = best_model

            metrics_records.append({
                'NOC': noc,
                'Best_Params': best_params,
                'CV_MSE': mean_mse_cv,
                'CV_RMSE': mean_rmse_cv,
                'Train_MSE': mse_all,
                'Train_RMSE': rmse_all,
                'Train_R2': r2_all
            })

        # 预测 2028
        group_sorted = group.sort_values(by='Year')
        row_2028 = {
            'Year': future_year,
            'NOC': noc
        }
        for feat in ['Total_Events', 'Total_Participants', 'Gender_Ratio', 'Olympic_events']:
            row_2028[feat] = simple_feature_extrapolation(group_sorted, feat, future_year)
        row_2028['Historical_Medals'] = group_sorted['Historical_Medals'].iloc[-1]
        row_2028['Is_Host'] = 1 if noc == host_noc else 0

        df_2028_single = pd.DataFrame([row_2028])
        X_2028_single = df_2028_single[feature_cols].values
        X_2028_s = scaler.transform(X_2028_single)

        pred_2028 = model.predict(X_2028_s)[0]
        if use_overall_model:
            residual_std = df[target_col].std()
        else:
            residuals = y - y_pred_all
            residual_std = np.std(residuals, ddof=1)

        ci_margin = 1.96 * residual_std
        lower_ci = pred_2028 - ci_margin
        upper_ci = pred_2028 + ci_margin

        actual_2024 = group.query('Year == 2024')
        if not actual_2024.empty:
            actual_2024_medals = actual_2024[target_col].values[0]
            diff = pred_2028 - actual_2024_medals
        else:
            diff = np.nan
        prob_first_medal = 0
        if target_col == 'Total_Medals':
            if row_2028['Historical_Medals'] == 0:
                z = (0 - pred_2028) / residual_std
                p = 1 - norm.cdf(z)
                prob_first_medal = p

        predictions_2028.append({
            'NOC': noc,
            f'Pred_{target_col}': pred_2028,
            f'{target_col}_LowerCI': lower_ci,
            f'{target_col}_UpperCI': upper_ci,
            '2024_Gold_Medals': group.query('Year == 2024')[target_col].values[0] if not group.query('Year == 2024').empty else np.nan,
            'CI_Margin': ci_margin,
            'Prob_FirstMedal': prob_first_medal  # 对金牌无意义，可留0
        })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(output_metrics_file, index=False, encoding='utf-8-sig')
    print(f"[训练指标] 已保存到 {output_metrics_file}, 共 {len(metrics_df)} 条记录.")

    preds_df = pd.DataFrame(predictions_2028)

    preds_df.sort_values(by=f'Pred_{target_col}', ascending=False, inplace=True)
    preds_df.to_csv(output_pred_file, index=False, encoding='utf-8-sig')
    print(f"[2028年预测结果] 已保存到 {output_pred_file}, 共 {len(preds_df)} 条记录.")

    return metrics_df, preds_df


if __name__ == "__main__":
    #构建特征
    final_df = build_feature_dataset('merged_olympic_data.csv')
    print("构建好的数据集(前10行):\n", final_df.head(10))

    ##对每个国家单独训练 Gold_Medals

    metrics_gold, pred_gold = train_and_predict_for_each_country(
        df=final_df,
        target_col='Gold_Medals',
        future_year=2028,
        host_noc='USA',
        output_metrics_file='metrics_per_country_gold.csv',
        output_pred_file='predict_2028_per_country_gold.csv'
    )

    #对每个国家单独训练 Total_Medals
    metrics_total, pred_total = train_and_predict_for_each_country(
        df=final_df,
        target_col='Total_Medals',
        future_year=2028,
        host_noc='USA',
        output_metrics_file='metrics_per_country_total.csv',
        output_pred_file='predict_2028_per_country_total.csv'
    )

    df_first = pred_total.query('Prob_FirstMedal > 0.5')
    print(f"\n预测2028年将首度获得奖牌(概率>0.5)的国家数量: {df_first.shape[0]}")
    print("这些国家如下:")
    print(df_first[['NOC','Prob_FirstMedal',f'Pred_Total_Medals']].head(30))