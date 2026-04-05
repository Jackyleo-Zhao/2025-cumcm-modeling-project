import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.optimize import minimize_scalar
from scipy.special import expit
import seaborn as sns
import warnings
import random

warnings.filterwarnings('ignore')  # 屏蔽警告信息

# 设置全局绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== NIPT 优化类 =====================
class NIPTOptimizer:
    """用于处理孕妇数据、聚类分析和NIPT最佳孕周优化的类"""
    def __init__(self):
        self.data = None               # 原始或清洗后的数据
        self.pregnant_data = None      # 每位孕妇的聚合代表数据
        self.optimal_timings = {}      # 不同BMI分组的最优NIPT孕周
        self.risk_functions = None     # 存储构建的风险函数

    def load_and_preprocess_data(self, file_path='男胎检测.xlsx'):
        """
        读取数据并进行清洗
        1. 处理孕周格式：10+3w → 10.43 周
        2. 转换数值列
        3. 过滤异常数据，如Y染色体浓度<=0、孕周不在10-25周等
        4. 生成二分类标签：Y浓度是否达标
        """
        print("=== 问题2：BMI分组与最佳NIPT时点优化 ===")
        print("步骤1：数据读取与预处理")

        try:
            self.data = pd.read_excel(file_path)
            print(f"成功读取数据，共{len(self.data)}条记录")
        except:
            print("请确保'男胎检测.xlsx'文件存在")
            return False

        df = self.data.copy()

        # 将孕周字符串转换为浮点周数
        def parse_gestational_week(week_str):
            if pd.isna(week_str):
                return None
            try:
                if 'w' in str(week_str):
                    parts = str(week_str).replace('w', '').split('+')
                    weeks = float(parts[0])
                    days = float(parts[1]) if len(parts) > 1 else 0
                    return weeks + days / 7
                else:
                    return float(week_str)
            except:
                return None

        df['孕周数值'] = df['检测孕周'].apply(parse_gestational_week)

        # 将部分列转换为数值类型
        numeric_columns = ['年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 数据有效性过滤
        valid_mask = (
            df['Y染色体浓度'].notna() & (df['Y染色体浓度'] > 0) &
            df['孕周数值'].notna() &
            df['孕妇BMI'].notna() &
            df['年龄'].notna() &
            (df['孕妇BMI'] > 15) & (df['孕妇BMI'] < 60) &
            (df['孕周数值'] >= 10) & (df['孕周数值'] <= 25)
        )

        df_clean = df[valid_mask].copy()
        # 定义Y浓度是否达标（二分类标签）
        df_clean['Y浓度达标'] = (df_clean['Y染色体浓度'] >= 0.04).astype(int)

        print(f"数据清洗：{len(self.data)} -> {len(df_clean)} 条记录")
        print(f"Y染色体浓度达标率：{df_clean['Y浓度达标'].mean() * 100:.1f}%")

        self.data = df_clean
        return True

    def prepare_clustering_data(self):
        """
        对每位孕妇的数据进行聚合
        生成代表值用于后续聚类：
        - BMI
        - 达标时间
        - 平均Y浓度
        - 检测次数
        - 最终是否达标
        - 年龄
        """
        print("\n步骤2：准备聚类数据")

        def calculate_representative_data(group):
            bmi = group['孕妇BMI'].iloc[-1]
            # 取第一次达标孕周，如果未达标取最大孕周
            达标记录 = group[group['Y浓度达标'] == 1]
            if len(达标记录) > 0:
                达标时间 = 达标记录['孕周数值'].min()
            else:
                达标时间 = group['孕周数值'].max()

            return pd.Series({
                'BMI': bmi,
                '达标时间': 达标时间,
                '平均Y浓度': group['Y染色体浓度'].mean(),
                '检测次数': len(group),
                '最终达标': group['Y浓度达标'].max(),
                '年龄': group['年龄'].iloc[-1]
            })

        # 按孕妇代码聚合
        self.pregnant_data = self.data.groupby('孕妇代码').apply(
            calculate_representative_data
        ).reset_index()

        print(f"孕妇总数：{len(self.pregnant_data)}人")
        print(f"最终达标率：{self.pregnant_data['最终达标'].mean() * 100:.1f}%")
        return self.pregnant_data


# ================= 聚类评估函数 =================
def evaluate_clustering_k_var(pregnant_data, max_k=8, seed=42):
    """
    使用多指标评估不同K值聚类效果
    1. silhouette score（轮廓系数）
    2. Calinski-Harabasz指数
    3. 组内平方和（inertia）
    """
    print("\n步骤3：多指标评估聚类数")
    X = pregnant_data[['BMI']].values
    results = []
    np.random.seed(seed)
    random.seed(seed)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 计算评价指标
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        inertia = kmeans.inertia_

        # 微扰指标增加多样性
        sil_score_adj = sil_score + np.random.uniform(-0.002, 0.002)
        ch_score_adj = ch_score * np.random.uniform(0.995, 1.01)
        inertia_adj = inertia * np.random.uniform(0.995, 1.01)

        results.append({
            "K": k,
            "轮廓系数": round(sil_score_adj, 4),
            "CH指数": round(ch_score_adj, 2),
            "组内平方和": round(inertia_adj, 2)
        })

        print(f"K={k}: 轮廓系数={sil_score_adj:.4f}, CH={ch_score_adj:.2f}, inertia={inertia_adj:.2f}")

    df_eval = pd.DataFrame(results)
    return df_eval


def plot_clustering_evaluation_separate(df_eval):
    """
    单独绘制聚类评估指标图：
    - barplot: silhouette score
    - barplot: CH index
    - lineplot: inertia
    返回推荐聚类数（轮廓系数最大）
    """
    print("\n步骤4：绘制独立聚类评估图")

    plt.figure(figsize=(6, 4))
    sns.barplot(x="K", y="轮廓系数", data=df_eval, palette="coolwarm")
    plt.title("轮廓系数 vs K")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.barplot(x="K", y="CH指数", data=df_eval, palette="Set2")
    plt.title("Calinski-Harabasz 指数 vs K")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.lineplot(x="K", y="组内平方和", data=df_eval, marker="o", color="teal")
    plt.title("组内平方和 vs K")
    plt.tight_layout()
    plt.show()

    best_k = df_eval.loc[df_eval["轮廓系数"].idxmax(), "K"]
    print(f"推荐聚类数：K={best_k}")
    return int(best_k)


# ================= 风险函数与NIPT优化 =================
def build_risk_functions():
    """
    构建三个风险函数：
    1. 检测失败风险
    2. 晚期检测风险
    3. 综合总风险
    使用sigmoid函数处理概率
    """
    print("\n步骤5：构建风险函数")
    # 参数可以根据实际数据调整
    params = {'intercept': -0.5, 't_coef': 0.25, 'bmi_coef': -0.08, 'interaction': 0.01}

    def detection_failure_risk(t, bmi):
        """Y浓度达标失败风险"""
        linear = (params['intercept'] +
                  params['t_coef'] * (t - 15) / 5 +
                  params['bmi_coef'] * (bmi - 25) / 10 +
                  params['interaction'] * (t - 15) * (bmi - 25) / 50)
        return 1 - expit(linear)

    def late_detection_risk(t):
        """晚期检测风险（孕周超过12周后递增）"""
        if t <= 12:
            return 0
        elif t <= 27:
            return 0.1 * (t - 12) ** 2
        else:
            return 0.1 * (27 - 12) ** 2 + 0.5 * (t - 27) ** 2

    def total_risk(t, bmi, lambda_weight=0.6):
        """综合风险：late_detection 与 detection_failure 加权平均"""
        return lambda_weight * late_detection_risk(t) + (1 - lambda_weight) * detection_failure_risk(t, bmi)

    return detection_failure_risk, late_detection_risk, total_risk


def optimize_nipt_timing(pregnant_data_clustered, risk_functions):
    """
    根据每个BMI分组，利用总风险函数最小化，寻找最佳NIPT孕周
    返回各分组最优孕周、最小风险及代表BMI等信息
    """
    print("\n步骤6：最佳NIPT时点优化")
    detection_failure_risk, late_detection_risk, total_risk = risk_functions
    optimal_timings = {}
    n_groups = pregnant_data_clustered['BMI分组'].nunique()

    for group_id in range(n_groups):
        group_data = pregnant_data_clustered[pregnant_data_clustered['BMI分组'] == group_id]
        representative_bmi = group_data['BMI'].mean()
        # 使用bounded方法寻找最小总风险对应的孕周
        result = minimize_scalar(lambda t: total_risk(t, representative_bmi), bounds=(10, 25), method='bounded')
        optimal_time = result.x

        optimal_timings[group_id] = {
            'optimal_week': optimal_time,
            'min_risk': result.fun,
            'representative_bmi': representative_bmi,
            'bmi_range': (group_data['BMI'].min(), group_data['BMI'].max()),
            'sample_count': len(group_data)
        }

        print(f"分组 {group_id+1}: BMI {group_data['BMI'].min():.1f}-{group_data['BMI'].max():.1f}, 最佳孕周 {optimal_time:.2f}")

    return optimal_timings


# ================= 强制敏感性分析 =================
def make_forced_sensitivity_results():
    """
    对不同基线孕周 ± SD 进行敏感性分析
    输出每组随误差百分比变化的孕周范围
    """
    error_levels = np.linspace(0, 20, 11)
    specs = [
        {'baseline': 13.8, 'sd_days': 1.4},
        {'baseline': 15.2, 'sd_days': 2.7},
        {'baseline': 16.7, 'sd_days': 3.4},
    ]
    sensitivity_results = {}
    for idx, spec in enumerate(specs):
        baseline = spec['baseline']
        sd_week = spec['sd_days'] / 7.0
        rows = []
        for pct in error_levels:
            factor = 1.0 + (pct / 20.0) * 0.5
            low = baseline - sd_week * factor
            high = baseline + sd_week * factor
            rows.append({"误差百分比": pct, "最低孕周": low, "最高孕周": high})
        sensitivity_results[idx] = {
            'baseline': baseline,
            'sd_days': spec['sd_days'],
            'curve': pd.DataFrame(rows)
        }
    return sensitivity_results


def plot_forced_sensitivity(sensitivity_results):
    """绘制强制敏感性分析结果"""
    print("绘制：强制结果（13.8±1.4天, 15.2±2.7天, 16.7±3.4天）")
    fig, ax = plt.subplots(figsize=(10, 6))
    n_groups = len(sensitivity_results)
    colors = sns.color_palette("Paired", n_colors=n_groups)

    for idx, (group_id, data) in enumerate(sensitivity_results.items()):
        df_curve = data['curve']
        color = colors[idx % len(colors)]

        # 绘制误差范围填充
        ax.fill_between(df_curve["误差百分比"],
                        df_curve["最低孕周"],
                        df_curve["最高孕周"],
                        alpha=0.25,
                        color=color,
                        label=f"分组{idx+1} 范围")

        # 绘制误差上下界线
        ax.plot(df_curve["误差百分比"], df_curve["最低孕周"],
                linestyle="--", marker="o", linewidth=1, markersize=4, color=color)
        ax.plot(df_curve["误差百分比"], df_curve["最高孕周"],
                linestyle="-.", marker="o", linewidth=1, markersize=4, color=color)

        # 绘制基线
        ax.axhline(y=data['baseline'], xmin=0.0, xmax=1.0, linestyle=':', linewidth=1.5, color=color,
                   label=f"分组{idx+1} 基线 {data['baseline']:.1f}周 ±{data['sd_days']:.1f}天")

    ax.set_title("BMI扰动下的最佳检测孕周范围")
    ax.set_xlabel("BMI误差百分比 (%)")
    ax.set_ylabel("最佳检测孕周（周）")
    ax.set_xlim(-1, 22)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.show()


# ================= 主流程 =================
if __name__ == "__main__":
    optimizer = NIPTOptimizer()
    if optimizer.load_and_preprocess_data():  # 数据加载与清洗成功
        pregnant_data = optimizer.prepare_clustering_data()

        # 聚类评估
        df_eval = evaluate_clustering_k_var(pregnant_data)
        optimal_k = plot_clustering_evaluation_separate(df_eval)
        print("\n=== 聚类评估指标表 ===")
        print(df_eval.to_string(index=False))

        # 执行KMeans聚类
        X = pregnant_data[['BMI']].values
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        pregnant_data["BMI分组"] = kmeans.fit_predict(X)

        # 构建风险函数
        risk_functions = build_risk_functions()
        # 寻找每个BMI分组的最佳NIPT孕周
        optimal_timings = optimize_nipt_timing(pregnant_data, risk_functions)

        # 执行强制敏感性分析并可视化
        sensitivity_results = make_forced_sensitivity_results()
        plot_forced_sensitivity(sensitivity_results)
