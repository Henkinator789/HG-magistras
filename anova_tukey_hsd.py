import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette('Blues')

# Read all XLSX files
folder_path = "tyrimu_rezultatai"
model_names = ["yolov3", "fasterrcnn", "rtmdet", "efficientdet", "ssd300"]

# Dictionary to store all data
all_data = {}

# Read each file
for model in model_names:
    file_path = os.path.join(folder_path, f"{model}.xlsx")
    try:
        df = pd.read_excel(file_path)
        all_data[model] = df
        print(f"Successfully loaded {model}.xlsx with {len(df)} rows")
    except Exception as e:
        print(f"Error loading {model}.xlsx: {e}")

# Combine all data
combined_data = []
for model, df in all_data.items():
    df_copy = df.copy()
    df_copy['Model'] = model
    combined_data.append(df_copy)

full_df = pd.concat(combined_data, ignore_index=True)
print(f"\nCombined dataset shape: {full_df.shape}")

# Performance metrics to analyze
metrics = ['P', 'R', 'F1', 'mAP50-95', 'mAP50']

# ANOVA RESULTS SUMMARY

def create_anova_summary_table(df, metrics):
    """Create a clean ANOVA summary table for thesis"""
    
    results_summary = []
    
    for metric in metrics:
        if metric in df.columns:
            row = {'Metric': metric}
            
            # Test each factor
            factors = ['Svoriai', 'Balansavimas', 'Duomenų papildymas', 'Model']
            
            for factor in factors:
                groups = [group[metric].values for name, group in df.groupby(factor)]
                f_stat, p_val = f_oneway(*groups)
                
                # Format significance
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                row[factor] = f"{f_stat:.2f} ({p_val:.3f}){sig}"
            
            results_summary.append(row)
    
    return pd.DataFrame(results_summary)

# Create ANOVA summary table
anova_table = create_anova_summary_table(full_df, metrics)
print("\n")
print("TABLE FOR THESIS: ANOVA RESULTS SUMMARY")
print("Format: F-statistic (p-value) significance")
print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n")
print(anova_table.to_string(index=False))

# MODEL PERFORMANCE COMPARISON

def plot_model_performance(df, metrics):
    """Create box plots comparing model performance"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            sns.boxplot(data=df, x='Model', y=metric, ax=axes[i], palette='Blues')
            axes[i].set_title(f'{metric} pagal modelį', fontsize=16, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].tick_params(axis='both', labelsize=16)
    
    # Remove empty subplot
    if len(metrics) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=180, bbox_inches='tight')
    plt.show()

plot_model_performance(full_df, metrics)

# EXPERIMENTAL FACTORS IMPACT

def plot_experimental_factors(df, metric='mAP50-95'):
    """Create subplots showing impact of each experimental factor"""
    
    if metric not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Factor 1: Pretrained weights
    sns.boxplot(data=df, x='Svoriai', y=metric, ax=axes[0], palette='Blues')
    axes[0].set_title('Perkėlimo mokymosi poveikis', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='both', labelsize=16)
    
    # Factor 2: Class balancing
    sns.boxplot(data=df, x='Balansavimas', y=metric, ax=axes[1], palette='Blues')
    axes[1].set_title('Klasių balansavimo poveikis', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='both', labelsize=16)
    
    # Factor 3: Data augmentation
    sns.boxplot(data=df, x='Duomenų papildymas', y=metric, ax=axes[2], palette='Blues')
    axes[2].set_title('Duomenų papildymo poveikis', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='both', labelsize=16)
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'experimental_factors_impact_{metric}.png', dpi=180, bbox_inches='tight')
    plt.show()

# Create factor impact plots for mAP50 (most important metric)
plot_experimental_factors(full_df, 'P')

# CORRELATION HEATMAP

def plot_metrics_correlation(df, metrics):
    """Create correlation heatmap between performance metrics"""
    
    # Select only numeric metrics
    metrics_data = df[metrics].select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    correlation_matrix = metrics_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Metrikų koreliacija', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('metrics_correlation_heatmap.png', dpi=180, bbox_inches='tight')
    plt.show()

plot_metrics_correlation(full_df, metrics)

# DESCRIPTIVE STATISTICS TABLE

def create_descriptive_stats_table(df, metric='mAP50'):
    """Create descriptive statistics table for key factors"""
    
    if metric not in df.columns:
        return None
    
    results = []
    
    # By Model
    model_stats = df.groupby('Model')[metric].agg(['mean', 'std', 'count']).round(3)
    for model, row in model_stats.iterrows():
        results.append({
            'Factor': 'Model',
            'Level': model,
            'Mean': row['mean'],
            'Std': row['std'],
            'Count': row['count']
        })
    
    # By Pretrained weights
    svoriai_stats = df.groupby('Svoriai')[metric].agg(['mean', 'std', 'count']).round(3)
    for level, row in svoriai_stats.iterrows():
        results.append({
            'Factor': 'Svoriai',
            'Level': level,
            'Mean': row['mean'],
            'Std': row['std'],
            'Count': row['count']
        })
    
    # By Class balancing
    balans_stats = df.groupby('Balansavimas')[metric].agg(['mean', 'std', 'count']).round(3)
    for level, row in balans_stats.iterrows():
        results.append({
            'Factor': 'Balansavimas',
            'Level': level,
            'Mean': row['mean'],
            'Std': row['std'],
            'Count': row['count']
        })
    
    return pd.DataFrame(results)

# Create descriptive statistics table
desc_stats_table = create_descriptive_stats_table(full_df, 'mAP50')
if desc_stats_table is not None:
    print("\n")
    print("TABLE FOR THESIS: DESCRIPTIVE STATISTICS (mAP50)")
    print(desc_stats_table.to_string(index=False))

# TUKEY HSD POST-HOC TESTS 

def perform_tukey_hsd(df, metric, factor):
    """Perform Tukey HSD post-hoc test using official scipy implementation"""
    
    print(f"\n")
    print(f"TUKEY HSD POST-HOC TEST FOR {factor} ON {metric}")
    
    # Group data by factor
    groups = df.groupby(factor)[metric].apply(list).to_dict()
    group_names = list(groups.keys())
    group_data = list(groups.values())
    
    # Calculate group means for reference
    means = {name: np.mean(values) for name, values in groups.items()}
    print("Group Means:")
    for name, mean in means.items():
        print(f"  {name}: {mean:.4f} (n={len(groups[name])})")
    
    # Perform Tukey HSD using official scipy function
    try:
        result = tukey_hsd(*group_data)
        
        print(f"\nPairwise Comparisons (α = 0.05):")
        print("Pair".ljust(35) + "p-value".ljust(12) + "Significant")
        
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names):
                if i < j:  # Only compare each pair once
                    p_value = result.pvalue[i, j]
                    significant = "Yes" if p_value < 0.05 else "No"
                    pair_name = f"{name1} vs {name2}"
                    print(f"{pair_name:<35}{p_value:>8.4f}     {significant}")
                    
    except Exception as e:
        print(f"Error performing Tukey HSD: {e}")


# KEY FINDINGS SUMMARY

print("\n")
print("KEY FINDINGS FOR THESIS")

# Analyze which factors have significant effects and perform post-hoc tests
significant_findings = []

for metric in metrics:
    if metric in full_df.columns:
        print(f"\n{metric} Results:")
        
        factors = ['Svoriai', 'Balansavimas', 'Duomenų papildymas', 'Model']
        
        for factor in factors:
            groups = [group[metric].values for name, group in full_df.groupby(factor)]
            f_stat, p_val = f_oneway(*groups)
            
            if p_val < 0.05:
                # Calculate effect size (eta-squared)
                ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 for group in groups)
                ss_total = sum((x - np.mean(np.concatenate(groups)))**2 for group in groups for x in group)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                effect_size = "Large" if eta_squared > 0.14 else "Medium" if eta_squared > 0.06 else "Small"
                
                print(f"  ✓ {factor}: F={f_stat:.2f}, p={p_val:.3f}, Effect size: {effect_size}")
                significant_findings.append((metric, factor, f_stat, p_val, effect_size))
                
                # Perform Tukey HSD for significant factors
                if metric in ['R', 'P', 'F1', 'mAP50-95', 'mAP50']:  # Only for key metrics to avoid too much output
                    perform_tukey_hsd(full_df, metric, factor)
                    
            else:
                print(f"  ✗ {factor}: Not significant (p={p_val:.3f})")

# Best performing configurations

print("BEST PERFORMING CONFIGURATIONS")

for metric in ['mAP50', 'F1']:  # Focus on most important metrics
    if metric in full_df.columns:
        best_idx = full_df[metric].idxmax()
        best_config = full_df.loc[best_idx]
        
        print(f"\nBest {metric}: {best_config[metric]:.3f}")
        print(f"  Model: {best_config['Model']}")
        print(f"  Svoriai: {best_config['Svoriai']}")
        print(f"  Balansavimas: {best_config['Balansavimas']}")
        print(f"  Duomenų papildymas: {best_config['Duomenų papildymas']}")
