import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, KBinsDiscretizer
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Set seaborn style for better visualizations
sns.set()


def distribution_value(nrow, ncols, data, x):
    """
    Plot distribution histograms for specified columns.

    Parameters:
    -----------
    nrow : int
        Number of rows in subplot grid
    ncols : int
        Number of columns in subplot grid
    data : pd.DataFrame
        Input dataframe
    x : list
        List of column names to plot
    """
    fig, axes = plt.subplots(nrow, ncols, figsize=(12, 5))

    if nrow * ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(x):
        sns.histplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')

    plt.tight_layout()
    plt.show()


# Initialize imputers for categorical and numerical data
impute_category = SimpleImputer(strategy='most_frequent')
impute_numerical = SimpleImputer(strategy='median')

# Define column names for echocardiogram dataset
column_names = ['Survival', 'Still-alive', 'Age-at-heart-attack',
                'Pericardial-effusion', 'Fractional-shortening', 'Epss', 'Lvdd',
                'Wall-motion-score', 'Wall-motion-index']

# Load dataset with missing values marked as '?'
df = pd.read_csv('../data/echocardiogram.csv',
                 usecols=column_names, na_values='?')

# Create a copy for data processing
data_processed = df.copy()

# Display raw data
print(data_processed)

# Impute missing values based on feature cardinality
for col in data_processed.columns:
    if data_processed[col].nunique() < 5:
        # Categorical features: use most frequent value
        data_processed[col] = impute_category.fit_transform(data_processed[[col]])
    else:
        # Numerical features: use median value
        data_processed[col] = impute_numerical.fit_transform(data_processed[[col]])

# Convert data types
data_processed['Still-alive'] = data_processed['Still-alive'].astype(int)
data_processed['Age-at-heart-attack'] = data_processed['Age-at-heart-attack'].astype(int)
data_processed["Survived_Year"] = (data_processed["Survival"] / 12).round(1)

# Print comprehensive statistical summary
print("Comprehensive Statistical Overview")
statistics_summary = data_processed.describe().T
statistics_summary["Skewness"] = data_processed.skew()
statistics_summary["Kurtosis"] = data_processed.kurt()
print(statistics_summary.to_string())

# Drop low-variance feature
data_processed = data_processed.drop("Pericardial-effusion", axis=1)

# Create age groups for analysis
data_processed['Age_at_heart_attack_group'] = pd.cut(data_processed['Age-at-heart-attack'],
                                                     bins=5, precision=0)

# Calculate average survival by age group
avg_survival_by_age = data_processed.groupby('Age_at_heart_attack_group', observed=True)['Survived_Year'].mean()

# Count survived patients by age group
survived_count_by_age = (data_processed[data_processed['Still-alive'] == 1]
                         .groupby('Age_at_heart_attack_group', observed=True)
                         .size()
                         .reindex(avg_survival_by_age.index, fill_value=0))

# Create dual-axis plot: survival and patient count by age group
plt.figure(figsize=(12, 6))

age_labels = avg_survival_by_age.astype(str)
x_positions = range(len(age_labels))

# Primary axis: Average survival in years
ax1 = plt.gca()
ax1.plot(x_positions, avg_survival_by_age.values, marker='o', linewidth=2.5,
         markersize=12, color='darkred', label='Avg Survival (Years)')
ax1.set_ylabel('Average Survival (Years)', color='darkred', fontsize=14)
ax1.tick_params(axis='y', labelcolor='darkred')

# Secondary axis: Number of survived patients
ax2 = ax1.twinx()
ax2.plot(x_positions, survived_count_by_age.values, marker='s', linewidth=2.5,
         markersize=12, color='green', linestyle='--', label='Survived Patients')
ax2.set_ylabel('Number of Survived Patients', color='green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='green')

# Set x-axis labels
plt.xticks(x_positions, age_labels, rotation=45)
plt.xlabel('Age Group at Heart Attack', fontsize=14)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Add value annotations on data points
for i, (survival_val, alive_val) in enumerate(zip(avg_survival_by_age.values, survived_count_by_age.values)):
    ax1.text(i, survival_val + 0.1, f'{survival_val:.1f}', ha='center', fontweight='bold', color='darkred')
    ax2.text(i, alive_val + 0.3, str(alive_val), ha='center', fontweight='bold', color='green')

plt.title('Survived Patients & Their Average Survival by Age Group', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# Create cross-tabulation of age groups vs survival status
mortality_crosstab = pd.crosstab(data_processed['Age_at_heart_attack_group'], data_processed['Still-alive'])

# Stacked bar chart: Dead vs Alive by Age Group
mortality_crosstab.plot(kind='bar', stacked=True, figsize=(10, 5), color=['red', 'green'])

plt.title('Dead vs Alive by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(['Dead After Attack', 'Alive After Attack'])
plt.xticks(rotation=0)
plt.show()

# Correlation analysis (drop categorical group column)
data_for_correlation = data_processed.drop("Age_at_heart_attack_group", axis=1)
correlation_matrix = data_for_correlation.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
heatmap.set_title("Feature Correlation Heatmap")
plt.show()

# Scatter plot: Wall Motion Index vs Survival
plt.scatter(data=data_processed, x="Wall-motion-index", y="Survival")
plt.xlabel('Wall-motion-index')
plt.ylabel("Survival")
plt.show()

# Select features for clustering
features_to_scale = ['Wall-motion-index', 'Fractional-shortening', 'Lvdd', 'Epss']

# Standardize features
scaler = StandardScaler()
scaled_features = data_processed[features_to_scale]
scaled_features = scaler.fit_transform(scaled_features)

# Calculate WCSS and Silhouette scores for different cluster numbers
wcss_scores = []
silhouette_scores = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    wcss_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))

# Plot Elbow Method and Silhouette Score
number_clusters = range(2, 10)
fig, ax1 = plt.subplots(figsize=(12, 6))

# WCSS curve
ax1.plot(number_clusters, wcss_scores, marker="o", linestyle="--", color="b")
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS", color="b")
ax1.tick_params(axis="y", labelcolor="b")

# Silhouette score curve
ax2 = ax1.twinx()
ax2.plot(number_clusters, silhouette_scores, marker="o", linestyle="--", color="r")
ax2.set_ylabel("Silhouette Score", color="r")
ax2.tick_params(axis="y", labelcolor="r")

plt.title("Elbow Method and Silhouette Score for Optimal Number of Clusters")
plt.grid(True)
plt.show()

# Perform K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data = data_processed[features_to_scale].copy()
clustering_data["Cluster"] = kmeans.fit_predict(scaled_features)
clustering_data["Survival"] = data_processed["Survival"].values
clustering_data = clustering_data.sort_values(by="Survival")

# Calculate mean survival by cluster
cluster_mean_survival = clustering_data.groupby("Cluster")["Survival"].mean().reset_index()

# Map clusters to risk categories
cluster_map = {1: 'High Risk', 0: 'Medium Risk', 2: 'Low Risk'}
cluster_mean_survival["Cluster"] = cluster_mean_survival["Cluster"].map(cluster_map)
cluster_mean_survival = cluster_mean_survival.sort_values(by="Survival")


# Bar plot: Mean survival by cluster
plt.figure(figsize=(8, 5))
sns.barplot(
    x='Cluster',
    y='Survival',
    data=cluster_mean_survival,
    hue="Cluster",
    palette=['red', 'orange', 'green'],
    legend=False
)

plt.title('Mean Survival by Cluster', fontsize=14, fontweight='bold')
plt.xlabel('Cluster')
plt.ylabel('Mean Survival (months)')

# Add value annotations
for i, val in enumerate(cluster_mean_survival['Survival']):
    plt.text(i, val + 0.5, f'{val:.1f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Map clusters in main dataframe to risk categories
clustering_data["Cluster"] = clustering_data["Cluster"].map(cluster_map)

# Interactive bar plot: Feature distribution by risk category
fig = px.bar(clustering_data,
             x="Cluster",
             y=features_to_scale,
             title="Feature Distribution by Risk Category",
             barmode="group",
             color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
             )

fig.show()

# Display raw dataframe info
print(df.head(50).to_string())
print(df.nunique())

# Prepare data for supervised learning
X = df.drop('Still-alive', axis=1)
y = df['Still-alive']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Check missing values in features
print(X_train.isna().sum())
print(X_test.isna().sum())

# Impute missing values in training and test features
for col in X_train.columns:
    if X_train[col].nunique() < 5:
        X_train[col] = impute_category.fit_transform(X_train[[col]])
        X_test[col] = impute_category.transform(X_test[[col]])
    else:
        X_train[col] = impute_numerical.fit_transform(X_train[[col]])
        X_test[col] = impute_numerical.transform(X_test[[col]])

# Impute missing values in target variable
y_train = y_train.to_frame()
y_test = y_test.to_frame()
for col in y_train.columns:
    y_train[col] = impute_category.fit_transform(y_train[[col]])
    y_train[col] = y_train[col].astype(int)
    y_test[col] = impute_category.fit_transform(y_test[[col]])
    y_test[col] = y_test[col].astype(int)

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

# Verify no missing values remain
print(X_train.isna().sum())
print(X_test.isna().sum())

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Plot feature distributions
distribution_value(3, 3, X_train, ['Survival', 'Age-at-heart-attack',
                                   'Pericardial-effusion', 'Fractional-shortening',
                                   'Epss', 'Lvdd', 'Wall-motion-score',
                                   'Wall-motion-index'])

# Scale features using RobustScaler
Scale = RobustScaler()
Scale.fit(X_train)
scale_train = Scale.transform(X_train)
scale_test = Scale.fit_transform(X_test)

# Train logistic regression model
logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(X_train_bal, y_train_bal)
y_hat = logistic_model.predict(X_test)

# Model evaluation
print(f'\nModel Accuracy: {logistic_model.score(X_test, y_test):.3f}')
print(classification_report(y_test, y_hat))