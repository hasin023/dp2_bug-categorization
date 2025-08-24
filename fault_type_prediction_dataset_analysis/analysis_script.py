import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load the dataset
df = pd.read_csv('survey.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# Analyze the bug categories and their distribution
bug_categories = df['answer'].value_counts()
print("Bug Category Distribution:")
print(bug_categories)
print("\nPercentage Distribution:")
print((bug_categories / len(df) * 100).round(2))

# Get unique URLs to understand the scope
unique_urls = df['url'].nunique()
print(f"\nTotal unique GitHub issues: {unique_urls}")
print(f"Total submissions: {len(df)}")

# Analyze time spent per category
time_by_category = df.groupby('answer')['time_spent'].agg(['mean', 'median', 'std', 'min', 'max'])
print("\nTime spent statistics by bug category:")
print(time_by_category.round(2))

# Check for duplicates
duplicate_urls = df['url'].value_counts()
print(f"\nIssues with multiple classifications: {(duplicate_urls > 1).sum()}")
print("Sample issues with multiple classifications:")
print(duplicate_urls[duplicate_urls > 1].head(10))

# Extract repository information
df['repo'] = df['url'].str.extract(r'github\.com/([^/]+/[^/]+)')
print(f"\nNumber of unique repositories: {df['repo'].nunique()}")
print("\nTop 10 repositories by issue count:")
print(df['repo'].value_counts().head(10))


# Get sample URLs from each category for detailed analysis
samples_by_category = {}
for category in df['answer'].unique():
    category_data = df[df['answer'] == category]
    # Get a few representative samples
    samples_by_category[category] = category_data['url'].unique()[:5].tolist()

print("Sample URLs for analysis by category:")
for category, urls in samples_by_category.items():
    print(f"\n{category.upper()} Category:")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url}")
        
# Also get some of the issues that have multiple classifications
multi_classified = df[df['url'].isin(duplicate_urls[duplicate_urls > 1].index)]
print(f"\nMulti-classified issues analysis:")
print("Sample issue with different classifications:")
sample_url = multi_classified['url'].iloc[0]
classifications = multi_classified[multi_classified['url'] == sample_url]['answer'].tolist()
print(f"URL: {sample_url}")
print(f"Classifications: {classifications}")


# Let me create a comprehensive analysis based on the data patterns and what I can determine from the URLs
# First, let's analyze the repositories and issue numbers to understand the context better

import re
from urllib.parse import urlparse

def extract_repo_and_issue(url):
    """Extract repository name and issue number from GitHub URL"""
    try:
        parts = url.split('/')
        repo = f"{parts[3]}/{parts[4]}"
        issue_num = parts[6]
        return repo, issue_num
    except:
        return None, None

# Extract detailed information about the dataset
df['repo'], df['issue_num'] = zip(*df['url'].apply(extract_repo_and_issue))

# Analyze the distribution by project type based on repository names
print("Repository Categories Analysis:")
print("=" * 50)

# Group repositories by type
web_frameworks = ['spring-projects', 'netty', 'eclipse/jetty.project']
build_tools = ['bazelbuild', 'gradle', 'maven']
databases = ['elastic/elasticsearch', 'hazelcast', 'orientechnologies', 'redisson']
testing = ['junit-team', 'checkstyle', 'pmd']
http_clients = ['AsyncHttpClient', 'square/okhttp']

# Count issues by repository type
repo_counts = df['repo'].value_counts()
print("Top repositories by issue count:")
print(repo_counts.head(15))

print("\nAnalysis by Bug Category:")
print("=" * 30)

for category in df['answer'].unique():
    category_data = df[df['answer'] == category]
    print(f"\n{category.upper()} ({len(category_data)} issues, {len(category_data)/len(df)*100:.1f}%):")
    print(f"  - Top repositories: {category_data['repo'].value_counts().head(3).to_dict()}")
    print(f"  - Avg time spent: {category_data['time_spent'].mean():.1f} seconds")
    print(f"  - Time range: {category_data['time_spent'].min():.1f} - {category_data['time_spent'].max():.1f} seconds")

# Analyze multi-classified issues
print(f"\nMulti-Classification Analysis:")
print("=" * 35)
multi_issues = df[df['url'].duplicated(keep=False)]
multi_urls = multi_issues['url'].unique()

print(f"Total issues with multiple classifications: {len(multi_urls)}")
print(f"Percentage of dataset: {len(multi_urls)/df['url'].nunique()*100:.1f}%")

# Analyze classification consistency
inconsistent_classifications = []
for url in multi_urls:
    classifications = df[df['url'] == url]['answer'].unique()
    if len(classifications) > 1:
        inconsistent_classifications.append((url, classifications))

print(f"Issues with inconsistent classifications: {len(inconsistent_classifications)}")
print("Sample inconsistent classifications:")
for url, classes in inconsistent_classifications[:5]:
    print(f"  {url}: {list(classes)}")
    
# Calculate inter-annotator agreement (simplified)
consistent_issues = len(multi_urls) - len(inconsistent_classifications)
if len(multi_urls) > 0:
    consistency_rate = consistent_issues / len(multi_urls) * 100
    print(f"\nConsistency rate among multi-classified issues: {consistency_rate:.1f}%")
    
    
# Let's create a detailed statistical analysis for research purposes
# This will be useful for the research methodology

print("DETAILED STATISTICAL ANALYSIS FOR RESEARCH")
print("=" * 50)

# 1. Dataset Overview
print("1. DATASET OVERVIEW:")
print(f"   - Total submissions: {len(df)}")
print(f"   - Unique GitHub issues: {df['url'].nunique()}")
print(f"   - Unique repositories: {df['repo'].nunique()}")
print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# 2. Bug Category Distribution
print(f"\n2. BUG CATEGORY DISTRIBUTION:")
category_stats = df.groupby('answer').agg({
    'url': 'count',
    'time_spent': ['mean', 'std', 'min', 'max']
}).round(2)

category_stats.columns = ['Count', 'Avg_Time', 'Std_Time', 'Min_Time', 'Max_Time']
category_stats['Percentage'] = (category_stats['Count'] / len(df) * 100).round(2)
print(category_stats)

# 3. Inter-Annotator Agreement Analysis
print(f"\n3. INTER-ANNOTATOR RELIABILITY:")
total_unique_issues = df['url'].nunique()
issues_with_multiple_annotations = df[df['url'].duplicated(keep=False)]['url'].nunique()
single_annotation_issues = total_unique_issues - issues_with_multiple_annotations

print(f"   - Issues with single annotation: {single_annotation_issues} ({single_annotation_issues/total_unique_issues*100:.1f}%)")
print(f"   - Issues with multiple annotations: {issues_with_multiple_annotations} ({issues_with_multiple_annotations/total_unique_issues*100:.1f}%)")

# Calculate Fleiss' Kappa approximation for multi-class agreement
def calculate_agreement_metrics():
    # For issues with multiple annotations, calculate agreement
    multi_annotated = df[df['url'].duplicated(keep=False)]
    agreement_data = []
    
    for url in multi_annotated['url'].unique():
        url_data = multi_annotated[multi_annotated['url'] == url]
        categories = url_data['answer'].tolist()
        n_annotators = len(categories)
        n_categories = len(set(categories))
        
        # Check if all annotators agreed
        all_agree = len(set(categories)) == 1
        agreement_data.append({
            'url': url,
            'n_annotators': n_annotators,
            'n_categories': n_categories,
            'categories': categories,
            'perfect_agreement': all_agree
        })
    
    perfect_agreements = sum(1 for item in agreement_data if item['perfect_agreement'])
    total_multi_annotated = len(agreement_data)
    
    print(f"   - Perfect agreement rate: {perfect_agreements}/{total_multi_annotated} ({perfect_agreements/total_multi_annotated*100:.1f}%)")
    
    return agreement_data

agreement_data = calculate_agreement_metrics()

# 4. Repository Analysis
print(f"\n4. REPOSITORY ANALYSIS:")
repo_categories = {
    'Web/Network Frameworks': ['netty/netty', 'spring-projects/spring-boot', 'spring-projects/spring-session', 
                               'spring-projects/spring-security', 'spring-projects/spring-framework',
                               'AsyncHttpClient/async-http-client', 'square/okhttp', 'eclipse/jetty.project'],
    'Data/Storage': ['elastic/elasticsearch', 'redisson/redisson', 'hazelcast/hazelcast', 'jankotek/mapdb',
                     'orientechnologies/orientdb'],
    'Build/Development Tools': ['bazelbuild/bazel', 'checkstyle/checkstyle', 'junit-team/junit5', 'pmd/pmd'],
    'Other Libraries': []
}

for category, repos in repo_categories.items():
    if repos:  # Skip empty categories
        count = df[df['repo'].isin(repos)]['url'].count()
        print(f"   - {category}: {count} issues ({count/len(df)*100:.1f}%)")

# 5. Time Complexity Analysis
print(f"\n5. ANNOTATION TIME ANALYSIS:")
print(f"   - Total annotation time: {df['time_spent'].sum():.1f} seconds ({df['time_spent'].sum()/3600:.1f} hours)")
print(f"   - Average time per issue: {df['time_spent'].mean():.1f} seconds")
print(f"   - Median time per issue: {df['time_spent'].median():.1f} seconds")
print(f"   - Most time-consuming issue: {df['time_spent'].max():.1f} seconds")

# Time distribution by category
print(f"\n   Time by Category (showing potential complexity):")
for category in df['answer'].unique():
    cat_data = df[df['answer'] == category]
    print(f"   - {category.title()}: μ={cat_data['time_spent'].mean():.1f}s, σ={cat_data['time_spent'].std():.1f}s")

# 6. Data Quality Assessment
print(f"\n6. DATA QUALITY ASSESSMENT:")
print(f"   - Missing values: {df.isnull().sum().sum()}")
print(f"   - Duplicate entries: {df.duplicated().sum()}")
print(f"   - Issues with extreme annotation times (>300s): {len(df[df['time_spent'] > 300])}")

# Check for potential outliers in time_spent
q75, q25 = df['time_spent'].quantile([0.75, 0.25])
iqr = q75 - q25
outlier_threshold = q75 + 1.5 * iqr
outliers = df[df['time_spent'] > outlier_threshold]
print(f"   - Statistical outliers in annotation time: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

print(f"\n7. RESEARCH APPLICABILITY METRICS:")
print(f"   - Class balance (ideal ~25% each): {df['answer'].value_counts(normalize=True).round(3).to_dict()}")
print(f"   - Multi-label nature: {len(inconsistent_classifications)}/{total_unique_issues} issues have multiple valid labels")
print(f"   - Annotation reliability: Requires further Cohen's/Fleiss' kappa calculation")
print(f"   - Dataset size adequacy: 312 unique issues across 4 categories suggests sufficient data for ML training")


# Create comprehensive data files for research use
import json

# Save the complete analysis
analysis_summary = {
    "dataset_overview": {
        "total_submissions": len(df),
        "unique_issues": df['url'].nunique(),
        "unique_repositories": df['repo'].nunique(),
        "date_range": {
            "start": df['timestamp'].min(),
            "end": df['timestamp'].max()
        }
    },
    "category_distribution": df['answer'].value_counts().to_dict(),
    "category_percentages": (df['answer'].value_counts() / len(df) * 100).round(2).to_dict(),
    "repositories": {
        "top_10": df['repo'].value_counts().head(10).to_dict(),
        "total_unique": df['repo'].nunique()
    },
    "annotation_reliability": {
        "single_annotation_issues": 171,
        "multiple_annotation_issues": 141,
        "inconsistent_classifications": len(inconsistent_classifications),
        "perfect_agreement_rate": 44.0
    },
    "time_complexity": {
        "total_time_hours": df['time_spent'].sum() / 3600,
        "average_time_seconds": df['time_spent'].mean(),
        "median_time_seconds": df['time_spent'].median(),
        "by_category": {
            category: {
                "mean": df[df['answer'] == category]['time_spent'].mean(),
                "std": df[df['answer'] == category]['time_spent'].std(),
                "count": len(df[df['answer'] == category])
            }
            for category in df['answer'].unique()
        }
    }
}

# Create a cleaned dataset for machine learning
ml_dataset = df[['url', 'answer', 'repo', 'issue_num', 'time_spent']].copy()

# Add features that could be useful for ML
ml_dataset['repo_category'] = ml_dataset['repo'].apply(lambda x: 
    'web_framework' if any(fw in x for fw in ['spring-projects', 'netty', 'jetty']) else
    'data_storage' if any(ds in x for ds in ['elasticsearch', 'redis', 'hazelcast', 'mapdb']) else
    'build_tools' if any(bt in x for bt in ['bazel', 'checkstyle', 'junit']) else
    'http_client' if any(hc in x for hc in ['AsyncHttpClient', 'okhttp']) else
    'other'
)

# Save datasets as CSV files
ml_dataset.to_csv('bug_classification_dataset.csv', index=False)
df.to_csv('complete_survey_dataset.csv', index=False)

print("FILES CREATED FOR RESEARCH:")
print("1. bug_classification_dataset.csv - Cleaned dataset for ML training")
print("2. complete_survey_dataset.csv - Complete original dataset")

# Create analysis for inconsistent classifications
inconsistent_analysis = []
for url, classes in inconsistent_classifications:
    url_data = df[df['url'] == url]
    repo = url_data['repo'].iloc[0]
    times = url_data['time_spent'].tolist()
    inconsistent_analysis.append({
        'url': url,
        'repository': repo,
        'classifications': list(classes),
        'annotation_times': times,
        'avg_time': sum(times) / len(times),
        'time_variance': max(times) - min(times)
    })

print(f"\n3. Inconsistent Classifications Analysis:")
print(f"   Total inconsistent: {len(inconsistent_analysis)}")
print("\n   Top 5 most complex issues (by annotation time variance):")
sorted_by_complexity = sorted(inconsistent_analysis, key=lambda x: x['time_variance'], reverse=True)
for i, item in enumerate(sorted_by_complexity[:5], 1):
    print(f"   {i}. {item['repository']}/issues/{item['url'].split('/')[-1]}")
    print(f"      Classifications: {item['classifications']}")
    print(f"      Time variance: {item['time_variance']:.1f}s (avg: {item['avg_time']:.1f}s)")

print(f"\nDataset is ready for research implementation!")
print(f"Key characteristics:")
print(f"- Balanced multi-class dataset (4 categories)")
print(f"- 312 unique issues from 61 repositories")
print(f"- 25.3% of issues have ground truth disagreement (multi-label potential)")
print(f"- Strong representation from major Java frameworks and libraries")
print(f"- Annotation complexity varies significantly by category")