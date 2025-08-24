## Dataset Analysis Script

**Purpose of the Script:**

1.  **Data Understanding:** Load and examine the `survey.csv` file to understand its structure, size, and content.
2.  **Exploratory Data Analysis (EDA):** Investigate the distribution of bug categories, time spent on classification, unique issues, repositories involved, and identify patterns or anomalies.
3.  **Quality Assessment:** Evaluate data quality, including checking for duplicates, missing values, and analyzing inter-annotator agreement (reliability) on issues classified multiple times.
4.  **Feature Engineering:** Extract relevant information (like repository name and issue number) from the GitHub URLs.
5.  **Dataset Preparation:** Clean and structure the data into a format suitable for machine learning tasks, potentially adding derived features.
6.  **Reporting & Export:** Summarize key findings, generate reports on data characteristics, and save cleaned/processed datasets for further research or model training.

### Section 1: Import Libraries and Load Data

- Imports necessary Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `collections`, `re`).
- Loads the `survey.csv` file into a Pandas DataFrame (`df`).

### Section 2: Initial Data Exploration

- Displays basic dataset information: shape, column names, first few rows (`df.head()`).
- Shows data types and potential missing values (`df.info()`).
- Provides basic descriptive statistics for numerical columns (`df.describe()`).

### Section 3: Core Analysis

- **Bug Category Analysis:**
  - Calculates the count and percentage distribution of each bug category (`df['answer'].value_counts()`).
  - Analyzes the time spent (`time_spent`) for each category (mean, median, std, min, max).
- **URL/Issue Analysis:**
  - Counts unique GitHub issue URLs (`df['url'].nunique()`) vs. total submissions.
  - Identifies issues that received multiple classifications (`df['url'].value_counts()`).
  - Extracts repository information from URLs (`df['repo'] = df['url'].str.extract(...)`).
  - Calculates the number of unique repositories and lists the top ones by issue count.
- **Sample Extraction:**
  - Retrieves a few sample URLs for each bug category for manual inspection.
  - Retrieves examples of issues with multiple classifications.

### Section 4: Detailed Analysis and Methodology Preparation

- Defines helper functions (e.g., `extract_repo_and_issue`).
- **Repository Categorization:** Attempts to group repositories into broader categories (e.g., Web Frameworks, Databases).
- **Multi-Classification Deep Dive:**
  - Analyzes issues with multiple annotations in detail.
  - Identifies cases where different classifications were given to the same issue (inconsistencies).
  - Calculates a basic consistency rate among multi-classified issues.
- **Statistical Summary for Research:**
  - Presents a structured overview of dataset metrics relevant for research (total issues, unique repos, date range).
  - Summarizes category distribution with counts, percentages, and time statistics.
  - Details inter-annotator reliability findings.
  - Analyzes repository types represented in the data.
  - Examines annotation time distribution overall and per category.
  - Assesses data quality (missing values, duplicates, outliers).
  - Evaluates dataset suitability for ML (class balance, multi-label nature, size).

### Section 5: Data Export and Finalization

- **Creates Summary Data:** Compiles key analysis results into a dictionary (`analysis_summary`).
- **Prepares ML Dataset:**
  - Creates a new DataFrame (`ml_dataset`) containing core columns.
  - Adds a derived feature (`repo_category`) based on the repository name.
- **Exports Files:**
  - Saves the prepared ML dataset (`bug_classification_dataset.csv`).
  - Saves the complete original dataset (`complete_survey_dataset.csv`).
- **Exports Analysis Results:**
  - Compiles detailed information about inconsistent classifications (`inconsistent_analysis`).
  - Prints a final summary highlighting the dataset's readiness and key characteristics for research/ML implementation.
