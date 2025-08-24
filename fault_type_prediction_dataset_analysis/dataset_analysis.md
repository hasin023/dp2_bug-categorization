# Survey.csv Dataset for LLM-based Classification

A manually annotated dataset of **510 GitHub issue classifications** across **312 unique issues** from **61 repositories**, providing a foundation for implementing **automatic bug categorization using Large Language Models (LLMs)**. The dataset contains four bug categories with significant annotation complexity and multi-label characteristics that make it ideal for advanced ML research.

## 1. Dataset Overview

### Core Statistics

- **Total Submissions**: 510 manual classifications
- **Unique Issues**: 312 GitHub issues
- **Repositories**: 61 distinct Java-based projects
- **Time Period**: December 2020 to May 2021
- **Total Annotation Time**: 5.2 hours (18,726 seconds)

### Bug Category Distribution

The dataset shows a **relatively balanced multi-class distribution**:

- **Semantic**: 164 issues (32.16%) - API misuse, incorrect logic, specification violations
- **Memory**: 125 issues (24.51%) - Memory leaks, OOM errors, resource management
- **Concurrency**: 102 issues (20.00%) - Thread safety, race conditions, deadlocks
- **Other**: 119 issues (23.33%) - Configuration, documentation, build issues

### Repository Analysis

The dataset primarily focuses on **enterprise Java frameworks and libraries**:

**Top Categories by Domain**:

- **Web/Network Frameworks** (35.9%): Spring Boot, Netty, Jetty, OkHttp
- **Data/Storage Systems** (19.4%): Elasticsearch, Redis, Hazelcast
- **Development Tools** (9.2%): Bazel, Checkstyle, JUnit

## 2. Some Issue Analysis and Classification Rationale

### Semantic Issues Example

**Issue**: `DiUS/pact-jvm/issues/196` - "NoSuchMethodError" during PactBroker loading

**Classification Rationale**:

- Root cause is **API version incompatibility** between HTTP client libraries
- Represents **semantic error** where code assumes newer API methods exist
- Typical of **interface contract violations** in dependency management

### Memory Issues Example

**Issue**: `spring-projects/spring-boot/issues/11338` - "spring-boot-starter-actuator caused OOM"

**Classification Rationale**:

- **OutOfMemoryError** caused by excessive `LatencyStats$PauseTracker` instances
- Classic **memory leak pattern** in monitoring/metrics collection
- Represents **resource management failure** with unbounded object creation

### Concurrency Issues Pattern

- Primarily involve **thread safety violations**, **race conditions**, and **deadlock scenarios**
- Often manifest in high-throughput systems (Netty, Hazelcast)
- **Shortest average annotation time** (20.2s) suggests clearer identification patterns

### Other Issues Pattern

- **Configuration errors**, **build system problems**, **documentation issues**
- **Highest annotation time variance** (σ=187.4s) indicating diverse problem types
- Often require **domain-specific knowledge** for proper classification

## 3. Annotation Reliability and Multi-Label Nature

### Inter-Annotator Agreement Analysis

**Key Findings**:

- **45.2% of issues** received multiple annotations from different annotators
- **Perfect agreement rate**: Only 44.0% among multi-annotated issues
- **79 issues (25.3%)** have genuinely **inconsistent classifications**
- This suggests **inherent multi-label characteristics** in bug classification

**Most Complex Issues** (highest annotation time variance):

1. `square/okhttp/issues/4875` - Both semantic and memory aspects
2. `google/ExoPlayer/issues/7273` - Memory management with semantic implications
3. `spring-projects/spring-boot/issues/2220` - Concurrency with semantic elements

## 4. Research Methodology for LLM Implementation

### 4.1 Data Preparation Strategy

**Multi-Label Approach**:

```python
# Convert to multi-label format for complex issues
def prepare_multilabel_dataset(df):
    # Issues with multiple valid classifications
    multilabel_issues = issues_with_disagreement
    # Single-label issues with high agreement
    singlelabel_issues = consistent_classifications
    return combined_dataset
```

**Feature Engineering**:

- **Repository context** (framework type, domain)
- **Issue metadata** (title length, description complexity)
- **Temporal features** (issue age, resolution time)

### 4.2 LLM Training Methodologies

**Approach 1: Few-Shot Learning with GPT-4/Claude**

```python
def create_few_shot_prompts(category_examples):
    """
    Create contextual examples for each bug category
    Include repository context and technical details
    """
    prompt = f"""
    Classify GitHub issues into: semantic, memory, concurrency, other

    Examples:
    {category_examples}

    Issue to classify: {issue_text}
    Classification:
    """
```

**Approach 2: Fine-Tuned Classification Models**

- **Base Model**: BERT, RoBERTa, or CodeBERT for code understanding
- **Multi-task Learning**: Joint classification + explanation generation
- **Domain Adaptation**: Pre-train on software engineering texts

**Approach 3: Hybrid Ensemble Method**

```python
def ensemble_classifier():
    # Combine multiple approaches
    rule_based_score = keyword_classifier(issue_text)
    llm_prediction = few_shot_llm(issue_text)
    finetuned_score = bert_classifier(issue_text)

    return weighted_average([rule_based_score, llm_prediction, finetuned_score])
```

## 5. Evaluation Methodology and Benchmarks

### 5.1 Evaluation Metrics Implementation

**Standard Classification Metrics**:

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred, multilabel=False):
    if multilabel:
        # Handle multi-label scenarios
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
    else:
        # Standard multi-class metrics
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### 5.2 Cross-Validation Strategy

**Repository-Aware Split**:

````python
def repository_stratified_split(df, test_size=0.2):
    """
    Ensure test set contains issues```om different repositories
    Prevents overfitting to specific```oject characteristics
    """
    repos = df['repo'].unique()
    test_repos = sample(repos, int(len(repos) * test_size))

    test_set = df[df['repo'].isin(test_repos)]
    train_set = df[~df['repo'].isin(test_repos)]

    return train_set, test_set
````

### 5.3 Expected Benchmark Results

**Baseline Expectations**:

- **Random Classifier**: ~25% accuracy (balanced classes)
- **Keyword-based**: 45-55% accuracy
- **Traditional ML**: 65-75% accuracy
- **Fine-tuned LLMs**: 80-85% accuracy target
- **GPT-4 Few-shot**: 75-80% accuracy expected

**Success Criteria**:

- **Accuracy**: >80% for single-label, >70% for multi-label
- **Precision/Recall**: >75% per class (macro-averaged)
- **F1-Score**: >78% overall
- **Inter-annotator Agreement**: Kappa >0.6 between model and human experts

## 6. Advanced Technical Methodologies

### 6.1 Prompt Engineering for Bug Classification

**Structured Prompt Template**:

````python
def create_classification_prompt(issue_data):
    return f"""
    **Repository Context**: {issue_data['repo']} - {get_repo_description(repo)}

    **Issue Title**: {issue_data['title']}

    **Issue Description**: {issue_data['body']}

    **Stack Trace/Code**: {extract_code_blocks(issue_data['body'])}

    **Task**: Classify this GitHub issue into one or more categories:
    - **SEMANTIC**: API misuse, logic errors, specification violations
    - **MEMORY**: Memory leaks, OOM errors, resource management issues```    - **CONCURRENCY**: Thread safety, race conditions, synchron```tion
    - **OTHER**: Configuration, build, documentation, environment issues

    **Classification**:
    **Confidence**:
    **Reasoning**:
    """
````

### 6.2 Multi-Modal Classification Approach

**Code + Text Analysis**:

````python
class MultiModalBugClassifier:
    def __init__(self):
        self.text_encoder = AutoModel.from_pretraine```microsoft/codebert-base')
        self.code_encoder = AutoModel.from_pretraine```microsoft/graphcodebert-base')

    def classify(self, issue_text, code_snippets):
        text_features = self.text_encoder(issue_text)
        code_features = self.code_encoder(code_snippets)

        combined_features = torch.cat([text_features, code_features], dim=-1)
        return self.classifier_head(combined_features)
````

### 6.3 Active Learning for Annotation Efficiency

```python
def active_learning_loop(unlabeled_pool, model, uncertainty_threshold=0.3):
"""
Select most informative samples for human annotation
Focus on disagreement cases and low-confidence predictions
"""
predictions = model.predict_proba(unlabeled_pool)
uncertainty_scores = calculate_entropy(predictions)

    # Select high uncertainty cases
    candidates = unlabeled_pool[uncertainty_scores > uncertainty_threshold]
    return candidates.sample(n=batch_size)

```

## 7. Path to Automatic Bug Triage

### 7.1 Integration with Bug Triage Workflows

**Automatic Classification Pipeline**:

````python
class BugTriageSystem:
    def __init__(self):
        self.classifier = load_trained_model()
        self.priority_estimator = PriorityModel()
        self.assignee_recommender = AssigneeModel()

    def process_new_issue(self, github_issue):
        # Step 1: Classify bug type
        bug_category = self.classifier.predict```thub_issue)
        confidence = self.classifier.predict_pr```(github_issue)

        # Step 2: Estimate priority based on category```      priority = self.priority_estimator.predict(bug_category, github_issue)

        # Step 3: Recommend assignee
        assignee = self.assignee_recommender.suggest(bug_category, github_issue.repo)

        return {
            'category': bug_category,
            'confidence': confidence,
            'priority': priority,
            'suggested_assignee': assignee,
            'requires_human_review': confidence < 0.8
        }
````

### 7.2 Continuous Learning System

**Model Improvement Loop**:

- **Feedback Collection**: Track classification accuracy from developer corrections
- **Model Retraining**: Periodic updates with new annotated data
- **Performance Monitoring**: Real-time accuracy tracking per repository/category

## 8. Additional Research Opportunities

### 8.1 Advanced Research Directions

**Temporal Analysis**:

- **Bug Evolution Patterns**: How issue descriptions change over time
- **Seasonal Trends**: Bug category frequencies across different time periods
- **Repository Maturity Impact**: Classification patterns in new vs. established projects

**Cross-Language Generalization**:

- **Multi-Language Dataset Creation**: Extend to Python, JavaScript, C++ projects
- **Transfer Learning**: Adapt Java-trained models to other languages
- **Universal Bug Patterns**: Identify language-agnostic bug characteristics

### 8.2 Human-AI Collaboration Research

**Annotation Quality Improvement**:

- **Guidelines Refinement**: Develop clearer category definitions based on disagreement analysis
- **Expert Calibration**: Training sessions to improve inter-annotator agreement
- **Confidence Modeling**: Predict when human review is necessary

### 8.3 Real-World Integration Studies

**Industry Deployment**:

- **A/B Testing**: Compare automated vs. manual triage efficiency
- **Developer Satisfaction**: Measure acceptance of automated classifications
- **Cost-Benefit Analysis**: Quantify time savings and accuracy improvements
