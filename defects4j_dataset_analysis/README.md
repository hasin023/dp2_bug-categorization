# Bug Categorization Using Defects4J Dataset

This research project focuses on **bug categorization** using the curated Defects4J dataset, enhanced by the comprehensive analysis provided in the "Studying-Data-Cleanness-in-Defects4J" repository. The goal is to explore **three bug categorization approaches**:

1. Based on **bug reports** (textual information)
2. Based on **bug reports summary**
3. Based on **source code and test evolution**

## Using This Dataset for Bug Categorization

1. **Bug Reports/Text Analysis**
   Use the `bug_repository.csv` and bug report URLs to extract textual data. Apply NLP methods to categorize bugs based on description, severity, or root cause.
2. **Test Evolution Patterns**
   Use the pattern classifications to classify bugs by development context (e.g., newly created tests indicate complex bugs). Temporal data like report and fix timestamps allow for severity and priority analysis.
3. **Source Code and Test Changes**
   Analyze code diffs in `manual_study/code_changes/` and statistical metadata to classify bugs by root cause or fault nature (e.g., logic errors, API misuse).

## Detailed Dataset Schema for Bug Categorization

| Field               | Description                                    | Usage for Categorization         |
| :------------------ | :--------------------------------------------- | :------------------------------- |
| `project`           | Project system name (e.g., Lang, Cli)          | Categorize by domain/system      |
| `bugid`             | Unique identifier of the bug                   | Primary key for linking data     |
| `test`              | Name of the fault-triggering test class        | Test-level analysis              |
| `test_method`       | Specific test method name                      | Granular test analysis           |
| `pattern`           | Integer 1-4 indicating test evolution pattern  | Core categorization dimension    |
| `datetime_modified` | Timestamps of test modifications               | Temporal and complexity analysis |
| `commit_modified`   | Commit hashes associated with modifications    | Code evolution tracking          |
| `created`           | Boolean: test created during bug fixing        | Creation context                 |
| `report_creation`   | Bug report creation timestamp                  | Timeline-based categorization    |
| `report_resolution` | Bug resolution timestamp                       | Timeline-based categorization    |
| `modified_in`       | Boolean indicating modification within bug fix | Modification context             |
| `created_in`        | Boolean indicating creation within bug fix     | Creation timing/context          |
| `report_url`        | URL pointing to original bug report            | Source for textual analysis      |

## Limitations \& Considerations

- Focused on **Java open source projects**; results may not generalize to other languages or proprietary codebases.
- Some bug severity and textual data require external issue tracker access (Jira, GitHub).
- Manual categorization covers a sample, so some test modifications require human judgment.
- Data represents **historical development practices**; evolving testing methodologies may affect applicability.

## References

- Defects4J main repository: [https://github.com/rjust/defects4j](https://github.com/rjust/defects4j)
- Studying Data Cleanness in Defects4J: [https://github.com/nakhlarafi/Studying-Data-Cleanness-in-Defects4J](https://github.com/nakhlarafi/Studying-Data-Cleanness-in-Defects4J)
