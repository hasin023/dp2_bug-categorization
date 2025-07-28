## Cross-Verification: "Studying-Data-Cleanness-in-Defects4J" vs. "rjust/defects4j"

This report verifies the data and project links used by the [Studying-Data-Cleanness-in-Defects4J](https://github.com/nakhlarafi/Studying-Data-Cleanness-in-Defects4J) repository using the official Defects4J repository ([rjust/defects4j](https://github.com/rjust/defects4j)), and provides updated/accurate links and procedural clarifications for defect datasets and project repositories.

### 1. **Defects4J Main Repository: Up-to-Date Structure**

The Defects4J repository organizes and manages defects from real-world Java projects. Its structure is as follows:

- **project_repos/**: The actual version control repositories (populated via `init.sh`)
- **framework/**
  - **bin/**: CLI for Defects4J
  - **core/**: Core Perl modules and project abstractions
    - **Projects.pm**: Lists all supported projects and their IDs
  - **bug-mining/**: Scripts for mining bugs from upstream repos
  - **projects/**: Project-specific resources, configs, and metadata
  - **test/**, **util/**, **lib/**, **doc/**
- **major/**: Major mutation framework
- **developer/**: Contributor resources

The full list of project IDs and mappings is programmatically encoded in the core modules (especially `Project.pm`).

### 2. **Defects4J Supported Project List \& Updated Repository Links**

#### Main Defects4J Projects and Their Official Upstream URLs :

| Identifier      | Project Name               | Upstream Repository URL                             |
| :-------------- | :------------------------- | :-------------------------------------------------- |
| Chart           | JFreeChart                 | https://github.com/jfree/jfreechart                 |
| Cli             | Apache Commons CLI         | https://github.com/apache/commons-cli               |
| Closure         | Google Closure Compiler    | https://github.com/google/closure-compiler          |
| Codec           | Apache Commons Codec       | https://github.com/apache/commons-codec             |
| Collections     | Apache Commons Collections | https://github.com/apache/commons-collections       |
| Compress        | Apache Commons Compress    | https://github.com/apache/commons-compress          |
| Csv             | Apache Commons CSV         | https://github.com/apache/commons-csv               |
| Gson            | Google Gson                | https://github.com/google/gson                      |
| JacksonCore     | Jackson Core               | https://github.com/FasterXML/jackson-core           |
| JacksonDatabind | Jackson Databind           | https://github.com/FasterXML/jackson-databind       |
| JacksonXml      | Jackson DataFormat XML     | https://github.com/FasterXML/jackson-dataformat-xml |
| Jsoup           | Jsoup                      | https://github.com/jhy/jsoup                        |
| JxPath          | Apache Commons JXPath      | https://github.com/apache/commons-jxpath            |
| Lang            | Apache Commons Lang        | https://github.com/apache/commons-lang              |
| Math            | Apache Commons Math        | https://github.com/apache/commons-math              |
| Mockito         | Mockito                    | https://github.com/mockito/mockito                  |
| Time            | Joda-Time                  | https://github.com/JodaOrg/joda-time                |

**Note:** All these repositories are still active, and the links above can be considered authoritative and up-to-date.

### 3. **Defects4J Bug Datasets and Standards**

- **Bug Data**: Each Defects4J bug contains a _buggy_ and _fixed_ version, verified to contain a reproducible failing test (the "triggering test").
- **Minimality**: The main repo ensures that bug fix commits are minimal (i.e., unrelated changes/features are removed), a process managed by maintainers.
- **Metadata**: Project IDs, bug IDs, revisions, triggering tests, and test oracles are all stored and accessible via both CLI and metadata CSVs in the Defects4J repo.

### 4. **Verifying "Studying-Data-Cleanness-in-Defects4J" Data Consistency**

#### _a. Project \& Bug IDs_

- The "Studying-Data-Cleanness-in-Defects4J" repository mirrors the main project list and bug IDs as in Defects4J—this is consistent with v2.0 and later.
- The counts (e.g., number of bugs/projects) are accurate and correlate with the current mainline as of Defects4J v3.0.1.

#### _b. Bug Repository Links_

- The `bug_repository.csv` in the analysis repository simply lists the official issue trackers, mainly JIRA and GitHub links, which are correct.

#### _c. Data Extraction Process_

- The data in "Studying-Data-Cleanness-in-Defects4J" was extracted using the same commit hashes and test names as in the official Defects4J metadata.
- The procedure for mapping bug IDs, collecting triggering tests, and extracting commit metadata is directly aligned with public instructions (see Defects4J documentation and scripts).

#### _d. Potential Missing/Updated Info_

- The main Defects4J repo sometimes deprecates or adds bugs (see `active-bugs.csv` and `deprecated-bugs.csv`). Ensure you always cross-reference the latest Defects4J data for bug activity status.
- Issue tracker links in CSV files should be periodically validated, as project hosting services or URLs may change.
- The official initialization script (`init.sh`) is **required** to populate the `project_repos/` directory with up-to-date code history for each project.

### 5. **Additional Useful Main-Repo Resources**

- **Bug-mining Documentation:** Detailed in `framework/bug-mining/README.md` for those interested in contributing new bugs or mining methodology.
- **Project Metadata**: Canonical project lists are always visible in the `framework/core/Project.pm` file and are accessible via CLI commands (see `defects4j info`, `defects4j query`).
- **New Project Integration**: If new projects are added, they will appear in newer Defects4J releases via updated code/metadata structures.
