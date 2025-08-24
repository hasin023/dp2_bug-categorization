# Setup and Commands for the defects4j CLI tool

## Setup Commands for Defects4J CLI Tool

1. **Clone the Defects4J repository**:

```bash
git clone https://github.com/rjust/defects4j.git
cd defects4j
```

2. **Run the initialization script** to download project repositories and external dependencies (this step is essential because the actual source code repos of projects are not bundled):

```bash
./init.sh
```

    - This script downloads all 17 project repositories with their histories and external libraries used by the projects.
    - It prepares the environment to work with all bug versions managed by Defects4J.

3. **Set up environment variable** to easily run Defects4J commands anywhere:
   - Add Defects4J `framework/bin` to your PATH, for example:

```bash
export PATH=$PATH:/path/to/defects4j/framework/bin
```

    - You can add this line to your `.bashrc` or `.zshrc` for persistent usage.

4. **Verify installation**:

```bash
defects4j
```

    - Running `defects4j` without any command will show the CLI tool's help and available commands.

## Core Defects4J CLI Commands

All commands follow the pattern:

```bash
defects4j <command> [options]
```

### Major commands with typical usage:

| Command    | Description                                                                         | Example Usage                                             |
| :--------- | :---------------------------------------------------------------------------------- | :-------------------------------------------------------- |
| `checkout` | Checkout a specific buggy or fixed project version                                  | `defects4j checkout -p Lang -v 1b -w /tmp/lang_1b`        |
| `compile`  | Compile the checked-out project version                                             | `defects4j compile -w /tmp/lang_1b`                       |
| `test`     | Run test suite or a specific test on the checked-out project version                | `defects4j test -w /tmp/lang_1b`                          |
| `info`     | Get metadata information about a project or bug ID                                  | `defects4j info -p Lang` or `defects4j info -p Lang -b 1` |
| `bids`     | List all active bug IDs for a project                                               | `defects4j bids -p Lang`                                  |
| `export`   | Export version-specific properties related to a project version                     | `defects4j export -p Lang -v 1b -o src.dir`               |
| `query`    | Query metadata for automation (e.g., get all bugs, tests, source directories, etc.) | `defects4j query -p Lang -q tests.all`                    |
| `coverage` | Run coverage analysis for a checked-out project version                             | `defects4j coverage -w /tmp/lang_1b`                      |
| `mutation` | Run mutation analysis on a checked-out project version                              | `defects4j mutation -w /tmp/lang_1b`                      |

## Detailed Frequently Used Commands

### 1. `checkout`

Checkout a buggy or fixed version of a project.

- `-p` : Project name (e.g. Lang, Closure, Cli...)
- `-v` : Version identifier, bug ID followed by 'b' (buggy) or 'f' (fixed), e.g. `1b`, `45f`
- `-w` : Working directory where the project will be checked out

Example:

```bash
defects4j checkout -p Lang -v 1b -w /tmp/lang_1b
```

This checks out the buggy version 1 of the Lang project to `/tmp/lang_1b`.

### 2. `compile`

Compile the sources and tests in a checked-out project.

- `-w` : The working directory of the checked-out project

Example:

```bash
defects4j compile -w /tmp/lang_1b
```

### 3. `test`

Run all tests or a specific test method on the checked-out project.

- `-w` : Working directory
- `-t` : Optional specific test case or test method to run (e.g., `org.apache.commons.lang3.StringUtilsTest`)

Example (run all tests):

```bash
defects4j test -w /tmp/lang_1b
```

Example (run a specific test method):

```bash
defects4j test -w /tmp/lang_1b -t org.apache.commons.lang3.StringUtilsTest::testIsEmpty
```

### 4. `info`

Get information about a project or a bug.

- `-p` : Project name
- `-b` : Bug id (optional)

Example (get project info):

```bash
defects4j info -p Lang
```

Example (get bug info):

```bash
defects4j info -p Lang -b 1
```

### 5. `bids`

List all bug IDs for a project.

- `-p` : Project name

Example:

```bash
defects4j bids -p Lang
```

### 6. `export`

Export specific properties for a project version.

- `-p` : Project name
- `-v` : Version (bug ID + status)
- `-o` : Property name (e.g., `src.dir`, `test.dir`, `build.dir`, `classes.dir`)

Example:

```bash
defects4j export -p Lang -v 1b -o src.dir
```

### 7. `query`

Query project metadata for automation, getting lists of all tests, all source files, etc.

- `-p` : Project name
- `-q` : Query string (e.g., `tests.all`, `src.classes`, `test.classes`)

Example (list all tests):

```bash
defects4j query -p Lang -q tests.all
```

### 8. `coverage`

Run test coverage analysis.

- `-w` : Working directory of the checked-out project

Example:

```bash
defects4j coverage -w /tmp/lang_1b
```

### 9. `mutation`

Run mutation testing on the checked-out project.

- `-w` : Working directory

Example:

```bash
defects4j mutation -w /tmp/lang_1b
```

## Additional Notes

- The typical workflow is:

1. `defects4j checkout` - get a buggy or fixed version.
2. `defects4j compile` - compile the project.
3. `defects4j test` - run tests to verify version correctness.

- The tool supports specifying buggy versions with suffix `b` and fixed versions with suffix `f`.
- The command options `-p`, `-v`, and `-w` are present in most commands where applicable.
- Run `defects4j <command> --help` or just `defects4j` for more details.
