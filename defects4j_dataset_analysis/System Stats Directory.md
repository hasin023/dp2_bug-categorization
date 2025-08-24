# System Stats Directory Overview

The `/system_stats` directory contains **code statistics for each of the 16 Java systems** studied from Defects4J, generated using the CLOC (Count Lines of Code) tool.

### **Directory Structure**

The directory contains individual `.txt` files for each system studied:

- `cli.txt` - Apache Commons CLI statistics
- `closure.txt` - Google Closure Compiler statistics
- `codec.txt` - Apache Commons Codec statistics
- `compress.txt` - Apache Commons Compress statistics
- `csv.txt` - Apache Commons CSV statistics
- `gson.txt` - Google Gson statistics
- `jsoup.txt` - Jsoup HTML parser statistics
- `jxpath.txt` - Apache Commons JXPath statistics
- `lang.txt` - Apache Commons Lang statistics
- `math.txt` - Apache Commons Math statistics
- `mockito.txt` - Mockito testing framework statistics
- `time.txt` - Joda Time statistics

### **Content Format \& Information**

Each file contains detailed **CLOC analysis results** showing:

**Example from `cli.txt`**:

```
Language    files   blank   comment   code
Java        38      1008    2593      4266
HTML        2       11      0         23
SUM:        40      1019    2593      4289
```

**Example from `math.txt`** (larger system):

```
Language    files   blank   comment   code
Java        813     18019   84273     84317
XML         2       6       28        61
SUM:        815     18025   84301     84378
```

### **Statistical Metrics Provided**

For each system, the files contain:

1. **Language Breakdown**: Different programming languages used (primarily Java, with some HTML/XML)
2. **File Count**: Total number of source files
3. **Blank Lines**: Number of empty lines
4. **Comment Lines**: Lines containing comments/documentation
5. **Code Lines**: Actual executable code lines
6. **Total Summary**: Aggregate statistics across all languages

### **Research Application**

This statistical information supports:

- **System Complexity Analysis**: Understanding the scale and complexity of each studied system
- **Comparative Studies**: Enabling comparisons between different systems based on size metrics
- **Contextualization**: Providing context for bug density and fault-triggering test ratios relative to system size
- **Baseline Metrics**: Establishing baseline measurements for future studies on similar systems

### **Key Insights from System Stats**

The statistics reveal significant variation in system sizes:

- **Smallest System**: CSV with ~2K lines of code
- **Largest System**: Closure with ~90K lines of code
- **Comment Density**: Some systems like Math have extensive documentation (84K comment lines vs 84K code lines)
- **File Distribution**: Systems range from 40 files (CLI) to 815 files (Math)
