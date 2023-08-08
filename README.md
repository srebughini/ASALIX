<p align="center">
  <a href=""><img src="https://i.imgur.com/sLJPVWS.png" title="source: imgur.com" /></a>
</p>

<p align="center">
  <a href="https://github.com/srebughini/ASALIX/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/srebughini/ASALIX"></a>
  <a href="https://www.codefactor.io/repository/github/srebughini/asalix"><img src="https://www.codefactor.io/repository/github/srebughini/asalix/badge" alt="CodeFactor" /></a>
  <a href="https://pypi.org/project/asalix/"><img src="https://img.shields.io/pypi/v/asalix"></a>
</p>


**ASALIX** is a **Python** package that collects **mathematical tools and utilities** designed to support **Lean Six
Sigma practitioners** in their process improvement journey. This repository aims to provide a robust **set of tools for
data analysis and statistical modeling**.  
Whether you are a Lean Six Sigma professional, a data analyst, or someone interested in process improvement, **ASALIX**
offers a wide range of functions and modules to aid you in your Lean Six Sigma projects.

## 1. Installation

To use the **ASALIX** library, you'll need to have Python installed on your system. You can install **ASALIX** using
pip:

```bash
pip install asalix
```

## 2. Examples

### 2.1 Extract dataset from Pandas DataFrame

```python
import pandas as pd
import numpy as np
from asalix import __init__ as ax

dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")
```

### 2.1 Calculate mean value and standard deviation

```python
import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")

# Print the calculated mean values on screen
print("\nMean value")
print("\u03BC:   ", ax.calculate_mean_value(dataset))  # Population mean value
print("xbar:", ax.calculate_mean_value(dataset))  # Sample mean value

# Print the calculated standard deviation on screen
print("\nStandard deviation")
print("\u03C3:", ax.calculate_standard_deviation(dataset, population=True))  # Population standard deviation
print("s:", ax.calculate_standard_deviation(dataset, population=False))  # Sample standard deviation
```

### 2.2 Perform normality test

```python
import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")

# Print the p-value of different normality test on screen
print("\nNormality test (P-value)")
print("Basic:              ", ax.normality_test(dataset))
print("Anderson-Darling:   ", ax.normality_test(dataset, test="anderson_darling"))
print("Kolmogorov-Smirnov: ", ax.normality_test(dataset, test="kolmogorov_smirnov"))
print("Shapiro-Wilk:       ", ax.normality_test(dataset, test="shapiro_wilk"))
```

### 2.3 Fit dataset with normal distribution

```python
import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")

# Fit dataset with a normal distribution
res = ax.normal_distribution_fit(dataset)
print("\nNormal fit")
print("p-value:", res.p_value)  # p-value
print("A:      ", res.normal_coefficient)  # Coefficient
print("\u03BC:      ", res.mean_value)  # Mean value
print("\u03C3:      ", res.standard_deviation)  # Standard deviation
```

### 2.4 Plot data using histogram

```python
import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")

# Create the histogram with a normal distribution fitted curve and plot it
ax.create_histogram(dataset, normal_distribution_fitting=True, plot=True, density=False)
```

### 2.5 Plot data using boxplot and calculate quartiles

```python
import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")

# Print the quartile values of the dataset on screen
quartiles = ax.create_quartiles(dataset, plot=True, fig_number=1)
print("\nQuartiles")
print("Minimum: ", quartiles.minimum)
print("1st:     ", quartiles.first)
print("Median:  ", quartiles.median)
print("3rd:     ", quartiles.third)
print("Maximum: ", quartiles.maximum)
```

### 2.6 Calculate confidence interval

```python
import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(100, 20, 20),
                                           "not_normal_dataset": list(range(1, 21))}),
                             data_column_name="normal_dataset")

# Print the confidence interval on screen
print("\n95% confidence internval")
print("\u03C3 known:  ", ax.calculate_confidence_interval(dataset, 0.95, population=True))
print("\u03C3 unknown:", ax.calculate_confidence_interval(dataset, 0.95, population=False))
```


## 3. For developers
To upload a new version of **ASALIX** on [PyPi](https://pypi.org/project/asalix/0.1.0/):
```bash
pip install --upgrade .
python example.py
python setup.py check
python setup.py sdist
twine upload dist/*
```