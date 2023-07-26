<p align="center">
  <a href=""><img src="https://i.imgur.com/sLJPVWS.png" title="source: imgur.com" /></a>
</p>

<p align="center">
  <a href="https://github.com/srebughini/ASALIX/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/srebughini/ASALIX"></a>
  <a href="https://www.codefactor.io/repository/github/srebughini/asalix"><img src="https://www.codefactor.io/repository/github/srebughini/asalix/badge" alt="CodeFactor" /></a>
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

### 2.1 Calculate mean value and standard deviation

```python
import pandas as pd
import numpy as np
import asalix as ax

dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0, 1000))}),
                             data_column_name="normal_dataset")

ax.create_histogram(dataset, normal_distribution_fitting=True, plot=True, density=False)

print("\nMean value")
print("\u03BC:   ", ax.calculate_mean_value(dataset))
print("xbar:", ax.calculate_mean_value(dataset))
print("\nStandard deviation")
print("\u03C3:", ax.calculate_standard_deviation(dataset, population=True))
print("s:", ax.calculate_standard_deviation(dataset, population=False))
print("\nNormality test (P-value)")
print("Basic:              ", ax.normality_test(dataset))
print("Anderson-Darling:   ", ax.normality_test(dataset, test="anderson_darling"))
print("Kolmogorov-Smirnov: ", ax.normality_test(dataset, test="kolmogorov_smirnov"))
print("Shapiro-Wilk:       ", ax.normality_test(dataset, test="shapiro_wilk"))
```

