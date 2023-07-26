import pandas as pd
import numpy as np
import asalix as ax

dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(10, 2, 1000),
                                           "not_normal_dataset": list(range(0,1000))}),
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
