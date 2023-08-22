import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
number_of_points = 150
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(100, 20, number_of_points),
                                           "not_normal_dataset": list(range(1, number_of_points + 1))}),
                             data_column_name="normal_dataset")

# Fit dataset with a normal distribution
res = ax.normal_distribution_fit(dataset)
print("\nNormal fit")
print("p-value:", res.p_value) #p-value
print("A:      ", res.normal_coefficient) # Coefficient
print("\u03BC:      ", res.mean_value)  # Mean value
print("\u03C3:      ", res.standard_deviation)  # Standard deviation

# Create the histogram with a normal distribution fitted curve and plot it
ax.create_histogram(dataset,
                    normal_distribution_fitting=False,
                    plot=True,
                    density=False,
                    fig_number=1)

# Print the calculated mean values on screen
print("\nMean value")
print("\u03BC:   ", ax.calculate_mean_value(dataset))  # Population mean value
print("xbar:", ax.calculate_mean_value(dataset))  # Sample mean value

# Print the calculated standard deviation on screen
print("\nStandard deviation")
print("\u03C3:", ax.calculate_standard_deviation(dataset, population=True))  # Population standard deviation
print("s:", ax.calculate_standard_deviation(dataset, population=False))  # Sample standard deviation

# Print the confidence interval on screen
print("\n95% confidence internval")
print("\u03C3 known:  ", ax.calculate_confidence_interval(dataset, 0.95, population=True))
print("\u03C3 unknown:", ax.calculate_confidence_interval(dataset, 0.95, population=False))

# Print the p-value of different normality test on screen
print("\nNormality test (P-value)")
print("Basic:              ", ax.normality_test(dataset))
print("Anderson-Darling:   ", ax.normality_test(dataset, test="anderson_darling"))
print("Kolmogorov-Smirnov: ", ax.normality_test(dataset, test="kolmogorov_smirnov"))
print("Shapiro-Wilk:       ", ax.normality_test(dataset, test="shapiro_wilk"))

# Print the quartile values of the dataset on screen
quartiles = ax.create_quartiles(dataset, plot=True, fig_number=2)
print("\nQuartiles")
print("Minimum: ", quartiles.minimum)
print("1st:     ", quartiles.first)
print("Median:  ", quartiles.median)
print("3rd:     ", quartiles.third)
print("Maximum: ", quartiles.maximum)

# Create control chart
control_chart = ax.create_control_charts(dataset, 'XbarR', plot=False, fig_number=3)
print("\nControl chart")
print("LCL:  ", control_chart.lcl)
print("UCL:  ", control_chart.ucl)
print("Mean: ", control_chart.mean)
print("\nNelson rules")
print("Rule 1: ", control_chart.rule1)
print("Rule 2: ", control_chart.rule2)
print("Rule 3: ", control_chart.rule3)
print("Rule 4: ", control_chart.rule4)
