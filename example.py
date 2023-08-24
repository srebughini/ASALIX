import pandas as pd
import numpy as np
import asalix as ax

# Extract the dataset from a Pandas Dataframe that contains normal and not normal data
number_of_points = 150
dataset = ax.extract_dataset(pd.DataFrame({"normal_dataset": np.random.normal(100, 20, number_of_points),
                                           "not_normal_dataset": list(range(1, number_of_points + 1))}),
                             data_column_name="not_normal_dataset")

# Fit dataset with a normal distribution
res = ax.normal_distribution_fit(dataset)
print("\nNormal fit")
print("p-value:", res.p_value)  # p-value
print("A:      ", res.normal_coefficient)  # Coefficient
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

# Nelson rules for time oriented dataset
nelson_rules = ax.check_dataset_using_nelson_rules(dataset)
print("\nNelson rules")
print("Rule 1: ", nelson_rules.rule1)
print("Rule 2: ", nelson_rules.rule2)
print("Rule 3: ", nelson_rules.rule3)
print("Rule 4: ", nelson_rules.rule4)
print("Rule 5: ", nelson_rules.rule5)
print("Rule 6: ", nelson_rules.rule6)
print("Rule 7: ", nelson_rules.rule7)

# Control chart
number_of_points = 150
number_of_timestamp = 10

data_matrix = np.random.rand(number_of_timestamp, number_of_points)
time_matrix = np.ones_like(data_matrix)

for i in range(0, number_of_timestamp):
    time_matrix[i, :] = time_matrix[i, :] * (i + 1)

time_dependent_dataset = ax.extract_time_dependent_dataset(pd.DataFrame({"time": time_matrix.ravel(),
                                                                         "data": data_matrix.ravel()}),
                                                           "data",
                                                           "time")

xbar_chart_limit, range_chart_limit = ax.create_control_charts(time_dependent_dataset,
                                                               'XbarR',
                                                               plot=True,
                                                               fig_number=3)
print("\nXbar chart")
print("LCL:  ", xbar_chart_limit.lcl)
print("UCL:  ", xbar_chart_limit.ucl)
print("CL:   ", xbar_chart_limit.cl)

print("\nRange chart")
print("LCL:  ", range_chart_limit.lcl)
print("UCL:  ", range_chart_limit.ucl)
print("CL:   ", range_chart_limit.cl)
