import asalix as ax

dataset = [1,2,3,4,5]

print("\nMean value")
print("\u03BC:\t", ax.calculate_mean_value(dataset))
print("xbar:\t", ax.calculate_mean_value(dataset))
print("\nStandard deviation")
print("\u03C3:\t", ax.calculate_standard_deviation(dataset, population=True))
print("s:\t", ax.calculate_standard_deviation(dataset, population=False))