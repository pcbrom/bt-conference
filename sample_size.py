# Re-import required module due to execution state reset
import statsmodels.stats.power as smp

# Define parameters
alpha = 0.05  # Significance level
power = 0.8   # Statistical power
effect_size = 0.3  # Medium effect size (Cohen's f for block ANOVA)

# Calculate the required sample size per group
analysis = smp.FTestAnovaPower()
n_samples_per_group = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)

# Round to the nearest integer
n_samples_per_group = int(round(n_samples_per_group))

# Define the number of models (k = number of repetitions per model)
k_repetitions = 5  # Five translation models

# Number of texts required (n)
n_texts = n_samples_per_group  

# Define the number of repetitions per text for consistency evaluation
r_repetitions = 3  

# Total number of observations considering consistency repetitions
total_observations = n_texts * k_repetitions * r_repetitions

print(f"Number of texts (n): {n_texts}")
print(f"Number of models (k): {k_repetitions}")
print(f"Number of repetitions per text (r): {r_repetitions}")
print(f"Total number of observations: {total_observations}")
