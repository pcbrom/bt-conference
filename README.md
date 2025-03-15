# Assessing BLEU Score Variability in LLMs - A Non-Parametric Comparison Using the Friedman Test

## Back-Translation Validation of Translation Performance for LLMs and Translation Tools - `sample_size.py`.

This subsection outlines the method for evaluating LLMs and translation tools using back-translation, with the BLEU score as the quality metric.

Each text selected for the experiment is considered an independent block, with \( n \) representing the number of texts (blocks). Each text is independently processed by all \( k = 5 \) models (e.g., Grok, DeepSeek-R1, GPT 4.5, Gemini 2.0 Flash Thinking, and Mistral Large). Texts must reflect diverse content to capture real variability in translation scenarios.

In the experimental procedure, each model translates the text from the source language to an intermediate language and then back to the original language. The BLEU score of the back-translation serves as a measure of semantic and syntactic preservation. The BLEU scores, represented by \( Y_{ij} \) for the \( j \)-th text (block) translated by the \( i \)-th model, are organized into a data matrix with \( k = 5 \) treatments (models) and \( n \) blocks (texts). The analysis aims to distinguish translation model effects while accounting for text variability, with the model formulated as:

\[
Y_{ij} = \eta + \tau_i + \beta_j + \epsilon_{ij},
\]

where:
- \( \eta \) represents the overall mean BLEU score,
- \( \tau_i \) denotes the effect of the \( i \)-th model (\( i = 1, \dots, 5 \)),
- \( \beta_j \) accounts for the effect of the \( j \)-th text (\( j = 1, \dots, n \)), controlling for inherent text variability,
- \( \epsilon_{ij} \) corresponds to the random error associated with the translation of text \( j \) by model \( i \).

Due to potential violations of normality and homoscedasticity, non-parametric methods are used: the Friedman test is applied to assess significant differences between translation models across texts. If significant differences are found, the Dunn post-hoc test, with adjustments for multiple comparisons (e.g., Bonferroni correction), is employed to identify statistically significant pairs of models.

The sample size was calculated for a significance level of 0.05 and a power of 0.8, considering a medium effect size (\( f = 0.3 \)) for non-parametric block designs. It required **\( n = 89 \) distinct texts**, each translated by **\( k = 5 \) models**, totaling **445 observations**.
