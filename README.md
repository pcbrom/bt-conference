# Assessing BLEU Score Variability in LLMs - A Non-Parametric Comparison Using the Friedman Test

## Methodology

This subsection outlines the method for evaluating LLMs and translation tools using back-translation, with the BLEU score as the quality metric.

Each text selected for the experiment is considered an independent block, with $k$ representing the number of texts (blocks) and each text being independently processed by all $n$ models (e.g., Grok, DeepSeek-R1, GPT 4.5, Gemini 2.0 Flash Thinking, and Mistral Large). Texts must reflect diverse content to capture real variability in translation scenarios.

In the experimental procedure, each model translates the text from the source language to an intermediate language and then back to the original language. The BLEU score of the back-translation serves as a measure of semantic and syntactic preservation. The BLEU scores, represented by $Y_{ij}$ for the $j$-th text (block) translated by the $i$-th model, are organized into a data matrix with $n$ treatments (models) and $k$ blocks (texts). The analysis aims to distinguish translation model effects while accounting for text variability, with the model formulated as:
\begin{equation}
    Y_{ij} = \eta + \tau_i + \beta_j + \epsilon_{ij},
\end{equation}
where $\eta$ represents the overall mean of the BLEU scores, $\tau_i$ denotes the effect of the $i$-th model, $\beta_j$ accounts for the effect of the $j$-th text (block) by controlling for inherent text variability, and $\epsilon_{ij}$ corresponds to the random error associated with the translation of text $j$ by model $i$.

Due to potential violations of normality and homoscedasticity, non-parametric methods are used: the Friedman test is applied to assess significant differences between translation models across texts. If significant differences are found, the Dunn post-hoc test, with adjustments for multiple comparisons (e.g., Bonferroni correction), is employed to identify statistically significant pairs of models.
