# Assessing BLEU Score Variability in LLMs - A Non-Parametric Comparison Using the Friedman Test

## Methodology

The study follows a **factorial experimental design** to analyze the performance differences among multiple **Large Language Models (LLMs)** in machine translation. The experiment evaluates three translation directions: **EN→ZH** (English to Chinese), **ZH→EN** (Chinese to English), and **Back Translation**. Each model was tested using an independent set of **\( n \)** texts, and translations were evaluated based on the **BLEU score**. The experiment follows a **repeated measures design**, where each text is translated by all models across all translation directions.

The LLMs analyzed in this study include: **Grok**, **DeepSeek-R1**, **GPT 4.5**, **Gemini 2.0 Flash Thinking**, **Mistral Large**, and **Google Translate** (possibly removed). Each model processed the same texts, allowing for direct performance comparisons.

### Mathematical Model
\[
Y_{ijl} = \mu + \tau_i + \delta_j + (\tau\delta)_{ij} + \gamma_l + \epsilon_{ijl}
\]

where:

- \( Y_{ijl} \) represents the BLEU score for the \( i \)-th LLM, in the \( j \)-th translation direction, for the \( l \)-th text.
- \( \mu \) is the overall mean of the experiment.
- \( \tau_i \) represents the effect of the \( i \)-th LLM.
- \( \delta_j \) is the effect of the \( j \)-th translation direction.
- \( (\tau\delta)_{ij} \) denotes the interaction effect between the LLM and translation direction.
- \( \gamma_l \) accounts for the random effect of text variability.
- \( \epsilon_{ijl} \) represents the residual error.

### Statistical Analysis
Since normality assumptions may not hold, a **non-parametric** approach was chosen using the **Friedman test**, followed by the **Dunn test with Bonferroni correction** for post hoc comparisons.

### Summary of Methodology
| Step | Description |
|---|---|
| **Experimental Factors** | LLM models and translation directions (EN→ZH, ZH→EN, Back Translation) |
| **Experimental Unit** | Independently translated texts |
| **Design** | Factorial with repeated measures |
| **Evaluation Metric** | BLEU score |
| **Statistical Test** | Friedman test for overall comparisons |
| **Post Hoc Test** | Dunn test with Bonferroni correction |
| **Sample Size Calculation** | Power analysis for Friedman test |

This methodology ensures a robust statistical approach for comparing LLM performance.
