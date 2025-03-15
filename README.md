# Assessing BLEU Score Variability in LLMs - A Non-Parametric Comparison Using the Friedman Test

## `sample_size.py`: Determining Sample Size for Statistical Significance

This script focuses on calculating the required sample size for a statistically robust comparison of Large Language Models (LLMs) and translation tools, employing back-translation and the BLEU score as the primary evaluation metric. The core objective is to determine the number of texts needed to reliably detect differences in translation quality between models, while controlling for variability inherent in the texts themselves.

### Methodology: Block Design and Statistical Model

The experimental design utilizes a block design, where each selected text serves as an independent block. This approach allows us to account for the variability in translation difficulty across different texts. Each text is translated by \( k = 5 \) different models (e.g., Grok, DeepSeek-R1, GPT 4.5, Gemini 2.0 Flash Thinking, and Mistral Large), ensuring that each model processes the same set of texts. The texts are chosen to represent a diverse range of content, reflecting the real-world variability encountered in translation tasks.

The translation process involves translating each text from the source language to an intermediate language and then back to the original language. The BLEU score of the back-translated text is used as a measure of how well the model preserves the semantic and syntactic integrity of the original text.

The BLEU scores are organized into a data matrix, where \( Y_{ij} \) represents the BLEU score for the \( j \)-th text (block) translated by the \( i \)-th model. The data matrix has \( k = 5 \) treatments (models) and \( n \) blocks (texts). The underlying statistical model is formulated as:

\[
Y_{ij} = \eta + \tau_i + \beta_j + \epsilon_{ij},
\]

where:
- \( \eta \) is the overall mean BLEU score across all models and texts.
- \( \tau_i \) represents the effect of the \( i \)-th model (\( i = 1, \dots, 5 \)), indicating the model's contribution to the BLEU score.
- \( \beta_j \) accounts for the effect of the \( j \)-th text (\( j = 1, \dots, n \)), controlling for the inherent variability in translation difficulty between texts.
- \( \epsilon_{ij} \) is the random error term associated with the translation of text \( j \) by model \( i \), capturing unexplained variability.

### Non-Parametric Testing: Friedman Test and Dunn Post-Hoc Test

Given the potential for violations of normality and homoscedasticity in the BLEU score data, non-parametric statistical methods are employed. Specifically, the Friedman test is used to assess whether there are significant differences between the translation models across the set of texts. The Friedman test is appropriate for analyzing data from a randomized block design when the data do not meet the assumptions of parametric tests.

If the Friedman test reveals significant differences, a Dunn post-hoc test is performed to identify which specific pairs of models exhibit statistically significant differences. The Dunn test includes adjustments for multiple comparisons (e.g., Bonferroni correction) to control the family-wise error rate and reduce the risk of false positives.

### Sample Size Calculation

The script calculates the required sample size to achieve sufficient statistical power for detecting meaningful differences between the translation models. The calculation is based on a significance level of 0.05 and a desired power of 0.8, and it considers a medium effect size (\( f = 0.3 \)) for non-parametric block designs. This effect size represents the expected magnitude of the differences between the models.

The calculation determines that **\( n = 89 \) distinct texts** are required, with each text translated by **\( k = 5 \) models**, resulting in a total of **445 observations**. This sample size ensures that the study has adequate power to detect statistically significant differences between the models, if such differences exist.


## `select_sample.py`: Data Sampling and Preparation

This script automates the process of extracting, filtering, and preparing a representative sample of abstracts from a larger text corpus for subsequent analysis, particularly in the context of evaluating machine translation models. It leverages regular expressions for pattern matching, the pandas library for efficient data manipulation, and file I/O operations to manage the data.

### Functionality:

1.  **Abstract Extraction:** The script begins by reading a text file (specifically, "data/Exact and Earth Sciences\_Chemistry.txt") and uses a regular expression (`r'Abstract:\s*(.*)'`) to identify and extract abstracts. The regular expression searches for lines starting with "Abstract:" (case-insensitive) and captures the subsequent text as the abstract content.

2.  **Data Structuring:** The extracted abstracts are stored in a list, which is then converted into a pandas DataFrame. This DataFrame provides a structured format for further processing and analysis. A column named 'abstract' is created to hold the extracted text.

3.  **Length Filtering:** To ensure the quality and relevance of the abstracts, a length filter is applied. Abstracts with a character length of 20 or less are discarded. This step is crucial for removing potentially incomplete or irrelevant entries. The length of each abstract is calculated and stored in a new column named 'nchar', and the DataFrame is filtered based on this column.

4.  **Random Sampling:** A random sample of 89 abstracts is selected from the filtered DataFrame using a fixed random state (random\_state=42) to ensure reproducibility. This sample size is determined by the `sample_size.py` script.

5.  **Data Augmentation for Repetition:** To facilitate consistency evaluation across different translation models, each sampled abstract is repeated three times. This is achieved using the `loc[sampled_df.index.repeat(3)]` method, which duplicates each row based on its index. A 'Repetition' column is added to indicate the repetition number (1, 2, or 3) for each abstract.

6.  **Column Initialization:** Two additional columns, 'EN\_ZH' and 'ZH\_EN', are created and initialized with empty strings. These columns are intended to store the translated versions of the abstracts (English to Chinese and Chinese to English, respectively) generated by the translation models in subsequent steps.

7.  **Dataframe Cleaning:** The temporary 'nchar' column, which was used for filtering, is removed from the DataFrame.

8.  **CSV Export:** Finally, the processed DataFrame is saved to a CSV file named "data/sampled Exact and Earth Sciences\_Chemistry abstracts.csv". The `index=False` argument ensures that the DataFrame index is not included in the output file.

### Purpose:

The primary purpose of this script is to create a well-defined and preprocessed dataset of abstracts suitable for evaluating the performance and consistency of different machine translation models. By extracting, filtering, sampling, and augmenting the data, the script ensures that the subsequent analysis is based on a representative and reliable set of texts. The repetition of abstracts is specifically designed to assess the consistency of translations produced by each model.

## `res_gemini.py` and `res_{model}.py`: Translation and Data Processing with Gemini and Other Models

**Note:** Similar scripts, named `res_{model}.py` (e.g., `res_dsr1.py`, `res_grok.py`), follow the same structure and functionality but utilize different translation models or APIs. The core logic of loading data, constructing prompts, handling API requests, and saving results remains consistent across these scripts, with variations primarily in the specific API endpoints, authentication methods, and model names used for translation.

This script leverages the Google Gemini API to translate a dataset of scientific abstracts from Chinese to English and back, facilitating an analysis of translation quality and consistency. It orchestrates the translation process, handles API interactions, and manages data input and output using pandas DataFrames.

### Functionality:

1.  **Environment Configuration:** The script begins by loading environment variables, specifically the Google API key, from a `.env` file. This key is essential for authenticating requests to the Gemini API. The path to the `.env` file is explicitly defined.

2.  **API Client Initialization:** It initializes a client for interacting with the Google Gemini API, using the loaded API key. This client is the primary interface for sending translation requests and receiving responses. The specific Gemini model used for translation is set to `'gemini-2.0-flash-thinking-exp'`.

3.  **Data Loading:** The script reads a CSV file (`data/sampled Exact and Earth Sciences_Chemistry abstracts.csv`) containing the abstracts to be translated. This file is assumed to be generated by the `select_sample.py` script. The data is loaded into a pandas DataFrame for efficient manipulation.

4.  **Data Type Enforcement:** To ensure proper handling of translated text, the script explicitly sets the data type of the 'ZH\_EN' and 'EN\_ZH' columns to `object` (string). This is crucial for accommodating the translated text, which may contain characters or formatting that could be misinterpreted if the columns had a different data type.

5.  **Translation Loop:** The script iterates through each row of the DataFrame, performing the following steps for each abstract:

    *   **ZH to EN Translation:** It constructs a prompt for translating the Chinese abstract to English. The prompt is designed to instruct the Gemini model to provide only the translated text, without any additional explanations or context. The `client.models.generate_content` method is used to send the translation request to the Gemini API. The translated text is extracted from the API response and stored in the 'ZH\_EN' column of the DataFrame. If no response is generated or an error occurs, a corresponding message is stored in the column.

    *   **EN to ZH Translation:** It constructs a prompt for translating the English translation back to Chinese. This step is crucial for evaluating the round-trip translation quality. The prompt is similar to the ZH to EN translation prompt, ensuring consistency in the instructions provided to the Gemini model. The translated text is extracted from the API response and stored in the 'EN\_ZH' column of the DataFrame. Error handling is implemented to capture any issues during the translation process.

    *   **Rate Limiting:** To comply with API rate limits, the script pauses for 60 seconds after processing every 100 abstracts. This prevents the script from exceeding the API's usage limits and ensures stable operation.

6.  **Error Handling:** The script includes comprehensive error handling to gracefully manage potential issues during the translation process. If an error occurs while translating a particular abstract, the error message is stored in both the 'ZH\_EN' and 'EN\_ZH' columns, allowing for subsequent analysis of translation failures.

7.  **Data Export:** After processing all abstracts, the script saves the DataFrame to a CSV file. The output filename includes the name of the Gemini model used for translation, allowing for easy identification of the results. The `index=False` argument ensures that the DataFrame index is not included in the output file.

### Purpose:

The primary purpose of this script is to automate the translation of a dataset of scientific abstracts using the Google Gemini API. By translating the abstracts from Chinese to English and back, the script enables a detailed analysis of translation quality, consistency, and potential information loss. The generated dataset can be used to evaluate the performance of the Gemini model and to identify areas for improvement in machine translation.
