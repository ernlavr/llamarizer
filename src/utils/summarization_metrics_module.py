# metrics from https://arxiv.org/pdf/2106.13822.pdf

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


class SummarizationMetrics:

    def preprocess_text(self, text):
            """
            Preprocesses the given text by removing stopwords, converting to lowercase, and tokenizing.

            Args:
                text (str): The input text to be preprocessed.

            Returns:
                list: The preprocessed tokens.
            """
            stop_words = set(stopwords.words('english'))
            tokens = [token.lower() for token in word_tokenize(text) if token.isalnum() and token.lower() not in stop_words]
            return tokens

    def calculate_novel_ngram_ratio(self, source_text, summary, n=2):
            """
            Calculates the ratio of novel n-grams in the summary compared to the source text.

            Args:
                source_text (str): The source text.
                summary (str): The summary text.
                n (int, optional): The size of n-grams. Defaults to 2.

            Returns:
                float: The ratio of novel n-grams in the summary.
            """
            
            source_tokens = self.preprocess_text(source_text)
            summary_tokens = self.preprocess_text(summary)
            source_ngrams = list(ngrams(source_tokens, n))
            summary_ngrams = list(ngrams(summary_tokens, n))
            source_freq_dist = FreqDist(source_ngrams)
            novel_ngrams_count = sum(1 for ngram in summary_ngrams if ngram not in source_freq_dist)
            novel_ngram_ratio = novel_ngrams_count / len(summary_ngrams)
            return novel_ngram_ratio

    def calculate_red_score(self, summary, n=2):
            """
            Calculates the redundancy score of a summary based on the frequency of repeated n-grams.
            
            Args:
                summary (str): The summary text.
                n (int): The number of tokens in each n-gram. Default is 2.
            
            Returns:
                float: The redundancy score, ranging from 0 to 1. A score of 0 indicates no redundancy, while a score of 1 indicates maximum redundancy.
            """
            summary_tokens = self.preprocess_text(summary)
            summary_ngrams = list(ngrams(summary_tokens, n))
            freq_dist = FreqDist(summary_ngrams)
            repeated_ngrams_count = sum(fi - 1 for fi in freq_dist.values())
            total_ngrams_count = len(summary_ngrams)
            red_score = 1 - (repeated_ngrams_count / total_ngrams_count)
            return red_score

    def calculate_compression_score(self, article, summary):
            """
            Calculates the compression score of a summary compared to the original article.

            Args:
                article (str): The original article.
                summary (str): The summary of the article.

            Returns:
                float: The compression score, which is a value between 0 and 1. 
                       A score closer to 1 indicates a higher compression rate.
            """
            article_tokens = self.preprocess_text(article)
            summary_tokens = self.preprocess_text(summary)
            article_length = len(article_tokens)
            summary_length = len(summary_tokens)
            compression_score = 1 - (summary_length / article_length)
            return compression_score

    def compute(self, source_texts, summarys):
        """
        Compute the summarization metrics for a list of source texts and their corresponding summaries.

        Args:
            source_texts (list): List of source texts.
            summarys (list): List of summaries.

        Returns:
            dict: A dictionary containing the computed metrics:
                - "red_score": The average reduction score.
                - "novel_1gram_ratio": The average ratio of novel 1-grams in the summary.
                - "novel_2gram_ratio": The average ratio of novel 2-grams in the summary.
                - "novel_3gram_ratio": The average ratio of novel 3-grams in the summary.
                - "compression_score": The average compression score.
        """
        red_scores = []
        novel_1gram_ratios = []
        novel_2gram_ratios = []
        novel_3gram_ratios = []
        compression_scores = []

        for source_text, summary in zip(source_texts, summarys):
            red_score = self.calculate_red_score(summary)
            red_scores.append(red_score)
            novel_1gram_ratio = self.calculate_novel_ngram_ratio(source_text, summary, n=1)
            novel_1gram_ratios.append(novel_1gram_ratio)
            novel_2gram_ratio = self.calculate_novel_ngram_ratio(source_text, summary, n=2)
            novel_2gram_ratios.append(novel_2gram_ratio)
            novel_3gram_ratio = self.calculate_novel_ngram_ratio(source_text, summary, n=3)
            novel_3gram_ratios.append(novel_3gram_ratio)
            compression_score = self.calculate_compression_score(source_text, summary)
            compression_scores.append(compression_score)

        return {
            "red_score": sum(red_scores) / len(red_scores),
            "novel_1gram_ratio": sum(novel_1gram_ratios) / len(novel_1gram_ratios),
            "novel_2gram_ratio": sum(novel_2gram_ratios) / len(novel_2gram_ratios),
            "novel_3gram_ratio": sum(novel_3gram_ratios) / len(novel_3gram_ratios),
            "compression_score": sum(compression_scores) / len(compression_scores),
        }




if __name__ == "__main__":
    # Example usage
    source_text = "Original source text with some information to be summarized in the summary information."
    summary = "Summary of the source text with some repeated summary information."

    red_score = compute_red_score(source_text, summary)
    print(f"RED Score: {red_score}")
    novel_2gram_ratio = calculate_novel_ngram_ratio(source_text, summary, n=2)
    print(f"Novel 2-gram Ratio: {novel_2gram_ratio}")
    novel_3gram_ratio = calculate_novel_ngram_ratio(source_text, summary, n=3)
    print(f"Novel 3-gram Ratio: {novel_3gram_ratio}")
    compression_score = calculate_compression_score(source_text, summary)
    print(f"Compression Score: {compression_score}")
