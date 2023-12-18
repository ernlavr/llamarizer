from transformers import pipeline, BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from summac.model_summac import SummaCConv
from ctc_score import SummarizationScorer
import torch
import pandas as pd
import evaluate as eval
import numpy as np
import logging
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk

class FactCC:
    def __init__(self, device = "cuda", model_path='manueldeprada/FactCC'):
        """
        Initializes the FactCC class.

        Args:
            model_path (str): The path or name of the pre-trained model to be used. Defaults to 'manueldeprada/FactCC'.
        """

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def compute(self, references,predictions):
            """
            Computes the FactCC score for a list of predicted sentences and their corresponding reference sentences.

            Args:
                predictions (list): A list of predicted sentences.
                references (list): A list of reference sentences.

            Returns:
                float: The FactCC score.
            """
            to_eval = list(zip(predictions, references))
            predictions = []
            for pred, ref in to_eval:
                input_dict = self.tokenizer(pred, ref, max_length=512, padding='max_length', truncation='longest_first', return_tensors='pt')
                input_dict = {key: val.to(self.device) for key, val in input_dict.items()}

                logits = self.model(**input_dict).logits
                softmax_probs = torch.nn.functional.softmax(logits, dim=1)
                predictions.append(softmax_probs[0][0].item())
                     
            FactCC_score = predictions
            return FactCC_score

class ANLI:

    def __init__(self, max_length = 512, model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def compute(self, sources, summaries):
        
        preds = []
        for source, summary in zip(sources, summaries):
            tokenized_input_seq_pair = self.tokenizer.encode_plus(source, summary, max_length=512,
                                                     return_token_type_ids=True, truncation=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)

            # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)
            predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
            preds.append(predicted_probability[0])
        return preds


class SummaC:

    def __init__(self,device = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=self.device, start_file="default", agg="mean") 

    def compute(self, sources, summaries):
        scores = self.model.score(sources, summaries)
        print(scores['scores'])
        return scores['scores']
    
class CTC:
    def __init__(self):
        self.scorer = SummarizationScorer(align='D-cnndm')

    def compute(self, docs, preds):
        
        scores = []
        for doc, pred in zip(docs, preds):
            score = self.scorer.score(doc=doc, refs=[], hypo=pred, aspect='consistency')
            scores.append(score)
        return np.mean(scores)

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
    def remove_padding(self,text):
        return text.replace("<s>", "").replace("</s>", "")

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
            if len(summary_ngrams) == 0:
                return 0
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
            summary_tokens = tokens = [token.lower() for token in word_tokenize(summary) if token.isalnum() and token.lower()]
            summary_ngrams = list(ngrams(summary_tokens, n))
            freq_dist = FreqDist(summary_ngrams)
            repeated_ngrams_count = sum(fi - 1 for fi in freq_dist.values())
            total_ngrams_count = len(summary_ngrams) + 0.001 # avoid division by zero

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
        
            source_text = self.remove_padding(source_text)
            summary = self.remove_padding(summary)
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
            "red_score": red_scores,
            "novel_1gram_ratio": novel_1gram_ratios,
            "novel_2gram_ratio": novel_2gram_ratios,
            "novel_3gram_ratio": novel_3gram_ratios,
            "compression_score": compression_scores,
        }


