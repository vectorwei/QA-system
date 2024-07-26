# Overview
In this repository, I build a question-answering system with the SQuAD v2.0 dataset, training by a‬ pre-trained DistilBERT model from Huggingface Transformers through Bash scripts and Python‬. And I deployed the trained question-answering model using Spark-NLP and integrated Kafka for real-time data streaming‬.
‭
## Introduction

### What is SQuAD2.0?
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering. Details of SQuAD2.0 can be checked at [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) and [Know What You Don’t Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822)

### What is DistilBERT Model 
DistilBERT is a distilled version of BERT (Bidirectional Encoder Representations from Transformers), created by Hugging Face. It uses knowledge distillation to retain most of BERT's language understanding capabilities while being smaller, faster, and more efficient. With only 60% of BERT's parameters, DistilBERT is ideal for resource-constrained environments, making it a popular choice for various NLP tasks like text classification, question answering, and named entity recognition. The details of DistilBERT model can be checked at [DistilBERT, a distilled version of BERT: smaller,
faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108). The pre-trained DistilBERT model is from [Huggingface Transformers](https://github.com/huggingface/transformers).
