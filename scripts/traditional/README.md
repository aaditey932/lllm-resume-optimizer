# Traditional Resume Optimization Workflow

This directory contains scripts for the traditional approach to resume optimization, using NLP and similarity-based techniques.

## Overview

The traditional approach uses text processing, feature extraction, and similarity measures to analyze and optimize resumes for job applications. Unlike LLM-based approaches, these methods rely on established NLP techniques to quantify the match between resumes and job descriptions.

## Scripts

### feature_extraction.py

Extracts features from preprocessed resumes and job descriptions, including:
- TF-IDF vectorization
- Keyword extraction
- n-gram analysis
- Skills and qualification identification

Usage:
```bash
python scripts/traditional/feature_extraction.py
```

### similarity_analysis.py

Calculates similarity scores between resumes and job descriptions using:
- Cosine similarity
- Jaccard similarity
- Word embedding-based similarity (Word2Vec, GloVe)
- Skills matching ratio

Usage:
```bash
python scripts/traditional/similarity_analysis.py
```

### optimization_suggestions.py

Generates recommendations for resume optimization based on similarity analysis:
- Key missing skills
- Terminology alignment suggestions
- Experience description improvements
- Overall match score

Usage:
```bash
python scripts/traditional/optimization_suggestions.py
```

## Workflow

1. Preprocessed data is loaded from the `data/processed` and `data/outputs` directories
2. Features are extracted from both resumes and job descriptions
3. Similarity analysis quantifies the match between resumes and job descriptions
4. Optimization suggestions are generated based on similarity gaps
5. Results are saved for comparison with LLM-based approaches

## Performance Metrics

The traditional approach can be evaluated using:
- Precision, recall, and F1 score for keyword matching
- Overall similarity score distribution
- Human evaluation of suggestion quality
- Time-to-completion compared to LLM methods

## Dependencies

- scikit-learn (for TF-IDF and similarity calculations)
- spaCy (for NER and linguistic processing)
- NLTK (for text preprocessing)
- pandas (for data handling)
- numpy (for numerical operations)
- matplotlib (for visualization) 