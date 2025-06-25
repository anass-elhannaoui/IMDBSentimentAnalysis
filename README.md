# IMDB Sentiment Analysis Project

This project implements sentiment analysis on the IMDB dataset using various machine learning models to classify movie reviews as positive or negative.

## Overview
The code processes the IMDB dataset, performs text preprocessing, and trains multiple models including SimpleRNN, LSTM, LSTM with Attention, and DistilBERT for sentiment classification.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `tensorflow`, `torch`, `transformers`, `spacy`, `sklearn`, `tqdm`
- Dataset: IMDB dataset (`aclImdb_v1.tar.gz`)

## Setup
1. Mount Google Drive in Colab and extract the IMDB dataset.
2. Install required libraries:
   ```bash
   pip install pandas numpy tensorflow torch transformers spacy sklearn tqdm
   python -m spacy download en_core_web_sm
   ```

## Data Preprocessing
1. **Load Data**: Extracts reviews and labels (positive/negative) from the IMDB dataset.
2. **Clean Text**: Removes HTML tags, converts to lowercase, and removes extra whitespace.
3. **Tokenization**: Uses spaCy for lemmatization and tokenization.
4. **Sequence Processing**: Converts text to sequences and pads them to a uniform length (120 tokens).
5. **Save Data**: Saves tokenized data and labels for reuse.

## Models
1. **SimpleRNN Model**:
   - Embedding layer (128 dimensions)
   - Two SimpleRNN layers (200 and 130 units)
   - Dropout layers (0.35 and 0.25)
   - Dense output layer (sigmoid)
   - Trained for 5 epochs, batch size 32

2. **LSTM Model**:
   - Embedding layer (128 dimensions)
   - Two LSTM layers (64 and 32 units)
   - Dropout layers (0.2)
   - Dense output layer (sigmoid)
   - Trained for 5 epochs, batch size 64
   - Saved as `sentiment_lstm_modelNATIVE.keras`

3. **LSTM with Attention**:
   - Embedding layer (128 dimensions)
   - Two LSTM layers (64 and 32 units, return sequences)
   - Custom Attention layer
   - Dense layers (32 units and sigmoid output)
   - Trained for 5 epochs, batch size 64

4. **DistilBERT Model**:
   - Uses `distilbert-base-uncased` pre-trained model
   - Tokenizes data with max length of 200
   - Fine-tuned for binary classification

## Usage
1. Run the preprocessing steps to load and clean the IMDB dataset.
2. Train any of the models (SimpleRNN, LSTM, LSTM with Attention, or DistilBERT).
3. Evaluate model performance on the test set.
4. Save and load models for further use.

## Saved Files
- `tokenized_imdb_reviews2.csv`: Tokenized reviews
- `padded_sequences.npy`: Padded sequence data
- `labels.npy`: Labels
- `tokenizer.pkl`: Keras tokenizer
- `sentiment_lstm_modelNATIVE.keras`: Trained LSTM model

## Results
- **SimpleRNN**: Test accuracy printed after training.
- **LSTM**: Test accuracy printed after training.
- **LSTM with Attention**: Test accuracy printed after training.
- **DistilBERT**: Model prepared for fine-tuning (training not shown).

## Notes
- Ensure sufficient memory for processing large datasets.
- GPU recommended for faster training, especially for DistilBERT.
- Adjust hyperparameters (e.g., `MAX_VOCAB_SIZE`, `MAX_SEQUENCE_LENGTH`) as needed.