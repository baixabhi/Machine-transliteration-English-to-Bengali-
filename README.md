# Machine transliteration(English to Bengali)

# Introduction

This project aims to develop a neural network model for transliterating Romanized text to Bengali script. So basically, transliteration is the process of converting text from one script to another while preserving its pronunciation.

# Dataset and Preprocessing

The dataset we have used for training and testing consists of Romanized and corresponding Bengali text pairs, stored in JSON format. The preprocessing steps involve reading the JSON files, converting them into Pandas DataFrames, adding start and end tokens to the target text, and tokenizing the words using Keras Tokenizer.

# Model Architecture

The model architecture comprises an encoder-decoder framework with LSTM layers. The encoder uses bidirectional LSTMs to capture context from both directions, while the decoder generates the output sequence in Bengali script. Embedding layers are utilized for word representations, and dense layers with softmax activation predict the next token in the sequence.

# Training Process

The model is trained using RMSprop optimizer with a learning rate of 0.01 and a batch size of 64 over 50 epochs. Validation split is set at 20% for monitoring model performance. Loss is calculated using Sparse Categorical Crossentropy, and metrics such as accuracy and BLEU score are evaluated.

# Results and Performance

The model achieves an accuracy of X% on the training set and Y% on the validation set after 50 epochs. The BLEU score, a measure of translation quality, is Z. Visualizations of training/validation loss curves and predicted vs. actual output demonstrate the model's performance.

# Deployment

To deploy the model, we saved as an HDF5 file and can be loaded for inference. Scripts or instructions for serving predictions in a production environment are provided.

# Usage and Examples

Any users can input Romanized text into the model for transliteration. For example, entering "muthukumaran" results in the Bengali transliteration "মুথুকুমারন".

#Conclusion

The developed neural network model successfully transliterates Romanized text to Bengali. Future improvements could include fine-tuning hyperparameters, augmenting the dataset, and exploring advanced architectures for better performance.
