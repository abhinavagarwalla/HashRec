Code for Information Retreival Project
--------------------------------------------------

HashTag Recommendation using Topical-Attention Based LSTM

File Description:
1. preprocessing.py, clean_new.py : preprocessing tweets
2. train.py: Naive-LSTM
3. trainAttention.py: LSTM with attention
4. trainAttentionLDA.py: LSTM with Topical attention
5. trainEmb: train on own embedding; do not run (not fully tested)
6. lda.py: For learning a LDA model

For training LSTMs, just call train()
For evaluation, evalu() function