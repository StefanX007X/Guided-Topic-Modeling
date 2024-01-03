# Guided-Topic-Modeling
Guided Topic Modeling (GTM) enables the generation of comprehensive topic word clusters (i.e. topic dictionaries) for a broad range of topics.

**Requirements:**  
* Python 3.11.4
* Conda
* Packages in requirements.yml

**Installation**  
* Clone the repository and cd to the folder containing the requirements.yml file.  
* Create a new virtual environment with the command:  
  *conda env create -f requirements.yml*
* Download the pre-trained Word2Vec embeddings:  
  https://drive.google.com/drive/folders/1AN1duLRZqN5cQOBpO0Ym3g1cw6Vh_gps?usp=sharing   
  and copy the downloaded .pkl file in ./models/

If the installation with the .yml file does not work install the packages manually:  
* Create a new environment: *conda create -n gtm_py11 python=3.11.4*    
* *conda activate gtm_py11*  
* *conda install numpy*  
* *conda install pandas*  
* *conda install scipy*  
* *conda install wordcloud*  
* *conda install -c conda-forge faiss*  

## Quick start
* Activate the virtual environment (conda activate gtm_py11)
* cd to the folder containing gtm.py

This command generates a 100 word topic from the two seed words 'iphone' and 'steve_jobs': 

*$python3 gtm.py --ps1 iphone --pw1 1.0 --ps2 steve_jobs --pw2 1.0 --size 100 --gravity 0.10*

should this not work, try:  
*>python gtm.py --ps1 iphone --pw1 1.0 --ps2 steve_jobs --pw2 1.0 --size 100 --gravity 0.10*

**Required Agruments:**  
--ps1      ... positive seed word 1  
--ps2      ... positive seed word 2  
--pw1      ... weight of the first positive seed word  
--pw2      ... weight of the second positive seed word  
--size     ... topic size  
--gravity  ... gravity parameter  

**Optional Agruments:**  
--ns1      ... negative seed word 1  
--ns2      ... negative seed word 2  
--nw1      ... weight of the first negative seed word  
--nw2      ... weight of the second negative seed word  

**Output:**  
* topic.csv in ./output/  
* WordCloud.png in ./output/WordClouds/  
* Log file in ./logs/  


<img width=500 height=auto src="https://github.com/quantista/Guided-Topic-Modeling/blob/main/output/Sample/topic_iphone_steve_jobs_15h_52m_01s.png?raw=true">


**Further notes**  
* All words in the vocabulary are lowercase



## Documentation

Guided Topic Modeling (GTM) is based on word embeddings that are obtained from a Word2Vec model which was trained on 10 million Thomson Reuters news articles (2.5 billion words) covering the period from 1996 to 2017. The vocabulary consists in total of 190.323 unique unigrams (single words) and bigrams (word pairs).

### Topic Generation

To generate your own topic you have to define two (positive) seed words that are indicative for a certain topic. For example, if you want to generate the topic "family" you could define the seed words "mother" and "father". Alternatively using "family" and "parents" or something similar would also be possible. Thus, you define two words that are both related to the desired topic.

### Parameters  

The GTM algorithm is designed to give the user control over the topic characteristics by specifying the parameters: seed word weights, negative seed words and the gravity parameter.  

**Weights**  
Each seed word is associated with a weight parameter, which is initially set to 1.0. By increasing the weight of one seed word relative to the other, the generated topic will be more closely centered around the word with the higher weight. If we for example define the seed words "iphone" with a weight of 2.0 and "steve_jobs" with a weight of 1.0 the topic will be centered around the concepts smartphone/iphone/electronic devices. Try it out to get a feeling. Usually, weights above 2 are not necessary.

**Negative Seed Words**  
Sometimes unwanted words appear in a topic. One solution for that would be the manual deletion of undesired words from the topic. Another option is to define negative seed words, i.e, words that should not be included in a topic. These negative seed words are associated with negative weights. Sensible values range from -0.1 to -0.5. For example if you try to generate the topic "recession" by defining the seed words "recession" and "crash" (cluster size = 100, gravity=0.15) you will notice the word "deadly_crash" appearing in the topic. This is because the word crash has multiple meanings, financial crash or car crash for example. We can avoid words unrelated to financial crashes by defining the negative seed word "deadly_crash" (e.g. weight = -0.3). With that, the topic is fully centered around the concept of a financial crash. An alternative solution would be to think about other, less ambiguous seed words.

**Gravity**  
Gravity allows to control how easily the topic is allowed to drift away from the initial seed words. If gravity is chosen to be very high (> 0.5) the topic will very likely stay centered around the initial seed words. If gravity is very low (e.g. 0) the topic will find a new center that is less determined by the seed words. Sensible values for gravity are in the range 0.05 to 0.2.

**Topic Size**  
Topic size is pretty much self-explaining, it defines the number of words collected for a specific topic.


### Word Embeddings

The hyperparameters of the Word2Vec algorithm are specifically adapted for the task of topic modeling. Thus we consider a rather low vector dimension of 64 to avoid data sparsity in vector space (curse of dimensionality). This also avoids the issue of too specific topic clusters we observed when working with higher-dimensional vectors. Also, for the task of topic modeling in mind we use the CBOW algorithm with a rather large window size of 18.

Standard Word2Vec embeddings do not contain any information about the word polarity which makes them unsuitable for the task of sentiment analysis. We trained and adapted Word2Vec embeddings, primarily for the financial domain. Thus, words that are considered as positive or negative in the financial context. Within a fully data driven method we take the feedback of the stock market to identify positive and negative words. We perform a PCA on the word embeddings obtained from Word2Vec an replace the least informative dimensions with the word polarity measure. Using these adapted embeddings allows the generation of polar topics, i.e. by defining the seed words "rise" and "surge" (positive topic) or "fall" and "decline" (negative topic) we obtain strictly polar topics.

### Model Data

**Polar Word Embeddings**  
To train the Word2Vec model we use the Python Gensim library. These embeddings are further enhanced by incorporating a measure of sentiment (we use the feedback of the stock market to obtain polar word embeddings), which allows a differentiation between positive and negative words. Details can be found in the <a target="_blank" href="https://ssrn.com/abstract=4575985">research paper</a> (Section 5.2.4 Polar Word Embeddings).

**Phrases Model for Bigrams**  
We trained a <a href="https://radimrehurek.com/gensim/models/phrases.html" target="_blank">Gensim Phrases</a> model on the Thomson Reuters news data from 1996 to 2017. Apply this model on your text data to identify the same bigrams as present in the GTM topics.
The trained Phrases Model is located in ./models/bigram_phrase_model_1996-2017.pkl

**Reference:**  
Dangl, Thomas and Salbrechter, Stefan, Guided Topic Modeling with Word2Vec: A Technical Note. Available at SSRN: <a target="_blank" href="https://ssrn.com/abstract=4575985">https://ssrn.com/abstract=4575985</a>
