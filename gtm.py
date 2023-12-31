import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import faiss
import argparse
import pickle
import json
from numpy.linalg import inv, norm
from wordcloud import WordCloud
from scipy import optimize
from datetime import datetime


class SimilarityMeasure():
    def __init__(self, sim_measure='cos_similarity'):
        self.sim_measure = sim_measure
    
    def calc_similarity(self, X, Y):
        
        if self.sim_measure == 'cos_similarity':
            """
            Calculate the Cosine Similarity between X and Y
            X and Y can be either one- or two dimensional arrays
            """       
            if X.ndim > 1: 
                norm_X = norm(X, axis=1) 
            else: 
                norm_X = norm(X)
            if Y.ndim > 1: 
                norm_Y = norm(Y, axis=1) 
            else: 
                norm_Y = norm(Y)        

            if Y.ndim > X.ndim:
                cs = np.dot(Y,X)/(norm_X*norm_Y)
            else:
                cs = np.dot(X,Y)/(norm_X*norm_Y)       
            return cs
    

        elif self.sim_measure == 'cos_angle':     
            """
            Calculate the angle between two vectors
            """
            a = np.sqrt(np.dot(X,X))
            b = np.sqrt(np.dot(Y,Y))
            if a > b:
                cosine = np.arccos(b/a)
            elif b > a:
                cosine = np.arccos(a/b)
            else:
                cosine = 0
            return cosine
            
        else:
            print(f"{self.sim_measure} not available!")


class Get_PCA_Embds(object):
    def __init__(self, pca_embds):
        self.pca_embds = pca_embds
    
    def __getitem__(self, keys):
        if isinstance(keys, list):
            return np.vstack([self.pca_embds[key] for key in keys])
        else:
            return self.pca_embds[keys]  
    

class GTM():
    """
    Guided Topic Modeling
    alpha_max    ... Stop iteration if the angle between the closest vector to X and it's projection on X exceeds alpha_max
    cluster_size ... Maximum Number of words in a topic
    embd_dim     ... embedding dimension

    Efficient similarity search (Faiss):
    nlist        ... number of cells for similarity search (https://www.pinecone.io/learn/faiss-tutorial/)
    nprobe       ... number of nearby cells to search too
    """
    def __init__(self, embd_dim=64, nlist=50, nprobe=8):    

        # Load polar word embeddings 
        with open('./models/w2v_cbow_64_neg_10_window_18_100_epochs_bigrams_1996_2018_polar.pkl', 'rb') as f:
            pca_embds = pickle.load(f)

        vocab_list           = list(pca_embds.keys())
        self.vocab_series    = pd.Series(vocab_list)
        self.embeddings_dict = Get_PCA_Embds(pca_embds)
        self.cos_angle       = SimilarityMeasure(sim_measure='cos_angle').calc_similarity
        self.cos_similarity  = SimilarityMeasure(sim_measure='cos_similarity').calc_similarity          
        
        # Efficient similarity search
        self.xb = self.embeddings_dict[vocab_list].astype(np.float32)
        quantizer  = faiss.IndexFlatL2(embd_dim)
        self.index = faiss.IndexIVFFlat(quantizer, embd_dim, nlist)
        self.index.train(self.xb)
        self.index.add(self.xb)  
        self.index.nprobe = nprobe                 


    def func(self, a, W_orth, I, X, C, weights, params):
        self.X_new = X + W_orth @ np.diag(a)
        H_A   = self.X_new @ np.linalg.inv(self.X_new.T @ self.X_new) @ self.X_new.T
        RSS   = np.sum((((I-H_A) @ C) @ np.diag(weights))**2)
        return RSS    
           
    def Unitvec(v):
        return v/norm(v)
    
    def UnitColumns(self, v):
        return v/norm(v, axis=0)

    def GenWordCloud(self, X):
        topics_dict = {}
        for i, w in enumerate(self.topic):
            v = self.embeddings_dict[w]
            b = np.linalg.inv(X.T@X) @ (X.T @ v)
            v_hat = X @ b
            topics_dict[w] = np.linalg.norm(v_hat)
            
        wordcloud = WordCloud(width=900, height=600, max_words=800, relative_scaling=1, normalize_plurals=False, background_color="rgba(255, 255, 255, 1)", mode="RGBA")
        wordcloud = wordcloud.generate_from_frequencies(topics_dict)
        sorted_topics_dict = dict(sorted(topics_dict.items(), key=lambda item: item[1], reverse=True))

        # Generate WordCloud
        fig = plt.figure(figsize=(9, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"./output/WordClouds/{self.filename}.png", dpi=150, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight')
        plt.close(fig)   

        return sorted_topics_dict

          
    def run(self, params, pos_seed, neg_seed): 
        run, j         = True, 0                   
        proj_subspace  = [pos_seed[i][0] for i,_ in enumerate(pos_seed)]
        
        neg_seed_word_str = ""
        try:
            neg_seed_words = [neg_seed[i][0] for i,_ in enumerate(neg_seed)]
            for ns in neg_seed_words:
                neg_seed_word_str += ns+"  "
        except:
            neg_seed_words = []
        pos_weights = np.array([pos_seed[i][1] for i,_ in enumerate(pos_seed)])        
        neg_weights = np.array([neg_seed[i][1] for i,_ in enumerate(neg_seed)])               
                    
        self.topic = [pos_seed[i][0] for i,_ in enumerate(pos_seed) if pos_seed[i][1] > 0]
         
        self.filename = f"topic_{self.topic[0]}_{self.topic[1]}_"+datetime.now().strftime('%Hh_%Mm_%Ss')

        
        # Create log file        
        with open(f"./logs/log_{self.filename}.txt", 'w') as f:
            f.write(f"Guided Topic Modeling\n\
                    \nTopic Size:            {params['cluster_size']}\
                    \nGravity:               {params['gravity']}\
                    \nPositive seed words:   {self.topic[0]}  {self.topic[1]}\
                    \nNegative seed words:   {neg_seed_word_str}\n\n"
            )        
    
        # Similarity Search
        xq = self.embeddings_dict[proj_subspace+neg_seed_words].astype(np.float32)   # query vectors
        _, sim_idx = self.index.search(xq, params['k-similar'])         
        bucket_idx = np.unique(sim_idx.flatten())   
       
        V_buckets  = pd.DataFrame(index = self.vocab_series[bucket_idx], 
                                  data  = {'vector': list(self.xb[bucket_idx,:])})
        
        self.V_bucket = V_buckets
        
        for i, w in enumerate(proj_subspace):           
            a = V_buckets.loc[w, 'vector'].reshape(-1,1)
            A = a if i == 0 else np.hstack((A, a))
            V_buckets = V_buckets.drop([w])
        
        if len(neg_weights) >= 1:
            for i, w in enumerate(neg_seed_words):
                b = V_buckets.loc[w, 'vector'].reshape(-1,1)
                N = b if i == 0 else np.hstack((N, b))
                V_buckets = V_buckets.drop([w])
     
            # Adjust A by the negative seed words
            if N.ndim == 1:
                A = self.UnitColumns(A @ np.diag(pos_weights) + N * neg_weights)
            else:      
                A = self.UnitColumns(A @ np.diag(pos_weights) + N @ np.diag(neg_weights) @ np.ones(shape=(len(neg_weights), len(proj_subspace)))*(1/len(neg_weights)) )
         
        V = np.vstack(V_buckets.vector).T    
        X, C = A.copy(), A.copy()
        C_orth, self.var, self.resid_mean = np.array([]), np.array([]), np.array([])
        I = np.identity(X.shape[0]) 
        weights = pos_weights   
        gravity = params['gravity']
        
        while run == True:
            j += 1            
            B     = np.linalg.inv(X.T@X) @ X.T @ V
            B_adj = np.diag(pos_weights) @ B                          # Scale projection coefficients with weights 
            sel_coeff = B_adj.sum(axis=0) > 0.5*max(pos_weights)      # Select words with positive projection coefficients    
            V_proj_adj= X @ B_adj[:,sel_coeff] 
            V_orth    = V[:,sel_coeff] - (X @ B[:,sel_coeff])         # Calculate the non-adjusted orthogonal vectors 
            norm_proj = norm(V_proj_adj, axis=0)  
            norm_orth = norm(V_orth, axis=0) 
            alpha     = np.arctan(norm_orth/norm_proj)   # Angle between the word vectors and their projection onto the plane
            min_idx   = alpha.argmin()
            true_idx  = np.where(sel_coeff==True)[0] 
            idx       = true_idx[min_idx]                # Index of the smalles value of alpha       
            alpha_min = np.min(alpha)                    # Smallest alpha
            new_word  = V_buckets.index[idx]             # New word that is added to the topic
            w = V[:, idx]                                # Vector of the new word           
            C = np.vstack([C.T, w]).T                    # Append new word(s) to the proj_subspace X and fit a new (hyper) plane through all points
            self.topic.append(new_word)                  # Append new word to topic   
            
            if ((j-1) % params['update_freq']) == 0:
                w_orth = V_orth[:, min_idx]
            else:
                w_orth = self.Unitvec(w_orth + V_orth[:, min_idx])
                
            weights = weights * (1+gravity)              # Increase weights of existing topic words
            weights = np.append(weights, 1)              # Add the weight of the newly added word
            
            if (j % params['update_freq']) == 0:                              
                W_orth = np.array([w_orth.T]*X.shape[1]).T           
                result = optimize.minimize(self.func, [0]*X.shape[1], method="CG", args=(W_orth, I, X, C, weights, params))   # Optimize
                X      = self.UnitColumns(self.X_new)                                                                         # Update X
            
            V_buckets = V_buckets.drop([new_word])                       
            V = np.vstack(V_buckets.vector).T    
            
            gravity = max(0, gravity - params['gravity']/params['cluster_size'])       # Decay gravity to 0 
                        
            if j == 1:
                C_orth = V_orth[:, min_idx]   
                self.resid_sum = np.array([norm(C_orth)])
            else:
                C_orth   = np.vstack([C_orth.T, V_orth[:, min_idx]]).T
                self.var = np.append(self.var, np.var(C_orth, axis=1).mean())
                self.resid_mean = np.append(self.resid_mean, norm(C_orth, axis=0).mean())
                                
            print(f"{new_word: <30} word #{j:<3}; angle: {alpha_min:.3f}")
            if ((alpha_min > params['alpha_max']) & (j >= 10)) or (C.shape[1] >= params['cluster_size']) or (len(true_idx) == 1):
                run = False

        topics_dict = self.GenWordCloud(X)
        topics_dict = pd.DataFrame.from_dict(topics_dict, orient='index', columns=["weight"])
        topics_dict.to_csv(f'./output/{self.filename}.csv', encoding='utf-8', index=True)

        temp_log = ""
        for i, word in enumerate(topics_dict.index):
            temp_log += f"#{i:<3} {word: <30} weight: {topics_dict.loc[word, 'weight']:.3f}\n"

        # Write topic to log file.
        with open(f"./logs/log_{self.filename}.txt", 'a') as f:   
            f.write(temp_log) 
        

                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps1', type=str, required=True)
    parser.add_argument('--ps2', type=str, required=True)
    parser.add_argument('--pw1', type=str, required=True)
    parser.add_argument('--pw2', type=str, required=True)
    parser.add_argument('--ns1', type=str, required=False)
    parser.add_argument('--ns2', type=str, required=False)
    parser.add_argument('--nw1', type=str, required=False)
    parser.add_argument('--nw2', type=str, required=False)
    parser.add_argument('--size', type=str, required=True)
    parser.add_argument('--gravity', type=str, required=True)
    run  = True
    args = parser.parse_args()

    print('Initialize GTM')
    gtm = GTM()

    pos_seed = [(args.ps1, float(args.pw1)), (args.ps2, float(args.pw2))]

    if ((type(args.ns1) != type(None)) & (type(args.ns2) != type(None))):
        neg_seed = [(args.ns1, float(args.nw1)), (args.ns2, float(args.nw2))]
    elif ((type(args.ns1) != type(None)) & (type(args.ns2) == type(None))):
        neg_seed = [(args.ns1, float(args.nw1))]
    else:
        neg_seed = []

    # Check if defined seed words are included in the vocabulary
    for word_i in pos_seed+neg_seed:
        if word_i[0] in gtm.vocab_series.values:
            pass
        else:
            print(f"{word_i[0]} is not part of the GTM vocabulary. Select a different seed word!")
            run = False

    if run:
        params = {
            'cluster_size': float(args.size),      
            'gravity'    :  float(args.gravity),
            'alpha_max':    2.0,      # Stop iteration if the next closest word has a projection angle exceeding alpha_max        
            'update_freq':  1,        # Re-fit the proj_subspace after every x words that are added to the topic
            'k-similar'  :  5000,     # Number of similar words per seed word obtained from faiss similarity search
        }

        print('Generate Topic')
        gtm.run(params, pos_seed, neg_seed)