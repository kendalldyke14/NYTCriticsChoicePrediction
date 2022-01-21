# Basics:
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import tqdm
import re
from collections import Counter

# SKLEARN:
from sklearn.metrics.pairwise import cosine_similarity

# GENSIM / Language Processing:
import nltk
from nltk.corpus import stopwords
import spacy
import gensim.models
import gensim.utils
import gensim.corpora as corpora
import gensim.downloader as api
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess,ClippedCorpus
import en_core_web_sm

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nltk.download('stopwords',quiet=True)

def get_texts_and_corpus(review_text,bigram_min_count=3,bigram_threshold=10,nlp=nlp):
    """This function returns the corpus, id2word, and text outputs needed for processing in Gensim.

    Args:
        review_text ([type]): [description]
        bigram_min_count (int, optional): [description]. Defaults to 3.
        bigram_threshold (int, optional): [description]. Defaults to 10.
        nlp ([type], optional): [description]. Defaults to nlp.

    Returns:
        [type]: [description]
    """

    # Remove punctuation
    review_text['review_text_processed'] = review_text['review_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    review_text['review_text_processed'] = review_text['review_text_processed'].map(lambda x: x.lower())

    # Get Data Words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    data = review_text["review_text_processed"].values.tolist()
    data_words = list(sent_to_words(data))
    
    # Build the bigram models
    bigram = gensim.models.Phrases(data_words, min_count=bigram_min_count, threshold=bigram_threshold) # higher threshold fewer phrases.
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    stop_words = stopwords.words('english')
    stop_words.extend(['re'])

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus,id2word,texts

def compute_coherence_values(corpus, dictionary, texts, k, a, b, min_prob):
    """[summary]

    Args:
        corpus ([type]): [description]
        dictionary ([type]): [description]
        texts ([type]): [description]
        k ([type]): [description]
        a ([type]): [description]
        b ([type]): [description]
        min_prob ([type]): [description]

    Returns:
        [type]: [description]
    """

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=42,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           minimum_probability=min_prob)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

def lda_gridsearch(review_text,grid_search_params):
    """[summary]

    Args:
        review_text ([type]): [description]
        grid_search_params ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Pct of Corpus to use:
    validation_set_corpus_pct = grid_search_params['validation_set_corpus_pct']
    # Topics range
    topics_range = grid_search_params["topics_range"]
    # Alpha parameter
    alpha = grid_search_params["alpha"]
    # Eta parameter
    eta = grid_search_params["eta"]
    # Minimum Probability Parameter:
    min_prob = grid_search_params["minimum_probability"]
    # 
    bigram_mincount=grid_search_params["bigram_min_count"]
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=len(validation_set_corpus_pct)*len(topics_range)*len(alpha)*len(eta)*len(min_prob)*len(bigram_mincount))
        
        corpus_title = ['100% Corpus']
        model_results = {'validation_set_corpus_pct': [],
                            'topics_range': [],
                            'alpha': [],
                            'eta': [],
                            'minimum_probability': [],
                            'bigram_min_count': [],
                            'coherence': []
                            }

        for m in bigram_mincount:
            corpus,id2word,texts = get_texts_and_corpus(review_text,bigram_min_count=m,bigram_threshold=10,nlp=nlp)

            # Validation sets
            corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(len(corpus)*x)) for x in validation_set_corpus_pct]
                        
            # iterate through validation corpuses
            for i in range(len(corpus_sets)):
                # iterate through number of topics
                for k in topics_range:
                    # iterate through alpha values
                    for a in alpha:
                        # iterare through Eta values
                        for b in eta:
                            # get the coherence score for the given parameters
                            for prob in min_prob:
                                cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                            texts=texts, k=k, a=a, b=b, min_prob=prob)
                                # Save the model results
                                model_results['validation_set_corpus_pct'].append(validation_set_corpus_pct[i])
                                model_results['topics_range'].append(k)
                                model_results['alpha'].append(a)
                                model_results['eta'].append(b)
                                model_results['minimum_probability'].append(prob)
                                model_results['bigram_min_count'].append(m)
                                model_results['coherence'].append(cv)

                                pbar.update(1)
        model_results_df=pd.DataFrame(model_results)
        pbar.close()
        return model_results_df


def get_top_distinct_words_per_topic(lda_model,num_words_to_show=10,num_words_init=10):
    """[summary]

    Args:
        lda_model ([type]): [description]
        num_words_to_show (int, optional): [description]. Defaults to 10.
        num_words_init (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """

    num_words=num_words_init
    words_per_topic_lst=[]
    nondistinct_topic_words=[]
    while [0 if len(words_per_topic_lst)<1 else min([len(words) for words in words_per_topic_lst])][0]<num_words_to_show:
        x=lda_model.show_topics(num_topics=-1, num_words=num_words,formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
        top_words_per_topic = [words for _,words in topics_words]
        top_words_per_topic_flat = [item for sublist in top_words_per_topic for item in sublist]

        # Get Distinct Words per Topic:
        word_count=Counter(top_words_per_topic_flat)
        nondistinct_topic_words.extend([key for key,val in word_count.items() if val >1])

        words_per_topic_lst_temp=[]
        for tp in x:
            curr_top_wrd_lst=[]
            for tup in tp[1]:
                if tup[0] not in nondistinct_topic_words:
                    curr_top_wrd_lst.append(tup[0])
            words_per_topic_lst_temp.append(curr_top_wrd_lst)
        if [0 if len(words_per_topic_lst_temp)<1 else min([len(words) for words in words_per_topic_lst_temp])][0]==num_words_to_show:
            words_per_topic_lst=[word_lst[:num_words_to_show] for word_lst in words_per_topic_lst_temp]
            break
        num_words+=1
    return words_per_topic_lst