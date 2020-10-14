# Document Classification and Topic Modeling of Scientific Papers
    
## Project Structure

<ol style="list-style-type: upper-roman;">
    <li> Introduction </li>
    <li> Data </li>
    <li> Text Preprocessing </li>
    <li> Models </li>
    <li> Evaluation </li>
    <li> Potential Improvements </li>
</ol>

## I. Introduction 
    
<p> In this project, I perform text classification on paper abstracts in Cornell University's arXiv database, a popular database of STEM papers. The objective is the following: </p>
    
<p style="text-align:center"> Given M documents constructed from a vocabulary of V unique words, find a set of K topics from which words in documents are drawn.</p>

<p> This statement of the problem assumes that documents may consist of terms that are borrowed from <b>one or more</b> of the K topics. Since we can assign one or more topic labels to each document, this a type of <i>soft clustering</i> problem. In hard clustering, we are constrained to assigning only one topic label per document. </p>
    
<p> The most intuitive way I can think of motivating this assumption is to convince yourself that academics working in different subject areas are speaking different languages. If we consider a term like "entropy," we know that it has use for at least two groups of people: physicists interested in thermodynamics, and statisticians interested in properties of probability distributions. By understanding associations to other words in the document, like "system" or "random", we can discriminate which topic the term "entropy" was more likely to have been drawn from. </p>

<p> In Data, I perform exploratory data analysis on the entire set of papers, then select the subset published from January 1st, 2019 to August 14th, 2020. 

<p> In Text Preprocessing, I explain how I transform raw text documents into vectors suitable for comparison and computation. I discuss how to convert raw text into a set of "tokens" suitable for further analysis, and only keeping tokens that are necessary to convey meaning in documents.</p>

<p>In Models, I discuss how Latent Semantic Analysis and Latent Dirichlet Allocation work as approaches to our soft classification problem.</p>

<p>In Evaluation, I discuss the results of the LSA and LDA models.  </p>

<p>Lastly, in Potential Improvements, I summarize the shortcomings of the methods used. I address some other interesting questions that arise from the earlier semantic analysis.</p>

## II. Data

The data can be found on Kaggle, <a href="https://www.kaggle.com/Cornell-University/arxiv"> here</a>. 
    
In my EDA, I answer the following questions:

1. How many papers are in the dataset?
2. What does the metadata for a paper look like?
3. What are the top categories as labelled by arXiv?
4. Who are the top authors?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.bag as db
import json
papers = db.read_text('arxiv-metadata-oai-snapshot-2020-08-14.json').map(json.loads)

### Count 

papers.count().compute()

### Example

print(json.dumps(papers.take(10)[4], indent=1))

### ArXiv Category Taxonomy 

from bs4 import BeautifulSoup
import requests

taxonomy_lk = 'https://arxiv.org/category_taxonomy'
text = requests.get(taxonomy_lk).text
soup = BeautifulSoup(text, 'html.parser')
subjects = soup.find_all('h4')[1:]
tax = {}
for subj in subjects:
    abbrev, name = subj.contents
    tax[abbrev.strip()] = name.contents[0].strip('()')
tax = pd.DataFrame(tax.values(), index=tax.keys(), columns=['subject'])
tax.loc['stat.ML'][0] = "Machine Learning (STAT)"
tax.loc['cs.LG'][0] = "Machine Learning (CS)"
tax.head()

### Top Categories 

def top_k_plot(df, k=10):
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    fig.tight_layout()
    ax.set_title('Top {} Categories'.format(k), fontsize='x-large')
    ax.barh(df['subject'][:k], df['count'][:k], edgecolor='black')
    ax.set_xlabel("Frequency", fontsize='large')
    ax.set_ylabel("Category", fontsize='large')
    plt.savefig("topk.png")
    plt.show()

ctgry_counts = (papers.map(lambda x: x['categories'].split(' '))
       .flatten()
       .frequencies(sort=True)
       .compute())
cc_df = pd.DataFrame(ctgry_counts, columns=['cat', 'count']).set_index('cat')
cc_df = tax.join(cc_df).sort_values('count', ascending=False)

%matplotlib inline
top_k_plot(cc_df, 25)

### Top Authors

def author_keys(x):
    return [' '.join(author[:2]).strip() for author in x['authors_parsed']]

(papers.map(lambda x: author_keys(x))
        .flatten()
        .frequencies(sort=True)
        .topk(10, key=1).compute())

### Subset: 2019-Present

<p> We now examine the subset of papers published from January 2019 to August 2020 (305,613 papers total). </p>

strip = lambda x: {
    'id': x['id'],
    'title': x['title'],
    'category':x['categories'],
    'abstract':x['abstract']}
last_version = lambda x: x['versions'][-1]['created']
paper_list = papers.filter(lambda x: int(last_version(x).split(' ')[3]) > 2018).map(strip).compute()
paper_df = pd.DataFrame(paper_list)
paper_df.head()

dd = {}
for subj in paper_df.category.values:
    for s in subj.split(" "):
        if s not in dd:
            dd[s] = 0
        else:
            dd[s] += 1
recent_topics = sorted(dd.items(), key= lambda x:x[1], reverse=True)
recent_topics = pd.DataFrame(recent_topics, columns=['subject', 'count']).set_index('subject')
recent_topics = tax.join(recent_topics).sort_values('count', ascending=False)

%matplotlib inline
top_k_plot(recent_topics, 20)

## III.  Text Preprocessing

### Terminology 

- **Term**: a meaningful substring in a document
- **Document**: text string
- **Corpus**: a collection of documents
- **Vocabulary**: the unique collection of terms in a corpus
    
 
Abstracts from arXiv are documents. A list of abstracts can be considered a corpus. In these new terms, we can succinctly state the goal of the text preprocessing stage as reducing document strings down to only elements of the vocabulary, $V$.

Three more steps are performed to reduce the size of $V$:
    
1. Tokenization
2. Lemmatization
3. Removal of Stop Words

### Tokenization

Tokenization is the process of chopping a string document into a set of <i>tokens</i>, or which symbols, that are representative of that document. 

For example:
<table style='font-size:12pt'>
    <thead>
        <th>Document</th>
        <th>Tokenization</th>
    </thead>
    <tbody>
    <tr>
        <td>
        "Would you like to eat soup?"
        </td>
        <td>
           ["would", "you", "like", "to", "eat", "soup", "?"]
        </td>
    </tr>
    </tbody>
</table>

<p> Tokenization of a document varies by task. Some tasks require fine details about context. A chatbot, for example, responding to sequences of user-submitted documents might need the question mark symbol included to understand that the user is asking a question. </p>
    
<p>In our specific task, we are not interested in a particular type of substring in the document: mathematical equations. So, our tokenization requires that we only accept substrings in a document that are not encapsulated by two dollar sign symbols.</p>

### Lemmatization

Languages have defined sets of rules that transform words from a root form to a new form containing an additional prefix or suffix to modify word features such as tense, number, or plurality. A simple example is Spanish, where the infinitive form of (regular) "-AR" verbs like "comprar" are transformed by concatenating suffixes from the set $\{-o,-as, \dots, -an\}$.
    
In linguistics, the root form of a word is called a _lemma_, and the transformed version is said to be _inflected_. Lemmatization takes all inflected forms of a word to their lemma, thus reducing the size of our vocabulary.

For example: 

<table style='font-size:12pt'>
    <thead>
        <th>Document</th>
        <th colspan="1">Lemmatization</th>
    </thead>
    <tbody>
        <td>
        "We shopped at several stores this afternoon"
        </td>
        <td>
           ["we", "<b>shop</b>", "at", "several", "<b>store</b>", "this", "afternoon"]
        </td>
    </tbody>
</table>

### Removal of Stop Words

After tokens are reduced to their lemmas, <i>stop words</i>, words used commonly enough to be deemed unimportant in our analysis, are removed. These removable words include some common classes of words like pronouns or determiners, which reveal little information about the conceptual content of documents.

For example: 

<table style="font-size: 12pt">
    <thead>
        <th>Document</th>
        <th colspan="1">Stop Words Removed</th>
    </thead>
    <tbody>
    <tr>
        <td>
        "We shopped at several store this afternoon"
        </td>
        <td>
           ["shopped", "several", "store", "afternoon"]
        </td>
    </tr>
    <tbody>
</table>

### In Terms of Fomal Languages
    
The framework of formal language theory helps us understand text preprocessing in general terms. We can define 

- **Alphabet** ($\Sigma$): a set of symbols 
- **Words** ($\Sigma^*$): the free monoid on the alphabet $\Sigma$
- **Language** (L): a subset of words $L\subset \Sigma^*$


    
An alphabet $\Sigma$, along with an associative operation called "concatenation" (+), and an identity element (the empty string, $\epsilon$) forms a monoid. We can then define  the set of all possible concatenations of a finite sequence of elements from $\Sigma$ as the free monoid of $\Sigma$. Our task in selecting a vocabulary $V$ from earlier can now be stated as finding a formal language, $L$, defined on an alphabet, $\Sigma$, suitable for our task.

import re
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))

def tokenize(txt):
    txt = txt.strip()
    txt = txt.replace('\n', ' ')
    txt = re.sub(r'\$(.*?)\$', '', txt) 
    txt = re.sub(r'[0-9]', '', txt)     
    txt = re.sub(r'[^a-zA-Z ]+', '', txt)
    tokens = txt.lower().split(' ')
    return [token for token in tokens if len(token) > 0]

def remove_stop(tokens):
    accepted = []
    for w in tokens:
        if (w not in stop_words):
            accepted.append(w)
    return accepted

class Tokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(t) for t in tokenize(doc)]
        return remove_stop(tokens)

### Example

<p>We can take a very ugly document from our corpus to show the result of applying the three preprocessing steps.</p>

eg = paper_df.abstract[20]
eg

eg_tok = tokenize(eg)
" ".join(eg_tok)

" ".join(remove_stop(eg_tok))

## IV. Models


<p>The corpus is now ready to be transformed into a matrix of document vectors that will be acceptable for modeling. In the above preprocessing, I took mathematical formulas (or, any characters placed between two dollar signs) to be  to be "noise" that should be removed from the data. </p>
    
    
<p>We can further reduce noise in the dataset by removing words that yield little information. Two more methods are employed to cut down the size of our vocabulary to $N$ terms.</p>

1. Using min_df to remove words that appear in less than (min_df)\% of the documents
2. Using max_df to remove words that appear more than (max_df)\% of the documents

<p>The first method removes words that have very poor representation in the corpus. This is clearly a good goal for our task, as we do not need very niche terms from some esoteric topic like Computational Geometry to understand widely represented topics in the corpus like statistical learning and ML algorithms.</p>
    

<p>The second method can be understood as the removal of "corpus-specific" stop words. Since the documents in this specific corpus are scientific vocabulary terms, we might view words like "equation" or "result," which are universally used across different scientific topics, as giving little information that is helpful in discriminating between topics. </p>


<p>Now, we can formally define the matrix we use for further analysis of the corpus.</p>

<p>Let $V$ be the vocabulary, where $|V|=N$. Consider a corpus of $M$ documents $D=\{d_1, \dots, d_M\}$, where $d_i\in \mathbb{R}^N$.</p>

<p>We need to introduce three statistics : the term frequency (TF), document frequency (DF), and inverse document frequency (IDF):</p>

<p style="text-decoration: underline;">TF </p>

tf$(t, d): V \times D \to \mathbb{R}$ defined by 

$$\text{tf}(t,d) = f_{t,d} = \text{# of occurrences of t in d}$$


<p style="text-decoration: underline;">DF </p>

df$(t, d): V \times D \to \mathbb{N}$ defined by 

$$n_t = |\{d \in D \mid t \in d\}| $$


<p style="text-decoration: underline;">IDF </p>

idf$(t, D): V \to \mathbb{R}$ defined by 

$$\text{idf}(t,D) = \log\left(\frac{N}{n_t}\right) = - \log\left(\frac{n_t}{N}\right)$$


Then, define the $\textit{term-document TF-IDF matrix}$ $X\in \mathbb{R}^{NXM}$ as

$$X = (\text{tf}(t,d) \times \text{idf}(t,D))_{t\in V, d\in D}$$



<!-- ### Information Theory

<font style="font-family: Baskerville; font-size: 14pt">

Full disclosure that this is a tangent. As an aside, I wanted to understand the concept of "divergence" between between two distributions $P$ and $Q$. A popular measure of divergence is the Kullback-Leibler divergence $D( P \| Q)$ the difference between the "cross-entropy" and "entropy" of a random variable:
    
$$D( P \| Q) = H(P, Q) - H(P).$$
    
A reader that is versed in information theory might have noticed that there is something special about the IDF function. We can view the conditional probability of observing term $t$ in the corpus as the relative document frequency:
    
$$P(t \mid D) = \frac{n_t}{N}$$ -->

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

### Vectorization

count_v = CountVectorizer(tokenizer=Tokenizer(),
                                   min_df=0.02,
                                   max_df=0.5)
X_tf = count_v.fit_transform(paper_df['abstract'])

tfidf_v = TfidfVectorizer(tokenizer=Tokenizer(),
                                   min_df=0.02,
                                   max_df=0.5)
X_tfidf = tfidf_v.fit_transform(paper_df['abstract'])

### Latent Semantic Analysis (LSA) 

#### The SVD 

Given $N$ terms in a corpus of $M$ documents, the document-term TF-IDF matrix is a real matrix $X\in \mathbb{R}^{M X N}$ which can be decomposed into a set of singular values $\sigma_1 \ge \dots \ge \sigma_d \ge 0$, left-singular vectors $\mathbf{u_1}, \dots, \mathbf{u_M}$ and right-singular vectors $\mathbf{v_1}, \dots, \mathbf{v_N}$ such that

$$ X = U \Sigma V^T = \begin{bmatrix} | & & | \\  \mathbf{u_1} & \dots & \mathbf{u_M} \\ | & & |  \end{bmatrix}    
\begin{bmatrix} D & \mathbf{0}  \\ \mathbf{0} & 0 \end{bmatrix}     
\begin{bmatrix} - & \mathbf{v_1}^T & - \\ & \vdots & \\ - & \mathbf{v_N}^T & - \end{bmatrix} $$
    
$$D = \begin{bmatrix} \mathbf{\sigma_1} & 0 & 0  \\ 0 & \ddots & 0  \\ 0 & 0 & \sigma_d  \end{bmatrix}$$

where $rank(X) = d\le N$, $\Sigma\in \mathbb{R}^{M X N}$ is diagonal, and $U, V$ are both orthogonal matrices.

Since we have that 
$$XX^T = U\Sigma V^T (V \Sigma^T U^T) = U \Sigma \Sigma^T U^T,$$
$$X^TX = V\Sigma^T U^T (U \Sigma V^T) = V \Sigma^T \Sigma V^T$$ 

this means that the columns of $U$ are eigenvectors of $XX^T$ which form an orthonormal basis for $\mathbb{R}^M$, and the columns of $V$ are eigenvectors of $X^TX$, forming an orthonormal basis for $\mathbb{R}^N$.

Now, we can form a good geometric intuition for what is happening. From the definition of SVD,

$$X V = U \Sigma V^T V$$
    
$$X [\mathbf{v_1} \dots \mathbf{v_N}] = [\sigma_1 \mathbf{u_1} \dots \sigma_d \mathbf{u_d} \  \mathbf{0} \dots \mathbf{0}]$$

The SVD neatly describes how a linear transformation $T: \mathbb{R}^N \to \mathbb{R}^M$ scales basis vectors for the domain, $\mathbb{R}^N$, to corresponding basis vectors for the codomain, $\mathbb{R}^M$. We produce an orthonormal basis for $col(X)$ by normalizing $X\mathbf{v_i}$ by its length: $\mathbf{u_i} = \frac{X \mathbf{v_i}}{\vert \vert X \mathbf{v_i} \vert \vert} = \frac{X \mathbf{v_i}}{\sigma_i}$. 

The first $k$ singular values, and their corresponding singular vectors, define the magnitudes and directions of maximal variance for the transformation $X$. 

Since $V$ consisted of a basis of eigenvectors of $X^T X$ corresponding to eigenvalues $\sigma_1^2 \ge \dots \ge \sigma_N^2\ge 0$,

$$ < v_i, (X^T X)v_i > = v_i^T (X^T X) v_i = v_i^T (V \Sigma^T \Sigma V^T) v_i  = e_i^T diag(\sigma_1^2, \dots,\sigma_N^2) e_i = \sigma_i^2 $$


#### Truncated SVD

Taking the first $k$ singular values, and their corresponding singular vectors, we can form the matrix

$$ X_k = U_k \Sigma_k V_k^T $$

which spans $k$ dimensional "topic" space. This matrix $X_k$ happens to be the best rank-k approximation to $X$.

svd = TruncatedSVD(n_components=100, random_state=0)

svd.fit(X_tfidf)

svd.explained_variance_ratio_.sum()

vocab = tfidf_v.get_feature_names()
all_topic_str = []
for i, topic in enumerate(svd.components_):
    vocab_topic = zip(vocab, topic)
    sorted_terms = sorted(vocab_topic, key= lambda x:x[1], reverse=True)[:15]
    topic_str = "Topic "+ str(i) + ": "
    for t in sorted_terms:
        topic_str += t[0] + " "
    all_topic_str.append(topic_str)
all_topic_str[:25]

%matplotlib inline
evr = np.array(list(enumerate(svd.explained_variance_ratio_))[:50])
fig = plt.figure(figsize=(20,5))
ax1, ax2 = plt.subplot(121), plt.subplot(122)
ax1.bar(evr[1:,0], evr[1:,1])
ax1.set_title('Explained Variance Ratio for Topics 1-K')
ax1.set_xlabel("Topic #")
ax1.set_ylabel("Explained Variance Ratio (%/100)")
ax2.bar(evr[1:,0], np.cumsum(evr[1:,1]))
ax2.set_title('Cumulative Variance in Topics 1-K')
ax2.set_xlabel("Topic #")
ax2.set_ylabel("Cumulative Variance Explained (%/100)")
plt.show()

### Latent Dirichlet Allocation (LDA) 

We want to discover the top $K$ topics from a corpus of $M$ documents. ***Latent Dirichlet Allocation*** is a probabilistic generative model used to discover a vector of "hidden" (latent) topics living among documents in the corpus. We are assuming that in reality the data was generated from the following process:

1. Draw $\mathbf{\theta}_i \sim Dirichlet(\alpha)$ for $i\in \{1, \dots, M\}$
2. Draw $\mathbf{\phi}_k \sim Dirichlet(\beta)$ for $k\in \{1, \dots, K\}$ 

For document $i$, with $N_i$ words total, and word position $j$ where $j\in\{1, \dots, N_i\}$,

3. Draw a topic $z_{i,j} \sim Multinomial(\mathbf{\theta}_i)$
4. Draw a word $w_{i,j} \sim Multinomial(\mathbf{\phi_{i,j}})$

lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)

lda_tf.fit(X_tf)

lda_tfidf = LatentDirichletAllocation(n_components=20, random_state=0)

lda_tfidf.fit(X_tfidf)

lda_tfidf_50 = LatentDirichletAllocation(n_components=50, random_state=0)

lda_tfidf_50.fit(X_tfidf)

## V. Evaluation 

### Visualization of LDA 

All credit goes to Carson Sievert and Kenneth E. Shirley for the creation of LDAvis <a href='https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf'>(Sievert and Shirley, 2014)</a>. The LDAVis Github can be found <a href='https://github.com/cpsievert/LDAvis'>here</a>. The tool lets us interactively understand the topics our LDA model generated in the following ways:



1. What is the meaning of each topic?
2. How prevalent is each topic in the overall corpus?
3. How are topics related?

import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

pyLDAvis.sklearn.prepare(lda_tf, X_tf, count_v)

pyLDAvis.sklearn.prepare(lda_tfidf, X_tfidf, tfidf_v)

pyLDAvis.sklearn.prepare(lda_tfidf_50, X_tfidf, tfidf_v)

## VI. Potential Improvements 

### Text Processing


- Consider adding n-grams
- Better lemmatization than NLTK (SpaCy is closer to production-grade, but more expensive)

### Modeling 

- Linear space is far too simple to understand complexities of language. Neural networks can do a better job capturing the complexity that comes with understanding context. 

### Further Questions

- Consider the tree formed by each topic as root node. What does that hierarchy look like? How to break a topic like ML into K latent sub-topics? What can we learn generally about the dependence structure of topics by recursing down to the end of this tree? (If I understand the content of a node/topic, I must understand the subtopics)