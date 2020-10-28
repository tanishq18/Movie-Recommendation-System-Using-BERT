"""!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
!pip install bert-serving-server --no-deps"""

import os
import tensorflow as tf
sesh = tf.InteractiveSession()

from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser


MODEL_DIR = '/content/uncased_L-12_H-768_A-12' 
GRAPH_DIR = '/content/graph/' 
GRAPH_OUT = 'extractor.pbtxt' 

POOL_STRAT = 'REDUCE_MEAN' 
POOL_LAYER = '-2' 
SEQ_LEN = '256' 

tf.gfile.MkDir(GRAPH_DIR)

parser = get_args_parser()
carg = parser.parse_args(args=['-model_dir', MODEL_DIR,
                               '-graph_tmp_dir', GRAPH_DIR,
                               '-max_seq_len', str(SEQ_LEN),
                               '-pooling_layer', str(POOL_LAYER),
                               '-pooling_strategy', POOL_STRAT])

tmp_name, config = optimize_graph(carg)
graph_fout = os.path.join(GRAPH_DIR, GRAPH_OUT)

tf.gfile.Rename(
    tmp_name,
    graph_fout,
    overwrite=True
)
print("\nSerialized graph to {}".format(graph_fout))

import logging
import numpy as np

from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.keras.utils import Progbar

from bert_serving.server.bert.tokenization import FullTokenizer
from bert_serving.server.bert.extract_features import convert_lst_to_features

log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.handlers = []

GRAPH_PATH = "/content/graph/extractor.pbtxt" 
VOCAB_PATH = "/content/uncased_L-12_H-768_A-12/vocab.txt" 

SEQ_LEN = 256 

INPUT_NAMES = ['input_ids', 'input_mask', 'input_type_ids']
bert_tokenizer = FullTokenizer(VOCAB_PATH)

def build_feed_dict(texts):
    
    text_features = list(convert_lst_to_features(
        texts, SEQ_LEN, SEQ_LEN, 
        bert_tokenizer, log, False, False))

    target_shape = (len(texts), -1)

    feed_dict = {}
    for iname in INPUT_NAMES:
        features_i = np.array([getattr(f, iname) for f in text_features])
        features_i = features_i.reshape(target_shape).astype("int32")
        feed_dict[iname] = features_i

    return feed_dict

def build_input_fn(container):
    
    def gen():
        while True:
          try:
            yield build_feed_dict(container.get())
          except:
            yield build_feed_dict(container.get())

    def input_fn():
        return tf.data.Dataset.from_generator(
            gen,
            output_types={iname: tf.int32 for iname in INPUT_NAMES},
            output_shapes={iname: (None, None) for iname in INPUT_NAMES})
    return input_fn

class DataContainer:
  def __init__(self):
    self._texts = None
  
  def set(self, texts):
    if type(texts) is str:
      texts = [texts]
    self._texts = texts
    
  def get(self):
    return self._texts

def model_fn(features, mode):
    with tf.gfile.GFile(GRAPH_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    output = tf.import_graph_def(graph_def,
                                 input_map={k + ':0': features[k] for k in INPUT_NAMES},
                                 return_elements=['final_encodes:0'])

    return EstimatorSpec(mode=mode, predictions={'output': output[0]})
  
estimator = Estimator(model_fn=model_fn)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def build_vectorizer(_estimator, _input_fn_builder, batch_size=128):
  container = DataContainer()
  predict_fn = _estimator.predict(_input_fn_builder(container), yield_single_examples=False)
  
  def vectorize(text, verbose=False):
    x = []
    bar = Progbar(len(text))
    for text_batch in batch(text, batch_size):
      container.set(text_batch)
      x.append(next(predict_fn)['output'])
      if verbose:
        bar.add(len(text_batch))
      
    r = np.vstack(x)
    return r
  
  return vectorize

bert_vectorizer = build_vectorizer(estimator, build_input_fn)

bert_vectorizer(64*['sample text']).shape

import nltk
from nltk.corpus import reuters

nltk.download("reuters")
nltk.download("punkt")

max_samples = 256
categories = ['wheat', 'tea', 'strategic-metal', 
              'housing', 'money-supply', 'fuel']

S, X, Y = [], [], []

for category in categories:
  print(category)
  
  sents = reuters.sents(categories=category)
  sents = [' '.join(sent) for sent in sents][:max_samples]
  X.append(bert_vectorizer(sents, verbose=True))
  Y += [category] * len(sents)
  S += sents
  
X = np.vstack(X) 
X.shape

with open("embeddings.tsv", "w") as fo:
  for x in X.astype('float16'):
    line = "\t".join([str(v) for v in x])
    fo.write(line + "\n")

with open("metadata.tsv", "w") as fo:
  fo.write("Label\tSentence\n")
  for y, s in zip(Y, S):
    fo.write("{}\t{}\n".format(y, s))

from IPython.display import HTML

HTML("""
<video width="900" height="632" controls>
  <source src="https://storage.googleapis.com/bert_resourses/reuters_tsne_hd.mp4" type="video/mp4">
</video>
""")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

Xtr, Xts, Ytr, Yts = train_test_split(X, Y, random_state=34)

mlp = LogisticRegression()
mlp.fit(Xtr, Ytr)

print(classification_report(Yts, mlp.predict(Xts)))

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

dim = X.shape[1]

Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])

squared_distance = tf.reduce_sum(tf.pow(Q - S, 2), reduction_indices=1)
distance = tf.sqrt(squared_distance)

top_k = 10

top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)
top_dists = tf.negative(top_neg_dists)

from sklearn.metrics.pairwise import euclidean_distances

top_indices.eval({Q:X[0], S:X})

np.argsort(euclidean_distances(X[:1], X)[0])[:10]

Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])

Qr = tf.reshape(Q, (1, -1))

PP = tf.keras.backend.batch_dot(S, S, axes=1)
QQ = tf.matmul(Qr, tf.transpose(Qr))
PQ = tf.matmul(S, tf.transpose(Qr))

distance = PP - 2 * PQ + QQ
distance = tf.sqrt(tf.reshape(distance, (-1,)))

top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)

top_indices.eval({Q:X[0], S:X})

Q = tf.placeholder("float", [dim])
S = tf.placeholder("float", [None, dim])
S_norm = tf.placeholder("float", [None, 1])

Qr = tf.reshape(Q, (1, -1))

PP = S_norm
QQ = tf.matmul(Qr, tf.transpose(Qr))
PQ = tf.matmul(S, tf.transpose(Qr))

distance = PP - 2 * PQ + QQ
distance = tf.sqrt(tf.reshape(distance, (-1,)))

top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=top_k)

X_norm = np.sum(X**2, axis=1).reshape((-1,1))

top_indices.eval({Q:X[0], S:X, S_norm: X_norm})

class L2Retriever:
    def __init__(self, dim, top_k=3, use_norm=False, use_gpu=True):
        self.dim = dim
        self.top_k = top_k
        self.use_norm = use_norm
        config = tf.ConfigProto(
            device_count={'GPU': (1 if use_gpu else 0)}
        )
        self.session = tf.Session(config=config)
        
        self.norm = None
        self.query = tf.placeholder("float", [self.dim])
        self.kbase = tf.placeholder("float", [None, self.dim])
        
        self.build_graph()

    def build_graph(self):
      
        if self.use_norm:
            self.norm = tf.placeholder("float", [None, 1])

        distance = dot_l2_distances(self.kbase, self.query, self.norm)
        top_neg_dists, top_indices = tf.math.top_k(tf.negative(distance), k=self.top_k)
        top_dists = tf.negative(top_neg_dists)

        self.top_distances = top_dists
        self.top_indices = top_indices

    def predict(self, kbase, query, norm=None):
        query = np.squeeze(query)
        feed_dict = {self.query: query, self.kbase: kbase}
        if self.use_norm:
          feed_dict[self.norm] = norm
        
        I, D = self.session.run([self.top_indices, self.top_distances],
                                feed_dict=feed_dict)
        
        return I, D
      
def dot_l2_distances(kbase, query, norm=None):
    query = tf.reshape(query, (1, -1))
    
    if norm is None:
      XX = tf.keras.backend.batch_dot(kbase, kbase, axes=1)
    else:
      XX = norm
    YY = tf.matmul(query, tf.transpose(query))
    XY = tf.matmul(kbase, tf.transpose(query))
    
    distance = XX - 2 * XY + YY
    distance = tf.sqrt(tf.reshape(distance, (-1,)))
    
    return distance

import pandas as pd
import json

# download and build the dataset 
"""!wget http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
!tar -xvzf MovieSummaries.tar.gz"""

plots_df = pd.read_csv('MovieSummaries/plot_summaries.txt', sep='\t', header=None)
meta_df = pd.read_csv('MovieSummaries/movie.metadata.tsv', sep='\t', header=None)

plot = {}
metadata = {}
movie_data = {}

for movie_id, movie_plot in plots_df.values:
  plot[movie_id] = movie_plot 

for movie_id, movie_name, movie_genre in meta_df[[0,2,8]].values:
  genre = list(json.loads(movie_genre).values())
  if len(genre):
    metadata[movie_id] = {"name": movie_name,
                          "genre": genre}
    
for movie_id in set(plot.keys())&set(metadata.keys()):
  movie_data[metadata[movie_id]['name']] = {"genre": metadata[movie_id]['genre'],
                                            "plot": plot[movie_id]}

X, Y, names = [], [], []

for movie_name, movie_meta in movie_data.items():
  X.append(movie_meta['plot'])
  Y.append(movie_meta['genre'])
  names.append(movie_name)

X_vect = bert_vectorizer(X, verbose=True)

X_square_norm = np.sum(X_vect**2, axis=1).reshape((-1,1))

test_id = 10060 # The Matrix

retriever = L2Retriever(768, use_norm=True, top_k=10, use_gpu=False)

retriever = L2Retriever(768, use_norm=False, top_k=10, use_gpu=False)

def buildMovieRecommender(movie_names, vectorized_plots, top_k=10):
  retriever = L2Retriever(vectorized_plots.shape[1], use_norm=True, top_k=top_k, use_gpu=False)
  vectorized_norm = np.sum(vectorized_plots**2, axis=1).reshape((-1,1))
  
  def recommend(query):
    try:
      idx = retriever.predict(vectorized_plots, 
                              vectorized_plots[movie_names.index(query)], 
                              vectorized_norm)[0][1:]
      for i in idx:
        print(names[i])
    except ValueError:
      print("{} not found in movie db. Suggestions:")
      for i, name in enumerate(movie_names):
        if query.lower() in name.lower():
          print(i, name)
          
  return recommend

recommend = buildMovieRecommender(names, X_vect)

recommend("The Matrix")
