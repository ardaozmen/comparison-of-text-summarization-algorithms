# Textrank Background Libraries
import networkx as nx
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# T5 Fine Tuned Libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Seq2Seq Libraries
import keras_preprocessing.text
import io
import json
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

# FastApi Libraries
from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import HTMLResponse

app = FastAPI()

templates = Jinja2Templates(directory="templates")

####################
##### Textrank #####
#################### 

def textrank(document):

    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
 
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
 
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    ranked = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    temp = dict(ranked)
    select_rank = [a_tuple[0] for a_tuple in ranked] 
    
    if len(ranked) == 1:
      res = [temp.get(t, 0) for t in select_rank[:1]]
      summary = ' '.join([str(elem) for elem in res])
    
    elif len(ranked) <= 5:
      res = [temp.get(t, 0) for t in select_rank[:int(len(select_rank)/2)]]
      summary = ' '.join([str(elem) for elem in res])
    
    else:
      
      res = [temp.get(t, 0) for t in select_rank[:5]]
      summary = ' '.join([str(elem) for elem in res])
       
    return summary


#######################
##### T5 Function #####
#######################

def getT5(text, max_length=150):
  
  output_dir = './output_dir'
  
  model = T5ForConditionalGeneration.from_pretrained(output_dir)
  tokenizer = T5Tokenizer.from_pretrained(output_dir)
  
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]

################################
###### T5 Function Turkish #####
################################

def getTR5(text, max_length=150):
  
  output_dir_tr = './output_dir_tr'
  
  model = T5ForConditionalGeneration.from_pretrained(output_dir_tr)
  tokenizer = T5Tokenizer.from_pretrained(output_dir_tr)
  
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]


######################
###### Seq2seq  ######
######################

class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def _init_(self, **kwargs):
        super(AttentionLayer, self)._init_(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

# For Turkish Seq2Seq
with open('Seq2Seq/tokenizer1.json') as f:
    data = json.load(f)
    x_tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
with open('Seq2Seq/tokenizer2.json') as f:
    data = json.load(f)
    y_tokenizer = keras_preprocessing.text.tokenizer_from_json(data)

encoder_model=tf.keras.models.load_model('Seq2Seq/encoder.h5')
decoder_model=tf.keras.models.load_model('Seq2Seq/decoder.h5',custom_objects={'AttentionLayer': AttentionLayer})

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index
max_text_len = 150
max_summary_len = 15

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

def summarize(array):
  seq2=[]
  seq=x_tokenizer.texts_to_sequences(array)
  seq2=pad_sequences(seq,  maxlen=max_text_len, padding='post') 
  return seq2


# For English Seq2Seq
with open('Seq2Seqen/tokenizer1_en.json') as f:
    data = json.load(f)
    x_tokenizer_en = keras_preprocessing.text.tokenizer_from_json(data)
with open('Seq2Seqen/tokenizer2_en.json') as f:
    data = json.load(f)
    y_tokenizer_en = keras_preprocessing.text.tokenizer_from_json(data)

encoder_model_en=tf.keras.models.load_model('Seq2Seqen/encoder_en.h5')
decoder_model_en=tf.keras.models.load_model('Seq2Seqen/decoder_en.h5',custom_objects={'AttentionLayer': AttentionLayer})

reverse_target_word_index_en=y_tokenizer_en.index_word
reverse_source_word_index_en=x_tokenizer_en.index_word
target_word_index_en=y_tokenizer_en.word_index
max_text_len_en = 200
max_summary_len_en = 30

def decode_sequence_en(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model_en.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index_en['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model_en.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index_en[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len_en-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2text_en(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index_en[i]+' '
    return newString

def summarize_en(array):
  seq2=[]
  seq=x_tokenizer_en.texts_to_sequences(array)
  seq2=pad_sequences(seq,  maxlen=max_text_len_en, padding='post') 
  return seq2

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
def texter(request: Request, text0:str = Form(...), values:str = Form(...)):
    
    if values == 'textrank':
      TextRank = textrank(text0)
      return templates.TemplateResponse("index.html", {"request": request, "summary": TextRank})
    
    elif values == 't5':
      t5 = getT5(text0)
      return templates.TemplateResponse("index.html", {"request": request, "summary": t5})

    elif values == 'tr5':
      tr5 = getTR5(text0)
      return templates.TemplateResponse("index.html", {"request": request, "summary": tr5})
    
    elif values == 'seq2seq':
      seq2seq = decode_sequence(summarize([text0]).reshape(1,150))
      return templates.TemplateResponse("index.html", {"request": request, "summary": seq2seq})

    elif values == 'seq2seqen':
      seq2seqen = decode_sequence_en(summarize_en([text0]).reshape(1,200))
      return templates.TemplateResponse("index.html", {"request": request, "summary": seq2seqen})
    
    else:
      return templates.TemplateResponse("index.html", {"request": request, "summary": 'Error'})


    
      