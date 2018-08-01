import numpy as np 
import tensorflow as tf 
import re 
import time 

# import Cornell movie corpus dataset 
lines = open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')


# STEP 1: DATA PREPROCESSING 
# create dictionary that maps line with its id (input-output)
id2line = {}
for line in lines: 
    _line = line.split(' +++$+++ ') #temp variable splits into line id & line
    if len(_line) == 5: 
        id2line[_line[0]] = _line[4] # maps line id w/its text

# create list of convos 
conversations_ids = []
for convo in conversations[:-1]: #exclude last empty row 
    # take last element & removes square brackets, quotes & spaces
    _convo = convo.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","") 
    conversations_ids.append(_convo.split(',')) # results in convo with just line id 
    
# acquire question & answer seperately 
questions = [] 
answers = [] 
for convo in conversations_ids:
    for i in range(len(convo) - 1):
        questions.append(id2line[convo[i]])
        answers.append(id2line[convo[i+1]]) 
        
# Initial cleaning of texts 
def clean_text(text): 
    text = text.lower() 
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"didn't", "did not",text)
    text = re.sub(r"don't", "does not",text)
    text = re.sub(r"doesn't", "does not",text)
    text = re.sub(r"it's","it is",text)
    text = re.sub(r"\'ll", " will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't","can not",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
    return text 

# Clean the questions 
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Clean the answers 
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
# Dict that maps each word to its # of occurences 
word2count = {}
for question in clean_questions:
    for word in question.split(): 
        if word not in word2count: # if first occurence
            word2count[word] = 1
        else:
            word2count[word] += 1 # increment # of occurences
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# Create two dictionaries that map the question words & answer words to unique int
# tokenization & filtering of words below threshold (20x--approx 5% of words)
threshold = 20 
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold: 
        questionswords2int[word] = word_number
        word_number += 1  
        
answerswords2int = {}
word_number = 0 
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

# Adding last tokens to two dicts 
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens: 
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating inverse dict of answerswords2int dict 
answersint2word = {w_i: w for w,w_i in answerswords2int.items()} #w_i = word integers

# Add EOS token to end of each answer 
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' # EOS token needed @ end of layers for seq2seq model 

# Translating all questions & answers into ints & replacing all words that were filtered out by <OUT> 
questions_into_int = []
for question in clean_questions: 
    ints = [] 
    for word in question.split(): 
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>']) # get unique int associated w/OUT 
        else: 
            ints.append(questionswords2int[word]) 
    questions_into_int.append(ints)
            
answers_into_int = []
for answer in clean_answers:
    ints = [] 
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
    
# Sorting questions & answers by length of questions --speed up training (reduce amt of padding) 
sorted_clean_questions = []
sorted_clean_answers = [] 

for length in range(1,25 +1): # go up to sentence length of 25 (add 1 bc of python range)
    for i in enumerate(questions_into_int): # we get index & question together with enumerate 
        if len(i[1]) == length: # if len of question is len to first for loop 
            sorted_clean_questions.append(questions_into_int[i[0]]) 
            sorted_clean_answers.append(answers_into_int[i[0]]) 



## PART 2 - BUILDING THE SEQ2SEQ MODEL 


# Create Placeholders for inputs & targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name='input')
    targets = tf.placeholder(tf.int32, [None,None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') # parameter that controls dropout rate
    return inputs, targets, lr, keep_prob

# Preprocessing the targets 
def preprocess_targets(targets, word2int, batch_size):
    # get left side of concatenation (up till last column)
    left_size = tf.fill([batch_size, 1], word2int['<SOS>']) # fill matrix with ids of SOS tokens
    right_size = tf.strided_slice(targets, [0,0], [batch_size,-1], [1,1]) # answers without last token (extracts subset of tensor)
    preprocessed_targets = tf.concat([left_size,right_size], axis=1)
    return preprocessed_targets

# Creating the Encoder RNN Layer 
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # create lstm 
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) #rnn size is num of tensor inputs in layers 
    # apply dropout
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    # we only need the encoder_state but this also returns encoder_ouput
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, 
                                                       cell_bw=encoder_cell, 
                                                       sequence_length=sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype=tf.float32) 
    return encoder_state
            
# Decoding the training set 
# Still to do: Update this section on decoding the training set. These classes are deprecated with TF 1.8/1.9...

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size]) # initialize
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option='bahdanau', 
                                                                                                                                    num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name='attn_dec_train') # training mode 
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    # apply final dropout to decoder_output
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
    
# Decode the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 

# Creating the Decoder RNN 
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        # dropout regularization 
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) # stacked lstm layers with dropout applied
        weights = tf.truncated_normal_initializer(stddev=0.1) # truncated normal distribution of weights 
        biases = tf.zeros_initializer() 
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state, 
                                                   decoder_cell, 
                                                   decoder_embedded_input, 
                                                   sequence_length, 
                                                   decoding_scope, 
                                                   output_function, 
                                                   keep_prob, 
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell, 
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1, 
                                           num_words, 
                                           decoding_scope, 
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions, test_predictions
    
    
# Build the Seq2Seq Model 
def seq2seq_model(inputs,targets,keep_prob,batch_size,sequence_length,answers_num_words,questions_num_words,encoder_embedding_size,decoder_embedding_size,rnn_size,num_layers,questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,#upper bound excluded)
                                                              encoder_embedding_size, 
                                                              initializer = tf.random_uniform_initializer(0,1))
    # variables to feed RNN 
    encoder_state = encoder_rnn(encoder_embedded_input,rnn_size,num_layers,keep_prob,sequence_length)
    preprocessed_targets = preprocess_targets(targets,questionswords2int,batch_size) 
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    # training and test predictions 
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, 
                                                         decoder_embeddings_matrix, 
                                                         encoder_state, 
                                                         questions_num_words, 
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions 


# Training the Seq2Seq Model 
    
# Setting the Hyperparameters
    
epochs = 100 # one whole iteration of the training
batch_size = 64 
rnn_size = 512 
num_layers = 3 #encoder & decoder RNN
encoding_embedding_size = 512  #num of columns in embedding matrix
decoding_embedding_size = 512 
learning_rate = 0.01 
learning_rate_decay = 0.9  # percentage reduced over iteration of training ~ 90% 
min_learning_rate = 0.0001 
keep_probability = 0.5 #based on Hinton article for optimal dropout 


# Defining a TF session 
tf.reset_default_graph()
session = tf.InteractiveSession()

# Load model inputs of seq2seq model 
inputs, targets, lr, keep_prob = model_inputs()

# Set sequence length 
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length') 

# Getting shape of inputs tensor 
input_shape = tf.shape(inputs)

# Getting training and test predictions 
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size, 
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)


# Loss Error, Optimizer, Gradient Clipping 
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, 
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error) 
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Pad Sequences with <PAD> token (equate length of question sentence with answer sentence)
def apply_padding(batch_of_sequences,word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Split data into batches of questions and answers 
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions)//batch_size): 
        start_index = batch_index * batch_size # first index of questions adding to batch
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int)) 
        yield padded_questions_in_batch, padded_answers_in_batch


# Split questions & answers into training and validating data 
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training 
batch_index_check_training_loss = 100 # check training loss every 100 batches
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1 
total_training_loss_error = 0 
list_validation_loss_error = []
early_stopping_check = 0 
early_stopping_stop = 1000 
checkpoint = "chatbot_weights.ckpt" 
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")









