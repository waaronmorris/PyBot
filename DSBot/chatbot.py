import numpy as np 
import tensorflow as tf 
import re 
import time 

# import Cornell movie corpus dataset 
lines = open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

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



                
            
            
            
            
            
            
            


