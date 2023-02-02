from syllable import Encoder #hecelere ayirmak icin
import re
from nltk.util import ngrams
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def generate_ngrams(s, n):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    dt = datetime.now()
    return [" ".join(ngram) for ngram in ngrams]


def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count



def markov_chain_twogram(sentence,twogram,unique_twogram,bigram,trigram,probability):
    probability_arr = generate_twogram_matrix(unique_twogram,twogram,bigram,trigram)
    unique_bigram = unique(bigram)
    probability_arr_bigram = generate_bigram_matrix(bigram,unique_bigram)
    sentence = encoder.tokenize(sentence)
    sentence_token_bigram = list(generate_ngrams(sentence, 1))

    #ilk hece
    exist_count = bigram.count(sentence_token_bigram[0])
    probability = probability*(exist_count/len(unique_bigram))
    #ikinci heceden sonra ilk hecenin gelmesi
    row = unique_bigram.index(sentence_token_bigram[0])
    col = unique_bigram.index(sentence_token_bigram[1])
    probability = probability*probability_arr_bigram[row][col]

    sentence_token_twogram = list(generate_ngrams(sentence, 2))

    for i in range(0,len(sentence_token_twogram)-1):
        row = unique_twogram.index(sentence_token_twogram[i])
        col = unique_bigram.index(sentence_token_bigram[i+2])
        probability = probability*probability_arr[row][col]
    
    return probability    

def markov_chain_trigram(sentence_token_bigram,trigram,unique_trigram,bigram,twogram,unique_twogram,fourgram,probability):
    probability_arr = generate_trigram_matrix(unique_trigram,bigram,trigram,fourgram)
    probability_arr_twogram = generate_twogram_matrix(unique_twogram,twogram,bigram,trigram)
    unique_bigram = unique(bigram)
    probability_arr_bigram = generate_bigram_matrix(bigram,unique_bigram)

    #ilk hece
    exist_count = bigram.count(sentence_token_bigram[0])
    probability = probability*(exist_count/len(unique_bigram))
    #ikinci heceden sonra ilk hecenin gelmesi
    row = unique_bigram.index(sentence_token_bigram[0])
    col = unique_bigram.index(sentence_token_bigram[1])
    probability = probability*probability_arr_bigram[row][col]
    #ilk iki heceden sonra diğer hecenin gelmesi
    sentence_token_twogram = list(generate_ngrams(sentence, 2))
    row = unique_twogram.index(sentence_token_twogram[0])
    col = unique_bigram.index(sentence_token_bigram[2])
    probability = probability*probability_arr_twogram[row][col]


    sentence_token_trigram = list(generate_ngrams(sentence, 3))

    for i in range(0,len(sentence_token_trigram)-1):
        row = unique_trigram.index(sentence_token_trigram[i])
        col = unique_bigram.index(sentence_token_bigram[i+3])
        probability = probability*probability_arr[row][col]
    
    return probability    

def markov_chain_bigram(sentence_token,bigram,unique_bigram,probability):
    probability_arr = generate_bigram_matrix(bigram,unique_bigram)
    for i in range(0,len(sentence_token)-1):
        row = unique_bigram.index(sentence_token[i])
        col = unique_bigram.index(sentence_token[i+1])
        probability = probability*probability_arr[row][col]
    
    exist_count = bigram.count(sentence_token[0])
    probability = probability*(exist_count/len(unique_bigram))
    return probability

def markov_chain_bigram_perp(sentence_token,bigram,unique_bigram,perplexity):
    probability_arr = generate_bigram_matrix(sentence_token,bigram,unique_bigram)
    
    for i in range(0,len(sentence_token)-1):
        row = unique_bigram.index(sentence_token[i])
        col = unique_bigram.index(sentence_token[i+1])
        perplexity = perplexity + np.log2(probability_arr[row][col])
    
    exist_count = bigram.count(sentence_token[0])
    perplexity = perplexity + np.log2(exist_count/len(unique_bigram))
    perplexity = np.power(2, -perplexity)
    return perplexity


def markov_chain_twogram_perp(sentence_token_twogram,sentence_token_bigram,twogram,unique_twogram,bigram,trigram,perplexity):
    probability_arr = generate_twogram_matrix(sentence_token_twogram,unique_twogram,twogram,bigram,trigram)
    unique_bigram = unique(bigram)
    probability_arr_bigram = generate_bigram_matrix(sentence_token_bigram,bigram,unique_bigram)
    
    #ilk hece
    exist_count = bigram.count(sentence_token_bigram[0])
    perplexity = perplexity + np.log2((exist_count/len(unique_bigram)))
    #ikinci heceden sonra ilk hecenin gelmesi
    row = unique_bigram.index(sentence_token_bigram[0])
    col = unique_bigram.index(sentence_token_bigram[1])
    perplexity = perplexity + np.log2(probability_arr_bigram[row][col])

    sentence_token_twogram = list(generate_ngrams(sentence, 2))

    for i in range(0,len(sentence_token_twogram)-1):
        row = unique_twogram.index(sentence_token_twogram[i])
        col = unique_bigram.index(sentence_token_bigram[i+2])
        perplexity = perplexity + np.log2(probability_arr[row][col])

    perplexity = perplexity/2
    perplexity = np.power(2, -perplexity)
    return perplexity  

def markov_chain_trigram_perp(sentence,trigram,unique_trigram,bigram,twogram,unique_twogram,fourgram,perplexity):
    sentence_token = encoder.tokenize(sentence)
    sentence_token_bigram = list(generate_ngrams(sentence_token, 1))

    sentence_token_trigram = list(generate_ngrams(sentence_token, 3))
    probability_arr = generate_trigram_matrix(sentence_token_trigram,unique_trigram,bigram,trigram,fourgram)
    sentence_token_twogram = list(generate_ngrams(sentence_token, 2))
    probability_arr_twogram = generate_twogram_matrix(sentence_token_twogram,unique_twogram,twogram,bigram,trigram)
    unique_bigram = unique(bigram)
    probability_arr_bigram = generate_bigram_matrix(sentence_token_bigram,bigram,unique_bigram)


    #ilk hece
    exist_count = bigram.count(sentence_token_bigram[0])
    perplexity = perplexity+np.log2((exist_count/len(unique_bigram)))
    #ikinci heceden sonra ilk hecenin gelmesi
    row = unique_bigram.index(sentence_token_bigram[0])
    col = unique_bigram.index(sentence_token_bigram[1])
    perplexity = perplexity+np.log2(probability_arr_bigram[row][col])
    #ilk iki heceden sonra diğer hecenin gelmesi
    sentence_token_twogram = list(generate_ngrams(sentence_token, 2))
    row = unique_twogram.index(sentence_token_twogram[0])
    col = unique_bigram.index(sentence_token_bigram[2])
    perplexity = perplexity+np.log2(probability_arr_twogram[row][col])


    for i in range(0,len(sentence_token_trigram)-1):
        row = unique_trigram.index(sentence_token_trigram[i])
        col = unique_bigram.index(sentence_token_bigram[i+3])
        perplexity = perplexity+np.log2(probability_arr[row][col])
    
    perplexity = perplexity/3
    perplexity = np.power(2, -perplexity)

    return perplexity      

def generate_sentece(sentence_token,unique_ngram,unique_bigram,gt_array,n):
    row = unique_ngram.index(sentence_token[len(sentence_token)-1])
    cols = gt_array.shape[1]
    random_sentence = ''
    max_prob = gt_array[row][0]
    max_col = 0
    print(gt_array)
    for i in range(0,cols):
        if gt_array[row][i] > max_prob:
            max_prob = gt_array[row][i]
            max_col = i
    if n==1:
        random_sentence = random_sentence + unique_bigram[max_col]
    elif n==2 : 
        random_sentence = random_sentence + unique_bigram[max_col-1]
    else: 
        random_sentence = random_sentence + unique_bigram[max_col-5]
    row = unique_bigram.index(unique_bigram[max_col])
    max_prob = max_prob = gt_array[row][0]
    max_col = 0
    for i in range(0,cols):
        if gt_array[row][i] > max_prob:
            max_prob = gt_array[row][i]
            max_col = i
    if n==1:
        random_sentence = random_sentence + unique_bigram[max_col]
    elif n==2 : 
        random_sentence = random_sentence + unique_bigram[max_col-2]
    else :
        random_sentence = random_sentence + unique_bigram[max_col-6]
    row = unique_bigram.index(unique_bigram[max_col])
    max_prob = max_prob = gt_array[row][0]
    max_col = 0
    for i in range(0,cols):
        if gt_array[row][i] > max_prob:
            max_prob = gt_array[row][i]
            max_col = i
    if n==1:
        random_sentence = random_sentence + unique_bigram[max_col]
    elif n==2 : 
        random_sentence = random_sentence + unique_bigram[max_col-3]
    else :
        random_sentence = random_sentence + unique_bigram[max_col-7]
    row = unique_bigram.index(unique_bigram[max_col])
    max_prob = max_prob = gt_array[row][0]
    max_col = 0
    for i in range(0,cols):
        if gt_array[row][i] > max_prob:
            max_prob = gt_array[row][i]
            max_col = i
    if n==1:
        random_sentence = random_sentence + unique_bigram[max_col]
    elif n==2 : 
        random_sentence = random_sentence + unique_bigram[max_col-4]
    else :
        random_sentence = random_sentence + unique_bigram[max_col-8]
    row = unique_bigram.index(unique_bigram[max_col])
    max_prob = max_prob = gt_array[row][0]
    max_col = 0
    for i in range(0,cols):
        if gt_array[row][i] > max_prob:
            max_prob = gt_array[row][i]
            max_col = i
    if n==1:
        random_sentence = random_sentence + unique_bigram[max_col]
    elif n==2 : 
        random_sentence = random_sentence + unique_bigram[max_col-5]
    else :
        random_sentence = random_sentence + unique_bigram[max_col-9]
    return random_sentence



def generate_bigram_matrix(sentence_token,bigram,unique_bigram):
    row = len(unique_bigram)
    column = len(unique_bigram)
    zeors_array = np.zeros( (row, column) )
    probability_arr = np.zeros( (row, column) )
    for i in range(0,len(bigram)-1):
        row = unique_bigram.index(bigram[i])
        col = unique_bigram.index(bigram[i+1])
        zeors_array[row][col]+=1
    
    sparse_mtr = csr_matrix(zeors_array)
    gt_array = generate_gt_bigram_matrix(sparse_mtr,zeors_array,bigram)
    random_sentence = generate_sentece(sentence_token,unique_bigram,unique_bigram,gt_array,1)
    print("cümle bigram::::",sentence_token,random_sentence)
    return gt_array

def generate_twogram_matrix(sentence_token,unique_twogram,twogram,bigram,trigram):
    unique_bigram = unique(bigram)
    row = len(unique_twogram)
    column = len(unique_bigram)
    zeors_array = np.zeros( (row, column) )
    probability_arr = np.zeros( (row, column) )
    for i in range(0,len(unique_twogram)):
        for j in range(0,len(unique_bigram)):
            row = unique_twogram.index(unique_twogram[i])
            col = unique_bigram.index(unique_bigram[j])
            searched = unique_twogram[i]+" "+unique_bigram[j]
            exist_count = trigram.count(searched)
            zeors_array[row][col]+=exist_count
    
    sparse_mtr = csr_matrix(zeors_array)
    gt_array = generate_gt_bigram_matrix(sparse_mtr,zeors_array,bigram)
    random_sentence = generate_sentece(sentence_token,unique_twogram,unique_bigram,gt_array,2)
    print("cümle twogram::::",sentence_token,random_sentence)
    return gt_array

def generate_trigram_matrix(sentence_token,unique_trigram,bigram,trigram,fourgram):
    unique_bigram = unique(bigram)
    row = len(unique_trigram)
    column = len(unique_bigram)
    zeors_array = np.zeros( (row, column) )
    probability_arr = np.zeros( (row, column) )
    for i in range(0,len(unique_trigram)):
        for j in range(0,len(unique_bigram)):
            row = unique_trigram.index(unique_trigram[i])
            col = unique_bigram.index(unique_bigram[j])
            searched = unique_trigram[i]+" "+unique_bigram[j]
            exist_count = fourgram.count(searched)
            zeors_array[row][col]+=exist_count
    
    print("cümle trigram::::",sentence_token,random_sentence)
    return zeors_array

def find_frequency(sparse_mtr,number): 
    count = 0
    for i in range(0,len(sparse_mtr.data)):
        if sparse_mtr.data[i] == number:
                count+=1
    return count


def generate_gt_bigram_matrix(sparse_mtr,freq_arr,bigram):
    rows = freq_arr.shape[0]
    cols = freq_arr.shape[1]
    gt_freq_arr = np.zeros( (rows, cols) )
    freq_one_result = find_frequency(sparse_mtr,1)
    
    for i in range(0,rows):
        for j in range(0,cols):
            if freq_arr[i][j] == 0:
                numerator = freq_one_result
                denumerator = len(bigram)
                gt_freq_arr[i][j] = numerator/denumerator 
            else :
                count = find_frequency(sparse_mtr,freq_arr[i][j])
                numerator = (freq_arr[i][j]+1) * (find_frequency(sparse_mtr,freq_arr[i][j]+1))
                denumerator = count
                gt_freq_arr[i][j] = (numerator/denumerator)
    return gt_freq_arr



choices = {"İ":"I", "ş" : "s", "ğ" : "g", "ü" : "u", "ö" : "o", "ç" : "c", "ı" : "i", "Ş" : "S", "Ğ" : "G", "Ü" : "U", "Ö" : "O", "Ç" : "C"}
encoder = Encoder(lang="tr", limitby="vocabulary", limit=3000) 
text_file = open("dataset.txt", "r",encoding="utf-8")


line = text_file.read()
line = line.lower()
for i in range (len(line)):
    line = line.replace(line[i:i+1],choices.get(line[i],line[i]))
tokens = encoder.tokenize(line)



sentence = "mücadele edecek"
sentence = sentence.lower()
for i in range (len(sentence)):
    sentence = sentence.replace(sentence[i:i+1],choices.get(sentence[i],sentence[i]))
sentence_token = encoder.tokenize(sentence)
sentence_token1 = list(generate_ngrams(sentence_token, 1))


sentence_token_trigram = list(generate_ngrams(sentence_token, 3))
bigram = list(generate_ngrams(tokens, 1))
print("okan")
two_gram = list(generate_ngrams(tokens, 2))
print("okann")
trigram = list(generate_ngrams(tokens, 3))
print("okannn")
fourgram = list(generate_ngrams(tokens, 4))
print("okannnn")
unique_trigram = unique(trigram)
print("okannnnnn")
probability_arr = generate_trigram_matrix(sentence_token_trigram,unique_trigram,bigram,trigram,fourgram)
#TEST FOR BIGRAM
"""bigram = list(generate_ngrams(tokens, 1))
unique_bigram = unique(bigram)
probability = 1
print("Bigram perplexity",markov_chain_bigram_perp(sentence_token1,bigram,unique_bigram,probability))"""

#TEST FOR TWO GRAM
"""tri_gram = list(generate_ngrams(tokens, 3))
two_gram = list(generate_ngrams(tokens, 2))
bigram = list(generate_ngrams(tokens, 1))
unique_twogram = unique(two_gram)
unique_bigram = unique(bigram)
sentence_token2 = list(generate_ngrams(sentence_token, 2))
probability = 1
print("twogram perplexity",markov_chain_twogram_perp(sentence_token2,sentence_token1,two_gram,unique_twogram,bigram,tri_gram,probability))"""


#TEST FOR TRIGRAM
"""four_gram = list(generate_ngrams(tokens, 4))
tri_gram = list(generate_ngrams(tokens, 3))
two_gram = list(generate_ngrams(tokens, 2))
bigram = list(generate_ngrams(tokens, 1))
unique_trigram = unique(tri_gram)
unique_twogram = unique(two_gram)
unique_bigram = unique(bigram)
probability = 1
print("trigram perplexity",markov_chain_trigram_perp(sentence,tri_gram,unique_trigram,bigram,two_gram,unique_twogram,four_gram,probability))"""




