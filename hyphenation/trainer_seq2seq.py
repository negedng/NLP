
# coding: utf-8

# In[1]:

import pyphen
import string
import re
import collections

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Embedding, LSTM
import keras

import langdetect as ld

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

np.random.seed(42)


# In[2]:

hun_chars = 'aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz' + '^$'  # ^,$


def hyph_tags(word, hypher, aslist=False):
    """Hyphenating classification of the characters in the word.
    {B(egin),M(iddle),E(nd),S(ingle)}"""
    if (len(word) == 0):
        raise IndexError("0 length word")
    ret = list('M' * len(word))
    ret[0] = 'B'
    ret[-1] = 'E'
    for i in hypher.positions(word):
        ret[i] = 'B'
        if(ret[i-1] == 'B'):
            ret[i-1] = 'S'
        else:
            ret[i-1] = 'E'
    if (aslist):
        return ret
    return "".join(ret)


def hyph_tags_4to2(word, aslist=False):
    """{B,M,E,S} to {B, M}"""
    ret = list(word)
    for i in range(len(ret)):
        if ret[i] == 'S':
            ret[i] = 'B'
        if ret[i] != 'B':
            ret[i] = 'M'
    if(aslist):
        return ret
    return "".join(ret)


def same_char_num(word, hypher):
    """Return true if the hyphenated word has as many chars as the original"""
    return len(hypher.inserted(word)) == len(word)+len(hypher.positions(word))
def only_hyphen_inserted(word, hypher):
    """Return true if the hyphenation is only hyphen addition"""
    target = hypher.inserted(word)
    i=0
    j=0
    while (i<len(word)) and (j<len(target)):
        if word[i]==target[j]:
            i+=1
            j+=1
        elif target[j]=='-':
            j+=1
        else:
            return False
    if i==len(word) and j==len(target):
        return True
    return False

def cleaning(data):
    """Text cleaning:
        lower the letters
        punctuation, digits ellimination"""
    formated_data = data.lower()
    formated_data = re.sub('['+string.punctuation+']', '', formated_data)
    formated_data = re.sub('['+string.digits+']', '', formated_data)
    return formated_data


# onehot: {'B','M','E','S'}
def one_hot_encode(char, dictionary='BM'):
    ret = [0]*len(dictionary)
    if char in dictionary:
        ret[dictionary.find(char)] = 1
        return ret
    raise ValueError('Value out of dictionary range: '+char)


def unison_shuffled_copies(a, b):
    """Randomize 2 same length array in the same permutation"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def one_hot_decode(arr, dictionary='BM'):
    assert len(arr) == len(dictionary)
    i = np.nonzero(arr)[0][0]
    return dictionary[i]


def generate_network_data(data, ret_input=[], ret_output=[],
                          length=2, length_after=0,
                          start_char='^', end_char='$',
                          chars=hun_chars, tag_chars='BMES'):
    """from [word,hyph_class(word) to length-long input-output data"""
    word = data[0]
    word_plus = start_char*(length-length_after-1)+word+end_char*length_after
    hyph_word = data[1]
    for i in range(0, len(word)):
        input_next_iter = []
        for c in word_plus[i:i+length]:
            input_next_iter.append(one_hot_encode(c, chars))
        output_next_iter = one_hot_encode(hyph_word[i], tag_chars)
        ret_input.append(input_next_iter)
        ret_output.append(output_next_iter)
    return

def generate_network_words(data, padding=None, start_char='^',
                           end_char='$', chars=hun_chars,
                           tag_chars='BM', tag_default=-1):
    """One-hot [word, hyph_class(word)]->[[[010],[010]],[[01],[01]]]
    padding to fixed size, if not null"""
    ret_input=[]
    ret_output=[]
    
    word = data[0]
    hyph_word = data[1]
    if padding != None:
        if len(word)>padding:
            raise IndexError("The word is longer than the fixed size")
        else:
            word = word + (padding-len(word))*end_char
            hyph_word = hyph_word + (padding-len(hyph_word)) * tag_chars[tag_default]
    for i in range(0,len(word)):
        input_next_iter = one_hot_encode(word[i],chars)
        output_next_iter = one_hot_encode(hyph_word[i], tag_chars)
        ret_input.append(input_next_iter)
        ret_output.append(output_next_iter)
    return ret_input, ret_output
    
def hyph_tupples(data, hypher,
                tag_chars='BM', filter_data='same_char'):
    """[words] -> [words, hyph_words]
    filter_data: same_char, only_hyphens, no_filter"""
    word_list = []
    c_all = 0
    c_same_char_num = 0
    for next_word in data:
        c_all += 1
        good_word = True
        if filter_data == 'same_char':
            good_word = same_char_num(next_word, hypher)
        elif filter_data == 'only_hyphens':
            good_word = only_hyphen_inserted(next_word, hypher)
        elif filter_data == 'no_filter':
            good_word = len(next_word)>0
        else:
            raise ValueError('filter_data not supported' + filter_data)
        if(len(next_word) != 0 and good_word):
            c_same_char_num += 1
            if(len(tag_chars) == 2):
                word_list.append([next_word,
                                  hyph_tags_4to2(hyph_tags(next_word, hypher=hypher))])
            else:
                word_list.append([next_word, hyph_tags(next_word, hypher=hypher)])
    return word_list, c_all, c_same_char_num

def tupple_to_train(word_list, window_length, length_after,
                 tag_chars='BM'):
    """[words, hyph_words] -> in[0,1,0...], out[0,1,0...]"""
    data_in = []
    data_out = []
    wrong_word = 0
    for word in word_list:
        try:
            generate_network_data(word, data_in, data_out,
                                  window_length, tag_chars=tag_chars,
                                  length_after=length_after)
        except ValueError:
            wrong_word += 1
    return data_in, data_out, wrong_word

def bigram_counter_from_file(filename):
    """creates bigram counter from file"""
    with open(filename) as f:
        word_list = []
        for words in f:
            words = words.strip()
            words = words.split()
            for w in words:
                w = cleaning(w)
                if len(w)>0:
                    word_list.append(w)

    bigram_counter = collections.Counter()
    for word in word_list:
        for i in range(2,len(word)):
            bigram_counter[word[i-2:i]] += 1
    return bigram_counter

def bigrams_in_word(word, bigram_counter, mc=100):
    bigrams = np.array(bigram_counter.most_common(mc))[:,0]
    w_bc = len(word)-1
    if w_bc<1:
        return 1.0
    w_bf = 0
    for i in range(2,len(word)):
        if word[i-2:i] in bigrams:
            w_bf +=1
    return w_bf/w_bc

def bigram_selector(word, bigram_counters,threshold=0.2, mc=100):
    """Choose the language of the word"""
    lang_likes = np.zeros(len(bigram_counters)+1)
    for i in range(0,len(bigram_counters)):
        lang_likes[i] = bigrams_in_word(word, bigram_counters[i], mc)
    lang_likes_max = np.argmax(lang_likes)
    
    for i in range(0,len(bigram_counters)):
        if i!=lang_likes_max:
            if lang_likes[lang_likes_max]-lang_likes[i]<=threshold:
                return len(bigram_counters)
    return lang_likes_max

# # Data import

# In[3]:

def data_reader(file, tail_cut=100000,
                lang_selector = False, lang_thr=0.6,
                lang_file_en='../wikipedia/angol/ossz_angol',
                lang_file_hu='../wikipedia/magyar/ossz_magyar'):
    """Read data from file"""

    if lang_selector:
        bigram_counter_en = bigram_counter_from_file(lang_file_en)
        bigram_counter_hu = bigram_counter_from_file(lang_file_hu)
        out_en_words = 0
    
    tail_cut_ptest_words = tail_cut + 500

    counter_hu_data = collections.Counter()
    with open(file, 'r',
              errors='ignore', encoding='latin2') as f:
        i = 0
        for line in f:
            i = i+1
            words = line.split()
            if len(words) > 1:
                if(words[1].isdigit()):
                    cword = cleaning(words[0])
                    if lang_selector:
                        lang = bigram_selector(cword,
                                            [bigram_counter_hu,
                                             bigram_counter_en],
                                            lang_thr)
                        if (lang!=1):
                            counter_hu_data[cword] += int(words[1])
                        else:
                            out_en_words +=1
                    else:
                        counter_hu_data[cword] += int(words[1])
            if i > tail_cut_ptest_words:
                break
    if lang_selector:
        print("Throwed english words: ", out_en_words)
    return counter_hu_data


# In[4]:

def train_data_generator(data_counter, window_length, length_after,
                         tag_chars='BM', tail_cut=100000,
                         valid_rate=0.2, test_rate=0.1,
                         language='hu'):
    """Generate training data from counter data
    unique words -> characters -> randomize -> cut"""
    if language == 'hu':
        hypher = pyphen.Pyphen(lang='hu_HU')
    elif language == 'en':
        hypher = pyphen.Pyphen(lang='en_EN')

    data_list = np.array(data_counter.most_common(tail_cut))[:,0]
    word_list, c_all, c_same_char_num = hyph_tupples(data_list,
                                                    tag_chars=tag_chars,
                                                    hypher=hypher)
    print('Data read successfully')
    print('non-standard hyphenation:')
    print(c_same_char_num, c_all, c_same_char_num/c_all)

    # Generate network data
    data_in = []
    data_out = []
    wrong_word = 0
    data_in, data_out, wrong_word = tupple_to_train(word_list,
                                                    window_length,
                                                    length_after,
                                                    tag_chars=tag_chars)
    print('Data len: ', len(data_in))
    print('Words with unrecognized caracter: ', wrong_word)

    data_len = len(data_in)

    data_in = np.array(data_in, dtype='float32')
    data_out = np.array(data_out, dtype='float32')
    data_in, data_out = unison_shuffled_copies(data_in, data_out)
    tests_input = data_in[0:int(data_len*test_rate)]
    tests_target = data_out[0:int(data_len*test_rate)]
    valid_input = data_in[int(data_len*test_rate):
                          int(data_len*(test_rate+valid_rate))]
    valid_target = data_out[int(data_len*test_rate):
                            int(data_len*(test_rate+valid_rate))]
    train_input = data_in[int(data_len*(test_rate+valid_rate)):]
    train_target = data_out[int(data_len*(test_rate+valid_rate)):]

    print('Training data size:', np.shape(train_input), np.shape(train_target))
    print('Validation data size:', np.shape(valid_input),
          np.shape(valid_target))
    print('Test data size:', np.shape(tests_input), np.shape(tests_target))

    train_input_flatten = np.reshape(
        train_input, (len(train_input), (window_length)*len(hun_chars)))
    valid_input_flatten = np.reshape(
        valid_input, (len(valid_input), (window_length)*len(hun_chars)))
    tests_input_flatten = np.reshape(
        tests_input, (len(tests_input), (window_length)*len(hun_chars)))
    print('Network data generated successfully')

    return [train_input_flatten, train_target,
            valid_input_flatten, valid_target,
            tests_input_flatten, tests_target]


def train_data_generator_uwords(data_counter, window_length, length_after,
                                tag_chars='BM', tail_cut=100000,
                                valid_rate=0.2, test_rate=0.1,
                                language='hu'):
    """Generate training data from counter data
        unique words -> randomize -> cut -> characters"""
    if language == 'hu':
        hypher = pyphen.Pyphen(lang='hu_HU')
    elif language == 'en':
        hypher = pyphen.Pyphen(lang='en_EN')
    data_list = np.array(data_counter.most_common(tail_cut))[:,0]
    np.random.shuffle(data_list)
    data_len = len(data_list)
    tests_data = data_list[0:int(data_len*test_rate)]
    valid_data = data_list[int(data_len*test_rate):
                           int(data_len*(test_rate+valid_rate))]
    train_data = data_list[int(data_len*(test_rate+valid_rate)):]
    
    c_all = 0
    c_same_char_num = 0
    tests_list, c_all_p, c_same_char_num_p = hyph_tupples(tests_data,
                                                          tag_chars=tag_chars,
                                                          hypher=hypher)
    c_all += c_all_p
    c_same_char_num += c_same_char_num_p
    valid_list, c_all_p, c_same_char_num_p = hyph_tupples(valid_data,
                                                          tag_chars=tag_chars,
                                                          hypher=hypher)
    c_all += c_all_p
    c_same_char_num += c_same_char_num_p
    train_list, c_all_p, c_same_char_num_p = hyph_tupples(train_data,
                                                          tag_chars=tag_chars,
                                                          hypher=hypher)
    c_all += c_all_p
    c_same_char_num += c_same_char_num_p
    
    print('Data read successfully')
    print('non-standard hyphenation:')
    print(c_same_char_num, c_all, c_same_char_num/c_all)
    
    wrong_word = 0
    tests_input, tests_target, wrong_w_p = tupple_to_train(tests_list,
                                                           window_length,
                                                           length_after,
                                                           tag_chars=tag_chars)
    wrong_word += wrong_w_p
    valid_input, valid_target, wrong_w_p = tupple_to_train(valid_list,
                                                           window_length,
                                                           length_after,
                                                           tag_chars=tag_chars)
    wrong_word += wrong_w_p
    train_input, train_target, wrong_w_p = tupple_to_train(train_list,
                                                           window_length,
                                                           length_after,
                                                           tag_chars=tag_chars)
    wrong_word += wrong_w_p
    print('Words with unrecognized caracter: ', wrong_word)

    print('Training data size:', np.shape(train_input), np.shape(train_target))
    print('Validation data size:', np.shape(valid_input),
          np.shape(valid_target))
    print('Test data size:', np.shape(tests_input), np.shape(tests_target))

    train_input_flatten = np.reshape(
        train_input, (len(train_input), (window_length)*len(hun_chars)))
    valid_input_flatten = np.reshape(
        valid_input, (len(valid_input), (window_length)*len(hun_chars)))
    tests_input_flatten = np.reshape(
        tests_input, (len(tests_input), (window_length)*len(hun_chars)))
    print('Network data generated successfully')

    return [train_input_flatten, train_target,
            valid_input_flatten, valid_target,
            tests_input_flatten, tests_target]
    

def train_data_generator_uchars(data_counter, window_length, length_after,
                                tag_chars='BM', tail_cut=100000,
                                valid_rate=0.2, test_rate=0.1,
                                language='hu'):
    """Generate training data from counter data
        unique words -> characters -> unique -> randomize -> cut"""
    if language == 'hu':
        hypher = pyphen.Pyphen(lang='hu_HU')
    elif language == 'en':
        hypher = pyphen.Pyphen(lang='en_EN')
    data_list = np.array(data_counter.most_common(tail_cut))[:,0]
    word_list, c_all, c_same_char_num = hyph_tupples(data_list,
                                                    tag_chars=tag_chars,
                                                     hypher=hypher)
    print('Data read successfully')
    print('non-standard hyphenation:')
    print(c_same_char_num, c_all, c_same_char_num/c_all)

    # Generate network data
    data_in = []
    data_out = []
    wrong_word = 0
    data_in, data_out, wrong_word = tupple_to_train(word_list,
                                                    window_length,
                                                    length_after,
                                                    tag_chars=tag_chars)
    print('Data len: ', len(data_in))
    print('Words with unrecognized caracter: ', wrong_word)

    #Unique
    data_len = len(data_in)

    data_in = np.array(data_in, dtype='float32')
    data_out = np.array(data_out, dtype='float32')
    
    shape_in = np.shape(data_in)
    shape_out = np.shape(data_out)
    
    data_in_flatten = np.reshape(
        data_in, (shape_in[0], shape_in[1]*shape_in[2]))
    shape_in_flatten = np.shape(data_in_flatten)
    
    data_iosum = np.concatenate((data_in_flatten, data_out), axis=1)
    data_iosum_unique = np.vstack({tuple(row) for row in data_iosum})
    
    data_in = data_iosum_unique[:,:-shape_out[1]]
    data_out = data_iosum_unique[:,-shape_out[1]:]
    print('Data unique len: ', np.shape(data_iosum_unique)[0])
    
    data_len = len(data_in)
    data_in, data_out = unison_shuffled_copies(data_in, data_out)
    tests_input = data_in[0:int(data_len*test_rate)]
    tests_target = data_out[0:int(data_len*test_rate)]
    valid_input = data_in[int(data_len*test_rate):
                          int(data_len*(test_rate+valid_rate))]
    valid_target = data_out[int(data_len*test_rate):
                            int(data_len*(test_rate+valid_rate))]
    train_input = data_in[int(data_len*(test_rate+valid_rate)):]
    train_target = data_out[int(data_len*(test_rate+valid_rate)):]

    print('Training data size:', np.shape(train_input), np.shape(train_target))
    print('Validation data size:', np.shape(valid_input),
          np.shape(valid_target))
    print('Test data size:', np.shape(tests_input), np.shape(tests_target))

    train_input_flatten = np.reshape(
        train_input, (len(train_input), (window_length)*len(hun_chars)))
    valid_input_flatten = np.reshape(
        valid_input, (len(valid_input), (window_length)*len(hun_chars)))
    tests_input_flatten = np.reshape(
        tests_input, (len(tests_input), (window_length)*len(hun_chars)))
    print('Network data generated successfully')

    return [train_input_flatten, train_target,
            valid_input_flatten, valid_target,
            tests_input_flatten, tests_target]
    


# In[5]:

def train_data_words(data_counter, tag_chars='BM', padding = 30, tail_cut=100000,
                     valid_rate=0.2, test_rate=0.1,
                     language='hu', no_split=False,
                     filter_data = 'same_char'):
    """Training data, example: alma -> {[[1,0..][0,0..][0,0...][0,0...]],[[1,0],[0,1][1,0][0,1]]}"""
    
    if language == 'hu':
        hypher = pyphen.Pyphen(lang='hu_HU')
        print("alma")
    elif language == 'en':
        hypher = pyphen.Pyphen(lang='en_EN')
        print("apple")
        
    data_list = np.array(data_counter.most_common(tail_cut))[:,0]
    word_list, c_all, c_same_char_num = hyph_tupples(data_list,
                                                    tag_chars=tag_chars,
                                                    hypher=hypher,
                                                    filter_data=filter_data)
    print('Data read successfully')
    print('non-standard hyphenation:')
    print(c_same_char_num, c_all, c_same_char_num/c_all)
    
    # Generate network data
    data_in = []
    data_out = []
    data_words = []
    wrong_word = 0
    long_word = 0
    for word in word_list:
        try:
            next_data_in, next_data_out = generate_network_words(word, padding = padding, tag_chars=tag_chars)
            next_data_in = np.array(next_data_in, dtype='float32')
            next_data_out = np.array(next_data_out, dtype='float32')
            data_in.append(next_data_in)
            data_out.append(next_data_out)
            data_words.append(word)
        except ValueError:
            wrong_word += 1
        except IndexError:
            long_word += 1
            print(word)
            
    print('Data len: ', len(data_in))
    print('Words with unrecognized caracter: ', wrong_word)
    print('Words longer than the padding: ', long_word)
    
    data_in = np.array(data_in)
    data_out = np.array(data_out)
    
    data_len = len(data_in)
    order = np.random.permutation(data_len)
    data_in = [data_in[k] for k in order]
    data_out = [data_out[k] for k in order]
    data_words = [data_words[k] for k in order]
    
    #data_in, data_out, word_list = unison_shuffled_copies(data_in, data_out, word_list)
    
    datas = {}
    
    if no_split:
        datas["input"] = np.array(data_in)
        datas["target"] = np.array(data_out)
        datas["words"] = np.array(data_words)
        return datas, wrong_word, long_word
    
    datas["tests_words"] = np.array(data_words[0:int(data_len*test_rate)])
    datas["tests_input"] = np.array(data_in[0:int(data_len*test_rate)])
    datas["tests_target"] = np.array(data_out[0:int(data_len*test_rate)])
    datas["valid_words"] = np.array(data_words[int(data_len*test_rate):
                                               int(data_len*(test_rate+valid_rate))])
    datas["valid_input"] = np.array(data_in[int(data_len*test_rate):
                                            int(data_len*(test_rate+valid_rate))])
    datas["valid_target"] = np.array(data_out[int(data_len*test_rate):
                                              int(data_len*(test_rate+valid_rate))])
    datas["train_words"] = np.array(data_words[int(data_len*(test_rate+valid_rate)):])
    datas["train_input"] = np.array(data_in[int(data_len*(test_rate+valid_rate)):])
    datas["train_target"] = np.array(data_out[int(data_len*(test_rate+valid_rate)):])
    
    return datas, wrong_word, long_word

# # Models

# In[6]:

def model_creator_dnn(window_length, output_length, num_layers=1,
                  num_hidden=10, chars=hun_chars):
    """Creates Keras model with the given input, output dimensions
    and layer number, hidden layer length"""
    
    input_shape = window_length*len(chars)
    
    model = Sequential()
    
    model.add(Dense(input_dim=(input_shape),
                    units=num_hidden, name='input_layer',
                    activation='sigmoid'))
    for i in range(1, num_layers):
        model.add(Dense(units=num_hidden, activation='sigmoid'))

    # model.add(Flatten())
    model.add(Dense(output_length, name='output_layer', activation='softmax'))

    if(output_length == 2):
        model.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def model_creator_cnn(output_length,
                      num_layers=1, num_hidden=516,
                      kernel_size=10, strides=1, word_length = 30, chars=hun_chars):
    """Creates Keras CNN model"""

    model = Sequential()
    
    model.add(Conv1D(num_hidden,kernel_size, strides=strides, padding="same",
                     activation='relu', input_shape=(word_length, len(chars))))
    for i in range(1,num_layers):
        model.add(Conv1D(num_hidden,kernel_size,strides=strides,
                         padding="same", activation='relu'))


    model.add(Dense((output_length), name = 'output_layer', activation='softmax'))
    
    if(output_length == 2):
        model.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def model_creator_lstm(output_length, 
                       num_layers=2, num_hidden=64,
                       word_length = 30, chars = hun_chars):
    """Creates Keras LSTM model"""
    
    model = Sequential()
    
    model.add(LSTM(num_hidden, activation='relu',return_sequences=True,
                   go_backwards=True,
                   input_shape=(word_length, len(chars))))
    for i in range(1,num_layers):
        model.add(LSTM(num_hidden, activation='relu',return_sequences=True,
                       go_backwards=True,))
    
    model.add(Dense((output_length), name = 'output_layer', activation='softmax'))

    
    if(output_length == 2):
        model.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# # Evaluation

# In[13]:

def hyph_predict(word, model,
                 length=2, length_after=0, tag_chars='BMES', aslist=False, model_type='dnn'):
    """Generate tagging from the input word according to the model"""
    word_in = []
    word_out = []
    generate_network_data([word, len(word)*tag_chars[0]],
                          word_in, word_out, length=length,
                          length_after=length_after, tag_chars=tag_chars)
    word_in = np.reshape(word_in, (len(word_in), (length)*len(hun_chars)))
    if model_type=='cnn':
        word_in = np.expand_dims(word_in, axis=1) # reshape (x, 1, 259) 
    word_out = model.predict(word_in)
    tag_list = np.array(list(tag_chars))
    temp = np.argmax(word_out, axis=1)
    temp = tag_list[temp]
    if(aslist):
        return temp
    return "".join(temp)


def hyph_insterted(word, model,
                   length=2, length_after=0, tag_chars='BMES', model_type='dnn'):
    tags = hyph_predict(word, model,length,
                        length_after, tag_chars, aslist=False, model_type=model_type)
    word_inserted = ""
    if tag_chars=='BM':
        for i in range(len(word)):
            if i != 0 and tags[i]=='B':
                word_inserted += '-'
            word_inserted += word[i]
    else:
        raise NotImplementedError('BM implemented only')
    return word_inserted


def evaluation(wtags_predicted, wtags_target, tag_chars='BM'): 
    """Compare BMBM with BBMB"""
    if tag_chars!='BM':
        raise NotImplementedError("Only BM available")
    tp = 0 # target: B prediction: B
    tn = 0 # target: M prediction: M
    fp = 0 # target: M prediction: B
    fn = 0 # target: B prediction: M
    for i in range(min(len(wtags_target),len(wtags_predicted))):
        c_t = wtags_target[i]
        c_p = wtags_predicted[i]
        if (c_t == 'B') and (c_p == 'B'):
            tp +=1
        elif (c_t == 'M') and (c_p == 'M'):
            tn +=1
        elif (c_t == 'M') and (c_p == 'B'):
            fp +=1
        elif (c_t == 'B') and (c_p == 'M'):
            fn +=1
        else:
            raise ValueError("Not expected tag!" + c_t + c_p)
    good = False
    if fn+fp == 0:
        good = True
    return tp, tn, fp, fn, good


def test_ev(model, model_type, model_params = None, num_tests=-1, verbose=1,
            hypher=pyphen.Pyphen(lang='hu_HU'), tests_data = None):
    """Evaulate the tests"""
    if tests_data:
        tests_input_cnn = tests_data["input_cnn"]
        tests_words = tests_data["words"]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    good = 0
    if (verbose>0):
        print("Prediction\tTarget")
    if model_type == 'cnn' or model_type =='lstm':
        test_result = model.predict(tests_input_cnn[0:num_tests])
        for i in range(len(test_result)):
            test_tags = result_decode(test_result[i])
            test_word = tests_words[i,0]
            ev = evaluation(test_tags, tests_words[i,1])
            tp += ev[0]
            tn += ev[1]
            fp += ev[2]
            fn += ev[3]
            if ev[4]:
                good+=1
            
            if (verbose>0) and (ev[4] == False):
                print(hyp_inserted(test_word,test_tags),'\t',
                      hypher.inserted(test_word))
                

    if model_type == 'dnn':
        test_range = num_tests
        if num_tests==-1:
            test_range = len(tests_words)
        window_length = model_params["window_length"]
        length_after = model_params["length_after"]
        tag_chars = model_params["tag_chars"]
        for i in range(test_range):
            test_word = tests_words[i,0]
            test_tags = hyph_predict(test_word, model_dnn,
                                       window_length, length_after,
                                       tag_chars,
                                       model_type=model_type)
            ev = evaluation(test_tags, tests_words[i,1])
            tp += ev[0]
            tn += ev[1]
            fp += ev[2]
            fn += ev[3]
            if ev[4]:
                good+=1
            
            if (verbose>0) and (ev[4] == False):
                print(hyp_inserted(test_word,test_tags),'\t',
                      hypher.inserted(test_word))
    
    test_precision = tp / (tp + fp)
    test_recall = tp / (tp + fn)
    test_Fscore = 2 * (test_precision *
                       test_recall) / (test_precision + test_recall)
    ret = {}
    ret["precision"] = test_precision
    ret["recall"] = test_recall
    ret["Fscore"] = test_Fscore
    ret["word_rate"] = good/len(tests_words)
    return ret
        
    
def result_decode(result, tag_chars='BM'):
    """[[0,1][0,1]] -> 'MM'"""
    tags = ""
    result = hardmax(result)
    for c in result:
        tags += one_hot_decode(c, tag_chars)
    return tags


def hardmax(arr, axis = -1):
    """Return 1 if the value is the max in the row, 0 otherwise
    [0.2,0.4,0.5]->[0,0,1]"""
    temp = arr - np.max(arr, axis=axis, keepdims=True)
    return np.round(1+temp)


def hyp_inserted(word, tags, tag_chars='BM'):
    """insert hyphen to the tags"""
    assert len(word)<=len(tags)
    s = ""
    for c in range(len(word)):
        if (c!=0) and tags[c]=='B':
            s+='-'
        s+=word[c]
    return s

# In[22]:

def read_hu_en_data(file):
    data_words = []
    with open(file) as f:
        for line in f:
            word = line.split()
            data_words.append(word)
    return data_words

def create_input_form_words(words, padding=30, dnn=False):
    data = {}
    data["words"] = np.array(words)
    data_input_cnn = []
    data_target_cnn = []
    for word in words:
        next_data_in, next_data_out = generate_network_words(word, padding = padding, tag_chars=tag_chars)
        data_input_cnn.append(np.array(next_data_in, dtype='float32'))
        data_target_cnn.append(np.array(next_data_out, dtype='float32'))
    data["input_cnn"] = np.array(data_input_cnn)
    data["target_cnn"] = np.array(data_target_cnn)
    if dnn:
        data_input, data_target, _ = tupple_to_train(words,
                                                        window_length,
                                                        length_after,
                                                        tag_chars=tag_chars)

        data_input_flatten = np.reshape(
            data_input, (len(data_input), (window_length)*len(hun_chars)))
        data["target"] = np.array(data_target)
        data["input_flatten"] = np.array(data_input_flatten)
    return data

def shuffle_concat_2D(arr1,arr2):
    ret = np.append(arr1,arr2,axis=0)
    order = np.random.permutation(len(ret))
    ret = [ret[k] for k in order]
    return ret


# # Main

# In[7]:

padding = 30
tail_cut = 100000
window_length = 7
length_after = 3
tag_chars = 'BM'
num_layers = 3
num_hidden = 150

# Data read and network data generate
counter_hu_data = data_reader('web2.2-freq-sorted-top200k.txt',tail_cut, lang_selector=False)

datas, wrong_words, long_word = train_data_words(counter_hu_data, tag_chars,
                                                 padding, tail_cut, language='hu',
                                                 filter_data='no_filter')

tests_input_cnn = datas["tests_input"]
tests_target_cnn = datas["tests_target"]
valid_input_cnn = datas["valid_input"]
valid_target_cnn = datas["valid_target"]
train_input_cnn = datas["train_input"]
train_target_cnn = datas["train_target"]

tests_words = datas["tests_words"]
valid_words = datas["valid_words"]
train_words = datas["train_words"]


print(np.shape(train_input_cnn), np.shape(valid_input_cnn), np.shape(tests_input_cnn))


# In[9]:

wrong_word = 0
tests_input, tests_target, wrong_w_p = tupple_to_train(tests_words,
                                                        window_length,
                                                        length_after,
                                                        tag_chars=tag_chars)
wrong_word += wrong_w_p
valid_input, valid_target, wrong_w_p = tupple_to_train(valid_words,
                                                        window_length,
                                                        length_after,
                                                        tag_chars=tag_chars)
wrong_word += wrong_w_p
train_input, train_target, wrong_w_p = tupple_to_train(train_words,
                                                        window_length,
                                                        length_after,
                                                        tag_chars=tag_chars)
wrong_word += wrong_w_p

print("Still wrong word (expected zero): ", wrong_word)

train_input_flatten = np.reshape(
    train_input, (len(train_input), (window_length)*len(hun_chars)))
valid_input_flatten = np.reshape(
    valid_input, (len(valid_input), (window_length)*len(hun_chars)))
tests_input_flatten = np.reshape(
    tests_input, (len(tests_input), (window_length)*len(hun_chars)))


# # Seq2Seq
#  - https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html


# In[30]:

padding = 30
padding_with_inserted = 40
input_characters = set('aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz')
input_characters = sorted(list(input_characters))
target_characters = set('aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz^$-')
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
max_encoder_seq_length = padding
max_decoder_seq_length = padding_with_inserted
# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def generate_seq2seq_data(words,
                          hypher=pyphen.Pyphen(lang='hu_HU')):
    # Vectorize the data.
    input_texts = []
    target_texts = []
    for word in words:
        input_text = word[0]
        target_text = hypher.inserted(input_text)
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '^' + target_text + '$'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                raise ValueError("Character not in input_charset: "+char)
        for char in target_text:
            if char not in target_characters:
                raise ValueError("Character not in output_charset: "+char)


    print('Number of samples:', len(input_texts))



    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    datas = {}
    datas["encoder_input_data"] = encoder_input_data
    datas["decoder_input_data"] = decoder_input_data
    datas["decoder_target_data"] = decoder_target_data
    return datas


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['^']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '$' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# In[21]:


batch_size = 1024  # Batch size for training.
epochs = 1000  # Number of epochs to train for.
latent_dim = 1024  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.

hypher = pyphen.Pyphen(lang='hu_HU')


train_data = generate_seq2seq_data(train_words)
encoder_input_data = train_data["encoder_input_data"]
decoder_input_data = train_data["decoder_input_data"]
decoder_target_data = train_data["decoder_target_data"]

valid_data = generate_seq2seq_data(valid_words)
valid_encoder_input_data = valid_data["encoder_input_data"]
valid_decoder_input_data = valid_data["decoder_input_data"]
valid_decoder_target_data = valid_data["decoder_target_data"]


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy')

earlyStopping_seq2seq = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7, verbose=0, mode='auto')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([valid_data["encoder_input_data"],
                            valid_data["decoder_input_data"]],
                           valid_data["decoder_target_data"]),
          callbacks=[earlyStopping_seq2seq])
# Save model
model.save('models/s2s'+str(latent_dim)+'.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)




tests_data = generate_seq2seq_data(tests_words)
tests_encoder_input_data = tests_data["encoder_input_data"]

tests_words.append(['asszonnyal',1])
tests_words.append(['rúzzsal',1])
tests_words.append(['mézzsír',1])
tests_words.append(['szivattyú',1])
tests_words.append(['nyúllyuk',1])
tests_words.append(['süllyedt',1])
tests_words.append(['késszár',1])
tests_words.append(['meggyek',1])



for seq_index in range(len(tests_words)):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = tests_encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    test_res = tests_words[seq_index,0]+'\t'+decoded_sentence+'\n'

    with open("results_seq2seq_"+str(latent_dim)+".txt", "a") as myfile:
        result = test_res
        myfile.write(result)



