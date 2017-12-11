
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

np.random.seed(42)



# In[2]:

hun_chars = 'aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz' + '^$'  # ^,$


def hyph_tags(word, hypher=pyphen.Pyphen(lang='hu_HU'), aslist=False):
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


def same_char_num(word, hypher=pyphen.Pyphen(lang='hu_HU')):
    """Return true if the hyphenated word has as many chars as the original"""
    return len(hypher.inserted(word)) == len(word)+len(hypher.positions(word))


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
    
def hyph_tupples(data, hypher=pyphen.Pyphen(lang='hu_HU'),
                tag_chars='BM'):
    """[words] -> [words, hyph_words]"""
    word_list = []
    c_all = 0
    c_same_char_num = 0
    for next_word in data:
        c_all += 1
        if(len(next_word) != 0 and same_char_num(next_word, hypher)):
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
                     language='hu', no_split=False):
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
                                                    hypher=hypher)
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

# In[7]:

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



# # HU-EN Data

# In[11]:

scale = 100

window_length = 7
length_after = 3
tag_chars = 'BM'
num_layers = 5
num_hidden = 110

tests_hu_words = read_hu_en_data("hu_en_datas/hu_tests.txt")
tests_en_words = read_hu_en_data("hu_en_datas/en_tests.txt")

tests_data_hu = create_input_form_words(tests_hu_words, padding=30)
tests_data_en = create_input_form_words(tests_en_words, padding=30)

valid_hu_words = read_hu_en_data("hu_en_datas/hu_valid.txt")
valid_en_words = read_hu_en_data("hu_en_datas/en_valid.txt")
train_hu_words = read_hu_en_data("hu_en_datas/hu_train.txt")
train_en_words = read_hu_en_data("hu_en_datas/en_train.txt")


# In[41]:

train_param = "HU0EN100"
print(train_param)

model_en = model_creator_cnn(len(tag_chars), num_layers=2, num_hidden=1024,kernel_size=8)
earlyStopping_cnn = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

valid_words = valid_en_words[:200*scale]
valid_data = create_input_form_words(valid_words, padding=30)
train_words = train_en_words[:1000*scale]
train_data = create_input_form_words(train_words, padding=30)

history_cnn = model_en.fit(train_data["input_cnn"], train_data["target_cnn"],
                            epochs=10*scale, batch_size=1024,
                            validation_data=(valid_data["input_cnn"], valid_data["target_cnn"]),
                            verbose=0, callbacks=[earlyStopping_cnn])
ev_hu = test_ev(model_en,"cnn", verbose=0, tests_data=tests_data_hu)
ev_en = test_ev(model_en,"cnn", verbose=0, tests_data=tests_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)

train_param = "HU20EN80"
print(train_param)

model_cnn = model_creator_cnn(len(tag_chars), num_layers=2, num_hidden=1024,kernel_size=8)

valid_words = shuffle_concat_2D(valid_en_words[:160*scale], valid_hu_words[:40*scale])
valid_data = create_input_form_words(valid_words, padding=30)
train_words = shuffle_concat_2D(train_en_words[:800*scale], train_hu_words[:200*scale])
train_data = create_input_form_words(train_words, padding=30)

history_cnn = model_cnn.fit(train_data["input_cnn"], train_data["target_cnn"],
                            epochs=10*scale, batch_size=1024,
                            validation_data=(valid_data["input_cnn"], valid_data["target_cnn"]),
                            verbose=0, callbacks=[earlyStopping_cnn])
ev_hu = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_hu)
ev_en = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)

    
train_param = "HU50EN50"
print(train_param)    

model_cnn = model_creator_cnn(len(tag_chars), num_layers=2, num_hidden=1024,kernel_size=8)

valid_words = shuffle_concat_2D(valid_en_words[:100*scale], valid_hu_words[:100*scale])
valid_data = create_input_form_words(valid_words, padding=30)
train_words = shuffle_concat_2D(train_en_words[:500*scale], train_hu_words[:500*scale])
train_data = create_input_form_words(train_words, padding=30)

history_cnn = model_cnn.fit(train_data["input_cnn"], train_data["target_cnn"],
                            epochs=10*scale, batch_size=1024,
                            validation_data=(valid_data["input_cnn"], valid_data["target_cnn"]),
                            verbose=0, callbacks=[earlyStopping_cnn])
ev_hu = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_hu)
ev_en = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)

    
train_param = "HU80EN20"
print(train_param)    

model_cnn = model_creator_cnn(len(tag_chars), num_layers=2, num_hidden=1024,kernel_size=8)

valid_words = shuffle_concat_2D(valid_en_words[:20*scale], valid_hu_words[:160*scale])
valid_data = create_input_form_words(valid_words, padding=30)
train_words = shuffle_concat_2D(train_en_words[:200*scale], train_hu_words[:800*scale])
train_data = create_input_form_words(train_words, padding=30)

history_cnn = model_cnn.fit(train_data["input_cnn"], train_data["target_cnn"],
                            epochs=10*scale, batch_size=1024,
                            validation_data=(valid_data["input_cnn"], valid_data["target_cnn"]),
                            verbose=0, callbacks=[earlyStopping_cnn])
ev_hu = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_hu)
ev_en = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)
    
train_param = "HU90EN10"
print(train_param)    

model_cnn = model_creator_cnn(len(tag_chars), num_layers=2, num_hidden=1024,kernel_size=8)

valid_words = shuffle_concat_2D(valid_en_words[:20*scale], valid_hu_words[:180*scale])
valid_data = create_input_form_words(valid_words, padding=30)
train_words = shuffle_concat_2D(train_en_words[:100*scale], train_hu_words[:900*scale])
train_data = create_input_form_words(train_words, padding=30)

history_cnn = model_cnn.fit(train_data["input_cnn"], train_data["target_cnn"],
                            epochs=10*scale, batch_size=1024,
                            validation_data=(valid_data["input_cnn"], valid_data["target_cnn"]),
                            verbose=0, callbacks=[earlyStopping_cnn])
ev_hu = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_hu)
ev_en = test_ev(model_cnn,"cnn", verbose=0, tests_data=tests_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)
    
train_param = "HU100EN0"
print(train_param)

model_hu = model_creator_cnn(len(tag_chars), num_layers=2, num_hidden=1024,kernel_size=8)

valid_words = valid_hu_words[:200*scale]
valid_data = create_input_form_words(valid_words, padding=30)
train_words = train_hu_words[:1000*scale]
train_data = create_input_form_words(train_words, padding=30)

history_cnn = model_hu.fit(train_data["input_cnn"], train_data["target_cnn"],
                            epochs=10*scale, batch_size=1024,
                            validation_data=(valid_data["input_cnn"], valid_data["target_cnn"]),
                            verbose=0, callbacks=[earlyStopping_cnn])
ev_hu = test_ev(model_hu,"cnn", verbose=0, tests_data=tests_data_hu)
ev_en = test_ev(model_hu,"cnn", verbose=0, tests_data=tests_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)
    

train_param = "LangSelect"
print(train_param)

tests_hu_detect = []
tests_en_detect = []
detect_else = 0

for word in tests_hu_words:
    if ld.detect(word[0])=='hu':
        tests_hu_detect.append(word)
    elif ld.detect(word[0])=='en':
        tests_en_detect.append(word)
    else:
        detect_else+=1
for word in tests_en_words:
    if ld.detect(word[0])=='hu':
        tests_hu_detect.append(word)
    elif ld.detect(word[0])=='en':
        tests_en_detect.append(word)
    else:
        detect_else+=1
train_param+= '\tAll:' + str(len(tests_hu_words) + len(tests_en_words)) 
train_param+= '\tHu:' + str(len(tests_hu_detect))
train_param+= '\tEn:' + str(len(tests_en_detect))
train_param+= '\tElse:'+ str(detect_else)

detect_data_hu = create_input_form_words(tests_hu_detect, padding=30)
detect_data_en = create_input_form_words(tests_en_detect, padding=30)

ev_hu = test_ev(model_hu,"cnn", verbose=0, tests_data=detect_data_hu)
ev_en = test_ev(model_en,"cnn", verbose=0, tests_data=detect_data_en)

with open("results_HU_EN.txt", "a") as myfile:
    result = train_param+'\n'+'HU:\t'+str(ev_hu)+'\nEN:\t'+str(ev_en)+'\n'
    myfile.write(result)





