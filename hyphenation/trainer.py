import pyphen
import string
import re
import collections

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
import keras


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
def one_hot_encode(char, dictionary='BMES'):
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


def one_hot_decode(arr, dictionary='BMES'):
    assert len(arr) == len(dictionary)
    i = np.nonzero(arr)[0][0]
    return dictionary[i]


hun_chars = 'aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz' + '^$'  # ^,$


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


def model_creator(window_length, output_length, num_layers=1,
                  num_hidden=10, chars=hun_chars):
    """Creates Keras model with the given input, output dimensions
    and layer number, hidden layer length"""
    model = Sequential()
    model.add(Dense(input_dim=((window_length)*len(chars)),
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


def data_reader(file, tail_cut=100000):
    """Read data from file"""

    tail_cut_ptest_words = tail_cut + 500

    counter_hu_data = collections.Counter()
    with open('web2.2-freq-sorted.txt', 'r',
              errors='ignore', encoding='latin2') as f:
        i = 0
        for line in f:
            i = i+1
            words = line.split()
            if len(words) > 1:
                if(words[1].isdigit()):
                    counter_hu_data[cleaning(words[0])] += int(words[1])
            if i > tail_cut_ptest_words:
                break
    return counter_hu_data


def train_data_generator(data_counter, window_length, length_after,
                         tag_chars='BM', tail_cut=100000):
    """Generate training data from counter data"""

    word_list = []
    c_all = 0
    c_same_char_num = 0

    for words in data_counter.most_common(tail_cut):
        c_all += 1
        next_word = words[0]
        if(len(next_word) != 0 and same_char_num(next_word)):
            c_same_char_num += 1
            if(len(tag_chars) == 2):
                word_list.append([next_word,
                                  hyph_tags_4to2(hyph_tags(next_word))])
            else:
                word_list.append([next_word, hyph_tags(next_word)])
    print('Data read successfully')
    print(c_same_char_num, c_all, c_same_char_num/c_all)

    # Generate network data
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
    print('Data len: ', len(data_in))
    print('Wrong words: ', wrong_word)

    valid_rate = 0.2
    test_rate = 0.1
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
            tests_input_flatten, tests_target, word_list]


if __name__ == "__main__":
    tail_cut = 100000
    window_length = 5
    length_after = 2
    tag_chars = 'BM'
    num_layers = 2
    num_hidden = 10

    for length_after in range(1, 6):
        window_length = length_after*2+1

        # Data read and network data generate
        counter_hu_data = data_reader('web2.2-freq-sorted.txt')
        [train_input_flatten, train_target,
         valid_input_flatten, valid_target,
         tests_input_flatten,
         tests_target,
         word_list] = train_data_generator(counter_hu_data,
                                              window_length,
                                              length_after,
                                              tag_chars)

        for num_layers in range(2, 12):
            for num_hidden in range(10, 101, 10):
                # Creating the keras model
                model = model_creator(window_length, len(tag_chars),
                                      num_layers, num_hidden)
                print('Model created. Start training...')

                earlyStopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, verbose=0, mode='auto')

                history = model.fit(train_input_flatten, train_target,
                                    epochs=1000, batch_size=1024,
                                    validation_data=(valid_input_flatten,
                                                     valid_target),
                                    verbose=0, callbacks=[earlyStopping])

                print('Training done')

                test_results = model.predict(tests_input_flatten)
                test_tp = 0
                test_fp = 0
                test_tn = 0
                test_fn = 0
                test_str = ""
                if(np.shape(test_results)[1] == 2):
                    for i in range(len(test_results)):
                        # positive
                        if np.argmax(test_results[i]) == 1:
                            if np.argmax(tests_target[i] == 1):
                                test_tp += 1
                            else:
                                test_fp += 1
                        else:
                            if np.argmax(tests_target[i] == 1):
                                test_fn += 1
                            else:
                                test_tn += 1
                    test_precision = test_tp / (test_tp + test_fp)
                    test_recall = test_tp / (test_tp + test_fn)
                    test_Fscore = 2 * (test_precision * 
                                       test_recall) / (test_precision +
                                                       test_recall)
                    test_str = str(test_precision) + '\t' + str(test_recall) 
                    test_str += '\t' + str(test_Fscore)
                else:
                    for i in range(len(test_results)):
                        if np.argmax(test_results[i]) == np.argmax(tests_target[i]):
                            test_success += 1
                        else:
                            test_fail += 1
                    test_str = str(test_fail/(test_fail+test_success))

                with open("results.txt", "a") as myfile:
                    result = ""
                    result += str(window_length) + '\t'
                    result += str(length_after) + '\t' + tag_chars
                    result += '\t' + str(num_layers) + '\t'
                    result += str(num_hidden) + '\t'
                    result += str(history.epoch[-1]) + '\t'
                    result += str(history.history['val_loss'][-1])
                    result += '\t' + test_str
                    result += '\n'
                    myfile.write(result)

