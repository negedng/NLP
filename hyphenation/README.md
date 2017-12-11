# Multilingual hyphenation using deep neural networks
## Project Laboratory(VIAUAL01) at BME-VIK AUT and BSC Thesis
For more details: 
* Course at AUT: https://www.aut.bme.hu/Course/VIAUAL01, https://www.aut.bme.hu/Course/VIAUA406
* Course at VIK: https://portal.vik.bme.hu/kepzes/targyak/VIAUAL01, https://portal.vik.bme.hu/kepzes/targyak/VIAUA406
* NLP: https://www.aut.bme.hu/Task/16-17-tavasz/Szotagolas-deep-learning

## Requirements
* Keras (with TensorFlow backend): https://keras.io/
* PyHyphen: https://pypi.python.org/pypi/PyHyphen/ 
* (Language detector): https://pypi.python.org/pypi/langdetect

## Install
Cloning the git repository, the Hyphenation can be started with running `hyphenator.py`

## Example files
* `trainer2_en.py`: training English hyphenator in DNN mode
* `trainer_hu_en.py`: training multiple English-Hungarian bilingual hyphenator
* `hu_en_data_creator.py`: creating English and Hungarian data examples using WebCorpus and UMBC WebBase
* `trainer_seq2seq.py`: Sequence-to-sequence model creator
* `hyph_test.ipynb`: Jupyter Notebook with examples and comments

## Pretrained models
* DNN, LSTM, CNN models
* Seq2seq and Seq2seqI (with non-standard hyphenation)

## Documentation
The BSc Thesis is available at `documentation.pdf`
