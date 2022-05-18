""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Обучение нейронной сети """
    #получаем известные ноты
    notes = get_notes()

    #получаем количество известных нот
    n_vocab = len(set(notes))

    #инициализируем входной и выходной слой нейросети
    network_input, network_output = prepare_sequences(notes, n_vocab)

    #инициализируем модель
    model = create_network(network_input, n_vocab)
    
    #обучаем модель
    train(model, network_input, network_output)

def get_notes():
    """ Получаем ноты и аккорды из папки ./midi_songs """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # файл имеет инструментальные части
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # файл имеет тональную структуру
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Создаем связи для нейросети """

    #количество связей
    sequence_length = 100

    # получаем названия полей
    pitchnames = sorted(set(item for item in notes))

     # создаем словарь для преобразования полей в целые числа
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # создайте входные последовательности и соответствующие выходные данные
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # преобразуйте входные данные в формат, совместимый со слоями LSTM
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # нормализация входных данных
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ создаем структуру нейросети """
    #полносвязная
    model = Sequential()
    #добавляем lstm слой с 512
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    #еще парочку
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    #функция нормализации
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    #функция активации relu
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    #функция активации softmax
    model.add(Activation('softmax'))
    #компилируем нейросеть по описанной структуре, выбирая функцию потерь - категориальную кроссэнтропию и оптимизируя 
    #метод оптимизации RMSprop похож на алгоритм градиентного спуска с импульсом.
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    #Антиградиент показывает куда убывает функция. Он нужен нам для того, чтобы стремиться к минимальной ошибке. 
    # Стоимость ошибки вычисляется в функции потерь.

    return model

def train(model, network_input, network_output):
    """ Обучение нейросети """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    #точка остановки
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    #запуск обучения
    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
