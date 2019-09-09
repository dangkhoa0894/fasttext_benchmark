import fasttext
from load_data import normalize

PATH='example.txt'

if __name__ == '__main__':
    classifier = fasttext.load_model('snapshots/model.bin')
    with open(PATH, errors='ignore') as f:
        str = f.read()
    str = normalize(str)
    predict = classifier.predict(str, k=3)
    print(predict)  