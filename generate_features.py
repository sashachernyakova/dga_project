import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import operator
import pandas as pd
import math
from collections import Counter

df_legit = pd.read_csv('legit_100k.csv')
df_legit.drop(columns=['index'], inplace=True)
# Изменим существующий столбец, удалив TLD из доменов
df_legit['domain'] = df_legit['domain'].apply(lambda x: x.split('.')[0])
# Преобразование всех строк в столбце 'domain' к нижнему регистру
df_legit['domain'] = df_legit['domain'].str.lower()
# Удаление строк с пропущенными значениями
df_legit = df_legit.dropna()
df_legit = df_legit.drop_duplicates()
df_legit['length'] = [len(x) for x in df_legit['domain']]
# словарь, состоящий из доменов Alexa Top Million
df_legit = df_legit[df_legit['length'] > 6]

# словарь, состоящий из наиболее употребительных слов и фраз
word_dataframe = pd.read_csv('words.csv')
# оставляем строки, содержащие только буквенные символы
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
# приведение слов к нижнему регистру и удаление начальных и конечных пробелов
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()


# инициализируется для создания символьных NGrams длины от 3 до 5 символов.
# NGrams, которые встречаются реже, чем в 0.01% доменных имен, будут исключены.
# NGrams, которые встречаются в более чем 100% доменных имен (что теоретически невозможно), также будут исключены.
alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-4, max_df=1.0)
# обучает векторизатор на данных и преобразует слова в матрицу счетчиков N-грамм
counts_matrix = alexa_vc.fit_transform(df_legit['domain'])
# суммирует частоты N-грамм по всем словам и преобразует их в одномерный массив (вектор) и
# применяет логарифм по основанию 10 к каждой сумме частот N-грамм
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())
# Частоты N-грамм могут иметь очень широкий диапазон значений (от очень редких до очень частых).
# Логарифмирование помогает сгладить этот разброс, делая распределение частот более компактным и управляемым.

dict_vc = CountVectorizer(analyzer='char', ngram_range=(3,5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())

# -----------------------------------------------
# если длина до первой точки 0, то домен начинается с точки. Значит, это точно dga

# Функция для удаления TLD и перевода в нижний регистр
def remove_tld_and_lower(domain):
    return domain.split('.')[0].lower()


def domain_length(domain):
    domain = remove_tld_and_lower(domain)
    # возвращаем число побольше, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 40
    return len(domain)


# отношение количества цифр в доменном имени к его общей длине
def count_numbers_to_len(domain):
    domain = remove_tld_and_lower(domain)
    # возвращаем 1.0, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 1.0
    count = 0
    for ch in domain:
        if ch.isdigit():
            count += 1
    return count / len(domain)


# 1 - если первый символ в доменном имени является цифрой, 0 по умолчанию
def is_first_number(domain):
    domain = remove_tld_and_lower(domain)

    # возвращаем 1, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 1

    if domain[0].isdigit():
        return 1
    return 0


# отношение количества символов, которые встречаются более, чем один раз в доменном имени,
# к общему количеству уникальных символов в доменном имени
def repeated_to_unique(domain):
    domain = remove_tld_and_lower(domain)
    symbols_counts = {}

    # возвращаем 1.0, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 1.0

    # Подсчитываем вхождения каждой буквы
    for char in domain:
        if char in symbols_counts:
            symbols_counts[char] += 1
        else:
            symbols_counts[char] = 1

    # Подсчитываем количество символов, встречающихся более одного раза
    more_than_once = sum(1 for count in symbols_counts.values() if count > 1)

    # Общее количество уникальных символов
    unique_symbols = len(symbols_counts)

    return more_than_once / unique_symbols


# отношение самой большой последовательности согласных и цифр в доменном имени ко всей длине
def longest_consonant_sequence(domain):
    domain = remove_tld_and_lower(domain)

    # возвращаем 1.0, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 1.0
    
    vowels = set('aeiouyAEIOUY')
    max_length = 0
    current_length = 0

    for char in domain:
        if (char.isalpha() and char not in vowels) or char.isdigit():
            current_length += 1
            if current_length > max_length:
                max_length = current_length
        else:
            current_length = 0

    return max_length / len(domain)


# отношение количества гласных к количеству согласных и цифр
def vowel_to_consonant_and_numbers(domain):
    domain = remove_tld_and_lower(domain)

    # возвращаем 0.0, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 0.0

    vowels = set('aeiouyAEIOUY')
    num_vowels = 0
    num_consonants_numbers = 0

    for char in domain:
        if char.isalpha():
            if char in vowels:
                num_vowels += 1
            else:
                num_consonants_numbers += 1
        elif char.isdigit():
            num_consonants_numbers += 1

    if num_consonants_numbers == 0:
        # Так как согласных(+цифры) в алфавите сильно больше, то у dga при генерации обычно согласных(+цифры) сильно
        # больше получается, поэтому у dga отношение num_vowels / num_consonants_numbers -> 0.
        # В данном if рассматривается случай всех гласных, на ноль делить нельзя.
        # Так как мало информативных слов, состоящих только из гласных, то скорее всего этот домен - dga
        # Поэтому в таком случае выбираем 0.
        return 0.0

    return num_vowels / num_consonants_numbers


# 1 - если домен первого уровня доменного имени есть в списке доменов первого уровня, распространенных среди
# злоумышленников, 0 по умолчанию
def is_malicious_tld(domain):
    dga = {
        "xyz", "xxx", "top", "club", "info", "site", "online", "pw", "loan", "net"
    }

    # Получаем TLD из доменного имени
    tld = domain.split('.')[-1]
    if tld in dga:
        return 1
    return 0


def entropy(domain):
    domain = remove_tld_and_lower(domain)

    # возвращаем 4.0, чтобы модель относила к dga, так как длина 0
    if len(domain) == 0:
      return 4.0

    # Подсчитываем частоту появления каждого символа
    freq = Counter(domain)
    total_chars = len(domain)

    # Рассчитываем энтропию
    return -sum((count / total_chars) * math.log2(count / total_chars) for count in freq.values())


# генерация признаков
def generate_features(df):
    df['len'] = df['domain'].apply(domain_length)
    df['count_numbers_to_len'] = df['domain'].apply(count_numbers_to_len)
    df['is_first_number'] = df['domain'].apply(is_first_number)
    df['repeated_to_unique'] = df['domain'].apply(repeated_to_unique)
    df['longest_consonant_sequence'] = df['domain'].apply(longest_consonant_sequence)
    df['vowel_to_consonant_and_numbers'] = df['domain'].apply(vowel_to_consonant_and_numbers)
    df['is_malicious_tld'] = df['domain'].apply(is_malicious_tld)
    df['entropy'] = df['domain'].apply(entropy)
    # каждое значение частоты N-граммы умножается на соответствующий логарифм частоты этой N-граммы
    # перемножение вектора на вектор дает число
    df['alexa_grams'] = [alexa_counts * alexa_vc.transform([domain]).T for domain in df['domain']]
    df['word_grams'] = [dict_counts * dict_vc.transform([domain]).T for domain in df['domain']]
    df['diff'] = df['alexa_grams'] - df['word_grams']
    return df
