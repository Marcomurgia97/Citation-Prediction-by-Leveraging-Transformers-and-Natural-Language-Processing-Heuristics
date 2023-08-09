import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from difflib import SequenceMatcher
from nltk.corpus import words as wr

import torch as tr
import re
import spacy
from spacy.tokenizer import Tokenizer

f = open('predictionNer.txt', 'r', encoding="utf-8")
f1 = open('predictionNerHeur.txt', 'w', encoding="utf-8")

def reorder(s, word, delta):
    words = s.split()
    oldpos = words.index(word)
    words.insert(oldpos + delta, words.pop(oldpos))
    return ' '.join(words)

def is_roman_number(num):
    pattern = re.compile(r"""   
                                    ^M{0,3}
                                    (CM|CD|D?C{0,3})?
                                    (XC|XL|L?X{0,3})?
                                    (IX|IV|V?I{0,3})?$
                """, re.VERBOSE)

    if re.match(pattern, num):
        return True

    return False

def computeratio(word):
    count = 0
    countcap = 0
    ratio = 0
    for c in word:
        if c.isalpha():
            count = count + 1
    if count > 0:
        countCap = sum(map(str.isupper, word))
        ratio = countCap / count
    if word in set(wr.words()):
        return 0
    elif is_roman_number(word):
        return 0
    elif len(word) == 1:
        return 0
    else:
        return ratio

def force1cit(toPrint):
    cont = 0
    splitted = toPrint.split()
    tmpSplitted = []
    for w in splitted:
        if '[cit]' in w:
            if cont > 0:
                w = w.replace('[cit]', '')
            cont = cont + 1
        tmpSplitted.append(w)
    toReturn = ' '.join(tmpSplitted).replace('  ', ' ')
    return toReturn


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)


def computeHeuristics(indCitSuggested, toPrintSplitted, toRemove):
    for c in indCitSuggested:
        if c < len(toPrintSplitted)-1:
            if toPrintSplitted[c + 1].startswith('('):
                acronym = toPrintSplitted[c + 1]
                listLetters = []
                regexForAcr = r'\(([^)]+)'
                acronym = re.findall(regexForAcr, acronym)[0]
                if acronym.find(' ') < 0:

                    lenAcr = len(acronym)
                    wordToCheck = toPrintSplitted[c - 1]
                    similarity = SequenceMatcher(None, ''.join([q for q in wordToCheck if q.isupper()]),
                                                 acronym).ratio()
                    if similarity >= 0.75:
                        toRemove.append(c)
                        print('ciao1')

                    else:
                        for i in range(lenAcr, 0, -1):
                            listLetters.append(toPrintSplitted[c - i][0])
                            similarity = SequenceMatcher(None, ''.join(listLetters).upper(),
                                                         acronym.upper()).ratio()

                        if similarity >= 0.75:
                            beforeAcr = True

                            toRemove.append(c)

            if toPrintSplitted[c - 1].lower() in tokensToExclude:
                toRemove.append(c)

    return toRemove


for line in f.readlines():
    toPrint = line

    toPrintSplitted = toPrint.split()
    flag_prev = False
    placeHolder = '[cit]'
    contWord = 0

    if 'prior work' in toPrint.lower() or 'prior study' in toPrint.lower() or 'previous work' in toPrint.lower() or 'previous study' in toPrint.lower() or 'prior works' in toPrint.lower() or 'prior studies' in toPrint.lower() or 'previous works' in toPrint.lower() or 'previous studies' in toPrint.lower():
        flag_prev = True

    toRemove = []
    indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']

    tokensToExclude = ['.', ',', ':', ';', 'by', 'in', 'from', 'into', 'tab.', 'table', 'fig.', 'figure', 'section']

    doc = nlp(toPrint)

    for token in doc:
        if token.pos_ == "VERB":
            start = token.idx  # Start position of token
            end = token.idx + len(token)  # End position = start + len(token)
            if not token.text.endswith('ing'):
                tokensToExclude.append(token.text)

    if flag_prev:
        if toPrintSplitted[len(toPrintSplitted) - 1] != placeHolder:
            toPrintSplitted.insert(len(toPrintSplitted) - 1, placeHolder)

    tmpToPrint = []
    toPrint = ' '.join(toPrintSplitted)
    indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']


    for word in toPrintSplitted:
        tmpToPrint.append(word)
        if computeratio(word) >= 0.60:
            if toPrintSplitted[contWord + 1] != placeHolder:
                tmpToPrint.append(placeHolder)
        contWord = contWord + 1


    toPrint = ' '.join(tmpToPrint)

    toPrintSplitted = toPrint.split()
    indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']

    toRemove = computeHeuristics(indCitSuggested, toPrintSplitted, toRemove)
    for el in sorted(toRemove, reverse=True):
        del toPrintSplitted[el]

    toPrint = ' '.join(toPrintSplitted)

    toPrintSplitted = toPrint.split()
    ids = [index for index, value in enumerate(toPrintSplitted) if value == '[cit]']
    indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']

    toRemoveIndex = []

    for i in range(0, len(ids)):
        if i < len(ids) - 1:
            if abs(ids[i] - ids[i + 1]) == 2:
                toRemoveIndex.append(ids[i])
                toRemoveIndex.append(ids[i + 1])

    toRemoveIndex = sorted(list(dict.fromkeys(toRemoveIndex)))
    aux = []
    for i in range(0, len(toRemoveIndex)):
        if i < len(toRemoveIndex) - 1:
            if abs(toRemoveIndex[i] - toRemoveIndex[i + 1]) != 2:
                aux.append(toRemoveIndex[i])

    for j in aux:
        toRemoveIndex.remove(j)

    toRemoveIndex = sorted(list(dict.fromkeys(toRemoveIndex)))[:-1]

    for el in sorted(toRemoveIndex, reverse=True):
        del toPrintSplitted[el]

    toPrint = ' '.join(toPrintSplitted)
    indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']

    flagIndexPlace = 0
    stringWithPlaceholder = toPrint
    for chunk in doc.noun_chunks:
        # print(chunk.text)
        if len(chunk.text.split()) > 1:
            start = chunk.text.split()[0]
            end = chunk.text.split()[len(chunk.text.split()) - 1]



            startIndexPlace = stringWithPlaceholder.split().index(start)

            endIndexPlace = stringWithPlaceholder.split().index(end)

            while startIndexPlace <= flagIndexPlace and startIndexPlace != 0:
                startIndexPlace = stringWithPlaceholder.split().index(start, startIndexPlace + 1)
            while endIndexPlace <= startIndexPlace:
                endIndexPlace = stringWithPlaceholder.split().index(end, endIndexPlace + 1)

            subString = ' '.join(stringWithPlaceholder.split()[startIndexPlace + 1:endIndexPlace])
            subString = subString + ' '
            toDel = subString
            if '[cit]' in subString:
                # print(array, "ciao", startIndex, endIndex)

                if len(stringWithPlaceholder.split()) == endIndexPlace + 1:
                    if stringWithPlaceholder.split()[endIndexPlace] == '[cit]':
                        subString = subString.replace('[cit] ', '', subString.count('[cit]'))
                        toDel = start + ' ' + toDel + ' ' + end
                        toDel = ' '.join(toDel.split())
                        subString = start + ' ' + subString + ' ' + end
                        subString = ' '.join(subString.split())

                        toPrint = stringWithPlaceholder.replace(toDel, subString)

                        stringWithPlaceholder = toPrint
                    elif stringWithPlaceholder.split()[endIndexPlace] != '[cit]':
                        if subString.count('[cit]') > 1:
                            subString = subString.replace('[cit] ', '', subString.count('[cit]') - 1)
                        Chunck = start + ' ' + subString + ' ' + end
                        stringWithPlaceholder = stringWithPlaceholder.replace(toDel, subString)
                        delta = Chunck.split().index(end) - Chunck.split().index('[cit]')
                        newChunck = reorder(Chunck, '[cit]', delta)
                        Chunck = ' '.join(Chunck.split())
                        newChunck = ' '.join(newChunck.split())
                        toPrint = stringWithPlaceholder.replace(Chunck, newChunck)

                        stringWithPlaceholder = toPrint

                elif len(stringWithPlaceholder.split()) < endIndexPlace + 1:
                    if stringWithPlaceholder.split()[endIndexPlace + 1] == '[cit]':

                        subString = subString.replace('[cit] ', '', subString.count('[cit]'))
                        toDel = start + ' ' + toDel + ' ' + end
                        toDel = ' '.join(toDel.split())
                        subString = start + ' ' + subString + ' ' + end
                        subString = ' '.join(subString.split())

                        toPrint = stringWithPlaceholder.replace(toDel, subString)

                        stringWithPlaceholder = toPrint


                    elif stringWithPlaceholder.split()[endIndexPlace + 1] != '[cit]':
                        if subString.count('[cit]') > 1:
                            subString = subString.replace('[cit] ', '', subString.count('[cit]') - 1)
                        Chunck = start + ' ' + subString + ' ' + end
                        stringWithPlaceholder = stringWithPlaceholder.replace(toDel, subString)
                        delta = Chunck.split().index(end) - Chunck.split().index('[cit]')
                        newChunck = reorder(Chunck, '[cit]', delta)
                        Chunck = ' '.join(Chunck.split())
                        newChunck = ' '.join(newChunck.split())
                        toPrint = stringWithPlaceholder.replace(Chunck, newChunck)

                        stringWithPlaceholder = toPrint



            flagIndexPlace = startIndexPlace



    toPrint = force1cit(toPrint)
    print(toPrint)
    f1.write(toPrint+'\n')
