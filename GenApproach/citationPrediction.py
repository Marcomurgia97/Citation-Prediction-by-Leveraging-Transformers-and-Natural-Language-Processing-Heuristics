import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from difflib import SequenceMatcher
from nltk.corpus import words as wr

import torch as tr
import re
import spacy
from spacy.tokenizer import Tokenizer
import sys


def selectIthWord(l, index):
    cont = 0
    toPrint = []
    for word in l:
        if word == '[cit]':
            if cont == index:
                toPrint.append(word)
            cont = cont + 1
        elif word != '[cit]':
            toPrint.append(word)

    return toPrint


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


def reorder(s, word, delta):
    words = s.split()
    oldpos = words.index(word)
    words.insert(oldpos + delta, words.pop(oldpos))
    return ' '.join(words)


def checkWordRepetead(word, string, cont):
    index = string.split().index(word)
    while index < cont:
        index = string.split().index(word, index + 1)

    return index


def concateInput(tmp, tokenizer, inputs):
    strInput = (tokenizer.decode(inputs[0])) + tmp
    newInput = tokenizer(strInput, return_tensors="pt").input_ids.to("cuda")

    return newInput


def manageProbHeuristic(arrayProb, array, toPrint):
    indexMaxProb = arrayProb.index(min(arrayProb))
    array = array[indexMaxProb:indexMaxProb + 1]
    tmpSplitted = selectIthWord(toPrint.split(), indexMaxProb)

    return [array, tmpSplitted]


def computeHeuristics(indCitSuggested, toPrintSplitted, toRemove, toRemoveMetric, contIndex):
    for c in indCitSuggested:
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
                    delete = [w for w in range(len(stringMetric.split())) if
                              stringMetric.split()[w].startswith('(' + acronym + ')')]

                    toRemoveMetric.append(delete[0])
                else:
                    for i in range(lenAcr, 0, -1):
                        listLetters.append(toPrintSplitted[c - i][0])
                        similarity = SequenceMatcher(None, ''.join(listLetters).upper(),
                                                     acronym.upper()).ratio()

                    if similarity >= 0.75:
                        delete = [w for w in range(len(stringMetric.split())) if
                                  stringMetric.split()[w].startswith('(' + acronym + ')')]
                        toRemoveMetric.append(delete[0])
                        toRemove.append(c)
        if toPrintSplitted[c - 1].lower() in tokensToExclude:

            if len(indCitPosition) > 0:
                toRemoveMetric.append(array[contIndex])
            toRemove.append(c)

        contIndex = contIndex + 1
    return [toRemove, toRemoveMetric]


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
# nlp.Defaults.stop_words.add('12345.')

# dev = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

# lysandre/arxiv-nlp
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(sys.argv[1])

model = model.to('cuda')

sentences = open(r"..\Gold Standard\GS.txt", 'r', encoding='UTF-8')

contSentences = 0

f1 = open('prediction.txt', 'w', encoding='utf8')

for line in sentences.readlines():

    tokensToExclude = ['.', ',', ':', ';', 'by', 'in', 'from', 'into', 'tab.', 'table', 'fig.', 'figure', 'section']
    punts = ['.', ',', ':', ';']

    stringBase = line
    startString = stringBase
    stringBase = ' .'.join(stringBase.rsplit('.', 1))
    if 'startHere' in stringBase:
        stringBase = stringBase.replace(' startHere', '')
    stringBaseMetric = stringBase
    flag_prev = False

    if 'prior work' in stringBase.lower() or 'prior study' in stringBase.lower() or 'previous work' in stringBase.lower() or 'previous study' in stringBase.lower() or 'prior works' in stringBase.lower() or 'prior studies' in stringBase.lower() or 'previous works' in stringBase.lower() or 'previous studies' in stringBase.lower():
        flag_prev = True

    string = ''
    choiches = [5]

    regex = r"(\(?\b(?:[A-Z][A-Za-z'-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'-]+)|(?:et al.?)))*(?:,? *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *(?:,? *\((?:19|20)[0-9][0-9]\)(?:, p\.? [0-9]+)?))\)?)|\[(.*?)\]"
    regex1 = r"(\(?\b(?:[A-Z][A-Za-z'-]+)(?:,? (?:(?:and |& )?(?:[A-Z][A-Za-z'-]+)|(?:et al.?)))*(?:,? *(?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?| *(?:,? *\((?:19|20)[0-9][0-9]\)(?:, p\.? [0-9]+)?))\)?)"


    placeHolder = '[cit]'
    stringMetric = ''
    matches = list(re.finditer(regex, stringBase))
    print(len(matches))
    if len(matches) > 0:
        for el in matches:
            string = stringBase.replace(el.group(0), '')
            stringMetric = stringBaseMetric.replace(el.group(0), '[citationHere] ')

            stringBase = string
            stringBaseMetric = stringMetric
    else:
        string = stringBase
        stringMetric = stringBase

    string = ' '.join(string.split())
    stringMetric = ' '.join(stringMetric.split())

    doc = nlp(string)

    for token in doc:
        if token.pos_ == "VERB":
            start = token.idx  # Start position of token
            end = token.idx + len(token)  # End position = start + len(token)
            if not token.text.endswith('ing'):
                tokensToExclude.append(token.text)

    # string = ' '.join(string.split())
    tmpIndCitPosition = [i for i, n in enumerate(stringMetric.split()) if n == '[citationHere]']

    for choice in choiches:
        contWord = 0
        indCitPosition = tmpIndCitPosition[:]
        started, found, foundSquare = False, False, False
        toSave, toSaveBracket = '', ''
        citationPredicted, array, toPrint, arrayProb = [], [], [], []

        startIndex, startIndexPlace = 0, 0
        if 'startHere' in startString:
            startIndex = startString.split().index('startHere')
            startIndexPlace = startString.split().index('startHere')

        if startIndex > 0:
            endContext = False
        else:
            endContext = True

        # input_ids = tokenizer('prova', return_tensors="pt").input_ids.to("cuda")
        with tr.no_grad():
            for word in stringMetric.split():
                
                if '[' in tokenizer.decode(toSave):

                    indexMetric = stringMetric.split().index(word)
                    # and len([i for i, n in enumerate(toPrint) if n == '[cit]']) < 2
                    if toPrint[len(toPrint) - 1] != placeHolder:
                        toPrint.append(placeHolder)
                    if len(matches) > 0:
                        array.append(checkWordRepetead(word, stringMetric, contWord))

                    toSave = ''
                    tmpSquare = ' ' + '['

                    tmpInputSquare = concateInput(tmpSquare, tokenizer, inputs)
                if tokenizer.decode(toSaveBracket) == '(' or tokenizer.decode(toSaveBracket) == ' (':
                    toSaveBracket = ''
                    Tmp = ' ' + '('

                    tmpInput = concateInput(Tmp, tokenizer, inputs)

                if word == stringMetric.split()[0] and not started:
                    inputs = tokenizer(stringMetric.split()[0], return_tensors="pt").input_ids.to('cuda')
                    started = True
                else:
                    if word != '[citationHere]':
                        tmp = ' ' + word
                        inputs = concateInput(tmp, tokenizer, inputs)
                if found:

                    output = model.generate(tmpInput, return_dict_in_generate=True, output_scores=True,
                                            max_new_tokens=9,
                                            pad_token_id=tokenizer.eos_token_id, num_beams=5)
                    # num_beams=5
                    # print(tokenizer.decode(output.sequences[0]))
                    for el in list(re.finditer(regex1, tokenizer.decode(output.sequences[0]))):
                        citationPredicted.append(el.group(0))
                    if len(list(re.finditer(regex1, tokenizer.decode(output.sequences[0])))) > 0:
                        if foundSquare:
                            if prob < arrayProb[len(arrayProb) - 1]:
                                arrayProb[len(arrayProb) - 1] = prob
                        elif not foundSquare:
                            arrayProb.append(prob)

                        if len(matches) > 0:
                            array.append(checkWordRepetead(word, stringMetric, contWord))

                        if toPrint[len(toPrint) - 1] != placeHolder:
                            toPrint.append(placeHolder)
                    found = False
                if foundSquare:
                    output = model.generate(tmpInputSquare, return_dict_in_generate=True, output_scores=True,
                                            max_new_tokens=1,
                                            pad_token_id=tokenizer.eos_token_id)
                    scores = output.scores[0]
                    next_token_logits = scores[0, :]
                    next_token_probability = tr.softmax(next_token_logits, dim=-1)
                    sorted_ids = tr.argsort(next_token_probability, dim=-1, descending=True)

                    for choice_idx in range(1):
                        token_id = sorted_ids[choice_idx]
                        token_probability = next_token_probability[token_id].cpu().numpy()
                        token_choice = f"{tokenizer.decode(token_id)}({100 * token_probability:.2f}%)"
                        print(token_choice)
                        '''
                        if 'cit' in tokenizer.decode(token_id):
                            indexMetric = stringMetric.split().index(word)
                            if toPrint[len(toPrint) - 1] != '[cit]':
                                toPrint.append('[cit]')
                            if len(matches) > 0:
                                indexMetric = stringMetric.split().index(word)
                                while indexMetric < contWord:
                                    indexMetric = stringMetric.split().index(word, indexMetric + 1)

                                array.append(indexMetric)

                        # print(token_choice)
                        '''
                    foundSquare = False

                if computeratio(stringMetric.split()[contWord - 1]) >= 0.60 and contWord > 0 and endContext:
                    if toPrint[len(toPrint) - 1] != placeHolder:
                        toPrint.append(placeHolder)
                        arrayProb.append(5)

                    if len(matches) > 0:
                        array.append(checkWordRepetead(stringMetric.split()[contWord], stringMetric, contWord))

                if word != '[citationHere]':
                    toPrint.append(word)

                if contWord >= startIndex:
                    endContext = True
                if endContext and word != '[citationHere]':
                    output = model.generate(inputs, return_dict_in_generate=True, output_scores=True,
                                            max_new_tokens=1,
                                            pad_token_id=tokenizer.eos_token_id)
                    scores = output.scores[0]
                    next_token_logits = scores[0, :]
                    next_token_probability = tr.softmax(next_token_logits, dim=-1)
                    sorted_ids = tr.argsort(next_token_probability, dim=-1, descending=True)
                    toSave = ''
                    toSaveBracket = ''
                    for choice_idx in range(choice):
                        token_id = sorted_ids[choice_idx]
                        token_probability = next_token_probability[token_id].cpu().numpy()
                        token_choice = f"{tokenizer.decode(token_id)}({100 * token_probability:.2f}%)"
                        if '[' in tokenizer.decode(token_id):
                            # print("Scelta {} dopo di {} con probabilita {}:" .format(choice_idx+1, word, (100 * token_probability)))
                            if contWord < len(stringMetric.split()) - 1:
                                arrayProb.append(choice_idx + 1)
                            foundSquare = True
                            toSave = token_id
                        if tokenizer.decode(token_id) == '(' or tokenizer.decode(token_id) == ' (':
                            # print("(...Scelta {} dopo di {} con probabilita {}:" .format(choice_idx+1, word, (100 * token_probability)))
                            found = True
                            toSaveBracket = token_id
                            prob = choice_idx + 1

                contWord = contWord + 1

        if flag_prev:
            if toPrint[len(toPrint) - 2] != placeHolder:
                if stringMetric.split()[len(stringMetric.split()) - 2] == '[citationHere]':
                    array.append(len(stringMetric.split()) - 2)
                else:
                    array.append(len(stringMetric.split()) - 1)
                arrayProb.append(1)
                toPrint.insert(len(toPrint) - 1, placeHolder)

        toPrint = ' '.join(toPrint)
        stringWithPlaceholder = toPrint
        array = list(dict.fromkeys(array))
        flagIndex = 0
        # startIndex = 0
        endIndex = 0
        flagIndexPlace = 0
        # startIndexPlace = 0
        endIndexPlace = 0

        if len(indCitPosition) >= 0:
            for chunk in doc.noun_chunks:
                # print(chunk.text)
                if len(chunk.text.split()) > 1 and startIndex >= 0:
                    start = chunk.text.split()[0]
                    end = chunk.text.split()[len(chunk.text.split()) - 1]

                    startIndex = stringMetric.split().index(start)
                    endIndex = stringMetric.split().index(end)

                    while startIndex <= flagIndex and startIndex != 0:
                        startIndex = stringMetric.split().index(start, startIndex + 1)
                    while endIndex <= startIndex:
                        endIndex = stringMetric.split().index(end, endIndex + 1)

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
                        indexProbChunck = []

                        test = [e for e in array if startIndex < e <= endIndex]
                        print(test)
                        for i in test:
                            indexProbChunck.append(array.index(i))
                        array = [e for e in array if e not in test]
                        if stringWithPlaceholder.split()[endIndexPlace + 1] == '[cit]':

                            subString = subString.replace('[cit] ', '', subString.count('[cit]'))
                            toDel = start + ' ' + toDel + ' ' + end
                            toDel = ' '.join(toDel.split())
                            subString = start + ' ' + subString + ' ' + end
                            subString = ' '.join(subString.split())

                            toPrint = stringWithPlaceholder.replace(toDel, subString)

                            stringWithPlaceholder = toPrint
                            for el in sorted(indexProbChunck, reverse=True):
                                del arrayProb[el]
                            # arrayProb.append(3)

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

                            for el in sorted(indexProbChunck, reverse=True):
                                del arrayProb[el]
                            if len(indCitPosition) > 0:
                                array.append(endIndex + 1)

                            arrayProb.append(3)

                    flagIndex = startIndex
                    flagIndexPlace = startIndexPlace
        
        array = sorted(list(dict.fromkeys(array)))
        tmpArray = array

        array = sorted(list(dict.fromkeys(array)))
        toPrintSplitted = toPrint.split()
        indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']
        tmpSplitted = toPrintSplitted
        toRemove = []
        toRemoveMetric = []
        contIndex = 0

        listHeurs = computeHeuristics(indCitSuggested, toPrintSplitted, toRemove, toRemoveMetric, contIndex)
        toRemove = listHeurs[0]
        toRemoveMetric = listHeurs[1]

        for el in sorted(toRemove, reverse=True):
            del tmpSplitted[el]
        if len(indCitPosition) > 0:
            for element in toRemoveMetric:
                if element in toRemoveMetric:
                    indexProb = array.index(element)
                    del arrayProb[indexProb]
                    array.remove(element)
        print(toRemove)

        toPrint = ' '.join(tmpSplitted)

        tmpToPrint = toPrint
        toPrintSplitted = toPrint.split()

        ids = [index for index, value in enumerate(toPrintSplitted) if value == '[cit]']
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
            del tmpSplitted[el]

        toPrint = ' '.join(tmpSplitted)
        
        if len(indCitPosition) > 0:
            for idx in toRemoveIndex:
                toFind = tmpToPrint.split()[idx + 1]
                toFindPrec = tmpToPrint.split()[idx - 1]
                toFindIdx = stringMetric.split().index(tmpToPrint.split()[idx + 1])
                toFindPrecIdx = stringMetric.split().index(tmpToPrint.split()[idx - 1])
                while toFindIdx < toFindPrecIdx:
                    toFindIdx = stringMetric.split().index(toFind, toFindIdx + 1)
                indexProb = array.index(toFindIdx)
                del arrayProb[indexProb]

                array.remove(toFindIdx)

        toPrintSplitted = toPrint.split()

        if len(arrayProb) > 0 and len(array) > 1:
            if len(arrayProb) == len(array):
                listHeurProb = manageProbHeuristic(arrayProb, array, toPrint)
                array = listHeurProb[0]
                tmpSplitted = listHeurProb[1]

        if len(arrayProb) > 1 and len(indCitPosition) == 0:
            listHeurProb = manageProbHeuristic(arrayProb, indCitSuggested, toPrint)
            indCitSuggested = listHeurProb[0]
            tmpSplitted = listHeurProb[1]

        toPrint = ' '.join(tmpSplitted)

        print("\n")
        toPrint = toPrint.replace(' .', '.')
        toPrint = toPrint.replace(' ,', ',')

        prediction = toPrint
        print(line)
        print(prediction)

        f1.write(prediction+'\n')

    contSentences = contSentences + 1

sentences.close()
