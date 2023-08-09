import os
import sys

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch as tr
import re


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


# dev = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

# lysandre/arxiv-nlp
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(sys.argv[1])

model = model.to('cuda')

sentences = open(r"..\Gold Standard\GS.txt", 'r', encoding='UTF-8')

contSentences = 0

f1 = open('prediction.txt', 'w', encoding='utf8')

for line in sentences.readlines():


    stringBase = line
    startString = stringBase
    stringBase = ' .'.join(stringBase.rsplit('.', 1))
    if 'startHere' in stringBase:
        stringBase = stringBase.replace(' startHere', '')
    stringBaseMetric = stringBase
    flag_prev = False


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

    # string = ' '.join(string.split())
    tmpIndCitPosition = [i for i, n in enumerate(stringMetric.split()) if n == '[citationHere]']

    for choice in choiches:
        contWord = 0
        indCitPosition = tmpIndCitPosition[:]
        started, found, foundSquare, foundString = False, False, False, False
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

        toPrint = ' '.join(toPrint)
        stringWithPlaceholder = toPrint
        array = list(dict.fromkeys(array))
        flagIndex = 0
        # startIndex = 0
        endIndex = 0
        flagIndexPlace = 0
        # startIndexPlace = 0
        endIndexPlace = 0

        array = sorted(list(dict.fromkeys(array)))
        tmpArray = array

        array = sorted(list(dict.fromkeys(array)))
        toPrintSplitted = toPrint.split()
        indCitSuggested = [i for i, n in enumerate(toPrint.split()) if n == '[cit]']
        tmpSplitted = toPrintSplitted
        toRemove = []
        toRemoveMetric = []
        contIndex = 0

        toPrint = ' '.join(tmpSplitted)

        tmpToPrint = toPrint
        toPrintSplitted = toPrint.split()

        ids = [index for index, value in enumerate(toPrintSplitted) if value == '[cit]']
        toRemoveIndex = []

        toPrint = ' '.join(tmpSplitted)
        
      

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

        f1.write(prediction + '\n')


    contSentences = contSentences + 1

sentences.close()
