import pdb, re, json, sys, math

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import random

def confusion_compute(tp, fp, fn, tn, label_set, do_print=True):
    mcc = (tn * tp - fn * fp) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    prec, rec = tp / (tp + fp) if tp + fp != 0 else 0, tp / (tp + fn) if tp + fn != 0 else 0
    if do_print: print(label_set[0] + ' = +:')
    if do_print: print('\tPrecision:\t', prec)
    if do_print: print('\tRecall:\t\t', rec)
    if do_print: print('\tF1:\t\t', 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0)

    prec, rec = tn / (tn + fn) if tn + fn != 0 else 0, tn / (tn + fp) if tn + fp != 0 else 0
    if do_print: print(label_set[1] + ' = +:')
    if do_print: print('\tPrecision:\t', prec)
    if do_print: print('\tRecall:\t\t', rec)
    if do_print: print('\tF1:\t\t', 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0)

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ba = (tpr + tnr) / 2
    if do_print: print("BA:", round(ba,4))
    if do_print: print("MCC:", round(mcc,4))

    return ba


class Task(object):
    def __init__(self, examples, prompter, prompt_library, prompt_type, use_query, dataset, label_set, do_print=False, start_num=0):
        self.prompter = prompter
        self.pl = prompt_library
        self.do_print = do_print
        self.prompt_type = prompt_type
        self.use_query = use_query
        self.dataset = dataset
        self.examples = []
        self.gold_labels = []
        self.label_set = label_set
        self.notes = []

        print("PREDICTION MODEL:", prompter.model)

        assert self.prompt_type in {'zero', 'few', 'qanli'}

        for x in examples:
            self.examples.append({'d1': x['d1'], 'd2': x['d2'], 'query': x['query'] if use_query else ''})
            is_normal_case = 'label' in x
            if not is_normal_case: assert 'labels' in x

            #pdb.set_trace()

            if is_normal_case:
                assert x['label'] in label_set
            else:
                for lab in x['labels']: assert lab in label_set

            self.gold_labels.append(x['label'] if is_normal_case else x['labels'])
            additional = {}
            for y in x:
                if y not in {'d1', 'd2', 'query', 'label'}:
                    additional[y] = x[y]
            if len(additional) > 0:
                self.notes.append(additional)
        if len(self.notes) == 0: self.notes = [{} for i in range(len(examples))]
        assert len(self.notes) == len(self.examples)

        self.preds = []

        self.save_path = 'artefacts/'+dataset+'/res_' +self.prompter.model+'_' + prompt_type + '_' + ('query' if use_query else 'noquery') + '.json'
        if start_num == 0: open(self.save_path, 'w').close() # reset file

        self.is_normal_case = is_normal_case

        # Read in so far and evaluate
        if start_num > 0:
            self.preds = []
            with open(self.save_path, 'r') as f:
                for line in f:
                    line_load = (json.loads(line))
                    self.preds.append(line_load['pred'])
            self.evaluate(partial_ok=True)
            assert len(self.preds) == start_num
        self.start_num = start_num


    def predict_all(self):
        for i in tqdm(range(self.start_num, len(self.examples))):
            prds = self.predict(self.examples[i])
            self.preds.append(prds)
            self.write_last_pred()

            if (i+1) % 10 == 0:
                print("Evaluating at", i+1)
                self.evaluate(partial_ok=True)
        print("Evaluating at", i+1)
        self.evaluate(partial_ok=True)

    def predict(self, ex):
        if self.prompt_type in {'zero', 'few'}:
            prompt_template = self.pl.task(self.label_set, self.prompt_type, self.use_query, self.dataset)
            prompt = prompt_template.format(doc1=ex['d1'], doc2=ex['d2'], query=ex['query'])
            full_response = self.prompter.prompt(prompt)
            # print(prompt)
            # print(full_response)
            # pdb.set_trace()
            response = re.findall('<answer>(.*?)</answer>', full_response, re.DOTALL)

            if len(response) != 1: return 'fail'
            response = response[0]
            response = response.strip('\n').lower()
            if response not in self.label_set: return 'fail'
        elif self.prompt_type in {'qanli'}:
            prompt_template = self.pl.task_answer_question_full_sentence()
            prompt = prompt_template.format(document=ex['d1'], query=ex['query'])

            full_response = self.prompter.prompt(prompt)
            response = re.findall('<answer>(.*?)</answer>', full_response, re.DOTALL)

            if len(response) != 1: return 'fail'
            answer1 = response[0]

            prompt_template = self.pl.task_answer_question_full_sentence()
            prompt = prompt_template.format(document=ex['d2'], query=ex['query'])

            full_response = self.prompter.prompt(prompt)
            response = re.findall('<answer>(.*?)</answer>', full_response, re.DOTALL)

            if len(response) != 1: return 'fail'
            answer2 = response[0]

            prompt_template = self.pl.task(self.label_set, self.prompt_type, self.use_query, self.dataset)
            prompt = prompt_template.format(doc1=answer1, doc2=answer2)
            full_response = self.prompter.prompt(prompt)
            # print(prompt)
            # print(full_response)
            # pdb.set_trace()
            response = re.findall('<answer>(.*?)</answer>', full_response, re.DOTALL)

            if len(response) != 1: return 'fail'
            response = response[0]
            response = response.strip('\n').lower()
            if response not in self.label_set: return 'fail'
        else: assert False
        return response


    def evaluate(self, partial_ok=False):
        assert self.preds is not None
        if not partial_ok:
            assert len(self.preds) == len(self.gold_labels)
            gold_labs = self.gold_labels
        else:
            assert len(self.preds) <= len(self.gold_labels)
            gold_labs = self.gold_labels[:len(self.preds)]
        preds = self.preds
        assert len(gold_labs) == len(preds)
        if not self.is_normal_case: # Have to expand it
            new_preds = []
            new_gold = []
            for i in range(len(gold_labs)):
                predi = preds[i]
                new_preds += ([predi] * len(gold_labs[i]))
                new_gold += gold_labs[i]
            preds, gold_labs = new_preds, new_gold
        else:
            preds = self.preds
        assert len(preds) == len(gold_labs)
        label_set = self.label_set + ['fail']
        cm = confusion_matrix(gold_labs, preds, labels = label_set)
        acc = accuracy_score(gold_labs, preds)

        print('Labels:', label_set)
        print(cm)
        print('Accuracy:', acc)

        tp, fp, fn, tn = cm[0, 0], cm[1,0], cm[0,1], cm[1, 1]
        if cm.shape[0] == 3:
            fp += cm[1,2]
            fn += cm[0,2]
        confusion_compute(tp, fp, fn, tn, label_set)

    def write_last_pred(self):
        i = len(self.preds) - 1
        dct = {**self.examples[i], **self.notes[i], 'pred': self.preds[i], 'gold': self.gold_labels[i]}
        json_string = json.dumps(dct)
        with open(self.save_path, 'a') as outfile:
            outfile.write(json_string + '\n')
        sys.stdout.flush()