from generator import ExampleGenerator
import json, pdb, re, random
import pandas as pd
import sqlite3

class RobustQa(ExampleGenerator):
    def __init__(self, **kwargs):
        self.dname = 'robustqa'
        super().__init__(**kwargs)

    def read_data(self):
        with open('robustqa/robustqa.yes-no_v1_20240701.json', 'r') as f:
            data = json.load(f)
        assert self.partition == 'all'
        for x in data:
            assert len(x) == 7
        return data

    def generate(self, idx):
        agreement_threshold = 0.75
        qc0 = 'Step 1: Extract groups'
        qc025 = 'Step 1.25: Two groups exist'
        qc05 = 'Step 1.5: Get documents for valid group indexes'
        qc1 = 'Step 2: Predict labels for each example'
        qc2 = 'Step 3: Match with groups @ agreement threshold ' + str(agreement_threshold)
        lst = [qc0, qc025, qc05, qc1, qc2]
        counters, totals = {}, {}
        for x in lst: counters[x], totals[x] = 0,0
        ex = self.data[idx]

        query, docs, mva = ex['question'], ex['documents'], ex['multi-view answer']
        assert len(mva) == 2

        def get_substrings(strg):
            sbs = re.findall(r'\[(.*?)\]', strg)
            resh = []
            for x in sbs:
                y = x.split(', ')
                resh += y
            resh = [x.replace('*', '') for x in resh]
            try:
                resh = [int(x) for x in resh]
            except:
                return None
            return sorted(list(set(resh)))

        group1_substrings = get_substrings(mva[0])
        group2_substrings = get_substrings(mva[1])

        totals[qc0] += 1
        if group1_substrings is None or group2_substrings is None:
            return [], counters, totals, [], [{'example': ex, 'failure': qc0}]
        counters[qc0] += 1

        group1 = [x for x in group1_substrings if x not in group2_substrings]
        group2 = [x for x in group2_substrings if x not in group1_substrings]

        totals[qc025] += 1
        if not(len(group1) > 0 and len(group2) > 0):
            return [], counters, totals, [], [{'example': ex, 'failure': qc025}]
        counters[qc025] += 1

        # These are 1-indexed, so make them 0-indexed!!!
        group1, group2 = [x-1 for x in group1], [x-1 for x in group2]
        idxes_to_keep = sorted(group1 + group2)
        totals[qc05] += 1
        if any([i >= len(docs) for i in idxes_to_keep]):
            return [], counters, totals, [], [{'example': ex, 'failure': qc05}]
        counters[qc05] += 1

        docs = [(docs[i], i) for i in idxes_to_keep]
        groups = [1 if x[1] in group1 else 2 for x in docs]
        docs = [x[0] for x in docs]
        labels = [None] * len(docs)

        totals[qc1] += 1
        for i in range(len(docs)):
            doc = docs[i]
            answer = doc['answers']
            answer = '; '.join(answer)
            label = self.prompt_get_label(query, answer)
            if label is None:
                return [], counters, totals, [],  [{'example': ex, 'failure': qc1}]
            labels[i] = (1 if label == 'yes' else 2)
        counters[qc1] += 1

        cnt_same, cnt_diff = 0,0
        for i in range(len(labels)):
            if groups[i] == labels[i]: cnt_same += 1
            else: cnt_diff += 1
        cnt_same, cnt_diff = cnt_same / len(labels), cnt_diff / len(labels)
        if cnt_same >= cnt_diff:
            confidence = cnt_same
        else:
            confidence = cnt_diff

        totals[qc2] += 1
        if confidence < agreement_threshold:
            return [], counters, totals, [], [{'example': ex, 'failure': qc2, 'groups': groups, 'binlabels': labels, 'confidence': confidence}]
        counters[qc2] += 1

        res = {query: {}}
        notes = {query: {}}
        for i in range(len(docs)):
            for j in range(i+1,len(docs)):
                i_is_yes, j_is_yes = labels[i] == 1, labels[j] == 1
                d1, d2 = docs[i]['text'], docs[j]['text']
                d1_first = random.random() < 0.5
                dpair = (d1, d2) if d1_first else (d2, d1)
                notes[query][dpair] = {'answer1': docs[i if d1_first else j]['answers'], 'answer2': docs[j if d1_first else i]['answers']}
                notes[query][dpair]['confidence'] = confidence
                if i_is_yes and j_is_yes or not i_is_yes and not j_is_yes:
                    res[query][dpair] = 'not_contradiction'
                else:
                    assert i_is_yes or j_is_yes and not (i_is_yes and j_is_yes)
                    res[query][dpair] = 'contradiction'

        return [res], counters, totals, [notes], []

    def prompt_get_label(self, query, answer):
        prompt_template = self.pl.robustqa_get_label()
        prompt = prompt_template.format(query=query, answer=answer)
        response = self.prompter.prompt(prompt)
        # print(prompt)
        # print(response)
        # pdb.set_trace()
        answer = re.findall('<response>(.*?)</response>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n')

        return answer


