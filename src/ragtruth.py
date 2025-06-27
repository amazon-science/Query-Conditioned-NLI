import pdb, json, re

from generator import ExampleGenerator

class RagTruth(ExampleGenerator):
    def __init__(self,**kwargs):
        self.dname = 'ragtruth'
        super().__init__(**kwargs)

    def read_data(self):
        source_info, response = [], []
        with open('RagTruth/source_info.jsonl', 'r') as f:
            for line in f:
                source_info.append(json.loads(line))
        with open('RagTruth/response.jsonl', 'r') as f:
            for line in f:
                response.append(json.loads(line))
        assert self.partition in {'train', 'test'}

        res = {}
        for x in source_info:
            if x['task_type'] != 'QA': continue
            res[x['source_id']] = x
            res[x['source_id']]['outputs'] = []

        for x in response:
            if x['source_id'] in res:
                res[x['source_id']]['outputs'].append(x)

        final = []
        for k,v in res.items():
            splts = [s['split'] for s in v['outputs']]
            assert len(set(splts)) == 1
            splt = splts[0]
            if splt == self.partition: final.append(v)

        return final

    def generate(self, idx):
        qc0 = 'Step 1 no labels: Predict output contains answer to query'
        qc1 = 'Step 2 no labels: Output contains answer to query'
        qc3 = 'Step 1 labels: Decision made - entailment'
        qc2 = 'Step 2 labels: Proceedable if contradiction was not fail'
        qc4 = 'Step 3 labels: Decision made - contradiction'
        qc5 = 'Step 4 labels: Proceedable if neutral was not fail'
        qc6 = 'Step 5 labels: Decision made - neutral'
        lst = [qc0, qc1, qc3, qc2, qc4, qc5, qc6]
        counters, totals = {}, {}
        for x in lst: counters[x], totals[x] = 0,0
        ex = self.data[idx]

        query, doc1 = ex['source_info']['question'], ex['source_info']['passages']
        outputs = ex['outputs']

        res, additional_notes, failed_examples = [], [], []
        for x in outputs:
            addnotesh = {'id': x['id'], 'source_id': x['source_id'], 'model': x['model'], 'temperature': x['temperature'], 'quality': x['quality'], 'labels': x['labels']}
            labels = x['labels']
            doc2 = x['response']
            if len(labels) == 0:
                is_relevant_to_query = self.prompt_entire_relevant(query, doc2)
                totals[qc0 ]+= 1
                if is_relevant_to_query is None or is_relevant_to_query not in {'yes', 'no'}:
                    failed_examples.append({**addnotesh, 'doc1': doc1, 'doc2': doc2, 'query': query, 'labels': labels, 'failure': qc0})
                    continue

                counters[qc0] += 1
                totals[qc1] += 1
                if is_relevant_to_query == 'no':
                    failed_examples.append({**addnotesh, 'doc1': doc1, 'doc2': doc2, 'query': query, 'failure': qc1})
                    continue
                counters[qc1] += 1
                res.append({'d1': doc1, 'd2': doc2, 'query': query, 'label': 'entailment'})
                additional_notes.append(addnotesh)
            else:
                labs = [y['label_type'] for y in labels]
                for y in labs:
                    assert y in {'Evident Baseless Info', 'Subtle Baseless Info', 'Evident Conflict', 'Subtle Evident Conflict', 'Subtle Conflict'}
                relevances = []
                for y in labels:
                    is_relevant_to_query = self.prompt_snippet_relevant(query, doc2, y['text'])
                    relevances.append(is_relevant_to_query)
                addnotesh = {**addnotesh, 'labs': labs,'relevances': relevances,}
                totals[qc3]+= 1
                if len(set(relevances)) == 1 and 'no' in relevances:
                    decision = 'entailment'
                    counters[qc3] += 1
                else:
                    yeses, fails = set(), set()
                    assert len(relevances) == len(labs)
                    for i in range(len(relevances)):
                        if relevances[i] == 'yes':  yeses.add(labs[i])
                        elif relevances[i] == 'no': continue
                        else: fails.add(labs[i])
                    totals[qc2] += 1
                    if len(fails) > 0 and 'Evident Conflict' in fails or 'Subtle Evident Conflict' in fails or 'Subtle Conflict' in fails:
                        failed_examples.append({**addnotesh, 'doc1': doc1, 'doc2': doc2, 'query': query, 'failure': qc2})
                        continue
                    counters[qc2] += 1
                    totals[qc4] += 1
                    if 'Evident Conflict' in yeses or 'Subtle Evident Conflict' in yeses or 'Subtle Conflict' in yeses:
                        decision = 'contradiction'
                        counters[qc4] += 1
                    else:
                        totals[qc5] += 1
                        if 'Evident Baseless Info' in fails or 'Subtle Baseless Info' in fails:
                            failed_examples.append({**addnotesh, 'doc1': doc1, 'doc2': doc2, 'query': query, 'failure': qc5})
                            continue
                        counters[qc5] += 1
                        totals[qc6] += 1
                        if 'Evident Baseless Info' in yeses or 'Subtle Baseless Info' in yeses:
                            decision = 'neutral'
                            counters[qc6] += 1
                        else:
                            failed_examples.append({**addnotesh, 'doc1': doc1, 'doc2': doc2, 'query': query, 'failure': qc6})
                            continue
                res.append({'d1': doc1, 'd2': doc2, 'query': query, 'label': decision})
                additional_notes.append(addnotesh)

        return res, counters, totals, additional_notes, failed_examples

    def prompt_entire_relevant(self, query, doc):
        prompt_template = self.pl.ragtruth_entire_relevant()
        prompt = prompt_template.format(query=query, response=doc)
        response = self.prompter.prompt(prompt)
        # print(prompt)
        # print(response)
        # pdb.set_trace()
        answer = re.findall('<answer>(.*?)</answer>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n')

        return answer


    def prompt_snippet_relevant(self, query, doc, snippet):
        prompt_template = self.pl.ragtruth_snippet_relevant()
        prompt = prompt_template.format(query=query, response=doc, snippet=snippet)
        response = self.prompter.prompt(prompt)
        # print(prompt)
        # print(response)
        # pdb.set_trace()
        answer = re.findall('<answer>(.*?)</answer>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n')

        return answer

