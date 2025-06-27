import os, pdb, json, fire, sys, random, pprint

from prompt_library import PromptLibrary
from prompt_utils import Prompter

from task import Task

import pandas as pd

def resolve_tupled_dataset(data): # type 2
    new_data = []
    assert len(data[0]) in {1, 2}
    if len(data[0]) == 1:
        assert 'tuple1' in data[0]
    elif len(data[0]) == 2:
        assert 'tuple1' in data[0] and 'additional_notes' in data[0]
    else:
        assert False
    for x in data:
        newx = {}
        correct_key = tuple(x['tuple1']['tuple'])
        assert len(correct_key) == 2
        newx[correct_key] = {}
        for k, v in x['tuple1'].items():
            if k == 'tuple': continue
            newx[correct_key][k] = v
        if 'additional_notes' in x: newx['additional_notes'] = x['additional_notes']
        assert len(newx) == len(x)
        new_data.append(newx)
    assert len(new_data) == len(data)

    return new_data

def resolve_type3_dataset(data):
    res = []
    for x in data:
        assert len(x) == 1
        query = list(x.keys())[0]
        rest = x[query]
        entry_here = {query: []}
        for tup, val in rest.items():
            d1, d2 = val['tuple'][0], val['tuple'][1]
            lab = val['label']
            entry_here[query].append({'d1': d1, 'd2': d2, 'label': lab, 'answer1': val['answer1'], 'answer2': val['answer2'], 'confidence': val['confidence']})
        res.append(entry_here)
    return res

def do_one_dataset(dataset, partition, num_examples, numper=None):
    assert dataset in {'snli', 'factscore_chatgpt', 'factscore_instructgpt', 'factscore_perplexityai', 'robustqa', 'ragtruth'} # 'docnli', 'factchd'
    data = []
    if 'factscore' in dataset:
        dset, part = 'factscore', dataset.split('_')[1]
    else:
        dset, part = dataset, partition
    with open('artefacts/' +dset+ '/' +part+'.json', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # figure out if type1 or type2; if type2 fix the format
    if dataset =='snli': typ = 1
    elif 'factscore' in dataset: typ = 2
    elif dataset == 'robustqa': typ = 3
    elif dataset == 'ragtruth': typ = 1
    else: assert False
    if typ == 2:
        data = resolve_tupled_dataset(data)
        assert numper is not None
        assert sum(numper) == num_examples
    elif typ == 3:
        data = resolve_type3_dataset(data)
        assert numper is not None
        assert sum(numper) == num_examples
    else:
        assert typ == 1
        assert numper is None

    random.shuffle(data)
    final_to_write = []
    data = data[:num_examples]
    for iii in range(len(data)):
        if len(final_to_write) >= num_examples: # case 2 or 3
            break
        x = data[iii]
        if typ == 1:
            newx = {}
            for k, v in x.items():
                if type(v) == str: v = v.replace('\n', '\\n')
                if k not in {'d1', 'd2', 'query', 'label', 'd1_ans', 'd2_ans', 'labels'}:
                    continue
                else:
                    newx[k] = v
            x = {'dataset': dataset, **newx, 'd1_coherent': ' ', 'd2_coherent': ' ', 'd1_answerable': ' ', 'd2_answerable': ' ',
                 'your_label': ' '}
            final_to_write.append(x)
        elif typ == 2:
            examples_here = []
            assert len(x) in {1,2}
            add_notes = x['additional_notes'] if 'additional_notes' in x else None
            ky = [pp for pp in list(x.keys()) if pp != 'additional_notes'][0]
            doc1, doc2 = ky[0], ky[1]
            qs = x[ky]
            qsh = list(qs.items())
            random.shuffle(qsh)
            for k,v in qsh[:numper[iii]]:
                add_notes_h = add_notes.get(k, {})
                if 'factscore' in dataset:
                    newx = {'query': k, 'label': v,
                            'fact': add_notes_h[0][0]}
                else: assert False
                if len(examples_here) == 0: xh = {'dataset': dataset, 'd1': doc1, 'd2': doc2}
                else: xh = {'dataset': ' ', 'd1': ' ', 'd2': ' '}
                xh = {**xh, **newx, 'd1_coherent': ' ', 'd2_coherent': ' ', 'd1_answerable': ' ', 'd2_answerable': ' ',
                 'your_label': ' '}
                examples_here.append(xh)
            final_to_write += examples_here
        elif typ == 3:
            examples_here = []
            assert len(x) == 1
            query = list(x.keys())[0]
            assert dataset == 'robustqa'
            if iii >= len(numper): nii = 2 # for now jsut set to 2, happens when not enough in one to really work
            else: nii = numper[iii]
            for ex in x[query]:
                if len(examples_here) == 0: resh = {'dataset': dataset, 'query': query}
                else: resh = {'dataset': ' ', 'query': ' '}
                resh = {**resh, **ex}
                if 'answer1' in resh: resh['answer1'] = '; '.join(resh['answer1'])
                if 'answer2' in resh: resh['answer2'] = '; '.join(resh['answer2'])
                resh = {**resh, 'd1_coherent': ' ', 'd2_coherent': ' ', 'd1_answerable': ' ', 'd2_answerable': ' ',
                      'your_label': ' '}
                examples_here.append(resh)
                if len(examples_here) ==nii: break# numper[iii]: break
            final_to_write += examples_here
        else: assert False

    if typ == 3:
        if len(final_to_write) > num_examples: final_to_write = final_to_write[:num_examples] # becasue of case where not enough in one so had to do extra
    assert len(final_to_write)  == num_examples, str(len(final_to_write)) + ' ' + str(num_examples)
    return final_to_write


def run(dataset,
        partition,
        num_examples,
        num_annotators=1,
        num_overlap=0):
    do_annotation=True

    assert dataset in {'snli', 'factscore_chatgpt', 'factscore_instructgpt', 'factscore_perplexityai', 'robustqa', 'ragtruth'}
    if do_annotation: assert num_annotators > 0
    else: assert num_annotators == 0 and num_overlap == 0
    if num_annotators == 1: assert num_overlap == 0
    assert dataset != 'all', 'not really gonna work now since have multiple formats'

    final_to_write = []
    do_numper = 'factscore' in dataset or 'robustqa' == dataset
    for x in [dataset]:
        if not do_annotation:
            assert False, 'phasing out'
        else:
            if num_overlap > 0:
                val_const = do_one_dataset(x, partition, num_overlap, numper = [3,4] if do_numper else None)
            for pers in range(num_annotators):
                if num_overlap > 0:
                    valh = [{'annotator': 'annotator_' + str(pers + 1), **y} for y in val_const]
                    final_to_write += valh

                val = do_one_dataset(x, partition, num_examples - num_overlap, numper = [3,4,3,4] if do_numper else None)
                val = [{'annotator': 'annotator_' + str(pers + 1), **y} if y['d1'] != ' ' else y for y in val]
                final_to_write += val

    if final_to_write is not None:
        df = pd.DataFrame.from_records(final_to_write, index=None)
        df.to_csv('temp.csv')



if __name__ == '__main__':
    print()
    print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
    fire.Fire(run)