import os, pdb, json, fire, sys

from prompt_library import PromptLibrary
from prompt_utils import Prompter

from task import Task
from print_examples import resolve_tupled_dataset, resolve_type3_dataset
import random
from task import confusion_compute
import numpy as np

def run(dataset,
        prompt_type,
        do_merge,
        use_query,
        start_num,
        model):
    assert prompt_type in {'zero', 'few', 'qanli', 'oracle'}
    if prompt_type in 'qanli': assert use_query
    if prompt_type == 'oracle': assert not use_query
    prompter = Prompter(model)
    pl = PromptLibrary()
    if not (dataset in {'snli', 'mctest', 'ragtruth'} or 'factscore' in dataset): assert not do_merge
    dset, partition = dataset, 'test'
    if dataset == 'snli':
        assert prompt_type != 'oracle'
        assert do_merge
        if not do_merge: label_set = ['entailment', 'contradiction', 'neutral']
        else: label_set = ['entailment', 'not_entailment']
        typ = 1
    elif dataset == 'ragtruth':
        assert prompt_type != 'oracle'
        assert do_merge
        if not do_merge: label_set = ['entailment', 'contradiction', 'neutral']
        else: label_set = ['entailment', 'not_entailment']
        typ = 1
    elif 'factscore' in dataset:
        assert do_merge, 'should always be merged, since contradiction is really not_entailment'
        label_set = ['entailment', 'not_entailment']
        typ = 2
        dset, partition = 'factscore', dataset.split('_')[1]
    elif dataset == 'robustqa':
        assert prompt_type != 'oracle'
        partition = 'all'
        assert not do_merge
        label_set = ['contradiction', 'not_contradiction']
        typ = 3
    else: assert False

    data = []

    with open('artefacts/' + dset + '/'+partition+'.json', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # figure out if type1 or type2; if type2 fix the format
    if typ == 2:
        data = resolve_tupled_dataset(data)
        print("Len orig data:", len(data))
        new_data = []
        for x in data:
            assert len(x) in {1,2}
            ky = [y for y in list(x.keys()) if y != 'additional_notes'][0]
            if use_query:
                for q in x[ky]:
                    newx = {'d1': ky[0], 'd2': ky[1], 'query': q, 'label': x[ky][q]}
                    new_data.append(newx)
            else:
                newx = {'d1': ky[0], 'd2': ky[1], 'labels': [x[ky][q] for q in x[ky]]}
                new_data.append(newx)
        data = new_data
    elif typ == 3:
        assert prompt_type != 'oracle'
        data = resolve_type3_dataset(data)
        print("Len orig data:", len(data))
        new_data = []
        for x in data:
            assert len(x) == 1
            query = list(x.keys())[0]
            for ex in x[query]:
                d1, d2, label = ex['d1'], ex['d2'], ex['label']
                new_data.append({'d1': d1, 'd2': d2, 'query': query, 'label': label})
        data = new_data
    print("Len data:", len(data))

    if (dataset in {'snli', 'mctest', 'ragtruth'} or 'factscore' in dataset) and do_merge:
        for x in data:
            if use_query or not typ == 2: x['label'] = 'not_entailment' if x['label'] != 'entailment' else x['label']
            else:
                for i in range(len(x['labels'])):
                    x['labels'][i] = 'not_entailment' if x['labels'][i] != 'entailment' else x['labels'][i]

    lab_distn = {}
    for x in data:
        if typ != 2 or typ == 2 and use_query: lab_distn[x['label']] = lab_distn.get(x['label'], 0) + 1
        else:
            assert typ == 2 and not use_query
            for lab in x['labels']:
                lab_distn[lab] = lab_distn.get(lab, 0) + 1
    print("Lab distribution:", lab_distn)

    # Compute best possible
    if prompt_type == 'oracle':
        assert not use_query and typ == 2
        print("BEST POSSIBLE SCORES")
        tp, fp, fn, tn = 0, 0, 0, 0
        ties_un, ties_no = 0, 0
        ties_by_group = []
        for x in data:
            labs = x['labels']
            cnt_ent, cnt_nent = sum([y == 'entailment' for y in labs]), sum([y == 'not_entailment' for y in labs])
            assert cnt_ent + cnt_nent == len(labs)
            if cnt_ent > cnt_nent:
                pred = 'entailment'
            elif cnt_ent < cnt_nent:
                pred = 'not_entailment'
            else: # tie
                assert cnt_ent == cnt_nent
                ties_un += 1
                ties_no += len(labs)
                pred = 'entailment' if random.random() < 0.5 else 'not_entailment'
                ties_by_group.append(len(labs))
                continue
            for y in labs:
                if y == 'entailment' and pred == 'entailment':
                    tp += 1
                elif y == 'entailment' and pred == 'not_entailment':
                    fn += 1
                elif y == 'not_entailment' and pred == 'entailment':
                    fp += 1
                elif y == 'not_entailment' and pred == 'not_entailment':
                    tn += 1
                else:
                    assert False
        if len(ties_by_group) > 0:
            import itertools
            combo, best_metric = None, -100000
            # Iterate over all 2^n combinations of "on" and "off"
            for combination in itertools.product([True, False], repeat=len(ties_by_group)):
                current_tp =tp
                current_tn = tn

                # Assign ties to tp or tn based on the combination
                for i, is_on in enumerate(combination):
                    if is_on:
                        current_tp += (ties_by_group[i])
                    else:
                        current_tn += (ties_by_group[i])

                # Compute the metric for the current combination
                combo = (current_tp, fp, fn, current_tn)
                metric = confusion_compute(current_tp, fp, fn, current_tn, ['entailment', 'not_entailment'], do_print=False)

                # Update the best combination if the current metric is better
                if metric > best_metric:
                    best_metric = metric
                    best_combo = combo
                print(metric, combo, best_metric, best_combo)
            tp, tn = combo[0], combo[3]
        confusion_compute(tp, fp, fn, tn, ['entailment', 'not_entailment'])
        print(np.array([[tp, fn], [fp, tn]]))

        print('Number ties unnormalized:', ties_un)
        print('Number ties normalized:', ties_no)

    else:
        # Run through simple prompting baseline to see confusion table.
        task = Task(data, prompter, pl, prompt_type, use_query, dataset=dataset,
                    label_set=label_set, do_print=False, start_num=start_num)
        task.predict_all()
        task.evaluate()

if __name__ == '__main__':
    print()
    print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
    fire.Fire(run)