import pdb, sys

import tqdm, json

class ExampleGenerator(object):
    def __init__(self, prompter, prompt_library, partition, start_num, num_examples=None, do_print=False):
        self.prompter = prompter
        self.pl = prompt_library
        self.start_num = start_num # put as current length if want to continue (this is 0 idxed, so if put as current length will start at next one)
        self.num_examples = num_examples
        self.partition = partition
        self.do_print = do_print
        self.paths = self.make_out_paths()

        self.examples = None # Final examples
        self.additional_notes = None # Additional notes from data generation to examine
        self.failed_examples = None

        self.data = []
        self.data = self.read_data()
        assert start_num < len(self.data), str(len(self.data))
        self.data = self.data[start_num:]

        if self.dname == 'snli': self.data_type = 1
        elif self.dname in {'mctest'} or 'factscore' in self.dname: self.data_type = 2
        elif self.dname == 'robustqa':self.data_type = 3
        elif self.dname == 'ragtruth': self.data_type = 1
        else:
            assert False
        if start_num == 0:
            for path in self.paths:
                open(self.paths[path], 'w').close()

    def make_out_paths(self):
        savepath = 'artefacts/' + self.dname + '/' + self.partition + '.json'
        failpath = 'artefacts/' + self.dname + '/'+ self.partition + '_fail.json'
        return {'savepath': savepath, 'failpath': failpath}

    def read_data(self):
        raise NotImplementedError("Implement in subclass.")

    def generate(self, idx):
        raise NotImplementedError("Implement in subclass.")

    def generate_examples(self):
        self.examples, self.additional_notes, self.failed_examples = [], [], []
        counters, totals = {}, {}
        num_examples = self.num_examples if self.num_examples is not None else len(self.data)
        N = min(len(self.data), num_examples)
        do_tqdm = False
        iterator = tqdm.tqdm(range(N)) if do_tqdm else range(N)
        for i in iterator:
            examples, cnts, tots, additional_notes, failed_examples = self.generate(i)
            if cnts is not None or tots is not None:
                assert cnts is not None and tots is not None
                assert sorted(cnts.keys()) == sorted(tots.keys())
                for k, v in cnts.items():
                    counters[k] = counters.get(k, 0) + v
                    totals[k] = totals.get(k, 0) + tots[k]
            self.examples += examples
            self.failed_examples += failed_examples
            if self.additional_notes is not None:
                assert len(additional_notes) == len(examples)
                self.additional_notes += additional_notes

            # Print info
            print('Completed', (i+1), '/', N, '=', str(round(100*(i+1)/N,2)) + '% of generations.')
            if len(counters) > 0: self.print_quality_checks(counters, totals)

            # Write data up till now
            if len(examples) > 0:
                if self.data_type == 2: assert len(examples[0].keys()) == 1 and type(list(examples[0].keys())[0]) == tuple
                elif self.data_type == 3: assert len(examples[0].keys()) == 1 and type(list(examples[0].keys())[0]) == str
                if self.data_type in {2,3}:
                    for iii in range(len(additional_notes)):
                        assert len(additional_notes[iii]) == 1
                        additional_notes[iii] = {'additional_notes': list(additional_notes[iii].values())[0]}
                self.write_examples(self.paths['savepath'], examples, additional_notes)
            if len(failed_examples) > 0:
                self.write_examples(self.paths['failpath'], failed_examples, is_fail=True)

            # Print stats of this batch
            self.print_stats()
            sys.stdout.flush()

    def print_stats(self):
        if len(self.examples) == 0: return
        lab_distn = {}
        for x in self.examples:
            if self.data_type == 1: lab_distn[x['label']] = lab_distn.get(x['label'], 0) + 1
            elif self.data_type == 2:
                assert len(x) == 1
                only_key = list(x.keys())[0]
                for val in x[only_key]:
                    lab_distn[x[only_key][val]] = lab_distn.get(x[only_key][val], 0) + 1
            else:
                assert self.data_type == 3
                assert len(x) == 1
                query = list(x.keys())[0]
                for y in x[query]:
                    lab_distn[x[query][y]] = lab_distn.get(x[query][y], 0) + 1

        print('    Examples kept:', sum(lab_distn.values()))
        print('    Examples failed:', len(self.failed_examples))
        print('    Label distribution:       ', lab_distn)
        if self.data_type == 1 and self.dname == 'snli': # Just snli for now
            lab_distn2 = {}
            for x in self.failed_examples:
                lab_distn2[x['label']] = lab_distn2.get(x['label'], 0) + 1
            print('    Label distribution tossed:', lab_distn2)


    def print_quality_checks(self, counters, totals):
        i = 1
        for k, v in counters.items():
            print('\t',k + ':', v, '/', totals[k], '=', (str(round(100 * v / totals[k], 2)) if totals[k] != 0 else 'None') + '%')
            i+=1

    def write_examples(self, path, examples, notes=None, is_fail=False):
        data_type = self.data_type
        if is_fail: data_type = 1 # use this for now
        assert type(examples) == list
        if notes is not None:
            assert type(notes) == list and len(examples) == len(notes)
            dct = [{**examples[i], **notes[i]} for i in range(len(examples))]
        else: dct = examples
        for i in range(len(dct)):
            # Can't write tuples unfortunately, do special thing
            newdcti = {}
            cnt=1
            if data_type == 3:
                add_notes = dct[i].get('additional_notes', {})
                if 'additional_notes' in dct[i]: dct[i].pop('additional_notes')
                try:
                    assert len(dct[i]) == 1
                except:
                    pdb.set_trace()
                query = list(dct[i].keys())[0]
                newdcti[query] = {}
                itr = dct[i][query]
            else: itr = dct[i]
            for x in itr: # x is a key
                if data_type == 2:
                    if type(x) == tuple:
                        assert type(itr[x]) == dict, 'for now'
                        newdcti['tuple' + str(cnt)] = {'tuple': list(x), **itr[x]}
                        cnt += 1
                    elif type(x) == str:
                        newdcti[x] = itr[x]
                    else: assert False
                elif data_type == 3:
                    # note here itr is different than the other cases
                    # x is a dpair
                    assert type(x) == tuple
                    assert type(itr[x]) == str
                    if add_notes is not None: dcth = {'tuple': list(x), 'label': itr[x], **add_notes[x]}
                    else: dcth = {'tuple': list(x), 'label': itr[x]}
                    newdcti[query]['tuple'+str(cnt)] = dcth
                    cnt += 1
                else:
                    assert data_type == 1
                    assert type(x) == str
                    newdcti[x] = dct[i][x]
            json_string = json.dumps(newdcti)
            with open(path, 'a') as outfile:
                outfile.write(json_string + '\n')

