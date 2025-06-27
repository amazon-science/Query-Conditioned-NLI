from generator import ExampleGenerator
import json, pdb, re
import pandas as pd

class Snli(ExampleGenerator):
    def __init__(self, **kwargs):
        self.dname = 'snli'
        super().__init__(**kwargs)

    def read_data(self):
        if self.partition == 'train':  pth = 'SNLI/' + 'snli_1.0_train.jsonl'
        elif self.partition == 'val':  pth = 'SNLI/' + 'snli_1.0_dev.jsonl'
        elif self.partition == 'test': pth = 'SNLI/' + 'snli_1.0_test.jsonl'
        elif self.partition == 'special': assert False # pth = 'special.csv'
        else: assert False
        data = []
        if self.partition != 'special':
            with open(pth, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            data = pd.read_csv(pth).to_dict('records')
            assert len(data) == 200
        res = []
        for i in range(len(data)):
            x = data[i]
            if self.partition != 'special':
                assert len(x) == 10
                dct = {'sentence1': x['sentence1'], 'sentence2': x['sentence2'], 'label': x['gold_label']}
            else: dct = {'sentence1': x['d1_ans'], 'sentence2': x['d2_ans'], 'label': x['label']}
            if dct['label'] not in {'entailment', 'contradiction', 'neutral'}:
                continue
            res.append(dct)
        return res

    def generate(self, idx):
        qc0 = 'Step 0: Throw out neutral'
        qc1 = 'Step 1: Paragraph/Question writing success'
        qc2 = 'Quality Check 1: Sentence 1 is a possible answer to question'
        qc3 = 'Quality Check 2: Sentence 2 is a possible answer to question'
        qc35 = 'Modification 1: Is not about location?'
        qc36 = '\tStep 2: Modify location on paragraph2'
        qc4 = 'Quality Check 3: Paragraph 1 contains answer to question'
        qc5 = 'Quality Check 4: Paragraph 2 contains answer to question'
        # qc6 = 'Quality Check 5: Neutral - sentence2 is not definitely true for paragraph 1'
        lst = [qc0, qc1, qc2, qc3, qc35, qc36, qc4, qc5]#, qc6]
        counters, totals = {}, {}
        for x in lst: counters[x], totals[x] = 0,0
        ex = self.data[idx]
        assert ex['label'] in {'entailment', 'contradiction', 'neutral'}

        totals[qc0] += 1
        if ex['label'] == 'neutral':
            failed_examples = [{'sent1': ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'label': ex['label'],
                                'failure': qc0}]
            return [], counters, totals, [], failed_examples
        counters[qc0] += 1

        paragraph1, paragraph2, question = self.prompt_write_all(ex['sentence1'], ex['sentence2'], ex['label'])
        totals[qc1] += 1
        if paragraph1 is None:
            failed_examples = [{'sent1':ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'label': ex['label'],
                                'failure': qc1}]
            return [], counters, totals, [], failed_examples
        counters[qc1] += 1

        pass_is_answer1 = self.prompt_is_an_answer(question, ex['sentence1'])
        totals[qc2] += 1
        if pass_is_answer1 is None or pass_is_answer1 in {'no', 'maybe'}:
            failed_examples = [{'sent1':ex['sentence1'],
                                'sent2': ex['sentence2'],
        'paragraph1': paragraph1,
        'paragraph2': paragraph2,
        'question': question,
                                'label': ex['label'],
                                'failure': qc2 + ' - '+ str(pass_is_answer1)}]
            return [], counters, totals, [], failed_examples
        counters[qc2] += 1

        pass_is_answer2 = self.prompt_is_an_answer(question, ex['sentence2'])
        totals[qc3] += 1
        if pass_is_answer2 is None or pass_is_answer2 in {'no', 'maybe'}:
            failed_examples = [{'sent1':ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'paragraph1': paragraph1,
                                'paragraph2': paragraph2,
                                'question': question,
                                'label': ex['label'],
                                'failure': qc3+ ' - ' + str(pass_is_answer2)}]
            return [], counters, totals, [], failed_examples
        counters[qc3] += 1

        is_about_location = self.prompt_is_about_location(question)
        totals[qc35] += 1
        if is_about_location is not None and is_about_location == 'no':
            pargraph2_orig = paragraph2
            paragraph2 = self.prompt_modify_location(paragraph2, question, ex['sentence2'])
            counters[qc35] += 1 # inside since want to count number of times modified this.
            totals[qc36] += 1
            if paragraph2 is None:
                failed_examples = [{'sent1': ex['sentence1'],
                                    'sent2': ex['sentence2'],
                                    'label': ex['label'],
                                    'paragraph1': paragraph1,
                                    'paragraph2': pargraph2_orig,
                                    'question': question,
                                    'failure': qc36}]
                return [], counters, totals, [], failed_examples
            counters[qc36] += 1

        pass_answerability1 = self.prompt_answerability(paragraph1, question)
        totals[qc4] += 1
        if pass_answerability1 is None or pass_answerability1 in {'no', 'maybe'}:
            failed_examples = [{'sent1':ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'paragraph1': paragraph1,
                                'paragraph2': paragraph2,
                                'question': question,
                                'label': ex['label'],
                                'failure': qc4 + ' - ' + str(pass_answerability1)}]
            return [], counters, totals, [], failed_examples
        counters[qc4] += 1

        pass_answerability2 = self.prompt_answerability(paragraph2, question)
        totals[qc5] += 1
        if pass_answerability2 is None or pass_answerability2 in {'no', 'maybe'}:
            failed_examples = [{'sent1':ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'paragraph1': paragraph1,
                                'paragraph2': paragraph2,
                                'question': question,
                                'label': ex['label'],
                                'failure': qc5 + ' - ' + str(pass_answerability2)}]
            return [], counters, totals, [], failed_examples
        counters[qc5] += 1

        # if ex['label'] == 'neutral':
        #     pass_neutral_check = self.prompt_is_def_true(paragraph1, ex['sentence2'])
        #     totals[qc6] += 1
        #     if pass_neutral_check is None or pass_neutral_check in {'yes', 'maybe'}: # don't want it to be definitely true
        #         failed_examples = [{'sent1': ex['sentence1'],
        #                             'sent2': ex['sentence2'],
        #                             'paragraph1': paragraph1,
        #                             'paragraph2': paragraph2,
        #                             'question': question,
        #                             'label': ex['label'],
        #                             'failure': qc6 + ' - ' + str(pass_neutral_check)}]
        #         return [], counters, totals, [], failed_examples
        #     counters[qc6] += 1


        if self.do_print: print('-----------------------------------------')
        res = [{'d1': paragraph1, 'd2': paragraph2, 'query': question, 'label': ex['label']}]
        notes = [{'d1_ans': ex['sentence1'], 'd2_ans': ex['sentence2']}]
        return res, counters, totals, notes, []

    def generate_old(self, idx):
        '''
        Generate paragrah/question for doc1; then paragraph for doc2 separately; no quality checks.
        '''
        qc1 = 'Step 1: Paragraph/Question writing success for d1'
        qc2 = 'Step 2: Paragraph/Question writing success for d2'
        counters, totals = {qc1: 0, qc2: 0}, {qc1: 0, qc2: 0}
        ex = self.data[idx]
        assert ex['label'] in {'entailment', 'contradiction', 'neutral'}
        paragraph, question = self.prompt_write_paragraph(ex['sentence1'])
        totals[qc1] += 1
        if paragraph is None:
            failed_examples = [{'sent1':ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'label': ex['label'],
                                'failure': qc1}]
            return [], counters, totals, [], failed_examples
        counters[qc1] += 1
        paragraph2 = self.prompt_write_paragraph_with_question(ex['sentence2'], question)
        totals[qc2] += 1
        if paragraph2 is None:
            failed_examples = [{'sent1': ex['sentence1'],
                                'sent2': ex['sentence2'],
                                'label': ex['label'],
                                'd1': paragraph,
                                'question': question,
                                'failure': qc2}]
            return [], counters, totals, [], failed_examples
        counters[qc2] += 1
        if self.do_print: print('-----------------------------------------')
        res = [{'d1': paragraph, 'd2': paragraph2, 'query': question, 'label': ex['label']}]
        notes = [{'d1_ans': ex['sentence1'], 'd2_ans': ex['sentence2']}]
        return res, counters, totals, notes, []

    def prompt_is_an_answer(self, query, answer):
        prompt_template = self.pl.snli_is_an_answer()
        prompt = prompt_template.format(query=query, answer=answer)
        response = self.prompter.prompt(prompt)
        answer = re.findall('<response>(.*?)</response>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n').lower()
        if answer not in {'yes', 'no', 'maybe'}: return None

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Answer:', answer)

        return answer

    def prompt_answerability(self, paragraph, query):
        prompt_template = self.pl.snli_answerability()
        prompt = prompt_template.format(paragraph=paragraph, query=query)
        response = self.prompter.prompt(prompt)

        answer = re.findall('<answer>(.*?)</answer>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n').lower()
        if answer not in {'yes', 'no', 'maybe'}: return None

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Answer:', answer)

        return answer

    def prompt_is_def_true(self, paragraph, sentence):
        prompt_template = self.pl.snli_is_def_true()
        prompt = prompt_template.format(paragraph=paragraph, sentence=sentence)
        response = self.prompter.prompt(prompt)

        answer = re.findall('<answer>(.*?)</answer>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n').lower()
        if answer not in {'yes', 'no', 'maybe'}: return None

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Answer:', answer)

        return answer

    def prompt_is_about_location(self, query):
        prompt_template = self.pl.snli_is_about_location()
        prompt = prompt_template.format(query=query)
        response = self.prompter.prompt(prompt)

        answer = re.findall('<answer>(.*?)</answer>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n').lower()
        if answer not in {'yes', 'no', 'maybe'}: return None

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Answer:', answer)

        return answer

    def prompt_modify_location(self, paragraph, query, sentence):
        prompt_template = self.pl.snli_modify_location()
        prompt = prompt_template.format(paragraph=paragraph, query=query, sentence=sentence)
        response = self.prompter.prompt(prompt)

        paragraph = re.findall('<modified_paragraph>(.*?)</modified_paragraph>', response, re.DOTALL)
        if len(paragraph) != 1: return None
        paragraph = paragraph[0]
        paragraph = paragraph.strip('\n')

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Answer:', paragraph)

        return paragraph

    def prompt_write_all(self, sentence1, sentence2, label):
        prompt_template = self.pl.snli_write_all(label)
        prompt = prompt_template.format(sent1=sentence1, sent2=sentence2, label=label)
        response = self.prompter.prompt(prompt)

        paragraph = re.findall('<paragraph1>(.*?)</paragraph1>', response, re.DOTALL)
        if len(paragraph) != 1: return None, None, None
        paragraph = paragraph[0]
        paragraph = paragraph.strip('\n')

        paragraph2 = re.findall('<paragraph2>(.*?)</paragraph2>', response, re.DOTALL)
        if len(paragraph2) != 1: return None, None, None
        paragraph2 = paragraph2[0]
        paragraph2 = paragraph2.strip('\n')

        question = re.findall('<query>(.*?)</query>', response, re.DOTALL)
        if len(question) != 1: return None, None, None
        question = question[0]
        question = question.strip('\n')

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Paragraph 1:', paragraph)
            print('Paragraph 2:', paragraph2)
            print('Question:', question)

        return paragraph, paragraph2, question

    def prompt_write_paragraph(self, sentence):
        prompt_template = self.pl.snli_write_paragraph()
        prompt = prompt_template.format(sentence=sentence)
        response = self.prompter.prompt(prompt)

        paragraph = re.findall('<paragraph>(.*?)</paragraph>', response, re.DOTALL)
        if len(paragraph) != 1: return None, None
        paragraph = paragraph[0]
        paragraph = paragraph.strip('\n')

        question = re.findall('<question>(.*?)</question>', response, re.DOTALL)
        if len(question) != 1: return None, None
        question = question[0]
        question = question.strip('\n')

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Paragraph:', paragraph)
            print('Question:', question)

        return paragraph, question

    def prompt_write_paragraph_with_question(self, sentence, question):
        prompt_template = self.pl.snli_write_paragraph_with_question()
        prompt = prompt_template.format(sentence=sentence, question=question)
        response = self.prompter.prompt(prompt)

        paragraph = re.findall('<paragraph>(.*?)</paragraph>', response, re.DOTALL)
        if len(paragraph) != 1: return None
        paragraph = paragraph[0]
        paragraph = paragraph.strip('\n')

        if self.do_print:
            print(prompt)
            print('\n')
            print(response)
            print('\n')
            print('Paragraph:', paragraph)

        return paragraph

