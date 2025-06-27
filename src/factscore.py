from generator import ExampleGenerator
import json, pdb, re
import pandas as pd
import sqlite3

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            assert False
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print(f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text) == str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip()) > 0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset + MAX_LENGTH])
                            offset += MAX_LENGTH

                psgs = [tokenizer.decode(tokens) for tokens in passages if
                        np.sum([t not in [0, 2] for t in tokens]) > 0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time() - start_time) / 60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert results is not None and len(
            results) == 1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results) > 0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results

class FactScore(ExampleGenerator):
    def __init__(self, **kwargs):
        self.dname = 'factscore'
        super().__init__( **kwargs)

    def read_data(self):
        db_path = 'factscore/enwiki-20230401.db'
        self.db = DocDB(db_path=db_path, data_path=None)
        pth = 'factscore/'
        if self.partition == 'chatgpt': pth = pth + 'ChatGPT.jsonl'
        elif self.partition == 'instructgpt': pth = pth + 'InstructGPT.jsonl'
        elif self.partition == 'perplexityai': pth = pth + 'PerplexityAI.jsonl'
        else: assert False
        data = []
        with open(pth, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        res = []
        for i in range(len(data)):
            x = data[i]
            assert len(x) == 5
            res.append(x)
        return res

    def generate(self, idx):
        qc0 = 'Step 1: Has annotations'
        qc12 = 'Step 2: Has human atomic facts'
        qc1 = 'Step 3: Is not irrelevant'
        lst = [qc0, qc12, qc1]
        counters, totals = {}, {}
        for x in lst: counters[x], totals[x] = 0,0
        ex = self.data[idx]

        inp, out, topic, cat, annotations = ex['input'], ex['output'], ex['topic'], ex['cat'], ex['annotations']
        top = topic.replace("'", "''")
        doc1 = self.db.connection.cursor().execute("select * from documents where title = '"+top+"'").fetchall()
        assert len(doc1) == 1
        doc1 = doc1[0]
        assert len(doc1) == 2
        doc1 = doc1[1]
        doc2 = out
        all_facts_by_question = {}
        totals[qc0] += 1
        if annotations is None:
            failed_examples = [{**ex, 'failure': qc0}]
            return [], counters, totals, [], failed_examples
        counters[qc0] += 1
        failed_examples = []
        assert annotations is not None
        for i in range(len(annotations)):
            ann = annotations[i]
            text = ann['text']
            haf = ann['human-atomic-facts']
            totals[qc12] += 1
            if haf is None:
                failed_examples += [{**ex, 'i': i, 'failure': qc0}]
                continue
            counters[qc12] += 1
            for x in haf:
                totals[qc1] += 1
                if x['label'] == 'IR': continue # don't need irrelevant facts
                counters[qc1] += 1
                question = self.prompt_write_question(x['text'])
                all_facts_by_question[question] = all_facts_by_question.get(question, [])
                all_facts_by_question[question].append((x['text'],'entailment' if x['label'] == 'S' else 'contradiction'))

        # Now, if a question is repeated, and any label is contradiction, mark it as contradiction
        dpair = (doc1, doc2)
        resh = {dpair: {}}
        newh = {dpair: {}}
        for question in all_facts_by_question:
            lst = all_facts_by_question[question]
            labels = [x[1] for x in lst]
            lab = 'contradiction' if 'contradiction' in labels else 'entailment'
            resh[dpair][question] = lab
            newh[dpair][question] = lst

        if len(resh[dpair]) > 0:
            return [resh], counters, totals, [newh], []

        return [], counters, totals, [], []

    def prompt_write_question(self, fact):
        prompt_template = self.pl.factscore_write_question()
        prompt = prompt_template.format(fact=fact)
        response = self.prompter.prompt(prompt)
        # print(prompt)
        # print(response)
        answer = re.findall('<question>(.*?)</question>', response, re.DOTALL)
        if len(answer) != 1: return None
        answer = answer[0]
        answer = answer.strip('\n')

        return answer


