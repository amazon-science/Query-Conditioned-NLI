import os, pdb, json, fire, sys

from factscore import FactScore
from prompt_library import PromptLibrary
from prompt_utils import Prompter

from snli import Snli
from robustqa import RobustQa
from ragtruth import RagTruth

def run(dataset,
        partition,
        start_num,
        model):
    prompter = Prompter(model)
    pl = PromptLibrary()
    assert type(start_num) == int
    if dataset == 'snli':
        assert partition in {'train', 'val', 'test'}
        generator = Snli(prompter, pl, partition, start_num, do_print=False)
    elif dataset == 'factscore':
        assert partition in {'chatgpt', 'instructgpt', 'perplexityai'}
        generator = FactScore(prompter, pl, partition, start_num, do_print=False)
    elif dataset == 'robustqa':
        assert partition == 'all'
        generator = RobustQa(prompter, pl, partition, start_num, do_print=False)
    elif dataset == 'ragtruth':
        assert partition in {'train', 'test'}
        generator = RagTruth(prompter, pl, partition, start_num, do_print=False)
    else: assert False
    generator.generate_examples()

if __name__ == '__main__':
    print(' '.join([sys.argv[0].split('/')[-1]] + sys.argv[1:]))
    sys.stdout.flush()
    fire.Fire(run)