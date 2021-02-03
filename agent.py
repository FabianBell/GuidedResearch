from transformers import T5Tokenizer, logging
logging.set_verbosity_error()
import re
import torch
import pandas as pd
import random

from model import *

class Agent:
    
    OKAY = '\033[92m'
    ERROR = '\033[91m'
    END = '\033[0m'

    def _get_tokenizer(self):
        EOB = '<EOB>'
        BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
        KB_PREFIX = ' DB: '
        EOKB = '<EOKB>'
        QUERY = 'query'
        tokenizer = T5Tokenizer.from_pretrained('t5-large',
            additional_special_tokens=[EOB, BELIEF_PREFIX, EOB, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', '<CTX>', QUERY, *[f'<extra_id_{i}>' for i in range(100)]])
        return tokenizer

    def __init__(self):
        #self.solist = DialogueRestyler()
        #self.solist.load_state_dict(torch.load('solist.pt'))
        self.textsettr = TextSETTR()
        self.textsettr.load_state_dict(torch.load('textsettr.pt'))
        self.tokenizer = self._get_tokenizer()
        self.bs_pattern = re.compile(r'[^{]*{\s\s?(?P<data>[^}]*)\s}(\squery\s{\s(?P<query>.*)\s})?\s<EOB>\s*')
        self.argument_pattern = re.compile(r':set\s(?P<arg>\w*)=(?P<argv>\w*)')
        self.kb = pd.read_pickle('kb.pkl')

    def _get_bs_dict(self, seq):
        match = self.bs_pattern.fullmatch(seq)
        if match is not None:
            if match.group('query') is not None:
                data = match.group('data')
                print(data)
                data = [elem.split(' = ') for elem in data.split('; ')]
                data = {k : v for k, v in data}
                return match.group('query'), data
            else:
                return None
        raise Exception(f'Invalid beliefe state {seq}')

    def db(self, bs_resp):
        if bs_resp is not None:
            query, bs_data = bs_resp
            print(query)
            args = list(bs_data.keys())
            query_data = self.kb[(self.kb.name == query) & (self.kb.args.apply(lambda x: x.issubset(args)))].apply(lambda x: x.apply(len) if x.name == 'args' else x).sort_values('args', ascending=False)
            if len(query_data) == 0:
                resp = ''
            else:
                query_data = query_data.iloc[0]
                resp = ' ; '.join([f'{query_data.response}_{i+1} = {elem}' for i, elem in enumerate(random.choice(query_data.data)[1])])
                resp = f'{query} { {resp} }'
        else:
            resp = ''
        return f' DB: {resp} <EOKB>'

    def _get_solist_result(self, inp):
        input_ids = self.tokenizer(inp, return_tensors='pt').input_ids
        pred = self.solist.generate(
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            input_ids,
            torch.ones(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long),
            torch.zeros(1, 512),
            max_length=512
        )
        beliefe_state = self.tokenizer.decode(pred[0])
        return beliefe_state

    def get_modified_response(self, response, source, target):
        print('S:', source)
        print('T:', target)
        input_ids = self.tokenizer(response, return_tensors='pt').input_ids
        source_ids, source_mask = self.tokenizer(source, return_tensors='pt').values()
        target_ids, target_mask = self.tokenizer(target, return_tensors='pt').values()
        pred = self.textsettr.generate(
            target_ids,
            target_mask,
            input_ids,
            torch.ones(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long),
            torch.tensor([[0.2, 0.4, 0.2, 0.4] + [0.]* 1020]),
            source_ids,
            source_mask,
            max_length=512
        )
        response = self.tokenizer.decode(pred[0])
        return response

    def write_okay(self, seq):
        print(self.OKAY + seq + self.END)

    def write_error(self, seq):
        print(self.ERROR + seq + self.END)

    def write_out(self, seq):
        print(f'>>>{seq}')

    def get_in(self):
        inp = input('>>>')
        return inp

    def __call__(self):
        history = ''
        target = ''
        source = ''
        while True:
            user = self.get_in()
            match = self.argument_pattern.fullmatch(user)
            if match is not None:
                arg, argv = match.groups()
                if arg == 'level':
                    try:
                        level = int(argv)
                        self.textsettr.set_style_level(level)
                        self.write_okay("New style level set")
                    except ValueError:
                        self.write_error(f"Invalid value {argv} for argument {arg}")
                else:
                    self.write_error(f"Unkown argument {arg}")
                continue
            if user == 'restart':
                history = ''
                target = ''
                source = ''
                self.write_okay('\n## Restart agent\n')
                continue
            if user == 'close':
                break
            target += ' ' + user
            history += 'user: ' + user
            beliefe_state = self._get_solist_result(history)
            #print('Beliefe state:', beliefe_state)
            bs_data = self._get_bs_dict(beliefe_state)
            kb_data = self.db(bs_data)
            intermediate = ''.join([history, beliefe_state, kb_data])
            response = self._get_solist_result(intermediate)
            print('Response:', response)
            source += ' ' + response
            history += ' assistant: ' + response
            response = self.get_modified_response(response, source, target)
            self.write_out(response)

agent = Agent()
#agent()
