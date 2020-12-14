from transformers import T5Tokenizer, logging
logging.set_verbosity_error()
import re
import torch

from model import *

CONTEXT_ID = 32109

class Agent:

    def _get_tokenizer(self):
        EOB = '<EOB>'
        BELIEF_PREFIX = ' => Belief State : Movie Tickets { '
        BELIEF_SUFFIX = ' } ' + EOB
        KB_PREFIX = ' DB: '
        EOKB = '<EOKB>'
        tokenizer = T5Tokenizer.from_pretrained('t5-small',
            additional_special_tokens=[EOB, BELIEF_PREFIX, BELIEF_SUFFIX, KB_PREFIX, EOKB, '{', '}', 'assistant:', 'user:', '<CTX>', *[f'<extra_id_{i}>' for i in range(100)]])
        return tokenizer

    def __init__(self):
        self.model = DialogueRestyler()
        self.model.load_state_dict(torch.load('prod_model.pt'))
        self.tokenizer = self._get_tokenizer()
        self.bs_pattern = re.compile(r'.*{\s\s(?P<data>.*)\s\s}.*')
        self.queried = []

    def _get_bs_dict(self, seq):
        match = self.bs_pattern.match(seq)
        if match is not None:
            data = match.group('data')
            data = [elem.split(' = ') for elem in data.split('; ')]
            data = {k : v for k, v in data}
            return data
        return dict()

    # name.movie = frozen; location = Koblenz; name.theater = KINOPOLIS Koblenz; date.showing = today
    def db(self, data):
        query = list(data.keys())
        if 0 not in self.queried and {'name.movie', 'location', 'date.showing', 'time.showing', 'num.tickets'}.issubset(set(query)):
            self.queried.append(0)
            return ' DB  book_tickets { status = success } <EOKB>'
        if 1 not in self.queried and {'name.movie', 'location'}.issubset(set(query)):
            self.queried.append(1)
            return ' DB:  find_theater { name.cinema_1 = KINOPOLIS Koblenz } <EOKB>'
        if 2 not in self.queried and {'name.movie','location', 'name.theater', 'date.showing'}.issubset(set(query)):
            self.queried.append(2)
            return ' DB: find_showtimes { time.showing_1 = 10 pm ; time.showing_2 = 10 am } <EOKB>'
        return ' DB:  <EOKB>'

    def _get_solist_result(self, inp):
        input_ids = self.tokenizer(inp, return_tensors='pt').input_ids
        pred = self.model.generate(
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            input_ids,
            torch.ones(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long),
            torch.zeros(1, 512),
            max_length=512
        )
        beliefe_state = self.tokenizer.decode(pred[0])
        return beliefe_state

    def _get_modified_response(self, response, source, target):
        input_ids = torch.tensor([[CONTEXT_ID] + self.tokenizer(response).input_ids])
        source_ids, source_mask = self.tokenizer(source, return_tensors='pt').values()
        target_ids, target_mask = self.tokenizer(target, return_tensors='pt').values()
        pred = self.model.generate(
            target_ids,
            target_mask,
            input_ids,
            torch.ones(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long),
            torch.tensor([[0.2, 0.4, 0.2, 0.4] + [0.]* 508]),
            source_ids,
            source_mask,
            max_length=512
        )
        response = self.tokenizer.decode(pred[0])
        return response

    def __call__(self):
        history = ''
        target = ''
        source = ''
        while True:
            user = str(input('>>>'))
            if user == 'restart':
                history = ''
                target = ''
                source = ''
                print('\n## Restart agent\n')
                continue
            if user == 'close':
                break
            target += user
            history += 'user: ' + user
            beliefe_state = self._get_solist_result(history)
            bs_data = self._get_bs_dict(beliefe_state)
            kb_data = self.db(bs_data)
            #print('Beliefe state:', beliefe_state)
            intermediate = ''.join([history, beliefe_state, kb_data])
            response = self._get_solist_result(intermediate)
            #print('Response:', response)
            source += ' ' + response
            history += ' assistant: ' + response
            response = self._get_modified_response(response, source, target)
            print(f'>>>{response}')

agent = Agent()
agent()
