import json
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

def read_squad(path):

    with open(path, 'rb') as f:
        squad_dict = json.load(f)


    contexts = []
    questions = []
    answers = []

    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa[access]:
                 
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    
    return contexts, questions, answers

def add_end_idx(answers, contexts):

    for answer, context in zip(answers, contexts):
        
        gold_text = answer['text']
   
        start_idx = answer['answer_start']
       
        end_idx = start_idx + len(gold_text)

      
        if context[start_idx:end_idx] == gold_text:
           
            answer['answer_end'] = end_idx
        else:
         
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift one token forward
        go_back = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end']-go_back)
            go_back +=1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)



if __name__ == "__main__":
    
    import sys
    import json
    import torch
    from transformers import DistilBertTokenizerFast
    from transformers import DistilBertForQuestionAnswering
    from torch.utils.data import DataLoader
    from transformers import AdamW
    from tqdm import tqdm

    epoches = int(sys.argv[1])
    batch_size = int(sys.argv[2])

    train_contexts, train_questions, train_answers = read_squad('Q1_data/train-v2.0.json')
    test_contexts, test_questions, test_answers = read_squad('Q1_data/dev-v2.0.json')
    val_contexts, val_questions, val_answers = train_contexts[-5000:], train_questions[-5000:], train_answers[-5000:]

    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    # apply function to our data
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)


    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)

    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
  
    model.train()

    optim = AdamW(model.parameters(), lr=5e-5)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epoches):
 
        model.train()
 
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
         
            optim.zero_grad()
           
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
     
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
          
            loss = outputs[0]
         
            loss.backward()
           
            optim.step()
         
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    model_path = 'QA_models/distilbert-custom1'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # switch model out of training mode
    model.eval()

    #val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=16)

    acc = []

    # initialize loop for progress bar
    loop = tqdm(val_loader)
    # loop through batches
    for batch in loop:
        # we don't need to calculate gradients as we're not training
        with torch.no_grad():
            # pull batched items from loader
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            # make predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            # pull preds out
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            # calculate accuracy for both and append to accuracy list
            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    # calculate average accuracy in total
    acc = sum(acc)/len(acc)

    sys.exit(acc)
