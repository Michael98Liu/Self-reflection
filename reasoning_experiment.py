import pandas as pd
import glob
from tqdm.notebook import tqdm
import os
import json
import time


def printShape(df, cols=[], msg=''):
    
    print(df.shape, end='  ')
    for col in cols:
        print(col, df[col].nunique(), end='  ')
    print(msg, flush=True)
    
    return df


#######################
###### important ######

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--temp", type=float)
parser.add_argument("--initial", type=str, default='both')
parser.add_argument("--epoch", type=int, default=-1)

args = parser.parse_args()
print('List of arguments:', args)

TASK = args.task
MODEL = args.model
TEMP = args.temp
INITIAL = args.initial
EPOCH = args.epoch

if EPOCH == -1:
    DIR = '.'
else:
    DIR = f'./round_{EPOCH}'

###### important ######
#######################


print(TASK, MODEL, TEMP, flush=True)


if MODEL == "LLAMA":
    import transformers
    import torch

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name, flush=True)
    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    print('Device:', pipeline.device)

else:

    from openai import OpenAI
    import google.generativeai as genai

    genai.configure(api_key="")
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    os.environ['OPENAI_API_KEY']=''
    client = OpenAI()


def promptGemini(userPrompt):

    try:
        response = model.generate_content(
            userPrompt,
            generation_config = genai.GenerationConfig(
                temperature=TEMP, top_p=1
            )
        )
    
    except Exception as e:
        print("<ERROR>", e, "*** User Prompt ***", userPrompt, "********** sleeping 30 seconds ...", flush=True)
        
        time.sleep(30)
        return ''
        
    print('*************************')
    print(userPrompt)
    print(response)
    print('*************************', flush=True)

    time.sleep(1)

    try:
        return response.text.strip()
    except Exception as e:
        print("<ERROR>", e, "*** User Prompt ***", userPrompt, "**********", flush=True)

        return ''


def promptGPT(userPrompt):

    try:
        mes=[{"role": "system", "content":''},{"role": "user", "content": userPrompt}]
        response = client.chat.completions.create(model='gpt-4o-mini', messages=mes, temperature=TEMP, top_p=1)
    
        print('*************************')
        print(userPrompt)
        print(response)
        print('*************************', flush=True)
    
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("<ERROR>", e, "*** User Prompt ***", userPrompt, "**********", flush=True)
        
        time.sleep(30)
        return ''

def promptLLAMA(userPrompt):
    
    outputs = pipeline(userPrompt, max_new_tokens=1024, temperature=TEMP, top_p=1)

    print('*************************')
    print(userPrompt)
    print(outputs)
    print('*************************', flush=True)

    return outputs[0]['generated_text'][len(userPrompt): ].strip()


promptLLM = promptGemini if MODEL == 'GEMINI' else promptGPT if MODEL == "GPT" else promptLLAMA if MODEL == "LLAMA" else None

if TASK == "MMLU":
    dataset = pd.read_csv('./MMLU_1000.csv')
    
elif TASK == "GSM":
    dataset = pd.read_csv('./math.csv')

elif TASK == "MED":
    dataset = pd.read_csv('./medical.csv')
    
else:
    raise Exception

print(TASK, dataset.shape)


if TASK == "GSM":
    output = "a single number "
    
    initial = f'''You are an expert problem solver. \
To solve the following math problem, you must think step by step, breaking it down into smaller parts and solving each part carefully.
Follow this structured approach:

1. Understand the problem: Restate the key information and what is being asked.
2. Plan a solution: Identify the best approach, considering different strategies.
3. Solve systematically: Carry out each step logically, showing all calculations or deductions.
4. Final answer: Summarize the solution concisely. Then output your final answer using {output} on a new line.'''
    
    initialSimple = f'''Solve the following math problem. Output your final answer using {output} on a new line.'''
    
elif TASK == 'MMLU' or TASK == 'MED':
    output = "'a', 'b', 'c', or 'd' only"
    initial = f'''You are an expert problem solver. \
To answer the following mutiple choice problem, you must think step by step, breaking it down into smaller parts and solving each part carefully.
Follow this structured approach:

1. Understand the problem: Restate the key information and what is being asked.
2. Plan a solution: Identify the best approach, considering different strategies.
3. Solve systematically: Carry out each step logically, showing all calculations or deductions.
4. Final answer: Summarize the solution concisely. Then output your final answer using {output} on a new line.'''
    
    initialSimple = f'''Answer the following mutiple choice problem. Output your final answer using using {output} on a new line.'''
else:
    raise Exception

Q='''
Question: {}
Your previous response to the Question is: {}
'''


reflectCritical=f'''
Critically evaluate your previous response for correctness and identify any potential errors or uncertainties. If needed, revise and improve your answer. \
Output your final answer with {output} on a new line.
'''

isCorrect=f'''
Is your previous response correct?
Answer in a single word using 'yes' or 'no'. Then provide a brief explanation of your reasoning. Then output your final answer with {output} on a new line.
'''

isWrong=f'''
Is your previous response wrong?
Answer in a single word using 'yes' or 'no'. Then provide a brief explanation of your reasoning. Then output your final answer with {output} on a new line.
'''

correctOrWrong=f'''
Is your previous response correct or wrong?
Answer in a single word using 'correct' or 'wrong'. Then provide a brief explanation of your reasoning. Then output your final answer with {output} on a new line.
'''

probability=f'''
Assess your previous response based on correctness and assign probabilities (ranging from 0 to 1) to the following two options:

a: The previous response was correct
b: The previous response was wrong

Ensure that the sum of both probabilities equals exactly 1, distributing the probability accordingly.

Format your answer as follows:

the probability of a: [value]
the probability of b: [value]

Then provide a brief explanation of your reasoning. Then output your final answer with {output} on a new line.
'''

update=f'''
Assess your previous response and determine if it is correct or wrong?
Choose one of the following:
1. Change your answer to a better one
2. Keep the current answer as it is

Then provide a brief explanation of your reasoning. Then output your final answer with {output} on a new line.
'''

isCorrectNoReason=f'''
Is your previous response correct?
Answer in a single word using 'yes' or 'no' only, do not use other words. Then output your final answer with {output} on a new line.
'''

isWrongNoReason=f'''
Is your previous response wrong?
Answer in a single word using 'yes' or 'no' only, do not use other words. Then output your final answer with {output} on a new line.
'''

correctOrWrongNoReason=f'''
Is your previous response correct or wrong?
Answer in a single word using 'correct' or 'wrong' only, do not use other words. Then output your final answer with {output} on a new line.
'''

prompts = {
    'initial': initial,
    'initialSimple': initialSimple,

    'isCorrect': isCorrect,
    'isWrong': isWrong,
    'correctOrWrong': correctOrWrong,
    'reflectCritical': reflectCritical,
    'update': update,
    'probability': probability,

    'isCorrectNoReason': isCorrectNoReason,
    'isWrongNoReason': isWrongNoReason,
    'correctOrWrongNoReason': correctOrWrongNoReason,
}

for k, v in prompts.items():
    print('*******', k, '*******')
    print(v)
    print(flush=True)


def getInitialResponse(promptName):

    assert('initial' in promptName)

    fileName = f'{DIR}/{TASK}_{MODEL}_{TEMP}_{promptName}.csv'

    if len(glob.glob(fileName)) != 0:
        df = pd.read_csv(fileName).fillna({promptName: ''})
    else:
        df = dataset.copy()
        df[promptName] = ''
        
    for ind, row in tqdm(df.iterrows()):
        
        if row[promptName] != '':
            print(ind, 'done', end='  ')
            continue

        # if ind>=10:
        #     break

        print(flush=True)
        
        userPrompt = prompts[promptName] + '\n\n' + row['question']
        if ind==0: print(userPrompt)
        
        response = promptLLM(userPrompt)

        df.loc[ind, promptName] = response
        
        df.to_csv(fileName, index=False)


def getReflection(promptName, initialToBeRef):
    
    assert('initial' not in promptName)
    assert('initial' in initialToBeRef)

    fileName = f'{DIR}/{TASK}_{MODEL}_{TEMP}_{initialToBeRef}_{promptName}.csv'

    if len(glob.glob(fileName)) != 0:
        df = pd.read_csv(fileName).fillna({promptName: ''})
    else:
        df = pd.read_csv(f'{DIR}/{TASK}_{MODEL}_{TEMP}_{initialToBeRef}.csv')
        df[promptName] = ''
        
    for ind, row in tqdm(df.iterrows()):
        
        if row[promptName] != '':
            print(ind, 'done', end='  ')
            continue

        if row[initialToBeRef] == '':
            print(ind, 'not ready', end='  ')
            continue

        # if ind>=10:
        #     break

        print(flush=True)
        
        userPrompt = prompts[promptName] + '\n\n' + Q.format(row['question'], row[initialToBeRef])
        if ind==0: print(userPrompt)
        
        response = promptLLM(userPrompt)

        df.loc[ind, promptName] = response
        
        df.to_csv(fileName, index=False)


if INITIAL == 'both':
    initialPrompts = ['initialSimple', 'initial']
else:
    initialPrompts = [INITIAL]

reflectionPrompts = [
    'isCorrect', 'isWrong', 'correctOrWrong', 'reflectCritical', 'update',
    'probability', 'isCorrectNoReason', 'isWrongNoReason', 'correctOrWrongNoReason'
]

print('*** Initial prompts ***', initialPrompts)
print('* Reflection prompts **', reflectionPrompts)


for iniP in initialPrompts:
    
    getInitialResponse(iniP) # get initial answers using different prompts

    for refP in reflectionPrompts:
        getReflection(refP, iniP) # for reach initial answer, reflect using different prompts

