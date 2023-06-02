from tqdm import tqdm
import argparse
import logging
import random
import numpy as np

from train.train_event_coref_train_loop import RteProcessor, YoucookProcessor
 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
 
import pandas as pd
def equal_set(set1, set2):
    if set1==set2:
        return True 
    else: 
        return False  
    
def clean(text):
    if text.endswith("."):
        return text[:-1]
    else:
        return text 

def add_into_set(sentence,left,right,one_set):
    word_list=sentence.split()[ left: right+1]
    a_args0= clean(' '.join(word_list))
    if len(word_list)>0:
        
        one_set.add(a_args0.lower())
    return a_args0,one_set
    
    


def gen_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default="youcook",
                        type=str,
                     
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="",
                        type=str,
                     
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    
    parser.add_argument('--start_pair_num',
                        type=int,default=0 )#100000
    parser.add_argument('--end_pair_num',
                        type=int,default=300000000000)#3000000

    args = parser.parse_args()
    return args


def _init(args):
    processors = {
        "rte": RteProcessor,"youcook":YoucookProcessor
    }

    output_modes = {
        "rte": "classification","youcook":"classification"
    }
    
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    return processor,output_mode,args

    
def main():
    args=gen_args()
    processor,output_mode,args=_init(args)
    test_examples = processor.get_ECB_plus_NLI_unlabeled_test(args.data_dir,start_pair_num=args.start_pair_num,end_pair_num=args.end_pair_num)
    output_coreference_list=[]
    for example in tqdm(test_examples):
        a_sentence=example.text_a
        b_sentence=example.text_b
        a_span = ' '.join(a_sentence.split()[int(example.span_a_left): int(example.span_a_right)+1])
        b_span = ' '.join(b_sentence.split()[int(example.span_b_left): int(example.span_b_right)+1])
        a_args_set=set()
        a_args0,a_args_set=add_into_set(a_sentence,example.a_arg0_left,example.a_arg0_right,a_args_set)
        a_args1,a_args_set=add_into_set(a_sentence,example.a_arg1_left,example.a_arg1_right,a_args_set)
        a_args2,a_args_set=add_into_set(a_sentence,example.a_arg2_left,example.a_arg2_right,a_args_set)
        b_args_set=set()
        b_args0,b_args_set=add_into_set(b_sentence,example.b_arg0_left,example.b_arg0_right,b_args_set)
        b_args1,b_args_set=add_into_set(b_sentence,example.b_arg1_left,example.b_arg1_right,b_args_set)
        b_args2,b_args_set=add_into_set(b_sentence,example.b_arg2_left,example.b_arg2_right,b_args_set)

        # a_args_set=set([a_args0.lower(),a_args1.lower(),a_args2.lower()])
        # b_args_set=        set([b_args0.lower(),b_args1.lower(),b_args2.lower()])
        if equal_set(a_args_set,b_args_set) and len(a_args_set)>0:
            output_coreference_list.append([example.pair_id,example.action_frame,a_sentence,a_span,a_args0,a_args1,
                                            a_args2,b_sentence,b_span,b_args0,b_args1,b_args2,"y"])
    df = pd.DataFrame (output_coreference_list, columns = ['paired_sent_ids','action_frame','a_sent', 'a_span','a_args0',
                                                           'a_args1','a_args2','setn2','b_span','b_args0','b_args1','b_args2','is_coreference'])
    df.to_csv(f'coreference_result.csv', index=False)
    df=df.sample(n=100)
    df.to_csv(f'coreference_result_100.csv', index=False)
    print(f"final coreference:{len(df)}")
 
    
if __name__ == "__main__":
    main()