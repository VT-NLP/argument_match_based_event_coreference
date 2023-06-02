import csv
import logging
import os
import random
import sys
import codecs
import numpy as np

from tqdm import tqdm, trange

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, span_a_left=None, span_a_right=None, a_arg0_left=None, a_arg0_right=None,
                 a_arg1_left=None, a_arg1_right=None, a_loc_left=None, a_loc_right=None, a_time_left=None,
                 a_time_right=None,a_arg2_left=None,a_arg2_right=None, text_b=None, span_b_left=None, span_b_right=None, b_arg0_left=None,
                 b_arg0_right=None,
                 b_arg1_left=None, b_arg1_right=None, b_loc_left=None, b_loc_right=None, b_time_left=None,
                 b_time_right=None,b_arg2_left=None,b_arg2_right=None, label=None, pair_id=None,action_frame=None,task_step_id_a=None,task_step_id_b=None):

        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.action_frame=action_frame
        self.text_a = text_a
        self.span_a_left = span_a_left
        self.span_a_right = span_a_right

        self.a_arg0_left = a_arg0_left
        self.a_arg0_right = a_arg0_right
        self.a_arg1_left = a_arg1_left
        self.a_arg1_right = a_arg1_right
        self.a_loc_left = a_loc_left
        self.a_loc_right = a_loc_right
        self.a_time_left = a_time_left
        self.a_time_right = a_time_right
        self.a_arg2_left=a_arg2_left
        self.a_arg2_right=a_arg2_right
        
        
        self.text_b = text_b
        self.span_b_left = span_b_left
        self.span_b_right = span_b_right
        self.label = label
        self.pair_id = pair_id

        self.b_arg0_left = b_arg0_left
        self.b_arg0_right = b_arg0_right
        self.b_arg1_left = b_arg1_left
        self.b_arg1_right = b_arg1_right
        self.b_loc_left = b_loc_left
        self.b_loc_right = b_loc_right
        self.b_time_left = b_time_left
        self.b_time_right = b_time_right
        self.b_arg2_left=b_arg2_left
        self.b_arg2_right=b_arg2_right        
        self.task_step_id_a=task_step_id_a
        self.task_step_id_b=task_step_id_b
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def compute_start_end_position( text,characterindex_left,characterindex_right):
    if characterindex_left==0:
        wordindex_left=0
    else:
        previous_text=text[0:characterindex_left]
        previous_word_num=len(previous_text.split())
        wordindex_left=previous_word_num
    
    surface_form_word_num=len(text[characterindex_left:characterindex_right].split())
    wordindex_right=surface_form_word_num+wordindex_left-1
    return wordindex_left,wordindex_right

import json
def save_example(pair_example_list,start_pair_num,end_pair_num):
    with  open(f'data/youcook/pairs_trigger_only_verb_{start_pair_num}_{end_pair_num}.txt' , 'w') as outfile :
        
        for pair_example in tqdm(pair_example_list):
            outstr=[]
            outstr.append(str(pair_example.guid))
            outstr.append(str(pair_example.text_a))
            outstr.append(str(pair_example.span_a_left))
            outstr.append(str(pair_example.span_a_right))
            outstr.append(str(pair_example.a_arg0_left))
            outstr.append(str(pair_example.a_arg0_right))
            outstr.append(str(pair_example.a_arg1_left))
            outstr.append(str(pair_example.a_arg1_right))
            outstr.append(str(pair_example.a_loc_left))
            outstr.append(str(pair_example.a_loc_right))
            outstr.append(str(pair_example.a_time_left))
            outstr.append(str(pair_example.a_time_right))
            outstr.append(str(pair_example.text_b))
            outstr.append(str(pair_example.span_b_left))
            outstr.append(str(pair_example.span_b_right))
            outstr.append(str(pair_example.b_arg0_left))
            outstr.append(str(pair_example.b_arg0_right))
            outstr.append(str(pair_example.b_arg1_left))
            outstr.append(str(pair_example.b_arg1_right))
            outstr.append(str(pair_example.b_loc_left))
            outstr.append(str(pair_example.b_loc_right))
            outstr.append(str(pair_example.b_time_left))
            outstr.append(str(pair_example.b_time_right))
            outstr.append(str(pair_example.label))
            outstr.append(str(pair_example.pair_id))
            outstr.append(str(pair_example.action_frame))
            outstr.append('\n')
            outfile.write('\t'.join(outstr))
        
        
def merge_example(example,example2,merge_id):
    
    merged_example=InputExample(merge_id,example.text_a,example.span_a_left, example.span_a_right, example.a_arg0_left, example.a_arg0_right, example.a_arg1_left,
                                        example.a_arg1_right, example.a_loc_left, example.a_loc_right, example.a_time_left, example.a_time_right,
                                        example.a_arg2_left,example.a_arg2_right,
                                        example2.text_a,example2.span_a_left, example2.span_a_right, example2.a_arg0_left, example2.a_arg0_right, example2.a_arg1_left,
                                        example2.a_arg1_right, example2.a_loc_left, example2.a_loc_right, example2.a_time_left, 
                                        example2.a_time_right,example2.a_arg2_left,example2.a_arg2_right,1,example.guid+ '&&' +example2.guid,example.action_frame,
                                        example.task_step_id_a,example2.task_step_id_a)
    merge_id+=1
    return merge_id,merged_example
    
class RteProcessor(DataProcessor):

    def get_ECB_plus_NLI(self, filename, is_train=True):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        pos_size = 0
        for row in readfile:

            line=row.strip().split('\t')
            if len(line) == 23:
                guid = "train-" + str(line_co - 1)
                text_a = line[0].strip()
                span_a_left = int(line[1].strip())
                span_a_right = int(line[2].strip())
                a_arg0_left = int(line[3].strip())
                a_arg0_right = int(line[4].strip())
                a_arg1_left = int(line[5].strip())
                a_arg1_right = int(line[6].strip())
                a_loc_left = int(line[7].strip())
                a_loc_right = int(line[8].strip())
                a_time_left = int(line[9].strip())
                a_time_right = int(line[10].strip())

                text_b = line[11].strip()
                span_b_left = int(line[12].strip())
                span_b_right = int(line[13].strip())
                b_arg0_left = int(line[14].strip())
                b_arg0_right = int(line[15].strip())
                b_arg1_left = int(line[16].strip())
                b_arg1_right = int(line[17].strip())
                b_loc_left = int(line[18].strip())
                b_loc_right = int(line[19].strip())
                b_time_left = int(line[20].strip())
                b_time_right = int(line[21].strip())

                label = int(line[22].strip())
                if label == 1:
                    pos_size += 1

                examples.append(
                    InputExample(guid, text_a, span_a_left, span_a_right, a_arg0_left, a_arg0_right, a_arg1_left,
                                 a_arg1_right, a_loc_left, a_loc_right, a_time_left, a_time_right,
                                 text_b, span_b_left, span_b_right, b_arg0_left, b_arg0_right, b_arg1_left,
                                 b_arg1_right, b_loc_left, b_loc_right, b_time_left, b_time_right, label, None))
                if is_train:
                    examples.append(
                        InputExample(guid=guid, text_a=text_b, span_a_left=span_b_left, span_a_right=span_b_right, text_b=text_a, span_b_left=span_a_left, span_b_right=span_a_right, label=label, pair_id=None))
            line_co+=1
            # if line_co > 20000:
            #     break
        readfile.close()
        print('data line: ', line_co)
        print('loaded  size:', len(examples), ' pos_size:', pos_size)
        return examples

    def get_ECB_plus_NLI_unlabeled_test(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        pos_size = 0
        for row in readfile:

            line=row.strip().split('\t')
            if len(line) == 25:
                guid = "test-" + str(line_co - 1)
                event_id_1 = line[0].strip()
                event_id_2 = line[1].strip()

                text_a = line[2].strip()
                span_a_left = int(line[3].strip())
                span_a_right = int(line[4].strip())
                a_arg0_left = int(line[5].strip())
                a_arg0_right = int(line[6].strip())
                a_arg1_left = int(line[7].strip())
                a_arg1_right = int(line[8].strip())
                a_loc_left = int(line[9].strip())
                a_loc_right = int(line[10].strip())
                a_time_left = int(line[11].strip())
                a_time_right = int(line[12].strip())

                text_b = line[13].strip()
                span_b_left = int(line[14].strip())
                span_b_right = int(line[15].strip())
                b_arg0_left = int(line[16].strip())
                b_arg0_right = int(line[17].strip())
                b_arg1_left = int(line[18].strip())
                b_arg1_right = int(line[19].strip())
                b_loc_left = int(line[20].strip())
                b_loc_right = int(line[21].strip())
                b_time_left = int(line[22].strip())
                b_time_right = int(line[23].strip())

                label = int(line[24].strip())
                if label == 1:
                    pos_size += 1

                examples.append(
                    InputExample(guid, text_a, span_a_left, span_a_right, a_arg0_left, a_arg0_right, a_arg1_left,
                                 a_arg1_right, a_loc_left, a_loc_right, a_time_left, a_time_right,
                                 text_b, span_b_left, span_b_right, b_arg0_left, b_arg0_right, b_arg1_left,
                                 b_arg1_right, b_loc_left, b_loc_right, b_time_left, b_time_right, label,
                                 pair_id=event_id_1 + '&&' + event_id_2))
            line_co+=1
            # if line_co > 20000:
            #     break
        readfile.close()
        print('data line: ', line_co)
        print('loaded  size:', len(examples), ' pos_size:', pos_size)
        return examples

def is_location(entity_frame):
    if entity_frame in ["Area",
            "Place",
            "Goal",
			"Goal_area", "Container", "Source", "Location","location",
			"Components","Enclosed_region",
			"Circumstances"]:
        return True
    else:
        return False
    
def is_time(entity_frame):
    if entity_frame == "Time":
        return True
    else:
        return False

def extract_left_right(text_a,one_entity_json):
    wordindex_left,wordindex_right,is_find=-1,-1,False
    entity_list=one_entity_json["Entities"]
    entity_surface_form=None
    for entity in entity_list:
        entity_surface_form=entity["Text"]
        wordindex_left,wordindex_right=entity["Source"]["Text_offset"]
        # wordindex_left,wordindex_right=compute_start_end_position( text_a,characterindex_left,characterindex_right)
        if wordindex_right>0:
            wordindex_right-=1
        is_find=True
    return wordindex_left,wordindex_right,is_find,entity_surface_form

class YoucookProcessor(RteProcessor):

    def get_ECB_plus_NLI_unlabeled_test(self, data_folder, is_train=False,start_pair_num=None,end_pair_num=None):
        example_list=[]
        guid=0
        for file_name in  os.listdir(data_folder):
            file_path=os.path.join(data_folder,file_name)
            # Opening JSON file
           
           
            with open(file_path,) as f:
                # returns JSON object as 
                # a dictionary
                data = json.load(f)
                task_id=data["Task_ID"]
                step_json_list=data["Steps"]
                # step_id_set=set()
                for one_step_json in step_json_list:
                    step_id=one_step_json["Step_ID"]
                    
                    task_step_id= task_id+"_"+str(step_id)
                    # if step_id in step_id_set:
                    #     continue 
                    # step_id_set.add(step_id)
                    text_a=one_step_json["Step_Description"]
                    action_surface_form=one_step_json["Action"]["Text"]
                    action_frame=one_step_json["Action"]["Action_Frame"]
                    characterindex_left,characterindex_right=one_step_json["Action"]["Source"]["Text_offset"]
                    # wordindex_left,wordindex_right=compute_start_end_position( text_a,characterindex_left,characterindex_right)
                    wordindex_left,wordindex_right=characterindex_left,characterindex_right
                    if wordindex_right>0:
                        wordindex_right-=1
                    span_a_left=wordindex_left
                    span_a_right=wordindex_right
                    
                    a_loc_left=-1
                    a_loc_right=-1
                    a_time_left=-1
                    a_time_right=-1
                    label=-1
                    entity_json=one_step_json["Entity"]
                    entity_num=0
                    a_arg0_left=-1
                    a_arg0_right=-1
                    a_arg1_left=-1
                    a_arg1_right=-1
                    a_arg2_left=-1
                    a_arg2_right=-1
                    # a_arg0_text=""
                    # a_arg1_text=""
                    # a_arg2_text=""
                    for entity_frame,one_entity_json in entity_json.items():
                        if is_time(entity_frame):
                            
                            a_temp_left,a_temp_right,is_find,entity_surface_form=extract_left_right(text_a,one_entity_json)
                            if   is_find:
                                a_time_left,a_time_right=a_temp_left,a_temp_right
                        elif is_location(entity_frame):
                            a_temp_left,a_temp_right,is_find,entity_surface_form=extract_left_right(text_a,one_entity_json)
                            if   is_find:
                                
                                a_loc_left,a_loc_right=a_temp_left,a_temp_right
                        else:
                            entity_list=one_entity_json["Entities"]
                            for entity in entity_list:
                                entity_surface_form=entity["Text"]
                                characterindex_left,characterindex_right=entity["Source"]["Text_offset"]
                                # wordindex_left,wordindex_right=compute_start_end_position( text_a,characterindex_left,characterindex_right)
                                wordindex_left,wordindex_right=characterindex_left,characterindex_right 
                                if wordindex_right>0:
                                    wordindex_right-=1
                                if entity_num==0:
                                
                                    a_arg0_left=wordindex_left
                                    a_arg0_right=wordindex_right
                                elif entity_num==1:
                                    a_arg1_left=wordindex_left
                                    a_arg1_right=wordindex_right
                                elif entity_num==2:
                                    a_arg2_left=wordindex_left
                                    a_arg2_right=wordindex_right
                                 
                                entity_num+=1
                    example_list.append(InputExample(file_name.split(".json")[0]+"_"+str(guid) , text_a, span_a_left, span_a_right, a_arg0_left, a_arg0_right, a_arg1_left,
                                        a_arg1_right, a_loc_left, a_loc_right, a_time_left, a_time_right,a_arg2_left,a_arg2_right,
                                        None, None, None, None, None, None,
                                        None, None, None, None, None,None, None,  label, None,action_frame,task_step_id,None))
                    guid+=1
        
        print(f"len_example_list: {len(example_list)}")
        pair_example_list=gen_pair(example_list,start_pair_num,end_pair_num)
        # print(f"len_pair_example_list: {len(pair_example_list)}")
        # save_example(pair_example_list,start_pair_num,end_pair_num)
        # print(len(pair_example_list))
        # if len(pair_example_list)>1000000:
        #     raise Exception("Huge data")
        return pair_example_list

def same_task_same_step(example,example2):
    if example.task_step_id_a==example2.task_step_id_a:
        return True
    else:
        return False

def gen_pair(example_list,start_pair_num,end_pair_num):
    pair_num=0
    merge_id=0
    pair_example_list=[]
    for i,example in tqdm(enumerate(example_list)):
        for j,example2 in enumerate(example_list):
            if i<j and isSameTopic(example,example2) and not same_task_same_step(example,example2):
                if start_pair_num>pair_num:
                    pair_num+=1
                    continue
                if pair_num>=end_pair_num:
                    print(f"early end. pair_num:{len(pair_example_list)}")
                    return   pair_example_list
                else:
                    pair_num+=1
                    merge_id,new_example=merge_example(example,example2,merge_id)
                    pair_example_list.append(new_example)
    print(f"pair_num:{len(pair_example_list)}")
    return pair_example_list
    
def isSameTopic(example,example2):
    if example.action_frame == example2.action_frame:
        return True  
    else:
        return False