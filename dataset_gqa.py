"""
Author: Aravind Kumar Arunachalam
Implementation for GQA dataset
"""

from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
import itertools
import math

def tfidf_from_questions_gqa(names, dictionary, dataroot='data', target=['gqa']):
    inds = [[], []]
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0])
                inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1])
                inds[1].append(c[0])

    # GQA
    if 'gqa' in target:
        for name in names:
            assert name in ['train_all', 'train_balanced', 'val_all', 'val_balanced', 'challenge_all', 'challenge_balanced', 'testdev_all', 'testdev_balanced', 'test_all', 'test_balanced']
            
            if name == 'train_all':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'train_all_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'train_balanced':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'train_balanced_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'val_all':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'val_all_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'val_balanced':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'val_balanced_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'challenge_all':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'challenge_all_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'challenge_balanced':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'challenge_balanced_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'testdev_all':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'testdev_all_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'testdev_balanced':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'testdev_balanced_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            elif name == 'test_all':
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'test_all_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))
            else:
                questions_path = os.path.join(dataroot, 'gqa', 'questions', 'test_balanced_questions.pkl')
                questions = pickle.load(open(questions_path, 'rb'))

            print (name)
            count = 0    
            for question in questions:
                count = count + 1
                populate(inds, df, question['question'])
            
            print (count)    
            
    # TF-IDF
    vals = np.ones((len(inds[1])))
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds),
                                     torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = dataroot+'/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0),tfidf.size(1)))

    return tfidf, weights

class ObjectSpatialFeature:
    def load_all_spatial_features(self, dataroot):   
        for file_num in list(range(16)):
            spatial_file_name = 'gqa_spatial_' + str(file_num) + '.h5'
            spatial_file_path = os.path.join(dataroot, 'gqa', 'spatial', str(spatial_file_name))
            spatial_file = h5py.File(spatial_file_path, 'r')
            self.normalized_bb.append(spatial_file['features'])
 
        print ("Loaded all spatial features.")
    
    def load_all_object_features(self, dataroot):
        for file_num in list(range(16)):
            objects_file_name = 'gqa_objects_' + str(file_num) + '.h5'
            objects_file_path = os.path.join(dataroot, 'gqa', 'objects', str(objects_file_name))
            objects_file = h5py.File(objects_file_path, 'r')
            self.features.append(objects_file['features'])
            self.bb.append(objects_file['bboxes'])
    
        print ("Loaded all object features.")
    
    def tensorize_all_spatial_features(self):
        for file_num in list(range(16)):
            self.normalized_bb[file_num] = torch.from_numpy(np.array(self.normalized_bb[file_num]))
            
        print ("Tensorized all spatial features.")
    
    def tensorize_all_object_features(self):
        for file_num in list(range(16)):
            self.features[file_num] = torch.from_numpy(np.array(self.features[file_num]))
            self.bb[file_num] = torch.from_numpy(np.array(self.bb[file_num]))
            
        print ("Tensorized all object features.")
    
    def __init__(self, dataroot):
        self.normalized_bb = []
        self.features = []
        self.bb = []
        self.adaptive = False
    
        print ("Starting to load all features to memory.")
        self.load_all_spatial_features(dataroot)
        self.load_all_object_features(dataroot)
        print ("Loaded all features to memory.")

    def get_features(self, object_file_number, object_file_index, spatial_file_number, spatial_file_index):
        feature = torch.from_numpy(self.features[object_file_number][object_file_index])
        normalized_bb = self.normalized_bb[spatial_file_number][spatial_file_index]
        
        numpy_array = np.zeros([2048,7], dtype=np.float32)
        for i in list(range(2048)):
            for j in list(range(7)):
                numpy_array[i][j] = (normalized_bb[i][j][0] + normalized_bb[i][j][1] + normalized_bb[i][j][2] + 
                                     normalized_bb[i][j][3] + normalized_bb[i][j][4] + normalized_bb[i][j][5] +
                                     normalized_bb[i][j][6])/7
        
        normalized_bb = torch.from_numpy(numpy_array)
        bb = torch.from_numpy(self.bb[object_file_number][object_file_index])
        
        return feature, normalized_bb, bb
    
    def get_dimension(self):
        feature_t = torch.from_numpy(self.features[0][0])
        v_dim = feature_t.size(1)
        
        normalized_bb_t = torch.from_numpy(self.normalized_bb[0][0])
        s_dim = normalized_bb_t.size(1)
        
        return v_dim, s_dim
        
class GQAFeatureDataset(Dataset):
    def load_question_answers(self, dataroot, name):
        if name == 'train_all':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'train_all_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'train_balanced':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'train_balanced_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'val_all':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'val_all_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'val_balanced':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'val_balanced_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'challenge_all':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'challenge_all_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'challenge_balanced':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'challenge_balanced_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'testdev_all':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'testdev_all_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'testdev_balanced':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'testdev_balanced_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        elif name == 'test_all':
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'test_all_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        else:
            questions_path = os.path.join(dataroot, 'gqa', 'questions', 'test_balanced_questions.pkl')
            self.entries = pickle.load(open(questions_path, 'rb'))
        
        answers_path = os.path.join(dataroot, 'gqa', 'questions', 'all_answers.pkl')
        self.dataset_answers = pickle.load(open(answers_path, 'rb'))
        
        answers_to_labels_path = os.path.join(dataroot, 'gqa', 'questions', 'answers_to_label.pkl')
        self.datatset_answers_to_label = pickle.load(open(answers_to_labels_path, 'rb'))
        
        print ("Loaded questions and answers for: " + name + ".")
    
    def load_objects_info(self, dataroot):
        objects_info_path = os.path.join(dataroot, 'gqa', 'objects', 'gqa_objects_info.pkl')
        self.objects_info = pickle.load(open(objects_info_path, 'rb'))
        
        print ("Loaded objects information.")
        
    def load_spatial_info(self, dataroot):
        spatial_info_path = os.path.join(dataroot, 'gqa', 'objects', 'gqa_objects_info.pkl')
        self.spatial_info = pickle.load(open(spatial_info_path, 'rb'))
        
        print ("Loaded spatial information.")
    
    def tokenize(self, max_length=14):
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad to the back of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            
        print ("Tokenization done.")
    
    def tensorize_questions_answers(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if answer is not None:
                answer_array = []
                answer_array.append(self.datatset_answers_to_label[answer])
                labels = np.array(answer_array)
                
                scores_array = []
                scores_array.append(1.)
                scores = np.array(scores_array, dtype=np.float32)
                
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer'] = {}
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None
                
        print ("Tensorized question and answers.")

    def __getitem__(self, index):
        entry = self.entries[index]
        
        
        raw_question = entry["question"]
        image_id = entry["image_id"]
        question = entry['q_token']
        question_id = entry['question_key']
        
        object_file_location = self.objects_info[image_id]
        spatial_file_location = self.spatial_info[image_id]
            
        spatial_adj_matrix = torch.zeros(1,1).double()
        semantic_adj_matrix = torch.zeros(1,1).double()
        
        features, normalized_bb, bb = self.object_spatial_feature.get_features(object_file_location['file_number'], 
                                                                               object_file_location['index_on_file'], 
                                                                               spatial_file_location['file_number'], 
                                                                               spatial_file_location['index_on_file'])
        answer = entry['answer']
        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, normalized_bb, question, target, question_id, image_id, bb, spatial_adj_matrix, semantic_adj_matrix

        else:
            return features, normalized_bb, question, question_id, question_id, image_id, bb, spatial_adj_matrix, semantic_adj_matrix
        
    def __init__(self, name, dictionary, osf_object, relation_type, dataroot='data',
                adaptive=False, pos_emb_dim=64, nongt_dim=100):
        super(GQAFeatureDataset, self).__init__()
        assert name in ['train_all', 'train_balanced', 'val_all', 'val_balanced', 
                        'challenge_all', 'challenge_balanced', 'testdev_all', 
                        'testdev_balanced', 'test_all', 'test_balanced']
        
        print ("Initializing...")
        
        self.object_spatial_feature = osf_object
        self.load_question_answers(dataroot, name)
        self.load_objects_info(dataroot)
        self.load_spatial_info(dataroot)
        self.data_root = dataroot
        self.num_ans_candidates = len(self.dataset_answers)

        self.dictionary = dictionary
        self.relation_type = relation_type
        self.adaptive = adaptive
        
        self.semantic_adj_matrix = None
        self.pos_boxes = None
        
        self.tokenize()
        self.tensorize_questions_answers()
        
        self.nongt_dim = nongt_dim
        self.emb_dim = pos_emb_dim
        self.v_dim, self.s_dim = self.object_spatial_feature.get_dimension()
        
        print ("Intialized.")
        
    def __len__(self):
        return len(self.entries)