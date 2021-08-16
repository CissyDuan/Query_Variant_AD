import csv
import json
import os
import copy
import numpy as np
import pickle
import util
import math
import copy


class Graph:
    def convert_to_idx(self, vocab):
        return [vocab.word2id(node) for node in self.nodes]
    def convert_to_idx_nei(self, vocab):
        return [vocab.word2id(node) for node in self.neibour]

    def sparse_to_dense(self):
        node_num = len(self.nodes)
        adj = [[0 for _ in range(node_num)] for _ in range(node_num)]
        for word_id_1 in self.graph:
            for word_id_2 in self.graph[word_id_1]:
                adj[word_id_1][word_id_2] = self.graph[word_id_1][word_id_2]
        return adj


class SubGraph_full(Graph):
    def __init__(self, query_keywords, title):
        #self.query = query
        self.query_keywords = query_keywords
        self.title_keywords = title
        # the core component of subgraph, holds the connection between keywords, stored in the format of dict[dict]
        self.graph = {}
        self.nodes = []
        self.word2node = {}
        self.adj = []
        self.word_type=[]

    def build_subgraph(self, kg):
        word_set_query=set(self.query_keywords)
        word_set_title=set(self.title_keywords)
        word_set = set(self.query_keywords).union(set(self.title_keywords))
        self.nodes = list(word_set)
        self.word2node = {self.nodes[i]: i for i in range(len(self.nodes))}
        for word in self.nodes:
            if word in word_set_query and word in word_set_title:
                self.word_type.append(1)
            elif word in word_set_query:
                self.word_type.append(2)
            elif word in word_set_title:
                self.word_type.append(3)
        for word_1 in self.nodes:
            for word_2 in self.nodes:
                if word_1 == word_2:
                    continue
                if word_1 in kg and word_2 in kg[word_1]:
                    if self.word2node[word_1] not in self.graph:
                        self.graph[self.word2node[word_1]] = {}
                    if self.word2node[word_2] not in self.graph:
                        self.graph[self.word2node[word_2]] = {}
                    self.graph[self.word2node[word_1]][self.word2node[word_2]] = kg[word_1][word_2]
                    self.graph[self.word2node[word_2]][self.word2node[word_1]] = kg[word_1][word_2]

class ExtendGraph_full(Graph):   #找到subgraph的neibour得到扩展后的新图[extend_node_num,extend_node_num]
    def __init__(self, subgraph,subgraph_full):
        self.nodes_min = copy.deepcopy(subgraph.nodes)
        self.nodes=copy.deepcopy(subgraph_full.nodes)
        self.graph = copy.deepcopy(subgraph_full.graph)
        self.word2node =copy.deepcopy(subgraph_full.word2node)
        self.adj = []
        self.word_type_extend=copy.deepcopy(subgraph_full.word_type)


    def extend(self, KG):
        node_set = list(set(self.nodes_min))
        neibour_add=[]

        for node in self.nodes_min:
            if node in KG:
                neighbors = KG[node].keys()
                for neighbor in neighbors:
                    if neighbor not in node_set and neighbor not in neibour_add:
                        node_set.append(neighbor)
                        neibour_add.append(neighbor)
                        self.word2node[neighbor] = len(self.nodes)+len(neibour_add)-1

        for neighbor in neibour_add:
            self.graph[self.word2node[neighbor]] = {}

            for node in self.nodes:
                if node in KG:
                    if neighbor in KG[node].keys():
                        if self.word2node[node] not in self.graph:
                            self.graph[self.word2node[node]] = {}
                        edge=KG[node][neighbor]
                        self.graph[self.word2node[node]][self.word2node[neighbor]] = edge
                        self.graph[self.word2node[neighbor]][self.word2node[node]] = edge

        for neibour in neibour_add:
            self.nodes.append(neibour)
            self.word_type_extend.append(4)



class ExtendGraph(Graph):   #找到subgraph的neibour得到扩展后的新图[extend_node_num,extend_node_num]
    def __init__(self, subgraph):
        self.nodes = copy.deepcopy(subgraph.nodes)
        self.graph = copy.deepcopy(subgraph.graph)
        self.word2node =copy.deepcopy(subgraph.word2node)
        self.adj = []
        self.word_type_extend=copy.deepcopy(subgraph.word_type)
        self.neibour = []

    def extend(self, KG):
        node_set = list(set(self.nodes))
        neibour_add=[]

        for node in self.nodes:
            if node in KG:
                neighbors = KG[node].keys()
                for neighbor in neighbors:
                    if neighbor not in node_set and neighbor not in neibour_add:
                        node_set.append(neighbor)
                        neibour_add.append(neighbor)
                        self.word2node[neighbor] = len(self.nodes)+len(neibour_add)-1

        for neighbor in neibour_add:
            self.graph[self.word2node[neighbor]] = {}

            for node in self.nodes:
                if node in KG:
                    if neighbor in KG[node].keys():
                        if self.word2node[node] not in self.graph:
                            self.graph[self.word2node[node]] = {}

                        edge=KG[node][neighbor]
                        self.graph[self.word2node[node]][self.word2node[neighbor]] = edge
                        self.graph[self.word2node[neighbor]][self.word2node[node]] = edge

        for neibour in neibour_add:
            self.neibour.append(neibour)
            self.nodes.append(neibour)
            self.word_type_extend.append(4)



class SubGraph(Graph):
    def __init__(self, query_keywords, title_keywords):
        #self.query = query
        self.query_keywords = query_keywords
        self.title_keywords = title_keywords
        # the core component of subgraph, holds the connection between keywords, stored in the format of dict[dict]
        self.graph = {}
        self.nodes = []
        self.word2node = {}
        self.adj = []
        self.word_type=[]

    def build_subgraph(self, kg):
        word_set_query=set(self.query_keywords)
        word_set_title=set(self.title_keywords)
        word_set = set(self.query_keywords).union(set(self.title_keywords))
        self.nodes = list(word_set)
        self.word2node = {self.nodes[i]: i for i in range(len(self.nodes))}
        for word in self.nodes:
            if word in word_set_query and word in word_set_title:
                self.word_type.append(1)
            elif word in word_set_query:
                self.word_type.append(2)
            elif word in word_set_title:
                self.word_type.append(3)
        for word_1 in self.nodes:
            for word_2 in self.nodes:
                if word_1 == word_2:
                    continue
                if word_1 in kg and word_2 in kg[word_1]:
                    if self.word2node[word_1] not in self.graph:
                        self.graph[self.word2node[word_1]] = {}
                    if self.word2node[word_2] not in self.graph:
                        self.graph[self.word2node[word_2]] = {}
                    self.graph[self.word2node[word_1]][self.word2node[word_2]] = kg[word_1][word_2]
                    self.graph[self.word2node[word_2]][self.word2node[word_1]] = kg[word_1][word_2]
                    # if self.word2node[word_2] in kg[self.word2node[word_1]]:
                    # if self.word2node[word_2] in kg[self.word2node[word_1]]:

class SubGraph_k(Graph):
    def __init__(self, query_keywords, title_keywords):
        self.query_keywords = query_keywords
        self.title_keywords = title_keywords
        # the core component of subgraph, holds the connection between keywords, stored in the format of dict[dict]
        self.graph = {}
        self.nodes = []
        self.word2node = {}
        self.adj = []
        self.word_type=[]


    def build_subgraph(self, kg):
        word_set_query = set(self.query_keywords)
        word_set_title = set(self.title_keywords)
        word_set=set(self.title_keywords)
        self.nodes = list(word_set)
        self.word2node = {self.nodes[i]: i for i in range(len(self.nodes))}
        for word in self.nodes:
            if word in word_set_query and word in word_set_title:
                self.word_type.append(1)
            elif word in word_set_query:
                self.word_type.append(2)
            elif word in word_set_title:
                self.word_type.append(3)
        for word_1 in self.nodes:
            for word_2 in self.nodes:
                if word_1 == word_2:
                    continue
                if word_1 in kg and word_2 in kg[word_1]:
                    if self.word2node[word_1] not in self.graph:
                        self.graph[self.word2node[word_1]] = {}
                    if self.word2node[word_2] not in self.graph:
                        self.graph[self.word2node[word_2]] = {}
                    self.graph[self.word2node[word_1]][self.word2node[word_2]] = kg[word_1][word_2]
                    self.graph[self.word2node[word_2]][self.word2node[word_1]] = kg[word_1][word_2]


class ExtendGraph_k(Graph):   #找到subgraph的neibour得到扩展后的新图[extend_node_num,extend_node_num]
    def __init__(self, subgraph_k):
        self.nodes = copy.deepcopy(subgraph_k.nodes)
        self.graph = copy.deepcopy(subgraph_k.graph)
        self.word2node =copy.deepcopy(subgraph_k.word2node)
        self.adj = []
        self.word_type_extend=copy.deepcopy(subgraph_k.word_type)
        self.neibour = []

    def extend(self, KG):
        node_set = list(set(self.nodes))
        neibour_add=[]

        for node in self.nodes:
            if node in KG:
                neighbors = KG[node].keys()
                for neighbor in neighbors:
                    if neighbor not in node_set and neighbor not in neibour_add:
                        node_set.append(neighbor)
                        neibour_add.append(neighbor)
                        self.word2node[neighbor] = len(self.nodes)+len(neibour_add)-1

        for neighbor in neibour_add:
            self.graph[self.word2node[neighbor]] = {}

            for node in self.nodes:
                if node in KG:
                    if neighbor in KG[node].keys():
                        if self.word2node[node] not in self.graph:
                            self.graph[self.word2node[node]] = {}
                        edge=KG[node][neighbor]
                        self.graph[self.word2node[node]][self.word2node[neighbor]] = edge
                        self.graph[self.word2node[neighbor]][self.word2node[node]] = edge

        for neibour in neibour_add:
            self.neibour.append(neibour)
            self.nodes.append(neibour)
            self.word_type_extend.append(4)




class KnowledgeGraph:
    def __init__(self, config):
        self.map={}
        self.total_word_appearance = 0
        self.word_appearance = {}
        self.pmi = {}
        # the core component of graph, holds the connection between words, stored in the format of dict[dict]
        self.KG = {}
        self.config = config

    def build_kg(self, data,vocab_file):
        vocab_words = json.load(open(vocab_file))
        for item in data:
            q_keywords, t_keywords, ad_keywords = item
            #print(q_keywords)
            #print(ad_keywords)
            words=list(set(q_keywords+ad_keywords))
            #words = list(set(ad_keywords))
            #print(words)
            removelist=[]
            for word in words:
                if word not in vocab_words and word not in removelist:
                    removelist.append(word)
            for i in removelist:

                words.remove(i)
                #print(word)

            self.map_keywords(words)

    def get_kg(self):
        kg_node=0
        kg_edge=0
        for word1 in self.pmi:
            for word2 in self.pmi[word1]:
                if self.pmi[word1][word2]>0:
                    if word1 not in self.KG:
                        self.KG[word1]={}
                        kg_node+=1
                    self.KG[word1][word2] = self.pmi[word1][word2]
                    kg_edge+=1
            if kg_node%1000==0 and kg_edge!=0:
                print(('进程{}，平均边{}').format(kg_node,kg_edge/kg_node))
        print(kg_node)
        print(kg_edge)

        #print(self.KG)



    def cal_pmi(self):

        def pmi(pmi_map, frequency_map):
            kg_edge = 0
            word_num=0
            for word_1 in frequency_map:
                pmi_map[word_1] = {}
                word_num+=1
                for word_2 in frequency_map[word_1]:
                    pmi_1_2= math.log((frequency_map[word_1][word_2]*self.total_word_appearance)/(self.word_appearance[word_1]* self.word_appearance[word_2]))
                    if pmi_1_2>0:
                        pmi_map[word_1][word_2]=pmi_1_2
                        kg_edge+=1
                if word_num%1000==0:
                    print('word_num:{},edge:{}'.format(word_num,kg_edge))
            print(kg_edge)

        #pmi(self.pmi, self.map)
        pmi(self.KG, self.map)

        word_sum=list(self.KG.keys())
        print(len(word_sum))
        #print(self.pmi)
        print('pmi get')


    def map_keywords(self,words):
        # Using self.q_t, q_q, t_t map
        def mapping(words,word_map):
            for word_1 in words:
                if word_1 not in word_map:
                    word_map[word_1] = {}
                for word_2 in words:
                    if word_2!=word_1:
                        if word_2 not in word_map[word_1]:
                            word_map[word_1][word_2] = 0
                        word_map[word_1][word_2] += 1

        self.total_word_appearance += len(words)
        for word in words:
            if word not in self.word_appearance:
                self.word_appearance[word] = 0
            self.word_appearance[word] += 1
        mapping(words, self.map)



if __name__ == '__main__':
    pass
