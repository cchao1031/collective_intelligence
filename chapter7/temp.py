# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:32:27 2016

@author: apple
"""

class decision_node:
    def __init__(self, col=None, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

import pandas as pd
import numpy as np

def load_tree_example():
    data = pd.read_csv('D:\\workfile\\Programming Collective Intelligence\\chapter7\\tree_example.txt',
                       names=('referrer', 'country', 'read_FAQ', 'webs_read', 'serve'))
    return data

def divideset(rows, column, value):
    judge = None
    if isinstance(value, np.int64) or isinstance(value, np.float64):
        judge = rows.ix[:, column] >= value
    else:
        judge = rows.ix[:, column] == value

    setT = rows[judge]
    setF = rows[-judge]
    return setT, setF

def Gini(rows):
    n = len(rows)
    frequency = rows.ix[:,-1].value_counts()/float(n)
    gini = 1 - sum(frequency * frequency)
    return gini, n

def Impurity(giniT, nT, giniF, nF):
    impurity = (nT * giniT + nF * giniF)/(nT + nF)
    return impurity

def buildTree(rows, scoref=Gini):
    (current_score, _) = scoref(rows)

    #记录最佳拆分变量
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows.columns) - 1
    #每一个Feature进行循环
    for col in range(column_count):
        column_values = rows.ix[:, col].unique()
        if len(column_values) == 1:
            continue
        else:

            #根据这个Feature的每个值，尝试拆分
            for value in column_values:
                (setT, setF) = divideset(rows, col, value)

                #信息增益
                (giniT, nT) = scoref(setT)
                (giniF, nF) = scoref(setF)
                gain = current_score - Impurity(giniT, nT, giniF, nF)
                if gain > best_gain:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (setT, setF)
    if best_gain > 0:
        trueBranch = buildTree(best_sets[0])
        falseBranch = buildTree(best_sets[1])
        return decision_node(col=rows.columns[best_criteria[0]], value=best_criteria[1],
                             tb=trueBranch, fb=falseBranch)
    else:
        return decision_node(results=rows.ix[:,-1].value_counts())

def printtree(tree, indent=''):
    #判断是否是节点
    if type(tree.results) == pd.Series:#不是节点
        print str(dict(tree.results))
    else:#是节点
        #打印条件
        print str(tree.col) + ':' + str(tree.value) + '?'

        #打印分支
        print indent + 'T->',
        printtree(tree.tb, indent + '\t')
        print indent + 'F->',
        printtree(tree.fb, indent + '\t')
