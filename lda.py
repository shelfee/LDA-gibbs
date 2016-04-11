#! /usr/bin/env python
# -*- coding: utf-8 -*-

'a LDA-gibbs module'

_author_='Zhang Xiaofei'
from mpi4py import MPI
import sys
comm=MPI.COMM_WORLD
import communicate as com
import numpy as np
import copy
#para:alpha_value,beta_value,K(topic_num)
K=100
alpha_value=50.0/K
beta_value=0.01
w_dic={}
doc_num=0
step=1000
sample_step=200
burnin=10
#processor_0 will divide documents evenly and assign tasks(w) to all processors
rank=comm.Get_rank()
size=comm.Get_size()
if rank==0:
	doc_list=[]
	len_doc_dic={}
	doc_no=0
#	s="mozilla_dataset.txt"
	s='sample.txt'
	for lines in open(s):
		x=eval(lines)
		doc_list.append(x[1])
		doc_len=len(x[1])
		if len_doc_dic.has_key(doc_len):
			len_doc_dic[doc_len].append(doc_no)
		else:
			len_doc_dic[doc_len]=[]
			len_doc_dic[doc_len].append(doc_no)
		doc_no+=1
	cpuload={}
	cputask={}
	for i in range(0,size):
		cpuload[i]=0
		cputask[i]={}
	import heapq
	while len(len_doc_dic)>0:
		LargestLen=heapq.nlargest(1,len_doc_dic.keys())[0]
		choose_doc=len_doc_dic[LargestLen].pop()
		if len(len_doc_dic[LargestLen])==0:
			del len_doc_dic[LargestLen]
		SmallestCPU=heapq.nsmallest(1,cpuload.values())[0]
		no=0
		while cpuload[no]!=SmallestCPU:
			no+=1
		cpuload[no]+=LargestLen
		cputask[no][choose_doc]=doc_list[choose_doc]
	for i in range(1,size):
		comm.send([cputask[i],doc_no],dest=i)
	w_dic=cputask[0]
	doc_num=doc_no
else:
	recv_data=comm.recv(source=0)
	w_dic=recv_data[0]
	doc_num=recv_data[1]

#initiate N_mk(num of words allocated for topic k in document m)
#initiate N_tk(num of term t allocated for topic k)
#initiate N_k(sum of terms for topic k)
#initiate N_m(sum of words for document m)
dic_topic={}
N_mk={}
N_tk={}
for i in range(0,K):
	N_tk[i]={}

for i in range(0,doc_num):
	dic_topic[i]={}
	N_mk[i]={}
	for j in range(0,K):
		N_mk[i][j]=0
w=w_dic.keys()
for doc_no in w:
	place=0
	for x in w_dic[doc_no]:
		dic_topic[doc_no][place]=0
		for i in range(0,K):
			N_tk[i][x]=0
		place+=1
document_no=0
N_k=np.zeros(K)
N_m=np.zeros(doc_num)
for document_no in w:
	place=0
	N_m[document_no]=len(w_dic[document_no])
	for x in w_dic[document_no]:
		ran_topic=np.random.randint(0,K-1)
		dic_topic[document_no][place]=ran_topic
		place+=1
		N_tk[ran_topic][x]=N_tk[ran_topic][x]+1
		N_mk[document_no][ran_topic]=N_mk[document_no][ran_topic]+1
		N_k[ran_topic]+=1
#communicate process
data=com.updateData(N_k,N_m,N_tk,N_mk,rank,size)
N_k=data[0]
N_m=data[1]
N_tk=data[2]
N_mk=data[3]
#Gibbs sampling process
term_list=N_tk[0].keys()
term_num=len(term_list)
if rank==0:
	thitas=np.zeros([doc_num,K])
	phis=[]
	collect_time=0
	for i in range(0,K):
       		phis.append({})
         	for t in term_list:
                	 phis[i][t]=0
	f=open("K100_allperplexity.txt",'w')
while step>0:
#	print('%d:%d'%(rank,step))
	step=step-1
	b_N_tk=None
	b_N_mk=None
	b_N_k=None
	
	if rank==0:
		b_N_tk=copy.deepcopy(N_tk)
		b_N_mk=copy.deepcopy(N_mk)
		b_N_k=copy.deepcopy(N_k)
		print(step)
	for doc_no in w:
		place=0
		for x in w_dic[doc_no]:
			topic=dic_topic[doc_no][place]
			N_tk[topic][x]-=1
			N_mk[doc_no][topic]-=1
#judge whether the topic only have this word			
			N_k[topic]-=1
#get new distribution and re-sample the topic of the term according to new distribution
			sump=0
			q=np.zeros(K)
			for i in range(0,K):
				t=(N_tk[i][x]+beta_value)*(alpha_value+N_mk[doc_no][i])/(N_k[i]+K*beta_value)
				sump+=t
				q[i]=sump
			ran=sump*np.random.random()
			i=0
			while i<K and q[i]<=ran:
				i+=1					
			sample_result=i
			N_mk[doc_no][sample_result]+=1
			N_tk[sample_result][x]+=1
			N_k[sample_result]+=1
			dic_topic[doc_no][place]=sample_result
			place+=1
	data=com.exchangeData(N_k,N_tk,N_mk,rank,size,b_N_k,b_N_tk,b_N_mk)
	N_k=data[0]
	N_tk=data[1]
	N_mk=data[2]
#compute thita and phi
#thita numofdoc*numoftopic type:2-D array
#phi numoftopic*numofword type:2-D dictionary
	if step<sample_step and step%burnin==0 :
		if rank==0:
			thita=np.zeros([doc_num,K])
			for i in range(0,doc_num):
				for j in range(0,K):
					thita[i][j]=(N_mk[i][j]+alpha_value)/(N_m[i]+alpha_value*K)
					thitas[i][j]+=thita[i][j]
			phi={}
			for i in range(0,K):
				phi[i]={}
				for t in term_list:
					phi[i][t]=(N_tk[i][t]+beta_value)/(N_k[i]+beta_value*term_num)
					phis[i][t]+=phi[i][t]
			collect_time+=1	
			sumsumlnpt=0
			sumn=0
			for i in range(0,doc_num):
				sumlnpt=0
				for t in doc_list[i]:
					p_t=0
					sumn+=1
					for k in range(0,K):
						p_t+=thita[i][k]*phi[k][t]
					lnpt=np.log(p_t)
					sumlnpt+=lnpt
				sumsumlnpt+=sumlnpt
			perplexity=np.exp(-sumsumlnpt/sumn)
			print('perplex%f:%d'%(perplexity,step))			
			f.write(str(perplexity))
			f.write('\n')
if rank==0:
	for i in range(0,doc_num):
		for j in range(0,K):
			thitas[i][j]/=collect_time
	for i in range(0,K):
		for t in term_list:
			phis[i][t]/=collect_time
	f=open("K100phiall.txt",'w')
	np.savetxt("K100thitaall.txt",thitas)
	import json
	phis=json.dumps(phis,sort_keys=True)
	f.write(phis)
	f.close()	
