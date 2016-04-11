
def updateData(N_k,N_m,N_tk,N_mk,rank,size):
	from mpi4py import MPI
	import sys
	comm=MPI.COMM_WORLD	
	if rank==0:
		allNk=[]
		allNm=[]
		allNtk=[]
		allNmk=[]
		for i in range(1,size):
			data=comm.recv(source=i)
			allNk.append(data[0])
			allNm.append(data[1])
			allNtk.append(data[2])
			allNmk.append(data[3])
		for i in range(0,size-1):
			N_k=N_k+allNk[i]
			N_m=N_m+allNm[i]
			topics=allNtk[i].keys()
			for topic in topics:
				terms=allNtk[i][topic].keys()
				for term in terms:
					if N_tk[topic].has_key(term):
						N_tk[topic][term]+=allNtk[i][topic][term]
					else:
						N_tk[topic][term]=allNtk[i][topic][term]
			docs=allNmk[i].keys()
			for doc in docs:
				topics=allNmk[i][doc].keys()
				for topic in topics:
					if N_mk[doc].has_key(topic):
						N_mk[doc][topic]+=allNmk[i][doc][topic]
					else:
						N_mk[doc][topic]=allNmk[i][doc][topic]
		b_data=[N_k,N_m,N_tk,N_mk]
	else:
		comm.send([N_k,N_m,N_tk,N_mk],dest=0)	
		b_data=None
	b_data=comm.bcast(b_data,root=0)
	return b_data

def exchangeData(N_k,N_tk,N_mk,rank,size,b_N_k,b_N_tk,b_N_mk):
	from mpi4py import MPI
        import sys
        comm=MPI.COMM_WORLD
        if rank==0:
                allNk=[]
                allNtk=[]
                allNmk=[]
                for i in range(1,size):
                        data=comm.recv(source=i)
                        allNk.append(data[0])
                        allNtk.append(data[1])
                        allNmk.append(data[2])
                for i in range(0,size-1):
                        N_k=N_k+allNk[i]-b_N_k
                        topics=allNtk[i].keys()
                        for topic in topics:
                                terms=allNtk[i][topic].keys()
                                for term in terms:
                                        N_tk[topic][term]+=allNtk[i][topic][term]
					N_tk[topic][term]-=b_N_tk[topic][term]
                                    
                        docs=allNmk[i].keys()
                        for doc in docs:
                                topics=allNmk[i][doc].keys()
                                for topic in topics:
                                        N_mk[doc][topic]+=allNmk[i][doc][topic]
					N_mk[doc][topic]-=b_N_mk[doc][topic]
                                        
                b_data=[N_k,N_tk,N_mk]
        else:
                comm.send([N_k,N_tk,N_mk],dest=0)
                b_data=None
        b_data=comm.bcast(b_data,root=0)
	return b_data

				
