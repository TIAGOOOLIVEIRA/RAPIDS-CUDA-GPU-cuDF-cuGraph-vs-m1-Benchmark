#pip install pyarrow
#conda install -c conda-forge fastparquet
#pip install Faker
#conda create -n rapids-22.02 -c rapidsai -c nvidia -c conda-forge rapids=22.02 python=3.8 cudatoolkit=11.2 dask-sql
#conda activate rapids-22.02
 
#to observe GPU card resources consumption
watch -n 1 nvidia-smi
 
from fastparquet import ParquetFile
import pyarrow as pa
 
import csv
import pandas as pd
 
import timeit
import time
from datetime import timedelta
 
import faker
import cugraph
import cudf, io
from io import StringIO
 
########################################################################################################################
########################################################################################################################
########################################################################################################################
#cuDF

filepathparquet = '/tmp/datasets/fgroup.parquet.gzip'
 
maxiter = 30000000
fams = list(fake.random_int(min=20000, max=33333333333) for i in range(maxiter))
docid = list(fake.ssn() for i in range(maxiter))
dict = {'famid': fams, 'docid': docid}
 
df = pd.DataFrame(dict)
 
#materializing parquet file
df.to_parquet(filepathparquet,compression='gzip',engine='pyarrow')
 
#reading parquet from regular pandas CPU based
start_time = time.monotonic()
pdd = pd.read_parquet(filepathparquet)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#HDD 0:00:20.143991
#SSD 0:00:16.735957
 
 
#playing with GPU card via cuDF
#loading parquet file via cuDF RAPIDS
start_time = time.monotonic()
fullgf = cudf.read_parquet(filepathparquet)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#HDD 0:00:03.775780
#SSD 0:00:03.374996
 
#sampling data from previous dataframe for a later joining operation
#simulate classic wide transformation problem in a distributed engine: joining very small dataframe with a large one
incoming = fullgf.sample(frac=0.01)
 
#joining both dataframes via join operation. *Not so sure join infers automatically any index for matching keys.
start_time = time.monotonic()
joindf = fullgf.join(incoming, how='inner', lsuffix='famid', rsuffix='famid')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#0:00:00.202807


#joining via merge operation
start_time = time.monotonic()
mergedf = fullgf.merge(incoming, on=['famid'], how='inner')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#0:00:00.144295
 
#same joining operation takes 4x slower in a CPU based Pandas Dataframe using same data scale.

########################################################################################################################
########################################################################################################################
########################################################################################################################
#cuGraph

#Patent citation network
#https://snap.stanford.edu/data/cit-Patents.html
#Nodes: 3774768
#Edges: 16518948
gfilepath = "/tmp/datasets/cit-Patents.txt"

start_time = time.monotonic()
gdf = cudf.read_csv(gfilepath, delimiter='\t', names=['src', 'dst'], dtype=['int32', 'int32'] )
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#HDD 0:00:01.898568

--louvain
gdf["data"] = 1.0

G = cugraph.Graph()

start_time = time.monotonic()
G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='data')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#HDD 0:00:00.523784

#Run PageRank on the Graph
start_time = time.monotonic()
df_PR = cugraph.pagerank(G)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#HDD 0:00:02.796918

#Exploring some ranked nodes
df_PR.sort_values(by='vertex', ascending=False, na_position='first')
df_PR.sort_values(by='pagerank', ascending=False, na_position='first').tail(3)
df_PR.loc[df_PR["vertex"] == 2836131]


########################################################################################################################
#Playing with others Graph algorithms

#Louvain
start_time = time.monotonic()
df_L, mod = cugraph.louvain(G)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#

#Ensemble Clustering for Graphs (ECG)
start_time = time.monotonic()
df_ECG = cugraph.ecg(G)
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#

part_ids = df_L["partition"].unique()
print(str(len(part_ids)) + " partition detected")

#Fetching nodes for one specific Louvain partition
vids = df_L.query("partition == 1")
v = cudf.Series(vids['vertex'])

subG = cugraph.subgraph(G, v)
print("\tNumber of Vertices: " + str(G.number_of_vertices()))
print("\tNumber of Edges:    " + str(G.number_of_edges()))


coo = cudf.DataFrame()
subDF = subG.view_edge_list()
