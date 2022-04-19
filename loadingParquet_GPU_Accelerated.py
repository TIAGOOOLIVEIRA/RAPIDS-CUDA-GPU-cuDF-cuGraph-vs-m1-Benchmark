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
 
#joining both dataframes
start_time = time.monotonic()
joindf = fullgf.join(incoming, how='inner', lsuffix='famid', rsuffix='famid')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
#0:00:00.202807
 
#same joining operation takes 4x slower in a CPU based Pandas Dataframe using same data scale.