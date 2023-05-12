import time
import xorbits._mars.tensor as mt
import xorbits._mars.dataframe as md

start = time.time()
size = 20000
da1 = mt.random.random((size, 2), chunk_size=(1, 2))
df1 = md.DataFrame(da1, columns=list("AB"))
df2 = df1 + 10
df3 = df2.sum()
ret = df3.execute()
print(time.time() - start)
