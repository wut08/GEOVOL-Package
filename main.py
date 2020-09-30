import numpy as np
import pandas as pd
import esquared as es
import initial_loadings_s as s
import baseline_x as x


your_file = input("Your price file csv ( format: date/ security A/ security B...)")
df=pd.read_csv(your_file)

# turn price file into log returns
simple_ret = df.iloc[:,1:10].pct_change()
log_ret = np.log(simple_ret+1)
for col in log_ret.columns:
    log_ret[col] = np.where(log_ret[col] >= 0.1, 0.1, np.where(log_ret[col] <= -0.1, -0.1, log_ret[col]))

log_ret = log_ret.iloc[1:].reset_index()

#calculate e_square
e_square = es.get_e(es.get_residual(log_ret))

print("e squared captured with dimension of:")
print(e_square.shape)

#estimate baseline s
s_prime = s.get_s(e_square)
print("Initial baseline for s is ")
print(s_prime)

#estimate baseline x
st_x = x.get_x(e_square)
print("Baseline x is ")
print(st_x)

