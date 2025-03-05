import pandas as pd
from tqdm import tqdm
import numpy as np

file_path = './AzureLLMInferenceTrace_code_1week.csv'  # 替换为你的 CSV 文件路径

data = pd.concat(tqdm(pd.read_csv(file_path, iterator=True, chunksize=1000), desc="Reading CSV", unit="chunk"))
data = data.head(2000)

data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])

req_rate = 3

np.random.seed(42)

arrival_times = []

data['SECOND'] = data['TIMESTAMP'].dt.floor('S')
unique_seconds = data['SECOND'].unique()

start_time = data['TIMESTAMP'].min().floor('D').timestamp()

for second in unique_seconds:

    requests_in_second = data[data['SECOND'] == second]
    

    if len(requests_in_second) >= req_rate:
        sampled_requests = requests_in_second.sample(n=req_rate, random_state=42)
        arrival_times.extend(sampled_requests['TIMESTAMP'].apply(lambda x: x.timestamp() - start_time).tolist())


