import pandas as pd

df = pd.read_csv(r'C:\Users\guilh\PycharmProjects\bacteria-cnpq\dataset-SNR\xor_graph_regression\output_global_add_pool\test_predictions_0.csv')

print(df['snr_true'].mean())