import pandas as pd
forecast_lindel_train_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\combine_forecast_lindel_data\forecast_lindel_train.csv'
forecast_lindel_valid_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\combine_forecast_lindel_data\forecast_lindel_valid.csv'
sprout_path = r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\new_key_df.csv'

forecast_lindel_sprout = [forecast_lindel_train_path, forecast_lindel_valid_path, sprout_path]

columns_all_sequences = []
columns_all_sequences.append(pd.read_csv(forecast_lindel_sprout[0], usecols=[1]))
columns_all_sequences.append(pd.read_csv(forecast_lindel_sprout[1], usecols=[1]))
columns_all_sequences.append(pd.read_csv(forecast_lindel_sprout[2], usecols=[0]))

columns_fl_sequences = []
columns_fl_sequences.append(pd.read_csv(forecast_lindel_sprout[0], usecols=[1]))
columns_fl_sequences.append(pd.read_csv(forecast_lindel_sprout[1], usecols=[1]))


combined_sequences_forecast_lindel_sprout = pd.concat(columns_all_sequences, ignore_index=False)
combined_sequences_forecast_lindel = pd.concat(columns_fl_sequences, ignore_index=False)

combined_sequences_forecast_lindel_sprout.to_csv(r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\allsequences\forecast_lindel_sprout_combined_sequences.csv', index=True, header=True)
combined_sequences_forecast_lindel.to_csv(r'C:\Users\Philipp\Desktop\Studium\BachelorArbeit\Thesis_v2\data\allsequences\forecast_lindel_combined_sequences.csv', index=True, header=True)