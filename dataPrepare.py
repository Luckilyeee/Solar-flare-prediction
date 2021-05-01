import pickle
import json
import pandas as pd
import numpy as np
from sklearn.utils import resample


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)


def dataPrepare(path,numbers):
    input1 = '/Users/peiyuli/PycharmProjects/TSC/train_partition1_data.json'
    input2 = '/Users/peiyuli/PycharmProjects/TSC/train_partition2_data.json'
    input3 = '/Users/peiyuli/PycharmProjects/TSC/train_partition3_data.json'

    inputs = [input1, input2, input3]

    MVTS_inputs = []
    MVTS_labels = []

    X_inputs = []
    M_inputs = []
    C_inputs = []
    B_inputs = []
    Q_inputs = []

    count = 0
    for p in inputs:
        with open(p) as infile:
            for line in infile:
                count += 1
                d = json.loads(line)
                for key, value in d.items():
                    label = value['label']
                    mvts = value['values']
                    values_df = pd.DataFrame.from_dict(mvts)
                    # deal with NAN value
                    pd.DataFrame(values_df).fillna(values_df.mean(), inplace=True)
                    values_np = values_df.to_numpy()
                    if np.any(np.isnan(values_np)):
                        pass
                    else:
                        if np.any(np.isnan(values_np)):
                            print("oops")
                        MVTS_inputs.append(values_np)
                        if label == "X":
                            label = 0
                            X_inputs.append(values_np)
                        elif label == "M":
                            label = 1
                            M_inputs.append(values_np)
                        elif label == "C":
                            label = 2
                            C_inputs.append(values_np)
                        elif label == "B":
                            label = 3
                            B_inputs.append(values_np)
                        elif label == "Q":
                            label = 4
                            Q_inputs.append(values_np)
                        MVTS_labels.append(label)

    # print(count)  #214023
    # print(len(MVTS_inputs)) #213057
    # print(len(MVTS_labels)) #213057
    # print(len(X_inputs)) #385
    # print(len(M_inputs)) #3860
    # print(len(C_inputs)) #21022
    # print(len(B_inputs)) #11876
    # print(len(Q_inputs)) #175914

    X_sample = resample(X_inputs, n_samples=numbers, random_state=22)
    M_sample = resample(M_inputs, n_samples=numbers*2, random_state=22)
    C_sample = resample(C_inputs, n_samples=numbers, random_state=22)
    B_sample = resample(B_inputs, n_samples=numbers, random_state=22)
    Q_sample = resample(Q_inputs, n_samples=numbers, random_state=22)
    print(len(X_sample))
    print(len(M_sample))
    print(len(C_sample))
    print(len(B_sample))
    print(len(Q_sample))

    Input_Sampled = np.concatenate([X_sample, M_sample, C_sample, B_sample, Q_sample])

    Input_Sampled = np.asarray(Input_Sampled, dtype=np.float_)
    labels_Sampled = [0]*numbers + [1]*numbers*2 + [2]*numbers + [3]*numbers + [4]*numbers
    labels_Sampled = np.asarray(labels_Sampled)

    save(Input_Sampled, output_path + "Sampled_inputs3.pck")
    save(labels_Sampled, output_path + "Sampled_labels3.pck")


output_path = "/Users/peiyuli/PycharmProjects/TSC/Data/"
numbers = 600
dataPrepare(output_path, numbers)


