import os
import tqdm
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import time
from collections import defaultdict
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA


def main():
    results = {}
    times = defaultdict(float)
    for dataset_name in opt['dataset']:
        results[dataset_name] = []
        print('===> process %s dataset\n' % dataset_name)
        benchmarks = os.listdir(os.path.join(opt['data_dir'], dataset_name, 'benchmarks'))
        benchmarks.sort()
        # benchmarks = benchmarks[:5]
        for benchmark in tqdm.tqdm(benchmarks):
            df = pd.read_csv(os.path.join(opt['data_dir'], dataset_name, 'benchmarks', benchmark))
            df = df.dropna()
            train_data = []
            train_label = []
            for _, data in df.iterrows():
                if data['ground.truth'] == 'nominal':
                    train_label.append(0)
                else:
                    train_label.append(1)
                train_data.append(np.array(data[opt['dataset'][dataset_name]]))
            train_data = np.array(train_data)
            train_label = np.array(train_label)

            res = {'benchmark': benchmark}
            new_time = time.time()
            for model_name, model in models.items():
                model.fit(train_data)
                if np.isnan(np.min(model.decision_scores_)):
                    auc = np.nan
                else:
                    auc = roc_auc_score(train_label, model.decision_scores_)
                # auc = roc_auc_score(train_label, model.decision_scores_)
                res[model_name] = auc
                times[model_name] += time.time() - new_time
                new_time = time.time()
            results[dataset_name].append(res)
    json.dump(results, open('./result.json', 'w'))

    for dataset_name in results:
        res_dir = os.path.join(opt['result_dir'], dataset_name)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        res_df = defaultdict(list)
        for res in results[dataset_name]:
            for k in res:
                res_df[k].append(res[k])
        res_df = pd.DataFrame(data=res_df)
        res_df.to_csv(os.path.join(res_dir, 'result.csv'), index=False)


if __name__ == "__main__":
    opt = {
        'data_dir': '../data/anomaly_detection',
        'result_dir': './result',
        'dataset': {
            'abalone': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'],
            'skin': ['R', 'G', 'B'],
        },
    }

    models = {
        'KNN largest': KNN(method='largest'),
        'KNN mean': KNN(method='mean'),
        'KNN median': KNN(method='median'),
        'CBLOF': CBLOF(),
        'LOF': LOF(),
        'FeatureBagging': FeatureBagging(),
        'HBOS': HBOS(),
        'IForest': IForest(),
        'MCD': MCD(),
        'OCSVM': OCSVM(),
        'PCA': PCA(),
    }

    main()
