# analyzer.py
# python analyzer.py --config config.yaml
# pip install PyYAML

import os
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from multiprocessing import Pool
from classifiers import *
from functools import partial

# ログの設定
def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# 設定ファイルの読み込み
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# スナップショットの抽出
class SnapshotExtractor:
    def __init__(self, config):
        self.topology_file = config['topology_file']
        self.trajectory_file = config['trajectory_file']
        self.selection1 = config['selection1']
        self.selection2 = config['selection2']
        self.distance_threshold = config['distance_threshold']
        self.snapshot_dir = config['snapshot_dir']
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.u = mda.Universe(self.topology_file, self.trajectory_file)
        self.close_frames = []
        self.snapshot_nearby_distance = config.get('snapshot_nearby_distance', 5.0)

    def find_close_frames(self):
        sel1 = self.u.select_atoms(self.selection1)
        sel2 = self.u.select_atoms(self.selection2)

        for ts in self.u.trajectory:
            min_dist = distances.distance_array(sel1.positions, sel2.positions).min()
            if min_dist <= self.distance_threshold:
                self.close_frames.append(ts.frame)
        logging.info(f"近接フレーム数: {len(self.close_frames)}")

    def save_snapshots(self):
        for i, frame in enumerate(self.close_frames):
            self.u.trajectory[frame]
            sel1 = self.u.select_atoms(self.selection1)  # 分子A
            sel2 = self.u.select_atoms(self.selection2)  # 分子B（ポリマー）

            # 分子Aから一定距離内のポリマー原子を選択
            polymer_nearby = self.u.select_atoms(f'({self.selection2}) and around {self.snapshot_nearby_distance} group sel1', sel1=sel1)

            # 周辺の水分子を選択（必要に応じて）
            water = self.u.select_atoms(f'resname SOL and around {self.snapshot_nearby_distance} group sel1_polymer', sel1_polymer=sel1 + polymer_nearby)

            # スナップショットに含める原子を結合
            selection = sel1 + polymer_nearby + water
            selection.write(f"{self.snapshot_dir}/snapshot_{i}.pdb")
            logging.info(f"スナップショット保存: snapshot_{i}.pdb")

# 解析器クラス
class InteractionAnalyzer:
    def __init__(self, config):
        self.classifiers = []
        self.config = config

    def add_classifier(self, classifier_func, name, params):
        self.classifiers.append({'func': classifier_func, 'name': name, 'params': params})

    def analyze_snapshot(self, snapshot_file):
        u = mda.Universe(snapshot_file)
        results = {}
        for classifier in self.classifiers:
            try:
                result = classifier['func'](u, classifier['params'])
            except Exception as e:
                logging.error(f"{snapshot_file} - {classifier['name']} の解析中にエラー: {e}")
                result = False
            results[classifier['name']] = result
        return results

    def classify_snapshot(self, results):
        for classifier in self.classifiers:
            name = classifier['name']
            if results.get(name):
                return name
        return '相互作用なし'

# スナップショットの解析（並列処理対応）
def process_snapshot(analyzer, snapshot_file):
    snapshot_path = os.path.join(analyzer.config['snapshot_dir'], snapshot_file)
    results = analyzer.analyze_snapshot(snapshot_path)
    category = analyzer.classify_snapshot(results)
    logging.info(f"{snapshot_file}: {category}")
    return {'snapshot': snapshot_file, 'category': category}

# メイン関数
def main(config_file):
    config = load_config(config_file)
    setup_logging(config['log_file'])
    logging.info("解析開始")

    # スナップショットの抽出
    extractor = SnapshotExtractor(config)
    extractor.find_close_frames()
    extractor.save_snapshots()

    # 解析器のセットアップ
    analyzer = InteractionAnalyzer(config)

    # 分類関数の追加
    for classifier_config in config['classifiers']:
        name = classifier_config['name']
        func_name = classifier_config['function']
        params = config.get(func_name, {})
        func = globals().get(func_name)
        if func:
            analyzer.add_classifier(func, name, params)
        else:
            logging.error(f"分類関数 {func_name} が見つかりません")

    # スナップショットの解析と分類
    snapshot_files = sorted(os.listdir(config['snapshot_dir']))
    num_processes = config.get('num_processes', 1)

    with Pool(processes=num_processes) as pool:
        func = partial(process_snapshot, analyzer)
        classification_results = pool.map(func, snapshot_files)

    # 結果の保存
    df = pd.DataFrame(classification_results)
    df.to_csv('classification_results.csv', index=False)
    logging.info("解析完了")

# エントリーポイント
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分子動力学解析ツール')
    parser.add_argument('--config', type=str, default='config.yaml', help='設定ファイルのパス')
    args = parser.parse_args()
    main(args.config)

