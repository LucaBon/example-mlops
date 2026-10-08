from typing import NamedTuple

from kfp.components import InputPath, OutputPath


def evaluate(test_path: InputPath('CSV'),
             run_id: str,
             model_version: str,
             metrics_path: OutputPath('JSON'),
             mlpipeline_metrics_path: OutputPath('Metrics'),
             mlpipeline_ui_metadata_path: OutputPath(),
             min_f1_macro: float = 0.6,
             min_class_recall: float = 0.5,
             min_improvement: float = 0.0,
             min_prob_better: float = 0.9,
             n_bootstrap: int = 1000,
             registered_model_name: str = 'WashingMachineModel',
             ) -> NamedTuple('Outputs', [('decision', str)]):
    """Score the candidate per washing cycle on the test split and gate deployment.

    Window predictions are reduced to one label per ``cycle_id`` by majority
    vote. Returns ``deploy`` when
      * the cycle-level CV macro-F1 logged by ``train`` reaches ``min_f1_macro``,
      * the CV recall of every class reaches ``min_class_recall``,
      * the test cycle accuracy beats always predicting the majority class, and
      * the candidate beats the current Production model on the test cycles
        the champion was not trained on: macro-F1 higher by more than
        ``min_improvement`` and higher in at least ``min_prob_better`` of the
        bootstrap resamples of those cycles (stratified by class). On a tie,
        the candidate must have a CV macro-F1 higher than the champion's by
        more than ``min_improvement``. Without a comparable champion (no list
        of its train cycles, it cannot score the test set, or every test cycle
        was in its train set) this check is skipped,
    and ``skip`` otherwise. The CV metrics carry the quality gates because a
    few test cycles per fault class make a single holdout noisy. Accuracy and
    recall are also reported per machine ``model``; they are not gates.
    Promotion happens in ``promote``, after a successful deploy.
    """
    import json
    import re
    import tempfile
    from collections import namedtuple

    import mlflow
    import mlflow.pyfunc
    import numpy as np
    import pandas as pd
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    target_col_name = 'target'
    cycle_col_name = 'cycle_id'
    machine_col_name = 'model'
    meta_cols = ['TIMESTAMP', 'DateTime', target_col_name, cycle_col_name,
                 'brand', machine_col_name]

    def sanitize(name):
        return re.sub(r'[^\w.\- ]', '_', str(name))

    def majority(s):
        # Most frequent label; ties go to the first label in sorted order
        counts = s.value_counts()
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    df = pd.read_csv(test_path, dtype={cycle_col_name: str})
    x = df.drop(columns=[c for c in meta_cols if c in df.columns])
    y = df[target_col_name].astype(str)
    groups = df[cycle_col_name]
    cycles = pd.DataFrame({'target': y.groupby(groups).agg(majority)})
    if machine_col_name in df.columns:
        cycles['machine'] = df[machine_col_name].astype(str).groupby(groups).first()
    y_cycle = cycles['target'].to_numpy()
    test_classes = sorted(set(y_cycle))

    def f1_macro(y_true, y_pred, labels=test_classes):
        return float(f1_score(y_true, y_pred, labels=labels, average='macro',
                              zero_division=0))

    def predict_cycles(model_uri):
        window_pred = pd.Series(
            np.asarray(mlflow.pyfunc.load_model(model_uri).predict(x)).astype(str),
            index=df.index)
        return window_pred, window_pred.groupby(groups).agg(majority)[cycles.index].to_numpy()

    def bootstrap(idx):
        # Resample cycles within each class; the same resamples score both models
        rng = np.random.default_rng(0)
        strata = [idx[y_cycle[idx] == c] for c in test_classes]
        strata = [s for s in strata if len(s)]
        return [np.concatenate([rng.choice(s, len(s)) for s in strata])
                for _ in range(n_bootstrap)]

    window_pred, candidate_pred = predict_cycles(f'runs:/{run_id}/model')
    report = classification_report(y_cycle, candidate_pred, labels=test_classes,
                                   zero_division=0)
    print(f'Cycle-level report ({len(cycles)} test cycles):')
    print(report)
    class_recall = dict(zip(test_classes, recall_score(
        y_cycle, candidate_pred, labels=test_classes, average=None, zero_division=0)))
    candidate = {
        'test_cycle_accuracy': float(accuracy_score(y_cycle, candidate_pred)),
        'test_cycle_f1_macro': f1_macro(y_cycle, candidate_pred),
        'test_window_accuracy': float(accuracy_score(y, window_pred)),
        'test_window_f1_macro': f1_macro(y, window_pred),
    }
    baseline_accuracy = float(cycles['target'].value_counts(normalize=True).iloc[0])

    client = MlflowClient()
    cv = {k: v for k, v in client.get_run(run_id).data.metrics.items()
          if k.startswith('cv_')}
    cv_recall = {k: v for k, v in cv.items()
                 if k.startswith('cv_recall_') and k.endswith('_mean')}

    candidate_boot = np.array([f1_macro(y_cycle[i], candidate_pred[i])
                               for i in bootstrap(np.arange(len(cycles)))])
    candidate['test_cycle_f1_macro_ci_low'] = float(np.quantile(candidate_boot, 0.05))
    candidate['test_cycle_f1_macro_ci_high'] = float(np.quantile(candidate_boot, 0.95))

    production = [v for v in client.get_latest_versions(
        registered_model_name, stages=['Production'])
        if str(v.version) != str(model_version)]
    champion, champion_version, comparison, champion_note = None, None, None, None
    if production:
        champion_version = str(production[0].version)
        champion_run_id = production[0].run_id
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with open(client.download_artifacts(
                        champion_run_id, 'train_cycles.json', tmp)) as f:
                    champion_train = set(json.load(f))
        except Exception as e:
            champion_note = f'champion run has no train_cycles.json ({e})'
        if champion_note is None:
            # Only cycles the champion has not seen; the candidate is scored on the same ones
            unseen = np.flatnonzero(~cycles.index.isin(champion_train))
            if len(unseen) == 0:
                champion_note = 'every test cycle is in the champion train set'
        if champion_note is None:
            try:
                _, champion_pred = predict_cycles(
                    f'models:/{registered_model_name}/{champion_version}')
            except Exception as e:
                champion_note = f'champion cannot score the test set ({e})'
        if champion_note is None:
            truth, cand, champ = y_cycle[unseen], candidate_pred[unseen], champion_pred[unseen]
            resamples = bootstrap(unseen)
            champion = {
                'test_cycle_accuracy': float(accuracy_score(truth, champ)),
                'test_cycle_f1_macro': f1_macro(truth, champ),
                'cv_f1_macro_mean': client.get_run(champion_run_id).data.metrics.get(
                    'cv_f1_macro_mean'),
            }
            cand_boot = np.array([f1_macro(y_cycle[i], candidate_pred[i]) for i in resamples])
            champ_boot = np.array([f1_macro(y_cycle[i], champion_pred[i]) for i in resamples])
            comparison = {
                'n_cycles': int(len(unseen)),
                'candidate_test_cycle_f1_macro': f1_macro(truth, cand),
                'improvement': f1_macro(truth, cand) - champion['test_cycle_f1_macro'],
                'prob_better': float(np.mean(cand_boot > champ_boot)),
            }
        else:
            print(f'WARNING: no comparable champion v{champion_version}: {champion_note}')

    if comparison is None:
        beats_champion = True
    elif comparison['improvement'] == 0:
        # Tie on test (e.g. both perfect): the candidate must do better in CV
        champion_cv = champion['cv_f1_macro_mean']
        beats_champion = (champion_cv is not None
                          and cv.get('cv_f1_macro_mean', -1.0) > champion_cv + min_improvement)
        comparison['tie_decided_by_cv'] = True
    else:
        beats_champion = (comparison['improvement'] > min_improvement
                          and comparison['prob_better'] >= min_prob_better)

    checks = {
        'cv_min_f1_macro': cv.get('cv_f1_macro_mean', -1.0) >= min_f1_macro,
        'cv_min_class_recall': (bool(cv_recall)
                                and min(cv_recall.values()) >= min_class_recall),
        'beats_majority_baseline': candidate['test_cycle_accuracy'] > baseline_accuracy,
        'beats_champion': bool(beats_champion),
    }
    decision = 'deploy' if all(checks.values()) else 'skip'

    # Per machine model, for information only: fault classes are concentrated on
    # few machine models, so a model can score by recognising the machine
    per_machine, multi_class = {}, None
    if 'machine' in cycles:
        cycles['pred'] = candidate_pred
        for machine, part in cycles.groupby('machine'):
            per_machine[machine] = {
                'n_cycles': int(len(part)),
                'accuracy': float(accuracy_score(part['target'], part['pred'])),
                'recall': {c: float((sub['pred'] == c).mean())
                           for c, sub in part.groupby('target')},
            }
        n_classes = cycles.groupby('machine')['target'].nunique()
        multi_models = sorted(n_classes[n_classes >= 2].index)
        part = cycles[cycles['machine'].isin(multi_models)]
        if len(part):
            multi_class = {
                'models': multi_models,
                'n_cycles': int(len(part)),
                'accuracy': float(accuracy_score(part['target'], part['pred'])),
                # Classes of these machines only: absent classes would score 0
                'f1_macro': f1_macro(part['target'], part['pred'],
                                     labels=sorted(part['target'].unique())),
            }

    labels = sorted(set(test_classes) | set(candidate_pred))
    matrix = confusion_matrix(y_cycle, candidate_pred, labels=labels)
    matrix_csv = '\n'.join(f'{t},{p},{matrix[i][j]}'
                           for i, t in enumerate(labels)
                           for j, p in enumerate(labels))

    machine_metrics = {}
    for machine, rep in per_machine.items():
        prefix = f'test_machine_{sanitize(machine)}'
        machine_metrics[f'{prefix}_accuracy'] = rep['accuracy']
        machine_metrics.update({f'{prefix}_recall_{sanitize(c)}': r
                                for c, r in rep['recall'].items()})
    if multi_class:
        machine_metrics.update({f'test_multi_class_models_{k}': multi_class[k]
                                for k in ('accuracy', 'f1_macro')})

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(candidate)
        mlflow.log_metric('baseline_test_cycle_accuracy', baseline_accuracy)
        mlflow.log_metrics({f'test_cycle_recall_{sanitize(c)}': float(r)
                            for c, r in class_recall.items()})
        mlflow.log_metrics(machine_metrics)
        mlflow.log_text(report, 'classification_report.txt')
        mlflow.log_text('target,predicted,count\n' + matrix_csv,
                        'confusion_matrix.csv')
        for prefix, values in (('champion_', champion), ('comparison_', comparison)):
            mlflow.log_metrics({prefix + k: float(v) for k, v in (values or {}).items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)})
        if champion_note:
            mlflow.set_tag('champion_note', champion_note[:250])
        mlflow.set_tag('evaluation_decision', decision)

    summary = {'candidate_version': model_version, 'candidate': candidate, 'cv': cv,
               'class_recall': {str(c): float(r) for c, r in class_recall.items()},
               'baseline_accuracy': baseline_accuracy, 'n_test_cycles': len(cycles),
               'per_machine': per_machine, 'multi_class_models': multi_class,
               'champion_version': champion_version, 'champion': champion,
               'champion_note': champion_note, 'comparison': comparison,
               'checks': checks, 'decision': decision}
    print(json.dumps(summary, indent=2))
    with open(metrics_path, 'w') as f:
        json.dump(summary, f)
    with open(mlpipeline_metrics_path, 'w') as f:
        json.dump({'metrics': [
            {'name': name.replace('_', '-'), 'numberValue': value, 'format': 'RAW'}
            for name, value in (('cv_f1_macro_mean', cv.get('cv_f1_macro_mean', 0.0)),
                                ('test_cycle_accuracy', candidate['test_cycle_accuracy']),
                                ('test_cycle_f1_macro', candidate['test_cycle_f1_macro']))
        ]}, f)

    def cell(text):
        return str(text).replace('|', '\\|')

    rows = ['| Machine model | Test cycles | Accuracy | Recall per class |',
            '| --- | --- | --- | --- |']
    rows += [f'| {cell(m)} | {r["n_cycles"]} | {r["accuracy"]:.2f} | '
             + ', '.join(f'{cell(c)}: {v:.2f}' for c, v in r['recall'].items()) + ' |'
             for m, r in per_machine.items()]
    if multi_class:
        rows.append(f'| models with 2+ classes ({cell(", ".join(multi_class["models"]))}) | '
                    f'{multi_class["n_cycles"]} | {multi_class["accuracy"]:.2f} | '
                    f'macro-F1 {multi_class["f1_macro"]:.2f} |')
    with open(mlpipeline_ui_metadata_path, 'w') as f:
        json.dump({'outputs': [
            {'type': 'confusion_matrix', 'format': 'csv', 'storage': 'inline',
             'schema': [{'name': 'target', 'type': 'CATEGORY'},
                        {'name': 'predicted', 'type': 'CATEGORY'},
                        {'name': 'count', 'type': 'NUMBER'}],
             'source': matrix_csv, 'labels': labels},
            {'type': 'markdown', 'storage': 'inline',
             'source': '## Test cycles per machine model\n\n' + '\n'.join(rows)},
        ]}, f)

    outputs = namedtuple('Outputs', ['decision'])
    return outputs(decision)
