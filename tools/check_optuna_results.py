#!/usr/bin/env python3
"""
Helper script to check Optuna optimization results.
"""

import os
import json
import optuna
from omegaconf import OmegaConf
import pandas as pd

# Paths
OPTUNA_DIR = "/home/smatsubara/documents/airlift/data/sandbox/optuna"
OUTPUTS_ROOT = "/home/smatsubara/documents/airlift/data/outputs_real"
STUDY_DB_PATH = os.path.join(OPTUNA_DIR, 'study.db')


def load_study():
    """Load Optuna study from database."""
    if not os.path.exists(STUDY_DB_PATH):
        print(f"[ERROR] Study database not found: {STUDY_DB_PATH}")
        return None
    
    storage = optuna.storages.RDBStorage(url=f'sqlite:///{STUDY_DB_PATH}')
    try:
        study = optuna.load_study(
            study_name='cnn_hyperparameter_optimization',
            storage=storage
        )
        return study
    except Exception as e:
        print(f"[ERROR] Failed to load study: {e}")
        return None


def show_best_trial(study):
    """Show best trial information."""
    if study is None:
        return
    
    best_trial = study.best_trial
    
    print("\n" + "="*60)
    print("BEST TRIAL INFORMATION")
    print("="*60)
    print(f"Trial Number: #{best_trial.number}")
    print(f"Validation Loss: {best_trial.value:.6f}")
    print(f"Test MSE: {best_trial.user_attrs.get('test_mse', 'N/A')}")
    print(f"Test MAE: {best_trial.user_attrs.get('test_mae', 'N/A')}")
    print(f"\nBest Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nOutput Directory: {best_trial.user_attrs.get('output_dir', 'N/A')}")
    print("="*60 + "\n")


def show_trials_summary(study, n_top=10):
    """Show top N trials summary."""
    if study is None:
        return
    
    # Sort trials by value
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda x: x.value)
    
    print(f"\n{'='*60}")
    print(f"TOP {min(n_top, len(completed_trials))} TRIALS")
    print(f"{'='*60}")
    print(f"{'Trial':<8} {'Val Loss':<12} {'Test MSE':<12} {'Test MAE':<12} {'State':<15}")
    print("-"*60)
    
    for i, trial in enumerate(completed_trials[:n_top]):
        test_mse = trial.user_attrs.get('test_mse', None)
        test_mae = trial.user_attrs.get('test_mae', None)
        print(f"#{trial.number:<7} {trial.value:<12.6f} "
              f"{test_mse if test_mse else 'N/A':<12} "
              f"{test_mae if test_mae else 'N/A':<12} "
              f"{trial.state.name:<15}")
    print("="*60 + "\n")


def show_study_statistics(study):
    """Show study statistics."""
    if study is None:
        return
    
    total_trials = len(study.trials)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"\n{'='*60}")
    print("STUDY STATISTICS")
    print(f"{'='*60}")
    print(f"Total Trials: {total_trials}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Pruned: {len(pruned_trials)}")
    print(f"Failed: {len(failed_trials)}")
    
    if completed_trials:
        values = [t.value for t in completed_trials]
        print(f"\nValidation Loss Statistics:")
        print(f"  Best: {min(values):.6f}")
        print(f"  Worst: {max(values):.6f}")
        print(f"  Mean: {sum(values)/len(values):.6f:.6f}")
        print(f"  Median: {sorted(values)[len(values)//2]:.6f}")
    
    print("="*60 + "\n")


def show_trial_details(trial_number):
    """Show details of a specific trial."""
    trial_dir = find_trial_output_dir(trial_number)
    if not trial_dir:
        print(f"[ERROR] Trial #{trial_number} not found")
        return
    
    print(f"\n{'='*60}")
    print(f"TRIAL #{trial_number} DETAILS")
    print(f"{'='*60}")
    
    # Load trial info
    trial_info_path = os.path.join(trial_dir, 'trial_info.yaml')
    if os.path.exists(trial_info_path):
        trial_info = OmegaConf.load(trial_info_path)
        print(f"State: {trial_info.get('state', 'N/A')}")
        print(f"Validation Loss: {trial_info.get('value', 'N/A')}")
        print(f"\nParameters:")
        params = trial_info.get('params', {})
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"\nUser Attributes:")
        user_attrs = trial_info.get('user_attrs', {})
        for key, value in user_attrs.items():
            if key != 'output_dir':
                print(f"  {key}: {value}")
    
    # Load metrics
    metrics_path = os.path.join(trial_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print(f"\nMetrics:")
        print(f"  Test MSE: {metrics.get('test_mse', 'N/A')}")
        print(f"  Test MAE: {metrics.get('test_mae', 'N/A')}")
        print(f"  Final Train Loss: {metrics.get('final_train_loss', 'N/A')}")
        print(f"  Final Val Loss: {metrics.get('final_val_loss', 'N/A')}")
    
    print(f"\nOutput Directory: {trial_dir}")
    print("="*60 + "\n")


def find_trial_output_dir(trial_number):
    """Find output directory for a given trial number."""
    for root, dirs, files in os.walk(OUTPUTS_ROOT):
        if 'trial_info.yaml' in files:
            trial_info_path = os.path.join(root, 'trial_info.yaml')
            try:
                trial_info = OmegaConf.load(trial_info_path)
                if trial_info.get('trial_number') == trial_number:
                    return root
            except Exception:
                continue
    return None


def export_to_csv(output_path=None):
    """Export all trials to CSV file."""
    study = load_study()
    if study is None:
        return
    
    if output_path is None:
        output_path = os.path.join(OPTUNA_DIR, 'trials_export.csv')
    
    trials_data = []
    for trial in study.trials:
        row = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'validation_loss': trial.value if trial.value is not None else None,
            'test_mse': trial.user_attrs.get('test_mse', None),
            'test_mae': trial.user_attrs.get('test_mae', None),
            **trial.params
        }
        trials_data.append(row)
    
    df = pd.DataFrame(trials_data)
    df.to_csv(output_path, index=False)
    print(f"[OK] Trials exported to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Optuna optimization results')
    parser.add_argument('--best', action='store_true', help='Show best trial')
    parser.add_argument('--summary', type=int, default=10, metavar='N',
                       help='Show top N trials summary (default: 10)')
    parser.add_argument('--stats', action='store_true', help='Show study statistics')
    parser.add_argument('--trial', type=int, metavar='N', help='Show details of trial N')
    parser.add_argument('--export', type=str, metavar='PATH', 
                       help='Export trials to CSV file')
    parser.add_argument('--all', action='store_true', help='Show all information')
    
    args = parser.parse_args()
    
    if args.all:
        args.best = True
        args.summary = 10
        args.stats = True
    
    study = load_study()
    
    if study is None:
        print("[ERROR] Could not load study. Make sure optuna_tutorial.py has been run.")
        return
    
    if args.best or args.all:
        show_best_trial(study)
    
    if args.summary or args.all:
        show_trials_summary(study, n_top=args.summary)
    
    if args.stats or args.all:
        show_study_statistics(study)
    
    if args.trial is not None:
        show_trial_details(args.trial)
    
    if args.export:
        export_to_csv(args.export)
    
    if not any([args.best, args.summary, args.stats, args.trial, args.export, args.all]):
        # Default: show all
        show_best_trial(study)
        show_trials_summary(study)
        show_study_statistics(study)


if __name__ == "__main__":
    main()

