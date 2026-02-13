import subprocess
import argparse
import sys
from utils.colors import prCyan, prGreen

def run_command(command):
    prCyan(f"\nExecuting: ", end='')
    prGreen(f"{' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with return code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Pipeline for Actor-Critic Training")
    parser.add_argument("--system", type=str, choices=["single_integrator", "double_integrator", "single_pendulum", "double_pendulum"], required=True)
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers to use")
    parser.add_argument("--evaluate", action="store_true", help="Run only the evaluation steps (5.1 and 5.2)")
    args = parser.parse_args()

    system = args.system

    # Paths configuration
    paths = {
        "critic_dataset": f"../datasets/critic_dataset_{system}.npz",
        "actor_dataset": f"../datasets/actor_dataset_{system}.npz",
        "critic_model": f"../models/critic_{system}.pth",
        "critic_feat_scaler": f"../models/critic_{system}_feature_scaler.pkl",
        "critic_targ_scaler": f"../models/critic_{system}_target_scaler.pkl",
        "actor_model": f"../models/actor_{system}.pth",
        "actor_model_rl": f"../models/actor_rl_{system}.pth",
        "actor_feat_scaler": f"../models/actor_{system}_feature_scaler.pkl",
        "actor_targ_scaler": f"../models/actor_{system}_target_scaler.pkl"
    }

    print("="*60)
    mode_text = "EVALUATION ONLY" if args.evaluate else "FULL PIPELINE"
    print(f"STARTING {mode_text} FOR: {system}")
    print("="*60)

    # Execute training steps only if NOT --evaluate
    if not args.evaluate:
        # 1) Build Critic Dataset
        step1 = ["python3", "build_critic_dataset.py", "--system", system, "--dataset", paths["critic_dataset"]]
        if args.workers is not None: step1.extend(["--workers", str(args.workers)])
        run_command(step1)

        # 2) Train Critic
        step2 = ["python3", "train.py", "--mode", "critic", "--dataset", paths["critic_dataset"], "--model", paths["critic_model"], "--feature_scaler", paths["critic_feat_scaler"], "--target_scaler", paths["critic_targ_scaler"]]
        run_command(step2)

        # 3) Build Actor Dataset
        step3 = ["python3", "build_actor_dataset.py", "--system", system, "--actor_dataset", paths["actor_dataset"], "--critic_dataset", paths["critic_dataset"], "--critic_model", paths["critic_model"], "--critic_feature_scaler", paths["critic_feat_scaler"], "--critic_target_scaler", paths["critic_targ_scaler"]]
        if args.workers is not None: step3.extend(["--workers", str(args.workers)])
        run_command(step3)

        # 4.1) Train Actor
        step4 = ["python3", "train.py", "--mode", "actor", "--dataset", paths["actor_dataset"], "--model", paths["actor_model"], "--feature_scaler", paths["actor_feat_scaler"], "--target_scaler", paths["actor_targ_scaler"]]
        run_command(step4)

        # 4.2) Train Actor RL
        step4_rl = ["python3", "train_actor_rl.py", "--system", system, "--model-critic", paths["critic_model"], "--model", paths["actor_model_rl"], "--critic-feature-scaler", paths["critic_feat_scaler"], "--critic-target-scaler", paths["critic_targ_scaler"]]
        run_command(step4_rl)

    # --- Evaluation steps are ALWAYS executed ---

    # 5.1) Evaluate Actor 
    step5 = ["python3", "evaluate_actor.py", "--system", system, "--dataset", paths["critic_dataset"], "--model", paths["actor_model"], "--feature_scaler", paths["actor_feat_scaler"], "--target_scaler", paths["actor_targ_scaler"]]
    if args.workers is not None: step5.extend(["--workers", str(args.workers)])
    run_command(step5)

    # 5.2) Evaluate Actor RL
    step5_rl = ["python3", "evaluate_actor.py", "--system", system, "--dataset", paths["critic_dataset"], "--model", paths["actor_model_rl"], "--feature_scaler", paths["critic_feat_scaler"]]
    if args.workers is not None: step5_rl.extend(["--workers", str(args.workers)])
    run_command(step5_rl)

    print("\n" + "="*60)
    print(f"PROCESS COMPLETED SUCCESSFULLY FOR: {system}")
    print("="*60)

if __name__ == "__main__":
    main()
