"""
Model Evaluation and Testing.

evaluate_agent
==============
Loads trained models and evaluates performance on the FAS-ISAC task.

Key Functions:
    - load_model(checkpoint_path): Load trained agent weights
    - evaluate(agent, env, num_episodes): Run evaluation episodes
    - compute_metrics(trajectories): Compute performance metrics
    - visualize_results(metrics): Generate performance plots

Performance Metrics:
    - Average episode return
    - Communication capacity (bits/s/Hz)
    - Sensing performance (detection rate, localization error)
    - Convergence analysis
    - Action statistics (antenna position distributions)

Evaluation Modes:
    - Deterministic: Use mean of policy (no exploration)
    - Stochastic: Sample from policy distribution
    - Worst-case: Test on adversarial channel conditions

Output:
    - Numeric results saved to CSV
    - Plots saved to results/ directory
"""
