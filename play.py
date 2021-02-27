from config import bandits, explorer, N_ROUNDS


if __name__ == "__main__":
    history_stats, regret_stats = explorer.experiment(bandits, N_ROUNDS)
    explorer.save_experiment(bandits, history_stats, regret_stats)













