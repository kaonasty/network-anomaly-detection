import optuna_dashboard
import optuna

storage = optuna.storages.RDBStorage("sqlite:///optuna.db")
optuna_dashboard.run_server(storage, host="127.0.0.1", port=8081)