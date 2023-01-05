# CatBosot
from catboost import CatBoostClassifier


def objective(trial):
    param = {
        "loss_function": trial.suggest_categorical("loss_function", ["Logloss"]),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.2),
        #"iterations": trial.suggest_int("depth", 50, 75),
        "depth": trial.suggest_int("depth", 5, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        #"min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20), 
    }
    # Conditional Hyper-Parameters
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    reg = CatBoostClassifier(**param, cat_features=columns_df[0])
    reg.fit(train_df[columns_df[0]+columns_df[1]+columns_df[2]], train_df['TARGET'],\
            eval_set=[(val_df[columns_df[0]+columns_df[1]+columns_df[2]], val_df['TARGET'])],\
            verbose=0, early_stopping_rounds=100)
    y_pred = reg.predict_proba(val_df[columns_df[0]+columns_df[1]+columns_df[2]])
    score = roc_auc_score(val_df['TARGET'], y_pred[:, 1])
    return score

study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=15, timeout=600) # Run for 10 minutes
print("Number of completed trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("\tBest Score: {}".format(trial.value))
print("\tBest Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train on the best trial
print(trial.params)
clf = CatBoostClassifier(**trial.params, iterations = 1500,cat_features=columns_df[0])
clf.fit(
    pd.concat([
        train_df[columns_df[0]+columns_df[1]+columns_df[2]],
        val_df[columns_df[0]+columns_df[1]+columns_df[2]]
    ], axis = 0),
    pd.concat([
        train_df['TARGET'],\
        val_df['TARGET']
    ], axis = 0)
)

# Make submit file
preds = list(clf.predict_proba(test_df[columns_df[0]+columns_df[1]+columns_df[2]])[:, 1])
ids = list(test_df['id_contract'])
submit_data = {'id_contract': ids, 'TARGET': preds}
submit_df = pd.DataFrame.from_dict(submit_data, )
submit_df.to_csv('submit_file.csv', sep=';', index=False)