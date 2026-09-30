from __future__ import annotations

import pytest

import optuna
from optuna.samplers import ConditionalGridSampler
from optuna.trial import Trial


def svm_objective(trial: Trial) -> float:
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    c = trial.suggest_int("C", 1, 10)
    if kernel == "rbf":
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        return c * gamma
    return float(c)


SVM_SEARCH_SPACE = {"kernel": ["linear", "rbf"], "C": [1, 10], "gamma": [0.1, 1.0]}
SVM_CONDITIONS = {"gamma": ("kernel", ["rbf"])}
SVM_CONFIGS = [
    {"kernel": "linear", "C": 1},
    {"kernel": "linear", "C": 10},
    {"kernel": "rbf", "C": 1, "gamma": 0.1},
    {"kernel": "rbf", "C": 1, "gamma": 1.0},
    {"kernel": "rbf", "C": 10, "gamma": 0.1},
    {"kernel": "rbf", "C": 10, "gamma": 1.0},
]


def test_enumeration_count_and_order() -> None:
    # All configurations can be listed by hand: gamma is only enumerated for rbf.
    sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)

    assert len(sampler) == 6
    assert sampler.configurations == SVM_CONFIGS


def test_configurations_are_copies() -> None:
    sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    sampler.configurations[0]["kernel"] = "mutated"

    assert sampler.configurations == SVM_CONFIGS


def test_multi_level_conditions() -> None:
    search_space = {"a": [0, 1], "b": [0, 1], "c": [0, 1]}
    conditions = {"b": ("a", [1]), "c": ("b", [1])}
    sampler = ConditionalGridSampler(search_space, conditions)

    assert sampler.configurations == [
        {"a": 0},
        {"a": 1, "b": 0},
        {"a": 1, "b": 1, "c": 0},
        {"a": 1, "b": 1, "c": 1},
    ]


def test_multiple_trigger_values() -> None:
    search_space = {"kernel": ["linear", "rbf", "poly"], "gamma": [0.1, 1.0]}
    conditions = {"gamma": ("kernel", ["rbf", "poly"])}
    sampler = ConditionalGridSampler(search_space, conditions)

    assert sampler.configurations == [
        {"kernel": "linear"},
        {"kernel": "rbf", "gamma": 0.1},
        {"kernel": "rbf", "gamma": 1.0},
        {"kernel": "poly", "gamma": 0.1},
        {"kernel": "poly", "gamma": 1.0},
    ]


def test_parent_may_appear_after_child_in_search_space() -> None:
    # The enumeration order is topologically sorted, so parents are always
    # enumerated before the parameters depending on them.
    search_space = {"gamma": [0.1], "kernel": ["linear", "rbf"]}
    conditions = {"gamma": ("kernel", ["rbf"])}
    sampler = ConditionalGridSampler(search_space, conditions)

    assert sampler.configurations == [
        {"kernel": "linear"},
        {"kernel": "rbf", "gamma": 0.1},
    ]


def test_condition_on_nonexistent_parameter() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ConditionalGridSampler({"a": [0]}, {"b": ("a", [0])})


def test_condition_on_nonexistent_parent() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        ConditionalGridSampler({"a": [0]}, {"a": ("b", [0])})


def test_circular_conditions() -> None:
    search_space = {"a": [0, 1], "b": [0, 1]}
    with pytest.raises(ValueError, match="circular"):
        ConditionalGridSampler(search_space, {"a": ("b", [0]), "b": ("a", [0])})


def test_self_referential_condition() -> None:
    with pytest.raises(ValueError, match="circular"):
        ConditionalGridSampler({"a": [0, 1]}, {"a": ("a", [0])})


def test_trigger_value_outside_parent_domain() -> None:
    with pytest.raises(ValueError, match="not in the domain"):
        ConditionalGridSampler({"a": [0], "b": [0]}, {"b": ("a", [1])})


def test_empty_trigger_values() -> None:
    with pytest.raises(ValueError, match="at least one trigger value"):
        ConditionalGridSampler({"a": [0], "b": [0]}, {"b": ("a", [])})


def test_empty_domain() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ConditionalGridSampler({"a": []})


def test_optimize_covers_all_configs_in_order() -> None:
    sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    study = optuna.create_study(sampler=sampler)
    # `n_trials` larger than the grid: the optimization stops after the grid is exhausted.
    study.optimize(svm_objective, n_trials=100)

    assert len(study.trials) == len(sampler)
    # Trials are evaluated in the enumeration order.
    for trial, expected in zip(study.trials, SVM_CONFIGS):
        assert trial.params == expected
    # Inactive parameters do not enter the corresponding trials.
    for trial in study.trials:
        if trial.params["kernel"] == "linear":
            assert "gamma" not in trial.params
        else:
            assert "gamma" in trial.params
    assert sampler.is_exhausted(study)


def test_suggest_inactive_param_raises() -> None:
    def objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        # gamma is suggested even for the linear kernel, violating the conditions.
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        return gamma if kernel == "rbf" else 0.0

    study = optuna.create_study(
        sampler=ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    )
    with pytest.raises(ValueError, match="not active"):
        study.optimize(objective)


def test_suggest_unknown_param_raises() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("unknown", 0.0, 1.0)

    study = optuna.create_study(
        sampler=ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    )
    with pytest.raises(ValueError, match="not found"):
        study.optimize(objective)


def test_out_of_range_value_warns() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_int("a", 0, 10)

    sampler = ConditionalGridSampler({"a": [100]})
    study = optuna.create_study(sampler=sampler)
    with pytest.warns(UserWarning):
        study.optimize(objective)


def test_resume_recognizes_equivalent_configs_regardless_of_dict_order() -> None:
    storage = optuna.storages.InMemoryStorage()
    sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    study = optuna.create_study(
        study_name="conditional-grid", storage=storage, sampler=sampler
    )
    study.optimize(svm_objective, n_trials=3)
    assert len(study.trials) == 3

    # Resume with a new sampler whose search space lists the parameters in a
    # different order. The already evaluated configurations must be recognized
    # by their parameter values and must not be re-evaluated.
    reordered_search_space = {
        "gamma": [0.1, 1.0],
        "C": [1, 10],
        "kernel": ["linear", "rbf"],
    }
    resumed_sampler = ConditionalGridSampler(reordered_search_space, SVM_CONDITIONS)
    resumed_study = optuna.load_study(
        study_name="conditional-grid", storage=storage, sampler=resumed_sampler
    )
    resumed_study.optimize(svm_objective)

    assert len(resumed_study.trials) == 6
    assert [t.params for t in resumed_study.trials] == [
        {"kernel": "linear", "C": 1},
        {"kernel": "linear", "C": 10},
        {"kernel": "rbf", "C": 1, "gamma": 0.1},
        {"kernel": "rbf", "C": 1, "gamma": 1.0},
        {"kernel": "rbf", "C": 10, "gamma": 0.1},
        {"kernel": "rbf", "C": 10, "gamma": 1.0},
    ]


def test_old_trials_compatibility_after_condition_change() -> None:
    storage = optuna.storages.InMemoryStorage()
    sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    study = optuna.create_study(
        study_name="conditional-grid", storage=storage, sampler=sampler
    )
    study.optimize(svm_objective)
    assert len(study.trials) == 6

    # Change the conditions: gamma is now active for the linear kernel instead.
    # All old trials become incompatible with the new reachable configurations
    # and must be ignored.
    new_conditions = {"gamma": ("kernel", ["linear"])}
    new_sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, new_conditions)
    new_study = optuna.load_study(
        study_name="conditional-grid", storage=storage, sampler=new_sampler
    )

    def new_objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        c = trial.suggest_int("C", 1, 10)
        if kernel == "linear":
            gamma = trial.suggest_float("gamma", 0.0, 1.0)
            return c * gamma
        return float(c)

    new_study.optimize(new_objective)

    assert len(new_study.trials) == 12
    new_params = [t.params for t in new_study.trials[6:]]
    assert new_params == [
        {"kernel": "linear", "C": 1, "gamma": 0.1},
        {"kernel": "linear", "C": 1, "gamma": 1.0},
        {"kernel": "linear", "C": 10, "gamma": 0.1},
        {"kernel": "linear", "C": 10, "gamma": 1.0},
        {"kernel": "rbf", "C": 1},
        {"kernel": "rbf", "C": 10},
    ]


def test_old_trials_partially_compatible_after_condition_change() -> None:
    storage = optuna.storages.InMemoryStorage()
    sampler = ConditionalGridSampler(SVM_SEARCH_SPACE, SVM_CONDITIONS)
    study = optuna.create_study(
        study_name="conditional-grid", storage=storage, sampler=sampler
    )
    study.optimize(svm_objective)
    assert len(study.trials) == 6

    # Add a new conditional parameter for the rbf kernel. The old linear trials
    # are still equivalent to reachable configurations and must not be repeated,
    # while the old rbf trials lack `degree` and are no longer compatible.
    new_search_space = dict(SVM_SEARCH_SPACE, degree=[2])
    new_conditions = {"gamma": ("kernel", ["rbf"]), "degree": ("kernel", ["rbf"])}
    new_sampler = ConditionalGridSampler(new_search_space, new_conditions)
    assert len(new_sampler) == 6  # 2 linear configs + 2 * 2 * 1 rbf configs.

    new_study = optuna.load_study(
        study_name="conditional-grid", storage=storage, sampler=new_sampler
    )

    def new_objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        c = trial.suggest_int("C", 1, 10)
        if kernel == "rbf":
            gamma = trial.suggest_float("gamma", 0.0, 1.0)
            degree = trial.suggest_int("degree", 1, 5)
            return c * gamma * degree
        return float(c)

    new_study.optimize(new_objective)

    # Only the 4 rbf configurations are re-evaluated.
    assert len(new_study.trials) == 10
    new_params = [t.params for t in new_study.trials[6:]]
    assert new_params == [
        {"kernel": "rbf", "C": 1, "gamma": 0.1, "degree": 2},
        {"kernel": "rbf", "C": 1, "gamma": 1.0, "degree": 2},
        {"kernel": "rbf", "C": 10, "gamma": 0.1, "degree": 2},
        {"kernel": "rbf", "C": 10, "gamma": 1.0, "degree": 2},
    ]


def test_no_conditions_is_plain_grid() -> None:
    sampler = ConditionalGridSampler({"a": [0, 1], "b": ["x", "y"]})

    assert len(sampler) == 4
    assert sampler.configurations == [
        {"a": 0, "b": "x"},
        {"a": 0, "b": "y"},
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
    ]

