from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import ValuesView
import itertools
from pathlib import Path

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.samplers._grid import GridValueType
from optuna.storages import RetryHeartbeatStaleTrialCallback
from optuna.testing.objectives import fail_objective
from optuna.testing.objectives import pruned_objective
from optuna.testing.storages import StorageSupplier
from optuna.trial import Trial
from optuna.trial import TrialState


def test_study_optimize_with_single_search_space() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 100)
        b = trial.suggest_float("b", -0.1, 0.1)
        c = trial.suggest_categorical("c", ("x", "y", None, 1, 2.0))
        d = trial.suggest_float("d", -5, 5, step=1)
        e = trial.suggest_float("e", 0.0001, 1, log=True)

        if c == "x":
            return a * d
        else:
            return b * e

    # Test that all combinations of the grid is sampled.
    search_space = {
        "b": np.arange(-0.1, 0.1, 0.05),
        "c": ("x", "y", None, 1, 2.0),
        "d": [-5.0, 5.0],
        "e": [0.1],
        "a": list(range(0, 100, 20)),
    }
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))  # type: ignore
    study.optimize(objective)

    def sorted_values(
        d: Mapping[str, Sequence[GridValueType]],
    ) -> ValuesView[Sequence[GridValueType]]:
        return dict(sorted(d.items())).values()

    all_grids = itertools.product(*sorted_values(search_space))  # type: ignore
    all_suggested_values = [tuple([p for p in sorted_values(t.params)]) for t in study.trials]
    assert set(all_grids) == set(all_suggested_values)

    # Test a non-existing parameter name in the grid.
    search_space = {"a": list(range(0, 100, 20))}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))  # type: ignore
    with pytest.raises(ValueError):
        study.optimize(objective)

    # Test a value with out of range.
    search_space = {
        "a": [110],  # 110 is out of range specified by the suggest method.
        "b": [0],
        "c": ["x"],
        "d": [0],
        "e": [0.1],
    }
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))  # type: ignore
    with pytest.warns(UserWarning):
        study.optimize(objective)


def test_study_optimize_with_exceeding_number_of_trials() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_int("a", 0, 100)

    # When `n_trials` is `None`, the optimization stops just after all grids are evaluated.
    search_space: dict[str, list[GridValueType]] = {"a": [0, 50]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=None)
    assert len(study.trials) == 2

    # If the optimization is triggered after all grids are evaluated, an additional trial runs.
    study.optimize(objective, n_trials=None)
    assert len(study.trials) == 3


def test_study_optimize_with_pruning() -> None:
    # Pruned trials should count towards grid consumption.
    search_space: dict[str, list[GridValueType]] = {"a": [0, 50]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(pruned_objective, n_trials=None)
    assert len(study.trials) == 2


def test_study_optimize_with_fail() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_int("a", 0, 100)

    # Failed trials should count towards grid consumption.
    search_space: dict[str, list[GridValueType]] = {"a": [0, 50]}
    study = optuna.create_study(sampler=samplers.GridSampler(search_space))
    study.optimize(fail_objective, n_trials=1, catch=ValueError)
    study.optimize(objective, n_trials=None)
    assert len(study.trials) == 2


def test_study_optimize_with_numpy_related_search_space() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 0, 10)
        b = trial.suggest_float("b", -0.1, 0.1)

        return a + b

    # Test that all combinations of the grid is sampled.
    search_space = {
        "a": np.linspace(0, 10, 11),
        "b": np.arange(-0.1, 0.1, 0.05),
    }
    with StorageSupplier("sqlite") as storage:
        study = optuna.create_study(
            sampler=samplers.GridSampler(search_space),  # type: ignore
            storage=storage,
        )
        study.optimize(objective, n_trials=None)


def test_study_optimize_with_multiple_search_spaces() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 100)
        b = trial.suggest_float("b", -100, 100)

        return a * b

    # Run 3 trials with a search space.
    search_space_0 = {"a": [0, 50], "b": [-50, 0, 50]}
    sampler_0 = samplers.GridSampler(search_space_0)
    study = optuna.create_study(sampler=sampler_0)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    for t in study.trials:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])

    # Run 2 trials with another space.
    search_space_1 = {"a": [0, 25], "b": [-50]}
    sampler_1 = samplers.GridSampler(search_space_1)
    study.sampler = sampler_1
    study.optimize(objective, n_trials=2)

    assert not sampler_0._same_search_space(sampler_1._search_space)
    assert len(study.trials) == 5
    for t in study.trials[:3]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[3:5]:
        assert sampler_1._same_search_space(t.system_attrs["search_space"])

    # Run 3 trials with the first search space again.
    study.sampler = sampler_0
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 8
    for t in study.trials[:3]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[3:5]:
        assert sampler_1._same_search_space(t.system_attrs["search_space"])
    for t in study.trials[5:]:
        assert sampler_0._same_search_space(t.system_attrs["search_space"])


def test_cast_value() -> None:
    samplers.GridSampler._check_value("x", None)
    samplers.GridSampler._check_value("x", True)
    samplers.GridSampler._check_value("x", False)
    samplers.GridSampler._check_value("x", -1)
    samplers.GridSampler._check_value("x", -1.5)
    samplers.GridSampler._check_value("x", float("nan"))
    samplers.GridSampler._check_value("x", "foo")
    samplers.GridSampler._check_value("x", "")

    with pytest.warns(UserWarning):
        samplers.GridSampler._check_value("x", [1])


def test_has_same_search_space() -> None:
    search_space: dict[str, list[int | str]] = {"x": [3, 2, 1], "y": ["a", "b", "c"]}
    sampler = samplers.GridSampler(search_space)
    assert sampler._same_search_space(search_space)
    assert sampler._same_search_space({"x": [3, 2, 1], "y": ["a", "b", "c"]})

    assert not sampler._same_search_space({"y": ["c", "a", "b"], "x": [1, 2, 3]})
    assert not sampler._same_search_space({"x": [3, 2, 1, 0], "y": ["a", "b", "c"]})
    assert not sampler._same_search_space({"x": [3, 2], "y": ["a", "b", "c"]})


def test_retried_trial() -> None:
    sampler = samplers.GridSampler({"a": [0, 50]})
    study = optuna.create_study(sampler=sampler)
    trial = study.ask()
    trial.suggest_int("a", 0, 100)

    callback = RetryHeartbeatStaleTrialCallback()
    callback(study, study.trials[0])

    study.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    assert len(study.trials) == 3
    assert study.trials[0].params["a"] == study.trials[1].params["a"]
    assert study.trials[0].system_attrs["grid_id"] == study.trials[1].system_attrs["grid_id"]


def test_enqueued_trial() -> None:
    sampler = samplers.GridSampler({"a": [0, 50]})
    study = optuna.create_study(sampler=sampler)
    study.enqueue_trial({"a": 100})

    study.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    assert len(study.trials) == 3
    assert study.trials[0].params["a"] == 100
    assert sorted([study.trials[1].params["a"], study.trials[2].params["a"]]) == [0, 50]


def test_same_seed_trials() -> None:
    grid_values = [0, 20, 40, 60, 80, 100]
    seed = 0

    sampler1 = samplers.GridSampler({"a": grid_values}, seed)
    study1 = optuna.create_study(sampler=sampler1)
    study1.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    sampler2 = samplers.GridSampler({"a": grid_values}, seed)
    study2 = optuna.create_study(sampler=sampler2)
    study2.optimize(lambda trial: trial.suggest_int("a", 0, 100))

    for i in range(len(grid_values)):
        assert study1.trials[i].params["a"] == study2.trials[i].params["a"]


def test_enqueued_insufficient_trial() -> None:
    sampler = samplers.GridSampler({"a": [0, 50]})
    study = optuna.create_study(sampler=sampler)
    study.enqueue_trial({})

    with pytest.raises(ValueError):
        study.optimize(lambda trial: trial.suggest_int("a", 0, 100))


def test_nan() -> None:
    sampler = optuna.samplers.GridSampler({"x": [0, float("nan")]})
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda trial: 1 if np.isnan(trial.suggest_categorical("x", [0, float("nan")])) else 0
    )
    assert len(study.get_trials()) == 2


def test_is_exhausted() -> None:
    search_space = {"a": [0, 50]}
    sampler = samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler)
    assert not sampler.is_exhausted(study)
    study.optimize(lambda trial: trial.suggest_categorical("a", [0, 50]))
    assert sampler.is_exhausted(study)


def test_conditional_grid_enumeration() -> None:
    search_space = {
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    conditions = {"gamma": {"kernel": ["rbf"]}}
    sampler = samplers.GridSampler(search_space, conditions=conditions)

    expected_grids = [
        {"kernel": "linear"},
        {"kernel": "rbf", "gamma": "scale"},
        {"kernel": "rbf", "gamma": "auto"},
    ]
    assert len(sampler) == 3
    assert sampler.all_grids == expected_grids
    assert list(sampler) == expected_grids


def test_conditional_grid_enumeration_with_multiple_levels() -> None:
    search_space = {
        "kernel": ["linear", "poly", "rbf"],
        "degree": [2, 3],
        "coef0": [0.0, 1.0],
        "gamma": ["scale", "auto"],
    }
    conditions = {
        "degree": {"kernel": ["poly"]},
        "coef0": {"kernel": ["poly"], "degree": [3]},
        "gamma": {"kernel": ["poly", "rbf"]},
    }
    sampler = samplers.GridSampler(search_space, conditions=conditions)

    expected_grids = [
        {"kernel": "linear"},
        {"kernel": "poly", "degree": 2, "gamma": "scale"},
        {"kernel": "poly", "degree": 2, "gamma": "auto"},
        {"kernel": "poly", "degree": 3, "gamma": "scale", "coef0": 0.0},
        {"kernel": "poly", "degree": 3, "gamma": "scale", "coef0": 1.0},
        {"kernel": "poly", "degree": 3, "gamma": "auto", "coef0": 0.0},
        {"kernel": "poly", "degree": 3, "gamma": "auto", "coef0": 1.0},
        {"kernel": "rbf", "gamma": "scale"},
        {"kernel": "rbf", "gamma": "auto"},
    ]
    assert len(sampler) == 9
    assert sampler.all_grids == expected_grids


@pytest.mark.parametrize(
    "conditions",
    [
        {"gamma": {"unknown": ["rbf"]}},
        {"unknown": {"kernel": ["rbf"]}},
        {"gamma": {"kernel": ["unknown"]}},
        {"kernel": {"gamma": ["scale"]}, "gamma": {"kernel": ["rbf"]}},
        {
            "gamma": {"kernel": ["rbf"]},
            "degree": {"gamma": ["scale"]},
            "kernel": {"degree": [2]},
        },
    ],
)
def test_conditional_grid_invalid_conditions(conditions: dict[str, object]) -> None:
    search_space = {
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "degree": [2],
    }
    with pytest.raises(ValueError):
        samplers.GridSampler(search_space, conditions=conditions)  # type: ignore[arg-type]


def test_conditional_grid_study_uses_only_active_parameters() -> None:
    def objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        if kernel == "rbf":
            trial.suggest_categorical("gamma", ["scale", "auto"])
        return 0.0

    search_space = {"kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    conditions = {"gamma": {"kernel": ["rbf"]}}
    study = optuna.create_study(
        sampler=samplers.GridSampler(search_space, conditions=conditions, seed=0)
    )
    study.optimize(objective)

    expected_grids = [
        {"kernel": "linear"},
        {"kernel": "rbf", "gamma": "scale"},
        {"kernel": "rbf", "gamma": "auto"},
    ]
    assert len(study.trials) == len(expected_grids)
    assert {frozenset(trial.params.items()) for trial in study.trials} == {
        frozenset(grid.items()) for grid in expected_grids
    }
    assert all(
        "gamma" not in trial.distributions
        for trial in study.trials
        if trial.params["kernel"] == "linear"
    )


def test_conditional_grid_inactive_parameter_is_not_sampled() -> None:
    def objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        trial.suggest_categorical("gamma", ["scale", "auto"])
        return 0.0 if kernel == "rbf" else 1.0

    search_space = {"kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    conditions = {"gamma": {"kernel": ["rbf"]}}
    study = optuna.create_study(
        sampler=samplers.GridSampler(search_space, conditions=conditions, seed=0)
    )
    with pytest.raises(ValueError, match="gamma"):
        study.optimize(objective)


def test_conditional_grid_resume_matches_equivalent_configurations(tmp_path: Path) -> None:
    search_space = {"kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    conditions = {"gamma": {"kernel": ["rbf"]}}
    storage = f"sqlite:///{tmp_path / 'conditional-grid.db'}"

    study = optuna.create_study(
        study_name="conditional-grid",
        storage=storage,
        sampler=samplers.GridSampler(search_space, conditions=conditions, seed=0),
    )
    study.add_trial(
        optuna.create_trial(
            state=TrialState.COMPLETE,
            value=0.0,
            params={"kernel": "rbf", "gamma": "scale"},
            distributions={
                "kernel": optuna.distributions.CategoricalDistribution(["linear", "rbf"]),
                "gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"]),
            },
            system_attrs={
                "search_space": {
                    "gamma": ["scale", "auto"],
                    "kernel": ["linear", "rbf"],
                },
                "conditions": {"gamma": {"kernel": ["rbf"]}},
                "grid_id": 2,
            },
        )
    )

    resumed_sampler = samplers.GridSampler(
        {"gamma": ["auto", "scale"], "kernel": ["rbf", "linear"]},
        conditions={"gamma": {"kernel": ["rbf"]}},
        seed=1,
    )
    resumed_study = optuna.load_study(
        study_name="conditional-grid", storage=storage, sampler=resumed_sampler
    )

    def objective(trial: Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        if kernel == "rbf":
            trial.suggest_categorical("gamma", ["scale", "auto"])
        return 0.0

    resumed_study.optimize(objective)
    assert len(resumed_study.trials) == 3
    assert {frozenset(trial.params.items()) for trial in resumed_study.trials} == {
        frozenset({"kernel": "rbf", "gamma": "scale"}.items()),
        frozenset({"kernel": "linear"}.items()),
        frozenset({"kernel": "rbf", "gamma": "auto"}.items()),
    }


def test_conditional_grid_compatibility_after_condition_change() -> None:
    search_space = {
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
    }
    old_conditions = {"gamma": {"kernel": ["rbf"]}}
    new_conditions = {
        "gamma": {"kernel": ["rbf"]},
        "shrinking": {"kernel": ["rbf"]},
    }
    sampler = samplers.GridSampler(search_space, conditions=new_conditions)
    study = optuna.create_study(sampler=sampler)

    compatible_old_trial = optuna.create_trial(
        state=TrialState.COMPLETE,
        value=0.0,
        params={"kernel": "linear"},
        distributions={
            "kernel": optuna.distributions.CategoricalDistribution(["linear", "rbf"]),
        },
        system_attrs={
            "search_space": {
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            },
            "conditions": old_conditions,
            "grid_id": 0,
        },
    )
    incompatible_old_trial = optuna.create_trial(
        state=TrialState.COMPLETE,
        value=0.0,
        params={"kernel": "rbf", "gamma": "scale"},
        distributions={
            "kernel": optuna.distributions.CategoricalDistribution(["linear", "rbf"]),
            "gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"]),
        },
        system_attrs={
            "search_space": search_space,
            "conditions": old_conditions,
            "grid_id": 1,
        },
    )
    study.add_trial(compatible_old_trial)
    study.add_trial(incompatible_old_trial)

    unvisited_params = [
        sampler.all_grids[grid_id] for grid_id in sampler._get_unvisited_grid_ids(study)
    ]
    assert {"kernel": "linear"} not in unvisited_params
    assert {"kernel": "rbf", "gamma": "scale", "shrinking": True} in unvisited_params
    assert {"kernel": "rbf", "gamma": "scale", "shrinking": False} in unvisited_params
