from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from numbers import Real
from typing import Any
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

from optuna._warnings import optuna_warn
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


GridValueType = Union[str, float, int, bool, None]
GridConfigType = dict[str, GridValueType]

if TYPE_CHECKING:
    GridConditionType = Mapping[str, Mapping[str, Sequence[GridValueType]]]


_logger = get_logger(__name__)


class GridSampler(BaseSampler):
    """Sampler that performs exhaustive search over user-specified grids.

    With :class:`~optuna.samplers.GridSampler`, the trials suggest all combinations of parameters
    in the given search space during the study. Conditional parameters can be declared with
    ``conditions`` so that parameters are included only in configurations where their parent
    parameters take the declared values.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
                if kernel == "rbf":
                    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
                return 0


            search_space = {"kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
            conditions = {"gamma": {"kernel": ["rbf"]}}
            sampler = optuna.samplers.GridSampler(search_space, conditions=conditions)
            assert sampler.all_grids == [
                {"kernel": "linear"},
                {"kernel": "rbf", "gamma": "scale"},
                {"kernel": "rbf", "gamma": "auto"},
            ]
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective)

    .. note::
        :class:`~optuna.samplers.BruteForceSampler` also provides the exhaustive search feature.
        Unlike :class:`~optuna.samplers.GridSampler`, :class:`~optuna.samplers.BruteForceSampler`
        does not ask users to provide search space and it supports hierarchical, aka dynamic search
        space, enabling users to seamlessly enjoy the exhaustive search.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_int("y", -100, 100)
                return x**2 + y**2


            search_space = {"x": [-50, 0, 50], "y": [-99, 0, 99]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective)

    Note:

        This sampler with :ref:`ask_and_tell` raises :exc:`RuntimeError` just after evaluating
        the final grid. This is because :class:`~optuna.samplers.GridSampler` automatically
        stops the optimization if all combinations in the passed ``search_space`` have already
        been evaluated, internally invoking the :func:`~optuna.study.Study.stop` method.
        As a workaround, we need to handle the error manually as in
        https://github.com/optuna/optuna/issues/4121#issuecomment-1305289910.

    Note:

        :class:`~optuna.samplers.GridSampler` does not take care of a parameter's quantization
        specified by discrete suggest methods but just samples one of values specified in the
        search space. E.g., in the following code snippet, either of ``-0.5`` or ``0.5`` is
        sampled as ``x`` instead of an integer point.

        .. testcode::

            import optuna


            def objective(trial):
                # The following suggest method specifies integer points between -5 and 5.
                x = trial.suggest_float("x", -5, 5, step=1)
                return x**2


            # Non-int points are specified in the grid.
            search_space = {"x": [-0.5, 0.5]}
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective, n_trials=2)

    Note:
        A parameter configuration in the grid is not considered finished until its trial is
        finished. Therefore, during distributed optimization where trials run concurrently,
        different workers will occasionally suggest the same parameter configuration.
        The total number of actual trials may therefore exceed the size of the grid.

    Note:
        All parameters must be specified when using :class:`~optuna.samplers.GridSampler` with
        :meth:`~optuna.study.Study.enqueue_trial`.

    Args:
        search_space:
            A dictionary whose key and value are a parameter name and the corresponding candidates
            of values, respectively.
        conditions:
            Activation rules for conditional parameters. Each key is a parameter name and its
            value maps parent parameter names to parent values for which the key is active. All
            listed parent rules must match. Parent references must exist in ``search_space`` and
            must not form a cycle.
        seed:
            A seed to fix the order of trials as the grid is randomly shuffled. This shuffle is
            beneficial when the number of grids is larger than ``n_trials`` in
            :meth:`~optuna.Study.optimize` to suppress suggesting similar grids. Please note
            that fixing ``seed`` for each process is strongly recommended in distributed
            optimization to avoid duplicated suggestions.
    """

    def __init__(
        self,
        search_space: Mapping[str, Sequence[GridValueType]],
        seed: int | None = None,
        *,
        conditions: GridConditionType | None = None,
    ) -> None:
        for param_name, param_values in search_space.items():
            for value in param_values:
                self._check_value(param_name, value)

        self._search_space = {}
        for param_name, param_values in sorted(search_space.items()):
            self._search_space[param_name] = list(param_values)

        self._conditions = self._validate_and_normalize_conditions(self._search_space, conditions)
        self._param_order = self._get_conditional_param_order(self._search_space, self._conditions)
        self._all_grids = self._generate_grids(
            self._search_space, self._conditions, self._param_order
        )
        self._n_min_trials = len(self._all_grids)
        self._rng = LazyRandomState(seed or 0)
        self._grid_order = list(range(self._n_min_trials))
        self._rng.rng.shuffle(self._grid_order)

    @property
    def all_grids(self) -> list[GridConfigType]:
        """Return all reachable grid configurations in a deterministic order."""
        return [grid.copy() for grid in self._all_grids]

    def __len__(self) -> int:
        return self._n_min_trials

    def __iter__(self) -> Iterator[GridConfigType]:
        for grid in self._all_grids:
            yield grid.copy()

    def reseed_rng(self) -> None:
        self._rng.rng.seed()

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Instead of returning param values, GridSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        # When the trial is created by RetryHeartbeatStaleTrialCallback or enqueue_trial, we
        # should not assign a new grid_id.
        if "grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return

        target_grids = self._get_unvisited_grid_ids(study)

        if 0 <= trial.number < self._n_min_trials and len(target_grids) == self._n_min_trials:
            grid_id = self._grid_order[trial.number]
        else:
            if len(target_grids) == 0:
                # This case may occur with distributed optimization or trial queue. If there is no
                # target grid, `GridSampler` evaluates a visited, duplicated point with the
                # current trial. After that, the optimization stops.
                _logger.warning(
                    "`GridSampler` is re-evaluating a configuration because the grid has been "
                    "exhausted. This may happen due to a timing issue during distributed "
                    "optimization or when re-running optimizations on already finished studies."
                )

                # One of all grids is randomly picked up in this case.
                target_grids = list(range(len(self._all_grids)))

            # In distributed optimization, multiple workers may simultaneously pick up the same
            # grid. To make the conflict less frequent, the grid is chosen randomly.
            grid_id = int(self._rng.rng.choice(target_grids))

        study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
        if self._conditions:
            study._storage.set_trial_system_attr(trial._trial_id, "conditions", self._conditions)
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if "grid_id" not in trial.system_attrs:
            message = "All parameters must be specified when using GridSampler with enqueue_trial."
            raise ValueError(message)

        if param_name not in self._search_space:
            message = f"The parameter name, {param_name}, is not found in the given grid."
            raise ValueError(message)

        grid_id = trial.system_attrs["grid_id"]
        grid = self._all_grids[grid_id]
        if param_name not in grid:
            message = (
                f"The parameter name, {param_name}, is not active in grid {grid_id} with "
                f"parameters {grid}. Please align the objective's parameter conditions with "
                "the sampler's `conditions`."
            )
            raise ValueError(message)

        param_value = grid[param_name]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            optuna_warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            study.stop()
        elif len(target_grids) == 1:
            grid_id = study._storage.get_trial_system_attrs(trial._trial_id).get("grid_id")
            if grid_id == target_grids[0]:
                study.stop()

    @staticmethod
    def _check_value(param_name: str, param_value: Any) -> None:
        if param_value is None or isinstance(param_value, (str, int, float, bool)):
            return

        message = (
            f"{param_name} contains a value with the type of {type(param_value)}, "
            "which is not supported by `GridSampler`. Please make sure a value is `str`, "
            "`int`, `float`, `bool` or `None` for persistent storage."
        )
        optuna_warn(message)

    def _get_unvisited_grid_ids(self, study: Study) -> list[int]:
        # List up unvisited grids based on already finished ones.
        visited_grids = []
        running_grids = []

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for t in trials:
            grid_ids = self._get_trial_grid_ids(t)
            if t.state.is_finished():
                visited_grids.extend(grid_ids)
            elif t.state == TrialState.RUNNING:
                running_grids.extend(grid_ids)

        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids) - set(running_grids)

        # If evaluations for all grids have been started, return grids that have not yet finished
        # because all grids should be evaluated before stopping the optimization.
        if len(unvisited_grids) == 0:
            unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)

    @staticmethod
    def _grid_value_equal(value1: GridValueType, value2: GridValueType) -> bool:
        value1_is_nan = isinstance(value1, Real) and np.isnan(float(value1))
        value2_is_nan = isinstance(value2, Real) and np.isnan(float(value2))
        return (value1 == value2) or (value1_is_nan and value2_is_nan)

    def _same_search_space(self, search_space: Mapping[str, Sequence[GridValueType]]) -> bool:
        if set(search_space.keys()) != set(self._search_space.keys()):
            return False

        for param_name in search_space.keys():
            if len(search_space[param_name]) != len(self._search_space[param_name]):
                return False

            for i, param_value in enumerate(search_space[param_name]):
                if not self._grid_value_equal(param_value, self._search_space[param_name][i]):
                    return False

        return True

    def _same_conditions(
        self, conditions: Mapping[str, Mapping[str, Sequence[GridValueType]]]
    ) -> bool:
        if set(conditions.keys()) != set(self._conditions.keys()):
            return False

        for param_name in self._conditions:
            expected = conditions[param_name]
            actual = self._conditions[param_name]
            if set(expected.keys()) != set(actual.keys()):
                return False
            for parent_name in actual:
                expected_values = list(expected[parent_name])
                actual_values = list(actual[parent_name])
                if len(expected_values) != len(actual_values):
                    return False
                if not all(
                    self._grid_value_equal(expected_value, actual_value)
                    for expected_value, actual_value in zip(expected_values, actual_values)
                ):
                    return False

        return True

    def _get_trial_grid_ids(self, trial: FrozenTrial) -> list[int]:
        system_attrs = trial.system_attrs
        grid_id = system_attrs.get("grid_id")
        search_space = system_attrs.get("search_space")
        conditions = system_attrs.get("conditions")

        if not self._conditions:
            if (
                search_space is not None
                and self._same_search_space(search_space)
                and isinstance(grid_id, int)
                and 0 <= grid_id < self._n_min_trials
            ):
                return [grid_id]
            return []

        uses_current_grid = (
            search_space is not None
            and self._same_search_space(search_space)
            and (
                (not self._conditions and "conditions" not in system_attrs)
                or (
                    self._conditions
                    and conditions is not None
                    and self._same_conditions(conditions)
                )
            )
        )
        if uses_current_grid and isinstance(grid_id, int) and 0 <= grid_id < self._n_min_trials:
            return [grid_id]

        return [
            grid_id
            for grid_id, grid in enumerate(self._all_grids)
            if self._same_grid_config(trial.params, grid)
        ]

    def _same_grid_config(self, params: Mapping[str, GridValueType], grid: GridConfigType) -> bool:
        if set(params.keys()) != set(grid.keys()):
            return False
        return all(self._grid_value_equal(params[name], value) for name, value in grid.items())

    @staticmethod
    def _value_in_values(value: GridValueType, values: Sequence[GridValueType]) -> bool:
        return any(GridSampler._grid_value_equal(value, candidate) for candidate in values)

    @classmethod
    def _validate_and_normalize_conditions(
        cls,
        search_space: Mapping[str, Sequence[GridValueType]],
        conditions: GridConditionType | None,
    ) -> dict[str, dict[str, list[GridValueType]]]:
        if conditions is None:
            return {}

        normalized_conditions: dict[str, dict[str, list[GridValueType]]] = {}
        for param_name, parent_conditions in conditions.items():
            if param_name not in search_space:
                raise ValueError(
                    f"Conditional parameter `{param_name}` is not found in the search space."
                )
            if not isinstance(parent_conditions, Mapping):
                raise ValueError(f"Conditions for `{param_name}` must be a mapping.")

            normalized_parent_conditions: dict[str, list[GridValueType]] = {}
            for parent_name, parent_values in parent_conditions.items():
                if parent_name not in search_space:
                    raise ValueError(
                        f"Condition parent `{parent_name}` of `{param_name}` is not found in "
                        "the search space."
                    )
                invalid_values = [
                    value
                    for value in parent_values
                    if not cls._value_in_values(value, search_space[parent_name])
                ]
                if invalid_values:
                    raise ValueError(
                        f"Condition values {invalid_values} for `{parent_name}` are not included "
                        f"in the grid of `{parent_name}`."
                    )
                normalized_parent_conditions[parent_name] = [
                    value
                    for value in search_space[parent_name]
                    if cls._value_in_values(value, parent_values)
                ]
            normalized_conditions[param_name] = normalized_parent_conditions

        return {
            param_name: normalized_conditions[param_name]
            for param_name in sorted(normalized_conditions)
        }

    @staticmethod
    def _get_conditional_param_order(
        search_space: Mapping[str, Sequence[GridValueType]],
        conditions: Mapping[str, Mapping[str, Sequence[GridValueType]]],
    ) -> list[str]:
        indegrees = {param_name: 0 for param_name in search_space}
        children: dict[str, list[str]] = {param_name: [] for param_name in search_space}
        for param_name, parent_conditions in conditions.items():
            indegrees[param_name] = len(parent_conditions)
            for parent_name in parent_conditions:
                children[parent_name].append(param_name)

        ready = deque(
            sorted(param_name for param_name, degree in indegrees.items() if degree == 0)
        )
        ordered_param_names = []
        while ready:
            param_name = ready.popleft()
            ordered_param_names.append(param_name)
            for child_name in sorted(children[param_name]):
                indegrees[child_name] -= 1
                if indegrees[child_name] == 0:
                    ready.append(child_name)

        if len(ordered_param_names) != len(search_space):
            cyclic_parameters = sorted(set(search_space) - set(ordered_param_names))
            raise ValueError(
                "Cyclic parameter conditions are not allowed. Parameters involved in the cycle: "
                f"{cyclic_parameters}."
            )

        return ordered_param_names

    @classmethod
    def _generate_grids(
        cls,
        search_space: Mapping[str, Sequence[GridValueType]],
        conditions: Mapping[str, Mapping[str, Sequence[GridValueType]]],
        param_order: Sequence[str],
    ) -> list[GridConfigType]:
        grids: list[GridConfigType] = [{}]
        for param_name in param_order:
            parent_conditions = conditions.get(param_name)
            next_grids: list[GridConfigType] = []
            for grid in grids:
                if parent_conditions is None:
                    active_values = search_space[param_name]
                elif not all(
                    parent_name in grid and cls._value_in_values(grid[parent_name], parent_values)
                    for parent_name, parent_values in parent_conditions.items()
                ):
                    next_grids.append(grid.copy())
                    continue
                active_values = search_space[param_name]

                for value in active_values:
                    next_grid = grid.copy()
                    next_grid[param_name] = value
                    next_grids.append(next_grid)
            grids = next_grids
        return grids

    def is_exhausted(self, study: Study) -> bool:
        """
        Return True if all the possible params are evaluated, otherwise return False.
        """
        return len(self._get_unvisited_grid_ids(study)) == 0
