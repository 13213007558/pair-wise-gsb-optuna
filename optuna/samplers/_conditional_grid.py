from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna._warnings import optuna_warn
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.samplers._grid import GridSampler
from optuna.samplers._grid import GridValueType
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_logger = get_logger(__name__)


class ConditionalGridSampler(BaseSampler):
    """Sampler that performs exhaustive search over a grid with conditional parameters.

    With :class:`~optuna.samplers.ConditionalGridSampler`, a parameter can be declared active
    only when another parameter takes specific values. For example, ``gamma`` is only
    meaningful for an RBF kernel, so it should be enumerated only when ``kernel="rbf"`` and
    never be evaluated for ``kernel="linear"``.

    The conditions are given explicitly via the ``conditions`` argument; this sampler never
    executes the objective function to discover its conditional structure. The objective is
    expected to suggest parameters following the same conditions: a parameter must be
    suggested only when it is active under the declared conditions.

    All valid (reachable) configurations are enumerated at construction time and can be
    inspected via :attr:`~ConditionalGridSampler.configurations` and
    ``len(sampler)`` before creating a study. Inactive parameters do not appear in the
    corresponding trials.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
                c = trial.suggest_float("C", 1e-2, 1e2, log=True)
                if kernel == "rbf":
                    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
                    return c * len(gamma)
                return c


            search_space = {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1.0],
                "gamma": ["scale", "auto"],
            }
            conditions = {"gamma": ("kernel", ["rbf"])}
            sampler = optuna.samplers.ConditionalGridSampler(search_space, conditions)
            assert len(sampler) == 6  # 2 linear configs + 2 * 2 rbf configs.
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective)

    Note:
        Conditions can be chained (multi-level dependencies), e.g. ``c`` may depend on ``b``
        which itself depends on ``a``. Cycles and references to parameters that do not exist
        in ``search_space`` are detected at construction time and raise :exc:`ValueError`.

    Note:
        When a study is resumed, already finished trials are matched to configurations by
        their parameter values, independently of dictionary ordering. Trials whose parameter
        sets are no longer reachable under the current conditions (e.g. after the conditions
        were changed) are simply ignored and do not count as visited.

    Note:
        A parameter configuration in the grid is not considered finished until its trial is
        finished. Therefore, during distributed optimization where trials run concurrently,
        different workers will occasionally suggest the same parameter configuration.
        The total number of actual trials may therefore exceed the number of configurations.

    Args:
        search_space:
            A dictionary whose key and value are a parameter name and the corresponding
            candidates of values, respectively. Every parameter must have a finite domain.
        conditions:
            A dictionary mapping a parameter name to a ``(parent_name, values)`` tuple.
            The parameter is active (enumerated and suggested) only when ``parent_name`` is
            itself active and takes one of ``values``. Parameters without an entry are
            always active. Defaults to no conditions, which is equivalent to
            :class:`~optuna.samplers.GridSampler`.
    """

    def __init__(
        self,
        search_space: Mapping[str, Sequence[GridValueType]],
        conditions: Mapping[str, tuple[str, Sequence[GridValueType]]] | None = None,
    ) -> None:
        self._search_space: dict[str, list[GridValueType]] = {
            param_name: list(param_values) for param_name, param_values in search_space.items()
        }
        for param_name, param_values in self._search_space.items():
            if len(param_values) == 0:
                raise ValueError(
                    f"The domain of the parameter `{param_name}` must not be empty."
                )
            for value in param_values:
                GridSampler._check_value(param_name, value)

        self._conditions: dict[str, tuple[str, list[GridValueType]]] = {
            param_name: (parent_name, list(trigger_values))
            for param_name, (parent_name, trigger_values) in (conditions or {}).items()
        }
        self._validate_conditions()

        self._param_names = self._topological_order()
        self._configs: list[dict[str, GridValueType]] = []
        self._enumerate(0, {})

    @property
    def configurations(self) -> list[dict[str, GridValueType]]:
        """All reachable configurations, in the deterministic order they are evaluated."""
        return [dict(config) for config in self._configs]

    def __len__(self) -> int:
        return len(self._configs)

    def _validate_conditions(self) -> None:
        for param_name, (parent_name, trigger_values) in self._conditions.items():
            if param_name not in self._search_space:
                raise ValueError(
                    f"The condition for `{param_name}` refers to a parameter that does not "
                    "exist in the search space."
                )
            if parent_name not in self._search_space:
                raise ValueError(
                    f"The condition for `{param_name}` refers to a parent parameter "
                    f"`{parent_name}` that does not exist in the search space."
                )
            if len(trigger_values) == 0:
                raise ValueError(
                    f"The condition for `{param_name}` must specify at least one trigger "
                    f"value of `{parent_name}`."
                )
            for value in trigger_values:
                if not any(
                    GridSampler._grid_value_equal(value, parent_value)
                    for parent_value in self._search_space[parent_name]
                ):
                    raise ValueError(
                        f"The trigger value `{value}` of the condition for `{param_name}` "
                        f"is not in the domain of the parent parameter `{parent_name}`."
                    )

        # Detect circular dependencies by walking each parent chain.
        for start in self._conditions:
            seen = set()
            node = start
            while node in self._conditions:
                if node in seen:
                    raise ValueError(
                        f"The conditions contain a circular dependency involving `{node}`."
                    )
                seen.add(node)
                node = self._conditions[node][0]

    def _topological_order(self) -> list[str]:
        # Parents must be enumerated before the parameters that depend on them. The order is
        # kept stable with respect to the user-given ``search_space`` order.
        order: list[str] = []
        placed: set[str] = set()
        remaining = list(self._search_space.keys())
        while remaining:
            for name in remaining:
                parent_name = self._conditions[name][0] if name in self._conditions else None
                if parent_name is None or parent_name in placed:
                    order.append(name)
                    placed.add(name)
                    remaining.remove(name)
                    break
            else:  # pragma: no cover - unreachable since cycles are rejected beforehand.
                raise ValueError("The conditions contain a circular dependency.")
        return order

    def _is_active(self, param_name: str, assignment: dict[str, GridValueType]) -> bool:
        node = param_name
        while node in self._conditions:
            parent_name, trigger_values = self._conditions[node]
            if parent_name not in assignment:
                return False
            if not any(
                GridSampler._grid_value_equal(assignment[parent_name], trigger_value)
                for trigger_value in trigger_values
            ):
                return False
            node = parent_name
        return True

    def _enumerate(self, index: int, assignment: dict[str, GridValueType]) -> None:
        if index == len(self._param_names):
            self._configs.append(dict(assignment))
            return
        param_name = self._param_names[index]
        if self._is_active(param_name, assignment):
            for value in self._search_space[param_name]:
                assignment[param_name] = value
                self._enumerate(index + 1, assignment)
            del assignment[param_name]
        else:
            self._enumerate(index + 1, assignment)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Instead of returning param values, the sampler puts the target configuration id as
        # a system attr, and the values are returned from `sample_independent`. This is
        # because the distribution object is hard to get at the beginning of trial, while we
        # need the access to the object to validate the sampled value.

        # When the trial is created by RetryHeartbeatStaleTrialCallback or enqueue_trial, we
        # should not assign a new configuration id.
        if "conditional_grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return

        target_configs = self._get_unvisited_config_ids(study)

        if len(target_configs) == 0:
            # This case may occur with distributed optimization or trial queue. If there is
            # no target configuration, the sampler evaluates a visited, duplicated point
            # with the current trial. After that, the optimization stops.
            _logger.warning(
                "`ConditionalGridSampler` is re-evaluating a configuration because the grid "
                "has been exhausted. This may happen due to a timing issue during "
                "distributed optimization or when re-running optimizations on already "
                "finished studies."
            )
            target_configs = list(range(len(self._configs)))

        config_id = target_configs[0]
        study._storage.set_trial_system_attr(trial._trial_id, "conditional_grid_id", config_id)
        study._storage.set_trial_system_attr(
            trial._trial_id, "conditional_grid", self._configs[config_id]
        )

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
        if "conditional_grid_id" not in trial.system_attrs:
            message = (
                "All parameters must be specified when using ConditionalGridSampler with "
                "enqueue_trial."
            )
            raise ValueError(message)

        if param_name not in self._search_space:
            message = (
                f"The parameter name, {param_name}, is not found in the given conditional "
                "grid."
            )
            raise ValueError(message)

        config_id = trial.system_attrs["conditional_grid_id"]
        config = self._configs[config_id]
        if param_name not in config:
            message = (
                f"The parameter `{param_name}` is not active in the configuration assigned "
                "to this trial. Please make sure the objective suggests parameters "
                "following the declared conditions."
            )
            raise ValueError(message)

        param_value = config[param_name]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            optuna_warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: "
                f"`{param_distribution}`."
            )

        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        target_configs = self._get_unvisited_config_ids(study)

        if len(target_configs) == 0:
            study.stop()
        elif len(target_configs) == 1:
            system_attrs = study._storage.get_trial_system_attrs(trial._trial_id)
            if system_attrs.get("conditional_grid_id") == target_configs[0]:
                study.stop()

    def _find_config_id(self, params: Mapping[str, Any]) -> int | None:
        for config_id, config in enumerate(self._configs):
            if self._same_config(params, config):
                return config_id
        return None

    @staticmethod
    def _same_config(params: Mapping[str, Any], config: Mapping[str, GridValueType]) -> bool:
        if set(params.keys()) != set(config.keys()):
            return False
        return all(
            GridSampler._grid_value_equal(params[name], config[name]) for name in config
        )

    def _get_unvisited_config_ids(self, study: Study) -> list[int]:
        # List up unvisited configurations based on already finished ones. Finished trials
        # are matched to configurations by their parameter values, so that equivalent
        # configurations are recognized independently of dictionary ordering and of the
        # sampler instance that produced them.
        visited_configs = set()
        running_configs = set()

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for t in trials:
            if t.state.is_finished():
                config_id = self._find_config_id(t.params)
                if config_id is not None:
                    visited_configs.add(config_id)
            elif t.state == TrialState.RUNNING:
                assigned = t.system_attrs.get("conditional_grid")
                if assigned is not None:
                    config_id = self._find_config_id(assigned)
                    if config_id is not None:
                        running_configs.add(config_id)

        all_configs = set(range(len(self._configs)))
        unvisited_configs = all_configs - visited_configs - running_configs

        # If evaluations for all configurations have been started, return configurations
        # that have not yet finished because all configurations should be evaluated before
        # stopping the optimization.
        if len(unvisited_configs) == 0:
            unvisited_configs = all_configs - visited_configs

        return sorted(unvisited_configs)

    def is_exhausted(self, study: Study) -> bool:
        """
        Return True if all the possible configurations are evaluated, otherwise return False.
        """
        return len(self._get_unvisited_config_ids(study)) == 0

