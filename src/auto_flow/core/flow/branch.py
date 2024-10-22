from typing import TypeVar, Tuple, Callable, Awaitable, Sequence, cast, Any, Iterator

from auto_flow.core.flow.flow import Flow, FlowLike, Flowable, to_flow

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


class BranchFlow(Flow[Input, Output]):
    """
    Flow that selects a branch to run based on a condition.
    Initialized with a list of (condition, Flow) pairs and a default branch.
    Runs the first Flow whose condition is True on the input,
    otherwise runs the default branch.
    """

    branches: Sequence[Tuple[Flowable[Input, bool], Flowable[Input, Output]]]
    default: Flowable[Input, Output]

    def __init__(self,
                 *branches: Tuple[Flow[Input, bool] |
                                  Callable[[Input], bool] |
                                  Callable[[Input], Awaitable[bool]],
                                  FlowLike[Input, Output]] | FlowLike[Input, Output],
                 name: str | None = None):
        if len(branches) < 2:
            raise ValueError("BranchFlow requires at least two branches.")

        branches_ = []
        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):  # type: ignore[arg-type]
                raise TypeError(
                    f"BranchFlow branches must be tuples or lists, not {type(branch)}"
                )

            if not len(branch) == 2:
                raise ValueError(
                    f"BranchFlow branches must be tuples or lists of length 2, not {len(branch)}"
                )
            condition, flow = branch
            condition = to_flow(condition)
            flow = to_flow(flow)
            branches_.append((condition, flow))

        default = to_flow(cast(FlowLike, branches[-1]))
        super().__init__(branches=branches_, default=default)  # type: ignore[call-arg]

    def invoke(self, inp: Input, **kwargs: Any) -> Output:
        """First evaluates the condition, then delegate to true or false branch."""

        for idx, (condition, flow) in enumerate(self.branches):
            local_config = self._get_local_config({"tags": [f"condition:{idx + 1}"]})
            if condition.invoke(inp, local_config):
                local_config = self._get_local_config({"tags": [f"branch:{idx + 1}"]})
                output = flow.invoke(inp, local_config, **kwargs)
                break
        else:
            local_config = self._get_local_config({"tags": [f"branch:default"]})
            output = self.default.invoke(inp, local_config, **kwargs)

        return output

    def transform(self, inp: Iterator[Input], **kwargs: Any) -> Iterator[Output]:
        for idx, (condition, flow) in enumerate(self.branches):
            local_config = self._get_local_config({"tags": [f"condition:{idx + 1}"]})
            if condition.invoke(inp, local_config):
                local_config = self._get_local_config({"tags": [f"branch:{idx + 1}"]})
                yield from flow.transform(inp, local_config, **kwargs)
                break
        else:
            local_config = self._get_local_config({"tags": [f"branch:default"]})
            yield from self.default.transform(inp, local_config, **kwargs)
