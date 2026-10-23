"""Shared simulation lifecycle and tool execution for both communication modes."""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from loguru import logger

from tau2.data_model.language import CallContext
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.user.user_simulator_base import UserState
from tau2.utils.utils import get_now

DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)

BaseAgentT = TypeVar("BaseAgentT")
BaseUserT = TypeVar("BaseUserT")
TrajectoryItemT = TypeVar("TrajectoryItemT")


class BaseOrchestrator(ABC, Generic[BaseAgentT, BaseUserT, TrajectoryItemT]):
    """
    Abstract base class for orchestrators.

    Provides the common infrastructure for managing simulations between Agent, User,
    and Environment. Subclasses implement specific communication patterns:
    - Orchestrator: Half-duplex (turn-based) communication, trajectory of Messages
    - FullDuplexOrchestrator: Full-duplex (streaming) communication, trajectory of Ticks

    Type Parameters:
        BaseAgentT: The agent type
        BaseUserT: The user type
        TrajectoryItemT: The trajectory item type (Message for half-duplex, Tick for full-duplex)

    Shared Responsibilities:
        - Environment initialization and tool execution
        - Termination tracking (max steps, max errors, done state)
        - Trajectory management
        - Simulation run lifecycle (initialize, step loop, finalize)

    Subclass Responsibilities:
        - Communication-specific initialization
        - Step implementation for their communication pattern
        - Mode-specific termination checks
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgentT,
        user: BaseUserT,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        simulation_id: Optional[str] = None,
        timeout: Optional[float] = None,
        call_context: Optional[CallContext] = None,
    ):
        """
        Initialize the base orchestrator.

        Args:
            domain: The domain name of the simulation (e.g., 'airline', 'retail', 'telecom').
            agent: The agent instance.
            user: The user instance.
            environment: The environment instance that handles tool execution.
            task: The task specification containing initial state, goals, and evaluation criteria.
            max_steps: Maximum number of simulation steps before termination. Defaults to 100.
            max_errors: Maximum number of tool execution errors before termination. Defaults to 10.
            seed: Optional random seed for reproducibility. Defaults to None.
            simulation_id: Optional simulation ID. Defaults to generated UUID.
            timeout: Maximum wallclock time in seconds. None means no timeout.
            call_context: Resolved language, persona and scoring context of a
                runner-built call; None for hand-built orchestrators.
        """
        self.call_context = call_context
        self.domain = domain
        self.agent: BaseAgentT = agent
        self.user: BaseUserT = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.simulation_id = simulation_id or str(uuid.uuid4())

        # State tracking
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None

        # Termination tracking
        self.max_steps: int = max_steps
        self.max_errors: int = max_errors
        self.timeout: Optional[float] = timeout
        self.step_count: int = 0
        self.done: bool = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors: int = 0
        self._run_start_time: Optional[str] = None
        self._run_start_perf: Optional[float] = None

    @property
    def initial_agent_message(self) -> Optional[AssistantMessage]:
        """The call's greeting for this mode; None uses the ordinary default."""
        if self.call_context is None:
            return None
        return AssistantMessage(
            role="assistant", content=self.call_context.language.agent_greeting
        )

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the orchestrator for simulation.

        Subclasses must implement mode-specific initialization:
        - Set up environment state
        - Initialize agent and user states
        - Set up initial messages/chunks
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Perform one step of the simulation.

        Subclasses implement their communication pattern:
        - Half-duplex: Turn-based message passing
        - Full-duplex: Simultaneous chunk generation
        """
        pass

    @abstractmethod
    def get_trajectory(self) -> list[TrajectoryItemT]:
        """
        Get the trajectory of the simulation.

        Returns:
            List of trajectory items. Type depends on orchestrator mode:
            - Orchestrator (half-duplex): list[Message]
            - FullDuplexOrchestrator: list[Tick]
        """
        pass

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """
        Get all messages from the simulation as a flat list.

        This provides a consistent way to get messages regardless of orchestrator mode.
        For half-duplex, this is the same as get_trajectory().
        For full-duplex, this returns linearized messages from all ticks.

        Returns:
            List of all messages sorted by timestamp with turn_idx assigned.
        """
        pass

    @abstractmethod
    def _validate_mode_compatibility(self) -> None:
        """
        Validate that the agent and user support this communication mode.

        Raises:
            ValueError: If agent or user don't support the required mode.
        """
        pass

    @abstractmethod
    def _check_termination(self) -> None:
        """
        Check for termination conditions specific to this communication mode.

        Sets self.done and self.termination_reason if termination conditions are met.
        """
        pass

    @abstractmethod
    def _finalize(self) -> SimulationRun:
        """
        Finalize the simulation and create the SimulationRun result.

        Called after the simulation loop completes. Should:
        - Send stop signals to agent and user
        - Calculate costs
        - Build and return SimulationRun

        Returns:
            SimulationRun with all simulation data.
        """
        pass

    def _check_timeout(self) -> None:
        if (
            self.timeout is not None
            and self._run_start_perf is not None
            and not self.done
        ):
            elapsed = time.perf_counter() - self._run_start_perf
            if elapsed >= self.timeout:
                self.done = True
                self.termination_reason = TerminationReason.TIMEOUT
                logger.info(
                    f"Simulation timed out after {elapsed:.1f}s (timeout={self.timeout}s)"
                )

    def _cleanup(self) -> None:
        """Best-effort cleanup of agent and user resources.

        Called from the ``finally`` block of :meth:`run` so that WebSocket
        connections, background threads, and other resources are released
        even when ``step()`` raises an unexpected exception.

        On the normal (non-error) path ``_finalize()`` handles cleanup
        as part of building the result, so this method is a no-op.
        """
        try:
            if hasattr(self, "agent") and self.agent is not None:
                self.agent.stop(None, getattr(self, "agent_state", None))
        except Exception as e:
            logger.warning(f"Error during agent cleanup: {e}")

        try:
            if hasattr(self, "user") and self.user is not None:
                self.user.stop(None, getattr(self, "user_state", None))
        except Exception as e:
            logger.warning(f"Error during user cleanup: {e}")

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Template method that orchestrates the simulation lifecycle:
        1. Initialize the simulation
        2. Step until done
        3. Check termination conditions after each step
        4. Finalize and return results

        Returns:
            SimulationRun: The simulation run with all results.
        """
        self._run_start_time = get_now()
        self._run_start_perf = time.perf_counter()
        self.initialize()

        finalized = False
        try:
            while not self.done:
                self.step()
                self._check_termination()
            result = self._finalize()
            finalized = True
            return result
        finally:
            if not finalized:
                logger.warning(
                    "Simulation loop exited with an exception — "
                    "running emergency cleanup"
                )
                self._cleanup()

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ) -> None:
        """
        Initialize the environment with the given state.

        Args:
            initialization_data: Optional data to initialize environment state.
            initialization_actions: Optional actions to execute during initialization.
            message_history: Message history for context.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        """
        Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            List of ToolMessage results from the environment.
        """
        tool_results = []
        for tool_call in tool_calls:
            tool_result = self.environment.get_response(tool_call)
            if tool_result.error:
                self.num_errors += 1
            tool_results.append(tool_result)
        return tool_results

    def _wrap_tool_results(self, tool_results: list[ToolMessage]) -> Message:
        """
        Wrap tool results in appropriate message type.

        Args:
            tool_results: List of tool message results.

        Returns:
            Single ToolMessage if one result, MultiToolMessage if multiple.
        """
        if len(tool_results) > 1:
            return MultiToolMessage(role="tool", tool_messages=tool_results)
        return tool_results[0]

    def _get_environment_info(self) -> EnvironmentInfo:
        """Get the environment info."""
        return self.environment.get_info()
