from typing import Optional, Type, Tuple

import cutlass.cute as cute

from cutlass.pipeline import (
    PipelineAsync,
    PipelineState,
    CooperativeGroup,
    Agent,
    PipelineUserType,
    make_pipeline_state,
)

from cutlass.cutlass_dsl import Boolean


class ImmutableResourceHandle:
    __origin: PipelineAsync
    __immutable_state: PipelineState

    def __init__(self, origin: PipelineAsync, immutable_state: PipelineState):
        self.__origin = origin
        self.__immutable_state = immutable_state

    @property
    def index(self):
        """Get the index of the current pipeline stage."""
        return self.__immutable_state.index

    @property
    def count(self):
        """Get the count of how many handles this producer has committed.
        This is useful for tracking the number of blocks that have been loaded from gmem.
        """
        return self.__immutable_state.count

    def get_origin(self):
        """Get the original pipeline this resource handle belongs to."""
        return self.__origin

    def __extract_mlir_values__(self):
        """Extract MLIR values from the current state.

        :return: List of MLIR values representing the current state
        :rtype: list
        """
        # TODO: need to handle pipeline as well
        return self.__immutable_state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new Producer instance from MLIR values.

        :param values: MLIR values to initialize the state
        :type values: Any
        :return: New Producer instance with state initialized from values
        :rtype: Producer
        """
        return self.__class__(
            self.__origin, self.__immutable_state.__new_from_mlir_values__(values)
        )


class PipelineProducer:
    """A class representing a producer in an asynchronous pipeline.

    The Producer class manages the producer side of an asynchronous pipeline, handling
    synchronization and state management for producing data. It provides methods for
    acquiring, committing, and advancing through pipeline stages.

    :ivar __pipeline: The asynchronous pipeline this producer belongs to
    :type __pipeline: PipelineAsync
    :ivar __state: The current state of the producer in the pipeline
    :type __state: PipelineState
    :ivar __group: The cooperative group this producer operates in
    :type __group: CooperativeGroup

    **Examples:**

        .. code-block:: python

            pipeline = PipelineAsync.create(...)
            producer = pipeline.create_producer(producer_group, stages)
            for i in range(iterations):
                handle = producer.acquire_and_advance()  # Wait for buffer to be empty
                # Produce data
                producer.commit(handle)   # Signal data is ready
                # An alternative way to do this is:
                # handle.commit()   # Signal data is ready
    """

    __pipeline: PipelineAsync
    __state: PipelineState
    __group: CooperativeGroup

    # {$nv-internal-release begin}
    # TODO: CFK-26943: uncomment dataclass decorator when fixing pre-processor's bug
    # @dataclass(frozen=True)
    # {$nv-internal-release end}
    class ImmutableResourceHandle(ImmutableResourceHandle):
        @property
        def barrier(self):
            """Get the barrier pointer for the current pipeline stage.

            :return: Pointer to the barrier for the current stage
            :rtype: cute.Pointer
            """
            return self.get_origin().producer_get_barrier(
                self._ImmutableResourceHandle__immutable_state
            )

        def commit(self):
            """Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.
            """
            self.get_origin().producer_commit(
                self._ImmutableResourceHandle__immutable_state
            )

    def __init__(self, pipeline, state, group: CooperativeGroup):
        """Initialize a new Producer instance.

        :param pipeline: The pipeline this producer belongs to
        :type pipeline: PipelineAsync
        :param state: Initial pipeline state
        :type state: PipelineState
        :param group: The cooperative group for synchronization
        :type group: CooperativeGroup
        """
        self.__pipeline = pipeline
        self.__state = state
        self.__group = group

    def acquire(
        self,
        try_acquire_token: Optional[Boolean] = None,
    ) -> ImmutableResourceHandle:
        """Wait for the current buffer to be empty before producing data.
        This is a blocking operation.

        :param try_acquire_token: Optional token to try to acquire the buffer
        :type try_acquire_token: Optional[Boolean]
        :return: A handle to the producer for committing the data
        :rtype: ImmutableResourceHandle
        """
        self.__pipeline.producer_acquire(self.__state, try_acquire_token)
        handle = PipelineProducer.ImmutableResourceHandle(
            self.__pipeline, self.__state.clone()
        )
        return handle

    def advance(self):
        """Move to the next pipeline stage."""
        self.__state.advance()

    def acquire_and_advance(
        self, try_acquire_token: Optional[Boolean] = None
    ) -> ImmutableResourceHandle:
        """Wait for the current buffer to be empty before producing data.
        Then advance to the next stage.
        This is a blocking operation.

        :param try_acquire_token: Optional token to try to acquire the buffer
        :type try_acquire_token: Optional[Boolean]
        :return: A handle to the producer for committing the data
        :rtype: ImmutableResourceHandle
        """
        handle = self.acquire(try_acquire_token)
        self.advance()
        return handle

    def try_acquire(self) -> Boolean:
        """Try to acquire the current buffer without blocking.

        :return: True if acquisition was successful, False otherwise
        :rtype: Boolean
        """
        return self.__pipeline.producer_try_acquire(self.__state)

    def commit(self, handle: Optional[ImmutableResourceHandle] = None):
        """Signal that data production is complete for the current stage.
        This allows consumers to start processing the data.
        """
        if handle is not None:
            assert handle.get_origin() is self, (
                "ResourceHandle does not belong to this PipelineProducer instance"
            )
            handle.commit()
        else:
            self.__pipeline.producer_commit(self.__state)

    def tail(self):
        """Ensure all used buffers are properly synchronized before producer exit.
        This should be called before the producer finishes to avoid dangling signals.
        """
        self.__pipeline.producer_tail(self.__state)

    def __extract_mlir_values__(self):
        """Extract MLIR values from the current state.

        :return: List of MLIR values representing the current state
        :rtype: list
        """
        # TODO: need to handle pipeline as well
        return self.__state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new Producer instance from MLIR values.

        :param values: MLIR values to initialize the state
        :type values: Any
        :return: New Producer instance with state initialized from values
        :rtype: Producer
        """
        return PipelineProducer(
            self.__pipeline, self.__state.__new_from_mlir_values__(values), self.__group
        )


class PipelineConsumer:
    """A class representing a consumer in an asynchronous pipeline.

    The Consumer class manages the consumer side of an asynchronous pipeline, handling
    synchronization and state management for consuming data. It provides methods for
    waiting, releasing, and advancing through pipeline stages.

    :ivar __pipeline: The asynchronous pipeline this consumer belongs to
    :type __pipeline: PipelineAsync
    :ivar __state: The current state of the consumer in the pipeline
    :type __state: PipelineState
    :ivar __group: The cooperative group this consumer operates in
    :type __group: CooperativeGroup

    **Examples:**
        .. code-block:: python

            pipeline = PipelineAsync.create(...)
            consumer = pipeline.create_consumer(consumer_group, stages)
            for i in range(iterations):
                handle = consumer.wait_and_advance()     # Wait for data to be ready
                # Consume data
                consumer.release(handle)  # Signal buffer is empty
                # An alternative way to do this is:
                # handle.release()  # Signal buffer is empty
    """

    __pipeline: PipelineAsync
    __state: PipelineState
    __group: CooperativeGroup

    # {$nv-internal-release begin}
    # TODO: CFK-26943: uncomment dataclass decorator when fixing pre-processor's bug
    # @dataclass(frozen=True)
    # {$nv-internal-release end}
    class ImmutableResourceHandle(ImmutableResourceHandle):
        def release(self):
            """Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.
            """
            self.get_origin().consumer_release(
                self._ImmutableResourceHandle__immutable_state
            )

    def __init__(self, pipeline, state: PipelineState, group: CooperativeGroup):
        """Initialize a new Consumer instance.

        :param pipeline: The pipeline this consumer belongs to
        :type pipeline: PipelineAsync
        :param state: Initial pipeline state
        :type state: PipelineState
        :param group: The cooperative group for synchronization
        :type group: CooperativeGroup
        """
        self.__pipeline = pipeline
        self.__group = group
        self.__state = state

    def wait(self, try_wait_token: Optional[Boolean] = None) -> ImmutableResourceHandle:
        """Wait for data to be ready in the current buffer.
        This is a blocking operation.

        :param try_wait_token: Optional token to try to wait for the buffer
        :type try_wait_token: Optional[Boolean]
        :return: A handle to the consumer for releasing the data
        :rtype: PipelineConsumerHandle
        """
        self.__pipeline.consumer_wait(self.__state, try_wait_token)
        handle = PipelineConsumer.ImmutableResourceHandle(
            self.__pipeline, self.__state.clone()
        )
        return handle

    def advance(self):
        """Move to the next pipeline stage."""
        self.__state.advance()

    def wait_and_advance(
        self, try_wait_token: Optional[Boolean] = None
    ) -> ImmutableResourceHandle:
        """Wait for data to be ready in the current buffer.
        Then advance to the next stage.
        This is a blocking operation.

        :param try_wait_token: Optional token to try to wait for the buffer
        :type try_wait_token: Optional[Boolean]
        :return: A handle to the consumer for releasing the data
        :rtype: PipelineConsumerHandle
        """
        handle = self.wait(try_wait_token)
        self.advance()
        return handle

    def try_wait(self) -> Boolean:
        """Try to check if data is ready without blocking.

        :return: True if data is ready, False otherwise
        :rtype: Boolean
        """
        return self.__pipeline.consumer_try_wait(self.__state)

    def release(self, handle: Optional[ImmutableResourceHandle] = None):
        """Signal that data consumption is complete for the current stage.
        This allows producers to start producing new data.
        """
        if handle is not None:
            assert handle.get_origin() is self, (
                "ResourceHandle does not belong to this PipelineConsumer instance"
            )
            handle.release()
        else:
            self.__pipeline.consumer_release(self.__state)

    def __extract_mlir_values__(self):
        """Extract MLIR values from the current state.

        :return: List of MLIR values representing the current state
        :rtype: list
        """
        return self.__state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new Consumer instance from MLIR values.

        :param values: MLIR values to initialize the state
        :type values: Any
        :return: New Consumer instance with state initialized from values
        :rtype: Consumer
        """
        # TODO: need to call pipeline.__new_from_mlir_values__ recursively
        return PipelineConsumer(
            self.__pipeline, self.__state.__new_from_mlir_values__(values), self.__group
        )


def make_pipeline_participants(
    pipeline_type: Type[PipelineAsync],
    barrier_storage: cute.Pointer,
    num_stages: int,
    producer_thread_count: int,
    consumer_thread_count: int,
    tx_count: int | None = None,
) -> Tuple[PipelineProducer, PipelineConsumer]:
    """
    Get a producer and consumer for a given pipeline type.
    This function helps to create CooperativeGroup by assigning Agent.Thread.
    With the CooperativeGroup, it can further help create & get producer & consumer.

    :param pipeline_type: The type of pipeline to create
    :type pipeline_type: Type[PipelineAsync]
    :param barrier_storage: The pointer to the barrier storage
    :type barrier_storage: cute.Pointer
    :param num_stages: The number of stages in the pipeline
    :type num_stages: int
    :param producer_thread_count: The number of threads in the producer group
    :type producer_thread_count: int
    :param consumer_thread_count: The number of threads in the consumer group
    :type consumer_thread_count: int
    :param tx_count: The number of transactions in the pipeline
    :type tx_count: int | None
    :return: A tuple of producer and consumer
    :rtype: Tuple[PipelineProducer, PipelineConsumer]
    """
    producer_group = CooperativeGroup(
        Agent.Thread,
        producer_thread_count,
        producer_thread_count,
    )
    consumer_group = CooperativeGroup(
        Agent.Thread,
        consumer_thread_count,
        consumer_thread_count,
    )
    if tx_count is not None:
        pipeline = pipeline_type.create(
            barrier_storage=barrier_storage,
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
        )
    else:
        pipeline = pipeline_type.create(
            barrier_storage=barrier_storage,
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
        )
    return make_pipeline_producer(pipeline, producer_group), make_pipeline_consumer(
        pipeline, consumer_group
    )


def make_pipeline_producer(pipeline, group: CooperativeGroup):
    state = make_pipeline_state(PipelineUserType.Producer, pipeline.num_stages)
    return PipelineProducer(pipeline, state, group)


def make_pipeline_consumer(pipeline, group: CooperativeGroup):
    state = make_pipeline_state(PipelineUserType.Consumer, pipeline.num_stages)
    return PipelineConsumer(pipeline, state, group)
