// sim/observer.go
package sim

// SimulationState contains a snapshot of the simulation state at a specific time.
// It provides a read-only view of instance states for observers.
type SimulationState struct {
	// Clock is the current simulation time in microseconds
	Clock int64
	// Instances contains state snapshots for each instance in the simulation
	Instances []InstanceState
}

// InstanceState contains observable state for a single instance.
// This is the state object passed to observers.
type InstanceState struct {
	// ID uniquely identifies this instance
	ID string
	// QueueDepth is the number of requests waiting to be scheduled
	QueueDepth int
	// BatchSize is the number of requests currently in the running batch
	BatchSize int
	// KVUtilization is the fraction of KV cache blocks in use (0.0 to 1.0)
	KVUtilization float64
	// CacheHitRate is the cache hit rate (0.0 to 1.0)
	CacheHitRate float64
	// CompletedRequests is the cumulative count of completed requests
	CompletedRequests int
	// TimedOutRequests is the cumulative count of timed out requests
	TimedOutRequests int
	// DroppedRequests is the cumulative count of dropped unservable requests
	DroppedRequests int
}

// Observer is called periodically during simulation to observe state.
// Implementations must not modify the simulation state.
type Observer interface {
	// Observe is called at regular intervals with a snapshot of simulation state.
	// The interval is configured when registering the observer.
	Observe(state SimulationState)
}

// RegisteredObserver holds an observer and its observation interval.
type RegisteredObserver struct {
	Observer     Observer
	Interval     int64 // observation interval in microseconds
	LastObserved int64 // last time this observer was called
}

// Made with Bob
