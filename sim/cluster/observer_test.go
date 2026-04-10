package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestObserver is a simple observer that records observations
type TestObserver struct {
	observations []sim.SimulationState
}

func (o *TestObserver) Observe(state sim.SimulationState) {
	o.observations = append(o.observations, state)
}

func TestClusterSimulator_RegisterObserver(t *testing.T) {
	// Create a simple cluster with 1 instance
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 10_000_000 // 10 seconds

	// Create observer
	observer := &TestObserver{}

	// Create cluster and register observer with 1 second interval
	requests := newTestRequests(100) // Enough requests to run for several seconds
	cluster := NewClusterSimulator(cfg, requests, nil)
	cluster.RegisterObserver(observer, 1_000_000) // 1 second

	// Run simulation
	mustRun(t, cluster)

	// Verify observer was called
	if len(observer.observations) == 0 {
		t.Error("Expected observer to be called at least once, but got 0 observations")
	}

	// Verify observations have correct structure
	for i, obs := range observer.observations {
		if obs.Clock <= 0 {
			t.Errorf("Observation %d: expected Clock > 0, got %d", i, obs.Clock)
		}
		if len(obs.Instances) != 1 {
			t.Errorf("Observation %d: expected 1 instance, got %d", i, len(obs.Instances))
		}
		if len(obs.Instances) > 0 {
			inst := obs.Instances[0]
			if inst.ID == "" {
				t.Errorf("Observation %d: instance ID is empty", i)
			}
		}
	}

	// Verify observations are spaced by approximately the interval
	if len(observer.observations) > 1 {
		for i := 1; i < len(observer.observations); i++ {
			timeDiff := observer.observations[i].Clock - observer.observations[i-1].Clock
			// Should be at least the interval (1 second = 1,000,000 microseconds)
			if timeDiff < 1_000_000 {
				t.Errorf("Observations %d and %d: time difference %d is less than interval 1,000,000",
					i-1, i, timeDiff)
			}
		}
	}
}

func TestClusterSimulator_RegisterObserver_MultipleObservers(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.Horizon = 10_000_000 // 10 seconds

	// Create two observers with different intervals
	fastObserver := &TestObserver{}
	slowObserver := &TestObserver{}

	requests := newTestRequests(200) // Enough requests for both instances to run for several seconds
	cluster := NewClusterSimulator(cfg, requests, nil)
	cluster.RegisterObserver(fastObserver, 500_000)   // 500ms
	cluster.RegisterObserver(slowObserver, 2_000_000) // 2 seconds

	mustRun(t, cluster)

	// Fast observer should have more observations than slow observer
	if len(fastObserver.observations) <= len(slowObserver.observations) {
		t.Errorf("Expected fast observer (%d observations) to have more than slow observer (%d observations)",
			len(fastObserver.observations), len(slowObserver.observations))
	}

	// Both should have been called
	if len(fastObserver.observations) == 0 {
		t.Error("Fast observer was never called")
	}
	if len(slowObserver.observations) == 0 {
		t.Error("Slow observer was never called")
	}
}

func TestClusterSimulator_RegisterObserver_PanicAfterRun(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 1_000_000

	cluster := NewClusterSimulator(cfg, nil, nil)
	mustRun(t, cluster)

	// Attempting to register observer after Run() should panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when registering observer after Run(), but didn't panic")
		}
	}()

	observer := &TestObserver{}
	cluster.RegisterObserver(observer, 1_000_000)
}

func TestClusterSimulator_RegisterObserver_InvalidInterval(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 1_000_000

	cluster := NewClusterSimulator(cfg, nil, nil)

	// Test zero interval
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for zero interval, but didn't panic")
		}
	}()

	observer := &TestObserver{}
	cluster.RegisterObserver(observer, 0)
}