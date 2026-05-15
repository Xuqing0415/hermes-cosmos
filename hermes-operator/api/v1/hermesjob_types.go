//go:generate controller-gen object paths="./..."

package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// HermesJobPhase represents the phase of a HermesJob
type HermesJobPhase string

const (
	HermesJobPending   HermesJobPhase = "Pending"
	HermesJobScheduled HermesJobPhase = "Scheduled"
	HermesJobRunning   HermesJobPhase = "Running"
	HermesJobCompleted HermesJobPhase = "Completed"
	HermesJobFailed    HermesJobPhase = "Failed"
	HermesJobRecovering HermesJobPhase = "Recovering"
)

// HermesJobPriority represents the priority of a HermesJob
type HermesJobPriority string

const (
	HermesJobPriorityLow      HermesJobPriority = "low"
	HermesJobPriorityNormal   HermesJobPriority = "normal"
	HermesJobPriorityHigh     HermesJobPriority = "high"
	HermesJobPriorityCritical HermesJobPriority = "critical"
)

// HermesJobSpec defines the desired state of HermesJob
type HermesJobSpec struct {
	// ModelName is the name of the model being trained
	ModelName string `json:"modelName"`

	// NumGPUs is the number of GPUs requested
	NumGPUs int32 `json:"numGPUs"`

	// GPUType is the type of GPU requested
	// +kubebuilder:default="A100"
	GPUType string `json:"gpuType,omitempty"`

	// CheckpointInterval is the interval (in steps) at which checkpoints are saved
	// +kubebuilder:default=100
	CheckpointInterval int32 `json:"checkpointInterval,omitempty"`

	// Priority is the priority of the job
	// +kubebuilder:default="normal"
	Priority HermesJobPriority `json:"priority,omitempty"`

	// MaxDuration is the maximum duration the job can run
	// +kubebuilder:default="72h"
	MaxDuration string `json:"maxDuration,omitempty"`

	// DataResidency specifies data residency requirements
	DataResidency string `json:"dataResidency,omitempty"`

	// RestartPolicy specifies the restart policy for failed pods
	// +kubebuilder:default="OnFailure"
	RestartPolicy corev1.RestartPolicy `json:"restartPolicy,omitempty"`
}

// HermesJobStatus defines the observed state of HermesJob
type HermesJobStatus struct {
	// Phase is the current phase of the job
	Phase HermesJobPhase `json:"phase,omitempty"`

	// CurrentGPUs is the number of GPUs currently allocated
	CurrentGPUs int32 `json:"currentGPUs,omitempty"`

	// CheckpointCount is the number of checkpoints saved
	CheckpointCount int32 `json:"checkpointCount,omitempty"`

	// StartTime is the time the job started
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// LastCheckpointTime is the time of the last checkpoint
	LastCheckpointTime *metav1.Time `json:"lastCheckpointTime,omitempty"`

	// RecoveryCount is the number of times the job has recovered
	RecoveryCount int32 `json:"recoveryCount,omitempty"`

	// Message provides additional information about the job status
	Message string `json:"message,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
//+kubebuilder:printcolumn:name="GPUs",type=integer,JSONPath=`.spec.numGPUs`
//+kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.modelName`
//+kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// HermesJob is the Schema for the hermesjobs API
type HermesJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   HermesJobSpec   `json:"spec,omitempty"`
	Status HermesJobStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// HermesJobList contains a list of HermesJob
type HermesJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []HermesJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&HermesJob{}, &HermesJobList{})
}
