//go:generate controller-gen object paths="./..."

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// HermesCheckpointPhase represents the phase of a HermesCheckpoint
type HermesCheckpointPhase string

const (
	HermesCheckpointCreating HermesCheckpointPhase = "Creating"
	HermesCheckpointReady    HermesCheckpointPhase = "Ready"
	HermesCheckpointFailed   HermesCheckpointPhase = "Failed"
)

// HermesCheckpointStorageType represents the storage type for checkpoints
type HermesCheckpointStorageType string

const (
	HermesCheckpointStorageRedis HermesCheckpointStorageType = "redis"
	HermesCheckpointStorageMinio HermesCheckpointStorageType = "minio"
	HermesCheckpointStorageLocal HermesCheckpointStorageType = "local"
)

// HermesCheckpointSpec defines the desired state of HermesCheckpoint
type HermesCheckpointSpec struct {
	// JobName is the name of the associated HermesJob
	JobName string `json:"jobName"`

	// Step is the training step at which this checkpoint was saved
	Step int32 `json:"step"`

	// StorageType is the type of storage to use for the checkpoint
	// +kubebuilder:default="redis"
	StorageType HermesCheckpointStorageType `json:"storageType,omitempty"`

	// Region is the region where the checkpoint is stored
	Region string `json:"region,omitempty"`
}

// HermesCheckpointStatus defines the observed state of HermesCheckpoint
type HermesCheckpointStatus struct {
	// Phase is the current phase of the checkpoint
	Phase HermesCheckpointPhase `json:"phase,omitempty"`

	// Size is the size of the checkpoint
	Size string `json:"size,omitempty"`

	// CreatedAt is the time the checkpoint was created
	CreatedAt *metav1.Time `json:"createdAt,omitempty"`

	// Replicas is the number of replicas of this checkpoint
	Replicas int32 `json:"replicas,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status
//+kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
//+kubebuilder:printcolumn:name="Step",type=integer,JSONPath=`.spec.step`
//+kubebuilder:printcolumn:name="Job",type=string,JSONPath=`.spec.jobName`
//+kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// HermesCheckpoint is the Schema for the hermescheckpoints API
type HermesCheckpoint struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   HermesCheckpointSpec   `json:"spec,omitempty"`
	Status HermesCheckpointStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// HermesCheckpointList contains a list of HermesCheckpoint
type HermesCheckpointList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []HermesCheckpoint `json:"items"`
}

func init() {
	SchemeBuilder.Register(&HermesCheckpoint{}, &HermesCheckpointList{})
}
