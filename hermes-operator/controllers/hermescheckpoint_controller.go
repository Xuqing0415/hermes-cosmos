package controllers

import (
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	trainingv1 "hermes-operator/api/v1"
)

// HermesCheckpointReconciler reconciles a HermesCheckpoint object
type HermesCheckpointReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

//+kubebuilder:rbac:groups=training.hermes.io,resources=hermescheckpoints,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=training.hermes.io,resources=hermescheckpoints/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=training.hermes.io,resources=hermescheckpoints/finalizers,verbs=update
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile is part of the main kubernetes reconciliation loop
func (r *HermesCheckpointReconciler) Reconcile(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
	log := log.FromContext(ctx)

	// Fetch the HermesCheckpoint instance
	var checkpoint trainingv1.HermesCheckpoint
	if err := r.Get(ctx, req.NamespacedName, &checkpoint); err != nil {
		if errors.IsNotFound(err) {
			return reconcile.Result{}, nil
		}
		return reconcile.Result{}, err
	}

	log.Info("Reconciling HermesCheckpoint", "name", checkpoint.Name, "phase", checkpoint.Status.Phase)

	// Handle finalization
	if checkpoint.GetDeletionTimestamp() != nil {
		return r.handleFinalization(ctx, &checkpoint)
	}

	// Add finalizer if not present
	if !containsString(checkpoint.GetFinalizers(), trainingv1.HermesCheckpointFinalizer) {
		return r.addFinalizer(ctx, &checkpoint)
	}

	// Reconcile based on phase
	switch checkpoint.Status.Phase {
	case trainingv1.HermesCheckpointCreating:
		return r.reconcileCreating(ctx, &checkpoint)
	case trainingv1.HermesCheckpointReady:
		return r.reconcileReady(ctx, &checkpoint)
	case trainingv1.HermesCheckpointFailed:
		return reconcile.Result{}, nil
	default:
		return r.initializeCheckpoint(ctx, &checkpoint)
	}
}

func (r *HermesCheckpointReconciler) initializeCheckpoint(ctx context.Context, checkpoint *trainingv1.HermesCheckpoint) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Initializing new HermesCheckpoint", "name", checkpoint.Name)

	checkpoint.Status.Phase = trainingv1.HermesCheckpointCreating
	checkpoint.Status.CreatedAt = &metav1.Time{Time: time.Now()}

	if err := r.Update(ctx, checkpoint); err != nil {
		return reconcile.Result{}, err
	}

	r.Recorder.Event(checkpoint, "Normal", "Initialized", "HermesCheckpoint initialized")

	return reconcile.Result{RequeueAfter: time.Second * 5}, nil
}

func (r *HermesCheckpointReconciler) reconcileCreating(ctx context.Context, checkpoint *trainingv1.HermesCheckpoint) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Creating checkpoint", "name", checkpoint.Name, "job", checkpoint.Spec.JobName)

	// Simulate checkpoint creation
	// In real implementation, this would interact with Redis/MinIO

	checkpoint.Status.Phase = trainingv1.HermesCheckpointReady
	checkpoint.Status.Size = "128MB"
	checkpoint.Status.Replicas = 3

	if err := r.Update(ctx, checkpoint); err != nil {
		return reconcile.Result{}, err
	}

	r.Recorder.Event(checkpoint, "Normal", "Ready", "Checkpoint created successfully")

	return reconcile.Result{}, nil
}

func (r *HermesCheckpointReconciler) reconcileReady(ctx context.Context, checkpoint *trainingv1.HermesCheckpoint) (reconcile.Result, error) {
	// Checkpoint is ready, nothing to do
	return reconcile.Result{}, nil
}

func (r *HermesCheckpointReconciler) handleFinalization(ctx context.Context, checkpoint *trainingv1.HermesCheckpoint) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Finalizing HermesCheckpoint", "name", checkpoint.Name)

	// Cleanup storage (Redis/MinIO)

	checkpoint.SetFinalizers(removeString(checkpoint.GetFinalizers(), trainingv1.HermesCheckpointFinalizer))
	if err := r.Update(ctx, checkpoint); err != nil {
		return reconcile.Result{}, err
	}

	return reconcile.Result{}, nil
}

func (r *HermesCheckpointReconciler) addFinalizer(ctx context.Context, checkpoint *trainingv1.HermesCheckpoint) (reconcile.Result, error) {
	checkpoint.SetFinalizers(append(checkpoint.GetFinalizers(), trainingv1.HermesCheckpointFinalizer))
	if err := r.Update(ctx, checkpoint); err != nil {
		return reconcile.Result{}, err
	}

	return reconcile.Result{Requeue: true}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *HermesCheckpointReconciler) SetupWithManager(mgr manager.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&trainingv1.HermesCheckpoint{}).
		Complete(r)
}
