package controllers

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	trainingv1 "hermes-operator/api/v1"
)

// HermesJobReconciler reconciles a HermesJob object
type HermesJobReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

//+kubebuilder:rbac:groups=training.hermes.io,resources=hermesjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=training.hermes.io,resources=hermesjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=training.hermes.io,resources=hermesjobs/finalizers,verbs=update
//+kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=pods;services;persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="",resources=events,verbs=create;patch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *HermesJobReconciler) Reconcile(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
	log := log.FromContext(ctx)

	// Fetch the HermesJob instance
	var job trainingv1.HermesJob
	if err := r.Get(ctx, req.NamespacedName, &job); err != nil {
		if errors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			return reconcile.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return reconcile.Result{}, err
	}

	log.Info("Reconciling HermesJob", "name", job.Name, "phase", job.Status.Phase)

	// Handle finalization
	if job.GetDeletionTimestamp() != nil {
		return r.handleFinalization(ctx, &job)
	}

	// Add finalizer if not present
	if !containsString(job.GetFinalizers(), trainingv1.HermesJobFinalizer) {
		return r.addFinalizer(ctx, &job)
	}

	// Reconcile based on phase
	switch job.Status.Phase {
	case trainingv1.HermesJobPending:
		return r.reconcilePending(ctx, &job)
	case trainingv1.HermesJobScheduled:
		return r.reconcileScheduled(ctx, &job)
	case trainingv1.HermesJobRunning:
		return r.reconcileRunning(ctx, &job)
	case trainingv1.HermesJobRecovering:
		return r.reconcileRecovering(ctx, &job)
	case trainingv1.HermesJobCompleted, trainingv1.HermesJobFailed:
		return reconcile.Result{}, nil
	default:
		return r.initializeJob(ctx, &job)
	}
}

func (r *HermesJobReconciler) initializeJob(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Initializing new HermesJob", "name", job.Name)

	job.Status.Phase = trainingv1.HermesJobPending
	job.Status.StartTime = &metav1.Time{Time: time.Now()}

	if err := r.Update(ctx, job); err != nil {
		return reconcile.Result{}, err
	}

	r.Recorder.Event(job, corev1.EventTypeNormal, "Initialized", "HermesJob initialized")

	return reconcile.Result{RequeueAfter: time.Second * 5}, nil
}

func (r *HermesJobReconciler) reconcilePending(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Scheduling HermesJob", "name", job.Name)

	// Check if StatefulSet already exists
	var sts appsv1.StatefulSet
	stsName := fmt.Sprintf("hermes-job-%s", job.Name)
	if err := r.Get(ctx, types.NamespacedName{Name: stsName, Namespace: job.Namespace}, &sts); err != nil {
		if errors.IsNotFound(err) {
			// Create StatefulSet
			if err := r.createStatefulSet(ctx, job); err != nil {
				return reconcile.Result{}, err
			}
			r.Recorder.Event(job, corev1.EventTypeNormal, "Scheduled", "StatefulSet created")
		} else {
			return reconcile.Result{}, err
		}
	}

	// Check if StatefulSet is ready
	if sts.Status.ReadyReplicas == job.Spec.NumGPUs {
		job.Status.Phase = trainingv1.HermesJobRunning
		job.Status.CurrentGPUs = job.Spec.NumGPUs
		if err := r.Update(ctx, job); err != nil {
			return reconcile.Result{}, err
		}
		r.Recorder.Event(job, corev1.EventTypeNormal, "Running", "Job is running")
	}

	return reconcile.Result{RequeueAfter: time.Second * 10}, nil
}

func (r *HermesJobReconciler) reconcileRunning(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	log := log.FromContext(ctx)

	// Check for failed pods
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(job.Namespace), client.MatchingLabels{"hermes-job": job.Name}); err != nil {
		return reconcile.Result{}, err
	}

	failedPods := 0
	for _, pod := range pods.Items {
		if pod.Status.Phase == corev1.PodFailed {
			failedPods++
		}
	}

	if failedPods > 0 {
		log.Info("Detected failed pods, initiating recovery", "name", job.Name, "failedPods", failedPods)
		job.Status.Phase = trainingv1.HermesJobRecovering
		job.Status.RecoveryCount++
		if err := r.Update(ctx, job); err != nil {
			return reconcile.Result{}, err
		}
		r.Recorder.Event(job, corev1.EventTypeWarning, "Recovering", fmt.Sprintf("Recovering from %d failed pods", failedPods))
	}

	return reconcile.Result{RequeueAfter: time.Second * 15}, nil
}

func (r *HermesJobReconciler) reconcileRecovering(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	log := log.FromContext(ctx)

	// Check if StatefulSet has recovered
	var sts appsv1.StatefulSet
	stsName := fmt.Sprintf("hermes-job-%s", job.Name)
	if err := r.Get(ctx, types.NamespacedName{Name: stsName, Namespace: job.Namespace}, &sts); err != nil {
		return reconcile.Result{}, err
	}

	if sts.Status.ReadyReplicas == job.Spec.NumGPUs {
		log.Info("Recovery complete", "name", job.Name)
		job.Status.Phase = trainingv1.HermesJobRunning
		if err := r.Update(ctx, job); err != nil {
			return reconcile.Result{}, err
		}
		r.Recorder.Event(job, corev1.EventTypeNormal, "Recovered", "Job has recovered")
	}

	return reconcile.Result{RequeueAfter: time.Second * 5}, nil
}

func (r *HermesJobReconciler) reconcileScheduled(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	// Transition to running when pods are ready
	return r.reconcileRunning(ctx, job)
}

func (r *HermesJobReconciler) createStatefulSet(ctx context.Context, job *trainingv1.HermesJob) error {
	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("hermes-job-%s", job.Name),
			Namespace: job.Namespace,
			Labels: map[string]string{
				"hermes-job": job.Name,
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas: &job.Spec.NumGPUs,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"hermes-job": job.Name,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"hermes-job": job.Name,
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy: job.Spec.RestartPolicy,
					Containers: []corev1.Container{
						{
							Name:  "trainer",
							Image: "hermes-trainer:latest",
							Env: []corev1.EnvVar{
								{Name: "JOB_NAME", Value: job.Name},
								{Name: "CHECKPOINT_INTERVAL", Value: fmt.Sprintf("%d", job.Spec.CheckpointInterval)},
							},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									"nvidia.com/gpu": resource.MustParse(fmt.Sprintf("%d", job.Spec.NumGPUs)),
								},
							},
						},
					},
				},
			},
		},
	}

	return ctrl.SetControllerReference(job, sts, r.Scheme)
}

func (r *HermesJobReconciler) handleFinalization(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Finalizing HermesJob", "name", job.Name)

	// Cleanup resources
	stsName := fmt.Sprintf("hermes-job-%s", job.Name)
	var sts appsv1.StatefulSet
	if err := r.Get(ctx, types.NamespacedName{Name: stsName, Namespace: job.Namespace}, &sts); err == nil {
		if err := r.Delete(ctx, &sts); err != nil {
			return reconcile.Result{}, err
		}
	}

	// Remove finalizer
	job.SetFinalizers(removeString(job.GetFinalizers(), trainingv1.HermesJobFinalizer))
	if err := r.Update(ctx, job); err != nil {
		return reconcile.Result{}, err
	}

	return reconcile.Result{}, nil
}

func (r *HermesJobReconciler) addFinalizer(ctx context.Context, job *trainingv1.HermesJob) (reconcile.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Adding finalizer", "name", job.Name)

	job.SetFinalizers(append(job.GetFinalizers(), trainingv1.HermesJobFinalizer))
	if err := r.Update(ctx, job); err != nil {
		return reconcile.Result{}, err
	}

	return reconcile.Result{Requeue: true}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *HermesJobReconciler) SetupWithManager(mgr manager.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&trainingv1.HermesJob{}).
		Owns(&appsv1.StatefulSet{}).
		WithOptions(controller.Options{MaxConcurrentReconciles: 2}).
		Complete(r)
}

// Helper functions
func containsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func removeString(slice []string, s string) []string {
	var result []string
	for _, item := range slice {
		if item != s {
			result = append(result, item)
		}
	}
	return result
}
