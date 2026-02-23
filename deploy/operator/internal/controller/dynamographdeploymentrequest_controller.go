/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"text/template"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	sigsyaml "sigs.k8s.io/yaml"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/observability"
)

const (
	// Condition types
	ConditionTypeValidation      = "Validation"
	ConditionTypeProfiling       = "Profiling"
	ConditionTypeSpecGenerated   = "SpecGenerated"
	ConditionTypeDeploymentReady = "DeploymentReady"

	// Event reasons
	EventReasonInitialized          = "Initialized"
	EventReasonValidationFailed     = "ValidationFailed"
	EventReasonProfilingJobCreated  = "ProfilingJobCreated"
	EventReasonProfilingJobFailed   = "ProfilingJobFailed"
	EventReasonAIConfiguratorFailed = "AIConfiguratorFailed"
	EventReasonSpecGenerated        = "SpecGenerated"
	EventReasonSpecChangeRejected   = "SpecChangeRejected"
	EventReasonDeploymentCreated    = "DeploymentCreated"
	EventReasonDeploymentReady      = "DeploymentReady"
	EventReasonDeploymentDegraded   = "DeploymentDegraded"
	EventReasonDeploymentDeleted    = "DeploymentDeleted"

	// Label keys
	LabelApp           = "app"
	LabelDGDR          = "dgdr"
	LabelDGDRName      = "dgdr.nvidia.com/name"
	LabelDGDRNamespace = "dgdr.nvidia.com/namespace"
	LabelManagedBy     = "nvidia.com/managed-by"

	// Label values
	LabelValueDynamoProfiler = "dynamo-profiler"
	LabelValueAICProfiler    = "aic-profiler"
	LabelValueDynamoOperator = "dynamo-operator"

	// Job naming
	JobNamePrefixOnline = "profile-online-"
	JobNamePrefixAIC    = "profile-aic-"

	// Container names
	ContainerNameProfiler     = "profiler"
	ContainerNameOutputCopier = "output-copier"

	// ServiceAccount
	ServiceAccountProfilingJob = "dgdr-profiling-job"

	// ConfigMap naming
	ConfigMapOutputPrefix = "dgdr-output-"

	// Annotation keys
	AnnotationAdditionalResources = "dgdr.nvidia.com/additional-resources"

	// Annotation keys for v1alpha1 backward compatibility
	annDGDRConfigMapRef     = "nvidia.com/dgdr-config-map-ref"
	annDGDROutputPVC        = "nvidia.com/dgdr-output-pvc"
	annDGDRProfilingConfig  = "nvidia.com/dgdr-profiling-config"
	annDGDRDeployOverrides  = "nvidia.com/dgdr-deployment-overrides"
	annDGDRDeploymentStatus = "nvidia.com/dgdr-deployment-status"

	// Size limits
	MaxAnnotationSize = 250000 // ~250KB, below K8s 256KB limit

	// Sidecar image
	SidecarImage = "bitnami/kubectl:latest"

	// Volume names
	VolumeNameProfilingConfig = "profiling-config"
	VolumeNameProfilingOutput = "profiling-output"
	VolumeNameModelCache      = "model-cache"

	// Volume paths
	ProfilingOutputPath        = "/data"
	ProfilingOutputFile        = "config_with_planner.yaml"
	ProfilingOutputFileMocker  = "mocker_config_with_planner.yaml"
	ProfilingConfigPath        = "/config"
	ProfilingConfigFile        = "disagg.yaml"
	DefaultModelCacheMountPath = "/opt/model-cache"

	// Command line arguments
	ArgModel   = "--model"
	ArgBackend = "--backend"
	ArgTTFT    = "--ttft"
	ArgITL     = "--itl"
	ArgConfig  = "--config"

	// Messages
	MessageInitialized               = "DGDR initialized successfully"
	MessageProfilingJobCreated       = "Profiling job created"
	MessageAICProfilingJobCreated    = "AIC profiling job created"
	MessageProfilingInProgress       = "Profiling is in progress"
	MessageSpecGenerated             = "DynamoGraphDeployment spec generated successfully"
	MessageSpecAvailable             = "Generated spec is available in status.profilingResults.selectedConfig"
	MessageDeploymentCreated         = "DynamoGraphDeployment %s created successfully"
	MessageDeploymentReady           = "DynamoGraphDeployment %s is ready"
	MessageDeploymentDegraded        = "DynamoGraphDeployment %s degraded from Ready to %s"
	MessageDeploymentDeleted         = "DGD %s was deleted. DGDR will not recreate it. Delete this DGDR and create a new one to redeploy."
	MessageInvalidState              = "Invalid state"
	MessageSpecChangeRejected        = "Cannot modify spec in phase '%s'. DynamoGraphDeploymentRequest is immutable once profiling starts. Create a new resource with a different name instead."
	MessageJobCreationFailed         = "JobCreationFailed"
	MessageDeploymentCreationFailed  = "DeploymentCreationFailed"
	MessageResultsRetrievalFailed    = "ResultsRetrievalFailed"
	MessageGenerationFailed          = "GenerationFailed"
	MessageAIConfiguratorCheckFailed = "AIConfiguratorCheckFailed"
	MessageProfilingCheckFailed      = "ProfilingCheckFailed"
	MessageConfigMapNotFound         = "ConfigMap %s not found in namespace %s"
	MessageConfigMapKeyNotFound      = "key %s not found in ConfigMap %s"
	MessageModelCachePVCNotFound     = "model cache PVC %s not found in namespace %s"

	// Validation messages
	ValidationErrorModelRequired  = "model is required"
	ValidationErrorITLPositive    = "sla.itl must be positive"
	ValidationErrorTTFTPositive   = "sla.ttft must be positive"
	ValidationErrorInvalidBackend = "invalid backend: %s (must be vllm, sglang, or trtllm)"

	// Valid backend values
	BackendVLLM   = "vllm"
	BackendSGLang = "sglang"
	BackendTRTLLM = "trtllm"

	// Profiling config field names
	ConfigKeyDeployment       = "deployment"
	ConfigKeyModelCache       = "modelCache"
	ConfigKeyPVCName          = "pvcName"
	ConfigKeyPVCPath          = "pvcPath"
	ConfigKeyMountPath        = "mountPath"
	ConfigKeyHardware         = "hardware"
	ConfigKeyEngine           = "engine"
	ConfigKeyOutputDir        = "output_dir"
	ConfigKeyNumGpusPerNode   = "numGpusPerNode"
	ConfigKeyGPUModel         = "gpuModel"
	ConfigKeyGPUVramMib       = "gpuVramMib"
	ConfigKeySystem           = "system"
	ConfigKeyMinNumGpusPerEng = "minNumGpusPerEngine"
	ConfigKeyMaxNumGpusPerEng = "maxNumGpusPerEngine"
	ConfigKeyBackend          = "backend"
	ConfigKeyConfig           = "config"
	ConfigKeyNamespace        = "namespace"
	ConfigKeyModel            = "model"
	ConfigKeyDGDImage         = "dgd_image"
	ConfigKeySLA              = "sla"
	ConfigKeySearchStrategy   = "searchStrategy"
)

// shell script template for the output copier sidecar
const sidecarScriptTemplate = `
set -e
set -o pipefail

# Wait for profiler container to terminate (no timeout - profiling can take hours)
echo "Waiting for profiler to complete..."
START_TIME=$(date +%s)
LAST_PROGRESS_LOG=$START_TIME
PROGRESS_INTERVAL=300

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  # Log progress every 5 minutes
  if [ $((CURRENT_TIME - LAST_PROGRESS_LOG)) -ge $PROGRESS_INTERVAL ]; then
    echo "Still waiting... ($(($ELAPSED / 60)) minutes elapsed)"
    LAST_PROGRESS_LOG=$CURRENT_TIME
  fi

  # Check if profiler container terminated
  CONTAINER_STATUS=$(kubectl get pod $HOSTNAME -n {{.Namespace}} -o jsonpath='{.status.containerStatuses[?(@.name=="profiler")].state}' 2>/dev/null || echo "")
  if echo "$CONTAINER_STATUS" | grep -q "terminated"; then
    echo "Profiler terminated (ran for $(($ELAPSED / 60)) minutes)"
    break
  fi
  sleep 5
done

# Check profiler status file (2 minute timeout)
echo "Checking profiler status..."
STATUS_FILE="{{.OutputPath}}/profiler_status.yaml"
TIMEOUT=120
CHECK_START=$(date +%s)

# Wait for status file to exist
while [ ! -f "$STATUS_FILE" ]; do
  ELAPSED=$(($(date +%s) - CHECK_START))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: Status file not found after ${TIMEOUT}s"
    exit 1
  fi
  sleep 2
done

# Read and parse status from YAML file
STATUS=$(grep "^status:" "$STATUS_FILE" | awk '{print $2}' | tr -d '"' | tr -d "'")

if [ -z "$STATUS" ]; then
  echo "ERROR: Invalid status file format"
  exit 1
fi

# Check status value
case "$STATUS" in
  success)
    MESSAGE=$(grep "^message:" "$STATUS_FILE" | sed 's/^message: *//' | tr -d '"' | tr -d "'")
    echo "Profiler succeeded: $MESSAGE"
    ;;
  failed)
    ERROR=$(grep "^error:" "$STATUS_FILE" | sed 's/^error: *//' | tr -d '"' | tr -d "'")
    MESSAGE=$(grep "^message:" "$STATUS_FILE" | sed 's/^message: *//' | tr -d '"' | tr -d "'")
    echo "ERROR: Profiler failed: ${ERROR:-$MESSAGE}"
    exit 1
    ;;
  running)
    echo "ERROR: Profiler still running (unexpected)"
    exit 1
    ;;
  *)
    echo "ERROR: Unknown status: $STATUS"
    exit 1
    ;;
esac

echo "Creating ConfigMap..."

# Start building ConfigMap YAML with DGD spec
cat >/tmp/cm.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{.ConfigMapName}}
  namespace: {{.Namespace}}
  labels:
    dgdr.nvidia.com/name: {{.DGDRName}}
    nvidia.com/managed-by: dynamo-operator
data:
  {{.OutputFile}}: |
EOF
sed 's/^/    /' {{.OutputPath}}/{{.OutputFile}} >> /tmp/cm.yaml

# Add mocker config (profiler always generates both real and mocker configs)
if [ -f {{.OutputPath}}/{{.MockerOutputFile}} ]; then
  echo "  {{.MockerOutputFile}}: |" >> /tmp/cm.yaml
  sed 's/^/    /' {{.OutputPath}}/{{.MockerOutputFile}} >> /tmp/cm.yaml
  echo "Added mocker config to ConfigMap"
fi

# Add profiler status file for debugging
if [ -f {{.OutputPath}}/profiler_status.yaml ]; then
  echo "  profiler_status.yaml: |" >> /tmp/cm.yaml
  sed 's/^/    /' {{.OutputPath}}/profiler_status.yaml >> /tmp/cm.yaml
fi

# Note: Profiling data (raw_data.npz converted to JSON) is included in the
# generated DGD YAML as a separate ConfigMap by the profiler, no need to add it here

kubectl apply -f /tmp/cm.yaml
echo "Saved profiling output to ConfigMap {{.ConfigMapName}}"
`

// DynamoGraphDeploymentRequestReconciler reconciles a DynamoGraphDeploymentRequest object
type DynamoGraphDeploymentRequestReconciler struct {
	client.Client
	Recorder record.EventRecorder
	Config   commonController.Config

	// RBACMgr handles RBAC setup for profiling jobs
	RBACManager RBACManager
}

// RBACManager interface for managing RBAC resources
type RBACManager interface {
	EnsureServiceAccountWithRBAC(ctx context.Context, targetNamespace, serviceAccountName, clusterRoleName string) error
}

// --------------------------------------------------------------------------
// Annotation helper types and functions for v1alpha1 backward compatibility
// --------------------------------------------------------------------------

// deploymentLifecycle tracks DGD deployment state in an annotation.
type deploymentLifecycle struct {
	Namespace string `json:"namespace,omitempty"`
	State     string `json:"state,omitempty"`
	Created   bool   `json:"created,omitempty"`
}

func getDeploymentLifecycle(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) *deploymentLifecycle {
	if dgdr.Annotations == nil {
		return nil
	}
	raw, ok := dgdr.Annotations[annDGDRDeploymentStatus]
	if !ok || raw == "" {
		return nil
	}
	var dl deploymentLifecycle
	if err := json.Unmarshal([]byte(raw), &dl); err != nil {
		return nil
	}
	return &dl
}

func setDeploymentLifecycle(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, dl *deploymentLifecycle) {
	if dgdr.Annotations == nil {
		dgdr.Annotations = make(map[string]string)
	}
	if dl == nil {
		delete(dgdr.Annotations, annDGDRDeploymentStatus)
		return
	}
	data, _ := json.Marshal(dl)
	dgdr.Annotations[annDGDRDeploymentStatus] = string(data)
}

// getConfigMapRef reads the v1alpha1 ConfigMapKeySelector from an annotation.
func getConfigMapRef(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) *nvidiacomv1alpha1.ConfigMapKeySelector {
	if dgdr.Annotations == nil {
		return nil
	}
	raw, ok := dgdr.Annotations[annDGDRConfigMapRef]
	if !ok || raw == "" {
		return nil
	}
	var ref nvidiacomv1alpha1.ConfigMapKeySelector
	if err := json.Unmarshal([]byte(raw), &ref); err != nil {
		return nil
	}
	return &ref
}

// getOutputPVC reads the output PVC name from an annotation.
func getOutputPVC(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) string {
	if dgdr.Annotations == nil {
		return ""
	}
	return dgdr.Annotations[annDGDROutputPVC]
}

// dgdOverrides holds metadata overrides for the generated DGD.
type dgdOverrides struct {
	Name        string            `json:"name,omitempty"`
	Namespace   string            `json:"namespace,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
	Annotations map[string]string `json:"annotations,omitempty"`
}

// getDGDOverrides extracts DGD metadata overrides from v1beta1 Overrides.DGD
// or from the backward-compat annotation.
func getDGDOverrides(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) *dgdOverrides {
	// Check Overrides.DGD (v1beta1 native path)
	if dgdr.Spec.Overrides != nil && dgdr.Spec.Overrides.DGD != nil && dgdr.Spec.Overrides.DGD.Raw != nil {
		var obj map[string]interface{}
		if err := json.Unmarshal(dgdr.Spec.Overrides.DGD.Raw, &obj); err == nil {
			ov := &dgdOverrides{}
			if m, ok := obj["metadata"].(map[string]interface{}); ok {
				ov.Name, _ = m["name"].(string)
				ov.Namespace, _ = m["namespace"].(string)
				if labels, ok := m["labels"].(map[string]interface{}); ok {
					ov.Labels = make(map[string]string)
					for k, v := range labels {
						if sv, ok := v.(string); ok {
							ov.Labels[k] = sv
						}
					}
				}
				if anns, ok := m["annotations"].(map[string]interface{}); ok {
					ov.Annotations = make(map[string]string)
					for k, v := range anns {
						if sv, ok := v.(string); ok {
							ov.Annotations[k] = sv
						}
					}
				}
			}
			return ov
		}
	}
	// Fallback: annotation for v1alpha1 backward compat
	if dgdr.Annotations != nil {
		if raw, ok := dgdr.Annotations[annDGDRDeployOverrides]; ok && raw != "" {
			var ov dgdOverrides
			if err := json.Unmarshal([]byte(raw), &ov); err == nil {
				return &ov
			}
		}
	}
	return nil
}

// isMockerEnabled checks whether mocker mode is enabled in the v1beta1 spec.
func isMockerEnabled(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) bool {
	return dgdr.Spec.Features != nil && dgdr.Spec.Features.Mocker != nil && dgdr.Spec.Features.Mocker.Enabled
}

// getOrCreateMap returns the sub-map for key, creating it if absent.
func getOrCreateMap(m map[string]interface{}, key string) map[string]interface{} {
	if v, ok := m[key]; ok {
		if vm, ok := v.(map[string]interface{}); ok {
			return vm
		}
	}
	nm := make(map[string]interface{})
	m[key] = nm
	return nm
}

// --------------------------------------------------------------------------
// Reconciler interface methods
// --------------------------------------------------------------------------

// GetRecorder implements commonController.Reconciler interface
func (r *DynamoGraphDeploymentRequestReconciler) GetRecorder() record.EventRecorder {
	return r.Recorder
}

// FinalizeResource implements commonController.Finalizer interface
func (r *DynamoGraphDeploymentRequestReconciler) FinalizeResource(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)

	logger.Info("DGDR finalized successfully", "name", dgdr.Name)
	return nil
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentrequests/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch

// Reconcile handles the reconciliation loop for DynamoGraphDeploymentRequest
func (r *DynamoGraphDeploymentRequestReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Reconciling DynamoGraphDeploymentRequest", "name", req.Name, "namespace", req.Namespace)

	// Fetch the DGDR instance as v1beta1
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	if err := r.Get(ctx, req.NamespacedName, dgdr); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("DGDR resource not found, ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get DGDR")
		return ctrl.Result{}, err
	}

	// Handle finalizer using common function
	finalized, err := commonController.HandleFinalizer(ctx, dgdr, r.Client, r)
	if err != nil {
		return ctrl.Result{}, err
	}
	if finalized {
		return ctrl.Result{}, nil
	}

	// Check for spec changes (immutability enforcement)
	if dgdr.Status.ObservedGeneration > 0 && dgdr.Status.ObservedGeneration != dgdr.Generation {
		phase := dgdr.Status.Phase
		if phase == nvidiacomv1beta1.DGDRPhaseProfiling || phase == nvidiacomv1beta1.DGDRPhaseDeploying ||
			phase == nvidiacomv1beta1.DGDRPhaseReady || phase == nvidiacomv1beta1.DGDRPhaseDeployed {
			logger.Info("Spec change detected in immutable phase",
				"phase", phase,
				"observedGeneration", dgdr.Status.ObservedGeneration,
				"currentGeneration", dgdr.Generation)

			r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonSpecChangeRejected,
				fmt.Sprintf(MessageSpecChangeRejected, phase))

			return ctrl.Result{}, nil
		}
	}

	// State machine using Phase
	switch dgdr.Status.Phase {
	case "", nvidiacomv1beta1.DGDRPhasePending:
		return r.handlePendingState(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseProfiling:
		return r.handleProfilingState(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseReady:
		return r.handleReadyState(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseDeploying:
		return r.handleDeployingState(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseDeployed:
		return r.handleDeployedState(ctx, dgdr)
	case nvidiacomv1beta1.DGDRPhaseFailed:
		return r.handleFailedState(ctx, dgdr)
	default:
		logger.Info("Unknown phase", "phase", dgdr.Status.Phase)
		return r.updatePhaseAndRequeue(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, MessageInvalidState)
	}
}

// handlePendingState handles first-time initialization (was handleInitialState)
// and starts the profiling process.
func (r *DynamoGraphDeploymentRequestReconciler) handlePendingState(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling pending state", "name", dgdr.Name)

	// First-time initialization (merged from handleInitialState)
	if dgdr.Status.ObservedGeneration == 0 {
		if err := r.validateSpec(ctx, dgdr); err != nil {
			r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonValidationFailed, err.Error())
			return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, ConditionTypeValidation, metav1.ConditionFalse, EventReasonValidationFailed, err.Error())
		}

		dgdr.Status.ObservedGeneration = dgdr.Generation
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonInitialized, MessageInitialized)
		// Fall through to profiling job creation
	}

	// Create profiling job (online or AIC)
	if err := r.createProfilingJob(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonProfilingJobFailed, err.Error())
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, ConditionTypeProfiling, metav1.ConditionFalse, MessageJobCreationFailed, err.Error())
	}

	// Record event with appropriate message
	if isOnlineProfiling(dgdr) {
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonProfilingJobCreated, MessageProfilingJobCreated)
	} else {
		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonProfilingJobCreated, MessageAICProfilingJobCreated)
	}

	// Update to Profiling phase
	return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseProfiling, ConditionTypeProfiling, metav1.ConditionFalse, "ProfilingRunning", MessageProfilingInProgress)
}

// handleProfilingState monitors profiling progress and generates spec when complete
func (r *DynamoGraphDeploymentRequestReconciler) handleProfilingState(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling profiling state", "name", dgdr.Name)

	completed, err := r.checkProfilingJobStatus(ctx, dgdr)
	if err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageProfilingCheckFailed, err.Error())
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, ConditionTypeProfiling, metav1.ConditionFalse, "ProfilingFailed", err.Error())
	}

	if !completed {
		logger.Info("Profiling job still running", "name", dgdr.Name)
		return ctrl.Result{}, nil
	}

	// Mark profiling as completed successfully
	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:               ConditionTypeProfiling,
		Status:             metav1.ConditionTrue,
		ObservedGeneration: dgdr.Generation,
		Reason:             "ProfilingCompleted",
		Message:            "Profiling job completed successfully",
	})

	// Retrieve profiling results and generate spec
	if err := r.generateDGDSpec(ctx, dgdr); err != nil {
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageGenerationFailed, err.Error())
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseFailed, ConditionTypeSpecGenerated, metav1.ConditionFalse, MessageGenerationFailed, err.Error())
	}

	r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonSpecGenerated, MessageSpecGenerated)

	// Create additional resources (ConfigMaps) immediately after profiling
	targetNamespace := dgdr.Namespace
	overrides := getDGDOverrides(dgdr)
	if overrides != nil && overrides.Namespace != "" {
		targetNamespace = overrides.Namespace
	}
	if err := r.createAdditionalResources(ctx, dgdr, targetNamespace); err != nil {
		logger.Error(err, "Failed to create additional resources after profiling")
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, "ConfigMapCreationFailed",
			fmt.Sprintf("Failed to create ConfigMaps from profiling output: %v", err))
	}

	// If autoApply is enabled, transition to Deploying
	if dgdr.Spec.AutoApply {
		logger.Info("AutoApply enabled, transitioning to Deploying phase")
		return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseDeploying, ConditionTypeSpecGenerated, metav1.ConditionTrue, EventReasonSpecGenerated, MessageSpecGenerated)
	}

	// Otherwise, transition to Ready
	return r.updatePhaseWithCondition(ctx, dgdr, nvidiacomv1beta1.DGDRPhaseReady, ConditionTypeSpecGenerated, metav1.ConditionTrue, EventReasonSpecGenerated, MessageSpecAvailable)
}

// handleReadyState handles DGDR in Ready phase.
// Ready means profiling completed and spec is available but no DGD has been
// created (autoApply=false) or a previously deployed DGD was deleted.
func (r *DynamoGraphDeploymentRequestReconciler) handleReadyState(_ context.Context, _ *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	// Terminal-ish state — user reviews results. Nothing to do.
	return ctrl.Result{}, nil
}

// handleDeployedState handles DGDR in Deployed phase — DGD exists and was healthy.
func (r *DynamoGraphDeploymentRequestReconciler) handleDeployedState(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	dl := getDeploymentLifecycle(dgdr)
	dgdNamespace := dgdr.Namespace
	if dl != nil && dl.Namespace != "" {
		dgdNamespace = dl.Namespace
	}

	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	err := r.Get(ctx, types.NamespacedName{Name: dgdr.Status.DGDName, Namespace: dgdNamespace}, dgd)

	if apierrors.IsNotFound(err) {
		return r.handleDGDDeleted(ctx, dgdr)
	}
	if err != nil {
		return ctrl.Result{}, err
	}

	// Update deployment lifecycle annotation with current DGD state
	needsAnnotationUpdate := false
	if dl != nil && dl.State != string(dgd.Status.State) {
		dl.State = string(dgd.Status.State)
		setDeploymentLifecycle(dgdr, dl)
		needsAnnotationUpdate = true
	}

	// If DGD degraded from Ready
	if dgd.Status.State != nvidiacomv1alpha1.DGDStateSuccessful {
		logger.Info("DGD degraded, transitioning to Deploying", "dgdState", dgd.Status.State)
		dgdr.SetPhase(nvidiacomv1beta1.DGDRPhaseDeploying)

		r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonDeploymentDegraded,
			fmt.Sprintf(MessageDeploymentDegraded, dgd.Name, string(dgd.Status.State)))

		meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
			Type:    ConditionTypeDeploymentReady,
			Status:  metav1.ConditionFalse,
			Reason:  EventReasonDeploymentDegraded,
			Message: fmt.Sprintf("Deployment degraded to %s", string(dgd.Status.State)),
		})

		savedStatus := *dgdr.Status.DeepCopy()
		if needsAnnotationUpdate {
			if err := r.Update(ctx, dgdr); err != nil {
				return ctrl.Result{}, err
			}
			dgdr.Status = savedStatus
		}
		return ctrl.Result{}, r.Status().Update(ctx, dgdr)
	}

	if needsAnnotationUpdate {
		if err := r.Update(ctx, dgdr); err != nil {
			return ctrl.Result{}, err
		}
	}
	return ctrl.Result{RequeueAfter: 30 * time.Second}, r.Status().Update(ctx, dgdr)
}

// handleDeployingState handles DGD creation and monitors deployment
func (r *DynamoGraphDeploymentRequestReconciler) handleDeployingState(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("Handling deploying state", "name", dgdr.Name)

	if !dgdr.Spec.AutoApply {
		logger.Info("AutoApply not enabled, transitioning to Ready")
		dgdr.SetPhase(nvidiacomv1beta1.DGDRPhaseReady)
		return ctrl.Result{}, r.Status().Update(ctx, dgdr)
	}

	// Check if we need to create DGD
	dl := getDeploymentLifecycle(dgdr)
	if dl == nil || !dl.Created {
		return r.createDGD(ctx, dgdr)
	}

	// DGD was already created, check its status
	dgdNamespace := dgdr.Namespace
	if dl.Namespace != "" {
		dgdNamespace = dl.Namespace
	}

	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      dgdr.Status.DGDName,
		Namespace: dgdNamespace,
	}, dgd)

	if apierrors.IsNotFound(err) {
		return r.handleDGDDeleted(ctx, dgdr)
	}
	if err != nil {
		return ctrl.Result{}, err
	}

	// Update deployment lifecycle annotation
	dl.State = string(dgd.Status.State)
	setDeploymentLifecycle(dgdr, dl)
	if err := r.Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}

	// Check if DGD is Ready → transition to Deployed
	if dgd.Status.State == nvidiacomv1alpha1.DGDStateSuccessful {
		logger.Info("DGD is Ready, transitioning to Deployed phase")
		dgdr.SetPhase(nvidiacomv1beta1.DGDRPhaseDeployed)

		r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonDeploymentReady,
			fmt.Sprintf(MessageDeploymentReady, dgd.Name))

		meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
			Type:    ConditionTypeDeploymentReady,
			Status:  metav1.ConditionTrue,
			Reason:  EventReasonDeploymentReady,
			Message: fmt.Sprintf(MessageDeploymentReady, dgd.Name),
		})
	}

	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// handleDGDDeleted handles the case when auto-created DGD is deleted by user
func (r *DynamoGraphDeploymentRequestReconciler) handleDGDDeleted(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGD was deleted by user, transitioning to Ready phase")

	dgdName := dgdr.Status.DGDName
	dgdr.SetPhase(nvidiacomv1beta1.DGDRPhaseReady)

	r.Recorder.Event(dgdr, corev1.EventTypeWarning, EventReasonDeploymentDeleted,
		fmt.Sprintf(MessageDeploymentDeleted, dgdName))

	dgdr.Status.DGDName = ""
	setDeploymentLifecycle(dgdr, nil)

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:    ConditionTypeDeploymentReady,
		Status:  metav1.ConditionFalse,
		Reason:  EventReasonDeploymentDeleted,
		Message: "Deployment was deleted by user. Create a new DGDR to redeploy.",
	})

	// Update annotations (clearing deployment lifecycle) and then status.
	// Save status before r.Update because the API server response overwrites local status changes.
	savedStatus := *dgdr.Status.DeepCopy()
	if err := r.Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}
	dgdr.Status = savedStatus
	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// createDGD creates a DynamoGraphDeployment with the generated spec
func (r *DynamoGraphDeploymentRequestReconciler) createDGD(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Extract DGD from ProfilingResults.SelectedConfig
	if dgdr.Status.ProfilingResults == nil || dgdr.Status.ProfilingResults.SelectedConfig == nil {
		return ctrl.Result{}, fmt.Errorf("profilingResults.selectedConfig is not set")
	}

	generatedDGD := &nvidiacomv1alpha1.DynamoGraphDeployment{}

	selectedConfig := dgdr.Status.ProfilingResults.SelectedConfig
	if selectedConfig.Object != nil {
		var ok bool
		generatedDGD, ok = selectedConfig.Object.(*nvidiacomv1alpha1.DynamoGraphDeployment)
		if !ok {
			return ctrl.Result{}, fmt.Errorf("selectedConfig.Object is not a DynamoGraphDeployment")
		}
	} else if selectedConfig.Raw != nil {
		if err := yaml.Unmarshal(selectedConfig.Raw, generatedDGD); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to unmarshal selected config: %w", err)
		}
	} else {
		return ctrl.Result{}, fmt.Errorf("selectedConfig has neither Object nor Raw set")
	}

	// Determine DGD name and namespace
	dgdName := generatedDGD.Name
	dgdNamespace := dgdr.Namespace

	// Apply overrides
	overrides := getDGDOverrides(dgdr)
	if overrides != nil {
		if overrides.Name != "" {
			dgdName = overrides.Name
		}
		if overrides.Namespace != "" {
			dgdNamespace = overrides.Namespace
		}
	}

	// Build labels (start with generated DGD's labels)
	labels := make(map[string]string)
	if generatedDGD.Labels != nil {
		for k, v := range generatedDGD.Labels {
			labels[k] = v
		}
	}
	labels[LabelDGDRName] = dgdr.Name
	labels[LabelDGDRNamespace] = dgdr.Namespace
	labels[LabelManagedBy] = LabelValueDynamoOperator

	// Merge custom labels from overrides
	if overrides != nil && overrides.Labels != nil {
		for k, v := range overrides.Labels {
			labels[k] = v
		}
	}

	// Build annotations (start with generated DGD's annotations)
	annotations := make(map[string]string)
	if generatedDGD.Annotations != nil {
		for k, v := range generatedDGD.Annotations {
			annotations[k] = v
		}
	}
	if overrides != nil && overrides.Annotations != nil {
		for k, v := range overrides.Annotations {
			annotations[k] = v
		}
	}

	// Create DGD from generated deployment
	dgd := &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        dgdName,
			Namespace:   dgdNamespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: generatedDGD.Spec,
	}

	logger.Info("Creating DynamoGraphDeployment", "name", dgdName, "namespace", dgdNamespace)

	dlState := &deploymentLifecycle{
		Namespace: dgdNamespace,
		State:     string(nvidiacomv1alpha1.DGDStatePending),
		Created:   true,
	}

	if err := r.Create(ctx, dgd); err != nil {
		if apierrors.IsAlreadyExists(err) {
			logger.Info("DGD already exists, updating status")
			dgdr.Status.DGDName = dgdName
			setDeploymentLifecycle(dgdr, dlState)
			savedStatus := *dgdr.Status.DeepCopy()
			if err := r.Update(ctx, dgdr); err != nil {
				return ctrl.Result{}, err
			}
			dgdr.Status = savedStatus
			return ctrl.Result{}, r.Status().Update(ctx, dgdr)
		}
		r.Recorder.Event(dgdr, corev1.EventTypeWarning, MessageDeploymentCreationFailed, err.Error())
		return ctrl.Result{}, err
	}

	// Update status
	dgdr.Status.DGDName = dgdName
	setDeploymentLifecycle(dgdr, dlState)

	r.Recorder.Event(dgdr, corev1.EventTypeNormal, EventReasonDeploymentCreated,
		fmt.Sprintf(MessageDeploymentCreated, dgdName))

	meta.SetStatusCondition(&dgdr.Status.Conditions, metav1.Condition{
		Type:    ConditionTypeDeploymentReady,
		Status:  metav1.ConditionFalse,
		Reason:  EventReasonDeploymentCreated,
		Message: fmt.Sprintf("DGD %s created, waiting for Ready", dgdName),
	})

	logger.Info("DynamoGraphDeployment created successfully", "name", dgdName)

	// Update annotations first, then status.
	// Save status before r.Update because the API server response overwrites local status changes.
	savedStatus := *dgdr.Status.DeepCopy()
	if err := r.Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}
	dgdr.Status = savedStatus
	return ctrl.Result{}, r.Status().Update(ctx, dgdr)
}

// createAdditionalResources creates ConfigMaps from the profiling output that should be deployed alongside the DGD
func (r *DynamoGraphDeploymentRequestReconciler) createAdditionalResources(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, targetNamespace string) error {
	logger := log.FromContext(ctx)

	if dgdr.Annotations == nil {
		return nil
	}

	resourcesYAML, exists := dgdr.Annotations[AnnotationAdditionalResources]
	if !exists || resourcesYAML == "" {
		return nil
	}

	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader([]byte(resourcesYAML)), 4096)
	resourceCount := 0

	for {
		obj := &unstructured.Unstructured{}
		if err := decoder.Decode(obj); err != nil {
			if err == io.EOF {
				break
			}
			logger.Error(err, "Failed to decode resource, skipping")
			continue
		}

		if obj.GetKind() == "" {
			continue
		}

		resourceCount++

		if obj.GetKind() != "ConfigMap" {
			logger.Info("Skipping non-ConfigMap resource from profiling output", "kind", obj.GetKind(), "name", obj.GetName())
			continue
		}

		cm := &corev1.ConfigMap{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, cm); err != nil {
			logger.Error(err, "Failed to convert to ConfigMap", "name", obj.GetName())
			continue
		}

		cm.Namespace = targetNamespace
		if cm.Labels == nil {
			cm.Labels = make(map[string]string)
		}
		cm.Labels[LabelDGDRName] = dgdr.Name
		cm.Labels[LabelDGDRNamespace] = dgdr.Namespace
		cm.Labels[LabelManagedBy] = LabelValueDynamoOperator

		if err := r.Create(ctx, cm); err != nil {
			if apierrors.IsAlreadyExists(err) {
				logger.Info("ConfigMap already exists, skipping", "name", cm.Name)
			} else {
				return fmt.Errorf("failed to create ConfigMap %s: %w", cm.Name, err)
			}
		} else {
			logger.Info("Created ConfigMap from profiling output", "name", cm.Name, "namespace", targetNamespace)
		}
	}

	if resourceCount > 0 {
		logger.Info("Deploying additional resources from profiling output", "count", resourceCount)
	}

	return nil
}

// handleFailedState handles DGDR in Failed phase
func (r *DynamoGraphDeploymentRequestReconciler) handleFailedState(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	logger.Info("DGDR is in failed state", "name", dgdr.Name)

	return ctrl.Result{}, nil
}

// getProfilingJobName returns the job name for a DGDR
func getProfilingJobName(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) string {
	return fmt.Sprintf("profile-%s", dgdr.Name)
}

// getOutputConfigMapName returns the ConfigMap name for profiling output
func getOutputConfigMapName(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) string {
	return fmt.Sprintf("%s%s", ConfigMapOutputPrefix, dgdr.Name)
}

// isOnlineProfiling determines whether online profiling or AI Configurator is being used.
func isOnlineProfiling(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) bool {
	// Check v1beta1 structured field first
	if dgdr.Spec.SearchStrategy == nvidiacomv1beta1.SearchStrategyThorough {
		return true
	}
	// Fallback: check annotation blob for v1alpha1 backward compat
	if dgdr.Annotations != nil {
		if rawBlob, ok := dgdr.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			var config map[string]interface{}
			if err := json.Unmarshal([]byte(rawBlob), &config); err == nil {
				if ss, ok := config[ConfigKeySearchStrategy].(string); ok {
					return ss == string(nvidiacomv1beta1.SearchStrategyThorough)
				}
				// Legacy: check sweep.useAiConfigurator / use_ai_configurator
				if sweep, ok := config["sweep"].(map[string]interface{}); ok {
					if v, ok := sweep["useAiConfigurator"].(bool); ok {
						return !v
					}
					if v, ok := sweep["use_ai_configurator"].(bool); ok {
						return !v
					}
				}
			}
		}
	}
	return false // default: rapid (AIC/offline)
}

// validateSpec validates the DGDR spec
func (r *DynamoGraphDeploymentRequestReconciler) validateSpec(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	// Validate ConfigMap if provided (from annotation for v1alpha1 compat)
	configMapRef := getConfigMapRef(dgdr)
	if configMapRef != nil {
		cm := &corev1.ConfigMap{}
		err := r.Get(ctx, types.NamespacedName{
			Name:      configMapRef.Name,
			Namespace: dgdr.Namespace,
		}, cm)

		if err != nil {
			if apierrors.IsNotFound(err) {
				return fmt.Errorf(MessageConfigMapNotFound, configMapRef.Name, dgdr.Namespace)
			}
			return err
		}

		key := configMapRef.Key
		if key == "" {
			key = "disagg.yaml"
		}
		if _, exists := cm.Data[key]; !exists {
			return fmt.Errorf(MessageConfigMapKeyNotFound, key, cm.Name)
		}
	}

	// Validate model cache PVC if provided
	modelCachePVC, _ := extractModelCachePVCConfig(dgdr)
	if modelCachePVC != "" {
		pvc := &corev1.PersistentVolumeClaim{}
		err := r.Get(ctx, types.NamespacedName{
			Name:      modelCachePVC,
			Namespace: dgdr.Namespace,
		}, pvc)

		if err != nil {
			if apierrors.IsNotFound(err) {
				return fmt.Errorf(MessageModelCachePVCNotFound, modelCachePVC, dgdr.Namespace)
			}
			return err
		}
	}

	if err := r.validateGPUHardwareInfo(ctx, dgdr); err != nil {
		return err
	}

	return nil
}

// toFloat64 converts a numeric value (int or float64) to float64.
func toFloat64(val interface{}) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return 0
	}
}

// validateGPUHardwareInfo ensures GPU hardware information is available when required for profiling
func (r *DynamoGraphDeploymentRequestReconciler) validateGPUHardwareInfo(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)

	// Check v1beta1 structured hardware fields
	var hasManualHardwareConfig bool
	if dgdr.Spec.Hardware != nil {
		hasManualHardwareConfig = dgdr.Spec.Hardware.GPUSKU != "" ||
			dgdr.Spec.Hardware.VRAMMB != nil ||
			dgdr.Spec.Hardware.NumGPUsPerNode != nil
	}

	// Check annotation blob for explicit GPU ranges (v1alpha1 backward compat)
	var hasExplicitGPURanges bool
	if dgdr.Annotations != nil {
		if rawBlob, ok := dgdr.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			var config map[string]interface{}
			if err := json.Unmarshal([]byte(rawBlob), &config); err == nil {
				// Check hardware config from blob
				if !hasManualHardwareConfig {
					if hardwareVal, ok := config[ConfigKeyHardware]; ok && hardwareVal != nil {
						if hardwareConfig, ok := hardwareVal.(map[string]interface{}); ok {
							_, hasGPUModel := hardwareConfig[ConfigKeyGPUModel]
							_, hasGPUVram := hardwareConfig[ConfigKeyGPUVramMib]
							_, hasNumGPUs := hardwareConfig[ConfigKeyNumGpusPerNode]
							hasManualHardwareConfig = hasGPUModel || hasGPUVram || hasNumGPUs
						}
					}
				}
				// Check engine config for explicit GPU ranges
				if engineVal, hasEngine := config[ConfigKeyEngine]; hasEngine && engineVal != nil {
					if engineConfig, ok := engineVal.(map[string]interface{}); ok {
						minGPUs, hasMin := engineConfig[ConfigKeyMinNumGpusPerEng]
						maxGPUs, hasMax := engineConfig[ConfigKeyMaxNumGpusPerEng]
						if hasMin && hasMax {
							minVal := toFloat64(minGPUs)
							maxVal := toFloat64(maxGPUs)
							if minVal > maxVal {
								return fmt.Errorf("invalid GPU range: %s (%v) cannot be greater than %s (%v)",
									ConfigKeyMinNumGpusPerEng, minVal, ConfigKeyMaxNumGpusPerEng, maxVal)
							}
							hasExplicitGPURanges = minVal > 0 && maxVal > 0
						}
					}
				}
			}
		}
	}

	if hasManualHardwareConfig || hasExplicitGPURanges {
		return nil
	}

	_, err := gpu.DiscoverGPUs(ctx, r.Client)
	if err == nil {
		return nil
	}

	logger.Info("GPU discovery not available", "reason", err.Error())

	isNamespaceScoped := r.Config.RestrictedNamespace != ""
	if isNamespaceScoped {
		tmpl := template.Must(template.New("nsGPUErr").Parse(
			`GPU hardware info required but cannot be auto-discovered.` +
				"\n\nOptions to resolve:" +
				"\n\n1. Re-enable GPU discovery (if it was disabled during Helm install):" +
				"\n   helm upgrade ... --set dynamo-operator.gpuDiscovery.enabled=true" +
				"\n\n2. Add hardware config to spec.hardware:" +
				"\n   {{.NumGPUs}}: 8" +
				"\n   {{.GPUModel}}: \"H100-SXM5-80GB\"" +
				"\n   {{.GPUVram}}: 81920" +
				"\n\n3. Or specify engine {{.MinGPUs}} and {{.MaxGPUs}} in the profiling config annotation for explicit GPU search ranges.",
		))
		var buf bytes.Buffer
		_ = tmpl.Execute(&buf, map[string]string{
			"NumGPUs":  ConfigKeyNumGpusPerNode,
			"GPUModel": ConfigKeyGPUModel,
			"GPUVram":  ConfigKeyGPUVramMib,
			"MinGPUs":  ConfigKeyMinNumGpusPerEng,
			"MaxGPUs":  ConfigKeyMaxNumGpusPerEng,
		})
		return fmt.Errorf("%s", buf.String())
	}

	return fmt.Errorf("GPU hardware info required but auto-discovery failed. Add hardware config to spec.hardware (%s, %s, %s) or specify %s.%s and %s.%s in profiling config",
		ConfigKeyNumGpusPerNode, ConfigKeyGPUModel, ConfigKeyGPUVramMib,
		ConfigKeyEngine, ConfigKeyMinNumGpusPerEng, ConfigKeyEngine, ConfigKeyMaxNumGpusPerEng)
}

// createProfilingJob creates a Kubernetes Job for profiling using SyncResource
func (r *DynamoGraphDeploymentRequestReconciler) createProfilingJob(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)

	// Delete any existing output ConfigMap to ensure fresh profiling results
	outputConfigMapName := getOutputConfigMapName(dgdr)
	existingCM := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      outputConfigMapName,
		Namespace: dgdr.Namespace,
	}, existingCM)
	if err == nil {
		logger.Info("Deleting existing output ConfigMap to ensure fresh profiling results", "configMap", outputConfigMapName)
		if err := r.Delete(ctx, existingCM); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "Failed to delete existing output ConfigMap", "configMap", outputConfigMapName)
			return fmt.Errorf("failed to delete existing output ConfigMap: %w", err)
		}
		logger.Info("Successfully deleted old output ConfigMap", "configMap", outputConfigMapName)
	} else if !apierrors.IsNotFound(err) {
		logger.Error(err, "Failed to check for existing output ConfigMap", "configMap", outputConfigMapName)
		return fmt.Errorf("failed to check for existing output ConfigMap: %w", err)
	}

	// Ensure profiling job RBAC exists (only for cluster-wide installation)
	if r.Config.RestrictedNamespace == "" {
		if err := r.RBACManager.EnsureServiceAccountWithRBAC(
			ctx,
			dgdr.Namespace,
			ServiceAccountProfilingJob,
			r.Config.RBAC.DGDRProfilingClusterRoleName,
		); err != nil {
			logger.Error(err, "Failed to ensure profiling job RBAC")
			return fmt.Errorf("failed to ensure profiling job RBAC: %w", err)
		}
	}

	// Run GPU discovery before creating job
	var gpuInfo *gpu.GPUInfo
	logger.Info("Attempting GPU discovery for profiling job")
	discoveredInfo, err := gpu.DiscoverGPUs(ctx, r.Client)
	if err != nil {
		logger.Info("GPU discovery not available, using manual hardware configuration",
			"reason", err.Error())
	} else {
		gpuInfo = discoveredInfo
		logger.Info("GPU discovery completed successfully",
			"gpusPerNode", gpuInfo.GPUsPerNode,
			"model", gpuInfo.Model,
			"vramMiB", gpuInfo.VRAMPerGPU,
			"system", gpuInfo.System)
	}

	// Use SyncResource to create/update the job
	modified, job, err := commonController.SyncResource(ctx, r, dgdr, func(ctx context.Context) (*batchv1.Job, bool, error) {
		jobName := getProfilingJobName(dgdr)
		outputConfigMapName := getOutputConfigMapName(dgdr)

		configYAML, err := r.prepareProfilingConfig(dgdr, gpuInfo)
		if err != nil {
			return nil, false, err
		}

		// Common environment variables
		profilerEnv := []corev1.EnvVar{
			{
				Name: "HUGGING_FACE_HUB_TOKEN",
				ValueFrom: &corev1.EnvVarSource{
					SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: "hf-token-secret",
						},
						Key: "HF_TOKEN",
					},
				},
			},
			{
				Name:  "NATS_SERVER",
				Value: fmt.Sprintf("nats://%s-nats:4222", dgdr.Namespace),
			},
			{
				Name:  "ETCD_ENDPOINTS",
				Value: fmt.Sprintf("%s-etcd:2379", dgdr.Namespace),
			},
			{
				Name:  "DGDR_NAME",
				Value: dgdr.Name,
			},
			{
				Name:  "DGDR_NAMESPACE",
				Value: dgdr.Namespace,
			},
			{
				Name:  "DGDR_UID",
				Value: string(dgdr.UID),
			},
		}

		// Build volume mounts
		volumeMounts := []corev1.VolumeMount{
			{
				Name:      VolumeNameProfilingOutput,
				MountPath: ProfilingOutputPath,
			},
		}

		// Add ConfigMap volume mount if provided
		configMapRef := getConfigMapRef(dgdr)
		if configMapRef != nil {
			volumeMounts = append(volumeMounts, corev1.VolumeMount{
				Name:      VolumeNameProfilingConfig,
				MountPath: ProfilingConfigPath,
				ReadOnly:  true,
			})
		}

		// Add model cache PVC mount if configured
		modelCachePVC, modelCacheMountPath := extractModelCachePVCConfig(dgdr)
		if modelCachePVC != "" {
			logger.Info("Mounting model cache PVC to profiler pod", "pvc", modelCachePVC, "mountPath", modelCacheMountPath)
			volumeMounts = append(volumeMounts, corev1.VolumeMount{
				Name:      VolumeNameModelCache,
				MountPath: modelCacheMountPath,
				ReadOnly:  true,
			})
		}

		// Profiler args
		profilerArgs := []string{
			"--profile-config", string(configYAML),
		}

		// Use image from spec
		imageName := dgdr.Spec.Image
		logger.Info("Using profiler image", "image", imageName)

		profilerContainer := corev1.Container{
			Name:         ContainerNameProfiler,
			Image:        imageName,
			Command:      []string{"python", "-m", "dynamo.profiler.profile_sla"},
			Args:         profilerArgs,
			Env:          profilerEnv,
			VolumeMounts: volumeMounts,
		}

		// Apply resource requirements from Overrides.ProfilingJob
		if dgdr.Spec.Overrides != nil && dgdr.Spec.Overrides.ProfilingJob != nil {
			podSpecOverride := &dgdr.Spec.Overrides.ProfilingJob.Template.Spec
			if len(podSpecOverride.Containers) > 0 {
				profilerContainer.Resources = podSpecOverride.Containers[0].Resources
			}
		}

		// Generate sidecar script from template
		tmpl, err := template.New("sidecar").Parse(sidecarScriptTemplate)
		if err != nil {
			return nil, false, fmt.Errorf("failed to parse sidecar script template: %w", err)
		}

		var scriptBuf bytes.Buffer
		err = tmpl.Execute(&scriptBuf, map[string]string{
			"OutputPath":       ProfilingOutputPath,
			"OutputFile":       ProfilingOutputFile,
			"MockerOutputFile": ProfilingOutputFileMocker,
			"ConfigMapName":    outputConfigMapName,
			"Namespace":        dgdr.Namespace,
			"DGDRName":         dgdr.Name,
		})
		if err != nil {
			return nil, false, fmt.Errorf("failed to execute sidecar script template: %w", err)
		}

		sidecarContainer := corev1.Container{
			Name:    ContainerNameOutputCopier,
			Image:   SidecarImage,
			Command: []string{"/bin/sh", "-c"},
			Args:    []string{scriptBuf.String()},
			VolumeMounts: []corev1.VolumeMount{{
				Name:      VolumeNameProfilingOutput,
				MountPath: ProfilingOutputPath,
				ReadOnly:  true,
			}},
		}

		// Use PVC if specified, otherwise use emptyDir for profiling output
		var profilingOutputVolume corev1.Volume
		outputPVC := getOutputPVC(dgdr)
		if outputPVC != "" {
			logger.Info("Using PVC for profiling output", "pvc", outputPVC)
			profilingOutputVolume = corev1.Volume{
				Name: VolumeNameProfilingOutput,
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: outputPVC,
					},
				},
			}
		} else {
			profilingOutputVolume = corev1.Volume{
				Name: VolumeNameProfilingOutput,
				VolumeSource: corev1.VolumeSource{
					EmptyDir: &corev1.EmptyDirVolumeSource{},
				},
			}
		}
		volumes := []corev1.Volume{profilingOutputVolume}

		// Add ConfigMap volume if provided
		if configMapRef != nil {
			key := configMapRef.Key
			if key == "" {
				key = ProfilingConfigFile
			}

			volumes = append(volumes, corev1.Volume{
				Name: VolumeNameProfilingConfig,
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{
							Name: configMapRef.Name,
						},
						Items: []corev1.KeyToPath{{
							Key:  key,
							Path: ProfilingConfigFile,
						}},
					},
				},
			})
		}

		// Add model cache PVC volume if configured
		if modelCachePVC != "" {
			volumes = append(volumes, corev1.Volume{
				Name: VolumeNameModelCache,
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: modelCachePVC,
						ReadOnly:  true,
					},
				},
			})
		}

		backoffLimit := int32(3)

		// Determine label based on whether AI Configurator is used
		labelValue := LabelValueDynamoProfiler
		if !isOnlineProfiling(dgdr) {
			labelValue = LabelValueAICProfiler
		}

		podSpec := corev1.PodSpec{
			ServiceAccountName: ServiceAccountProfilingJob,
			RestartPolicy:      corev1.RestartPolicyNever,
			SecurityContext: &corev1.PodSecurityContext{
				RunAsNonRoot: ptr.To(true),
				RunAsUser:    ptr.To[int64](1000),
				RunAsGroup:   ptr.To[int64](1000),
				FSGroup:      ptr.To[int64](1000),
			},
			Containers: []corev1.Container{profilerContainer, sidecarContainer},
			Volumes:    volumes,
			ImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "nvcr-imagepullsecret"},
			},
		}

		// Apply tolerations and nodeSelector from Overrides.ProfilingJob
		if dgdr.Spec.Overrides != nil && dgdr.Spec.Overrides.ProfilingJob != nil {
			podSpecOverride := &dgdr.Spec.Overrides.ProfilingJob.Template.Spec
			if len(podSpecOverride.Tolerations) > 0 {
				podSpec.Tolerations = podSpecOverride.Tolerations
			}
			if len(podSpecOverride.NodeSelector) > 0 {
				podSpec.NodeSelector = podSpecOverride.NodeSelector
			}
		}

		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      jobName,
				Namespace: dgdr.Namespace,
				Labels: map[string]string{
					LabelApp:       labelValue,
					LabelDGDR:      dgdr.Name,
					LabelManagedBy: LabelValueDynamoOperator,
				},
			},
			Spec: batchv1.JobSpec{
				BackoffLimit: &backoffLimit,
				Template: corev1.PodTemplateSpec{
					Spec: podSpec,
				},
			},
		}

		return job, false, nil
	})

	if err != nil {
		return err
	}

	if modified {
		logger.Info("Profiling job created/updated", "job", job.Name)
	}

	return nil
}

// prepareProfilingConfig builds the profiling config from v1beta1 structured fields
// with annotation blob fallback for v1alpha1 backward compat.
func (r *DynamoGraphDeploymentRequestReconciler) prepareProfilingConfig(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, gpuInfo *gpu.GPUInfo) ([]byte, error) {
	var config map[string]interface{}

	// Start from annotation blob if present (v1alpha1 backward compat)
	if dgdr.Annotations != nil {
		if rawBlob, ok := dgdr.Annotations[annDGDRProfilingConfig]; ok && rawBlob != "" {
			if err := json.Unmarshal([]byte(rawBlob), &config); err != nil {
				return nil, fmt.Errorf("failed to parse profiling config annotation: %w", err)
			}
		}
	}
	if config == nil {
		config = make(map[string]interface{})
	}

	// Ensure nested maps exist
	deploymentConfig := getOrCreateMap(config, ConfigKeyDeployment)
	engineConfig := getOrCreateMap(config, ConfigKeyEngine)
	slaConfig := getOrCreateMap(config, ConfigKeySLA)

	// Always inject model and backend
	deploymentConfig[ConfigKeyModel] = dgdr.Spec.Model
	engineConfig[ConfigKeyBackend] = string(dgdr.Spec.Backend)

	// Image → dgd_image
	if dgdr.Spec.Image != "" {
		deploymentConfig[ConfigKeyDGDImage] = dgdr.Spec.Image
	}

	// Default namespace if not set
	if _, ok := deploymentConfig[ConfigKeyNamespace]; !ok {
		deploymentConfig[ConfigKeyNamespace] = dgdr.Namespace
	}

	// Default output_dir
	if _, ok := config[ConfigKeyOutputDir]; !ok {
		config[ConfigKeyOutputDir] = ProfilingOutputPath
	}

	// SLA fields (v1beta1 structured → profiler's "sla" section)
	if dgdr.Spec.SLA != nil {
		if dgdr.Spec.SLA.TTFT != nil {
			slaConfig["ttft"] = *dgdr.Spec.SLA.TTFT
		}
		if dgdr.Spec.SLA.ITL != nil {
			slaConfig["itl"] = *dgdr.Spec.SLA.ITL
		}
	}
	// ISL/OSL go under "sla" section
	if dgdr.Spec.Workload != nil {
		if dgdr.Spec.Workload.ISL != nil {
			slaConfig["isl"] = float64(*dgdr.Spec.Workload.ISL)
		}
		if dgdr.Spec.Workload.OSL != nil {
			slaConfig["osl"] = float64(*dgdr.Spec.Workload.OSL)
		}
	}

	// ModelCache
	if dgdr.Spec.ModelCache != nil {
		mcMap := make(map[string]interface{})
		if dgdr.Spec.ModelCache.PVCName != "" {
			mcMap[ConfigKeyPVCName] = dgdr.Spec.ModelCache.PVCName
		}
		if dgdr.Spec.ModelCache.PVCModelPath != "" {
			mcMap[ConfigKeyPVCPath] = dgdr.Spec.ModelCache.PVCModelPath
		}
		if dgdr.Spec.ModelCache.PVCMountPath != "" {
			mcMap[ConfigKeyMountPath] = dgdr.Spec.ModelCache.PVCMountPath
		}
		if len(mcMap) > 0 {
			deploymentConfig[ConfigKeyModelCache] = mcMap
		}
	}

	// SearchStrategy — top-level key in profiler config
	if dgdr.Spec.SearchStrategy != "" {
		config[ConfigKeySearchStrategy] = string(dgdr.Spec.SearchStrategy)
	}

	// ConfigMapRef from annotation (v1alpha1 compat)
	configMapRef := getConfigMapRef(dgdr)
	if configMapRef != nil {
		engineConfig[ConfigKeyConfig] = fmt.Sprintf("%s/%s", ProfilingConfigPath, ProfilingConfigFile)
	}

	// GPU info injection from cluster discovery
	if gpuInfo != nil {
		hardwareConfig := getOrCreateMap(config, ConfigKeyHardware)
		if _, ok := hardwareConfig[ConfigKeyNumGpusPerNode]; !ok && gpuInfo.GPUsPerNode > 0 {
			hardwareConfig[ConfigKeyNumGpusPerNode] = gpuInfo.GPUsPerNode
		}
		if _, ok := hardwareConfig[ConfigKeyGPUModel]; !ok && gpuInfo.Model != "" {
			hardwareConfig[ConfigKeyGPUModel] = gpuInfo.Model
		}
		if _, ok := hardwareConfig[ConfigKeyGPUVramMib]; !ok && gpuInfo.VRAMPerGPU > 0 {
			hardwareConfig[ConfigKeyGPUVramMib] = gpuInfo.VRAMPerGPU
		}
		if _, ok := hardwareConfig[ConfigKeySystem]; !ok && gpuInfo.System != "" {
			hardwareConfig[ConfigKeySystem] = gpuInfo.System
		}
	}

	// Inject from v1beta1 Hardware spec if set (overrides GPU discovery)
	if dgdr.Spec.Hardware != nil {
		hardwareConfig := getOrCreateMap(config, ConfigKeyHardware)
		if dgdr.Spec.Hardware.NumGPUsPerNode != nil {
			hardwareConfig[ConfigKeyNumGpusPerNode] = *dgdr.Spec.Hardware.NumGPUsPerNode
		}
		if dgdr.Spec.Hardware.GPUSKU != "" {
			hardwareConfig[ConfigKeyGPUModel] = dgdr.Spec.Hardware.GPUSKU
		}
		if dgdr.Spec.Hardware.VRAMMB != nil {
			hardwareConfig[ConfigKeyGPUVramMib] = *dgdr.Spec.Hardware.VRAMMB
		}
	}

	// Write back nested maps
	config[ConfigKeyDeployment] = deploymentConfig
	config[ConfigKeyEngine] = engineConfig
	config[ConfigKeySLA] = slaConfig

	return sigsyaml.Marshal(config)
}

// extractModelCachePVCConfig extracts model cache PVC settings.
// Returns (pvcName, mountPath) - both empty if not configured.
func extractModelCachePVCConfig(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (string, string) {
	// Check v1beta1 structured fields first
	if dgdr.Spec.ModelCache != nil && dgdr.Spec.ModelCache.PVCName != "" {
		mountPath := dgdr.Spec.ModelCache.PVCMountPath
		if mountPath == "" {
			mountPath = DefaultModelCacheMountPath
		}
		return dgdr.Spec.ModelCache.PVCName, mountPath
	}
	// Fallback: check annotation blob for v1alpha1 backward compat
	if dgdr.Annotations == nil {
		return "", ""
	}
	rawBlob, ok := dgdr.Annotations[annDGDRProfilingConfig]
	if !ok || rawBlob == "" {
		return "", ""
	}
	var config map[string]interface{}
	if err := json.Unmarshal([]byte(rawBlob), &config); err != nil {
		return "", ""
	}
	deployment, ok := config[ConfigKeyDeployment].(map[string]interface{})
	if !ok {
		return "", ""
	}
	modelCache, ok := deployment[ConfigKeyModelCache].(map[string]interface{})
	if !ok {
		return "", ""
	}
	pvcName, _ := modelCache[ConfigKeyPVCName].(string)
	if pvcName == "" {
		return "", ""
	}
	mountPath, _ := modelCache[ConfigKeyMountPath].(string)
	if mountPath == "" {
		mountPath = DefaultModelCacheMountPath
	}
	return pvcName, mountPath
}

// checkProfilingJobStatus checks if the profiling job has completed
func (r *DynamoGraphDeploymentRequestReconciler) checkProfilingJobStatus(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) (bool, error) {
	logger := log.FromContext(ctx)
	jobName := getProfilingJobName(dgdr)

	job := &batchv1.Job{}
	if err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: dgdr.Namespace}, job); err != nil {
		return false, err
	}

	for _, condition := range job.Status.Conditions {
		if condition.Type == batchv1.JobComplete && condition.Status == corev1.ConditionTrue {
			logger.Info("Profiling job completed", "job", jobName)
			return true, nil
		}
		if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
			detailedError := r.getProfilingJobErrorDetails(ctx, dgdr, job)
			if detailedError != "" {
				return false, fmt.Errorf("profiling job failed: %s. Details: %s", condition.Message, detailedError)
			}
			return false, fmt.Errorf("profiling job failed: %s", condition.Message)
		}
	}

	return false, nil
}

// getProfilingJobErrorDetails retrieves detailed error information from failed profiling job pods
func (r *DynamoGraphDeploymentRequestReconciler) getProfilingJobErrorDetails(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, job *batchv1.Job) string {
	logger := log.FromContext(ctx)

	podList := &corev1.PodList{}
	labelSelector := client.MatchingLabels{
		"job-name": job.Name,
	}

	if err := r.List(ctx, podList, client.InNamespace(dgdr.Namespace), labelSelector); err != nil {
		logger.Error(err, "Failed to list pods for profiling job")
		return ""
	}

	for _, pod := range podList.Items {
		if pod.Status.Phase == corev1.PodFailed {
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if containerStatus.Name == ContainerNameProfiler && containerStatus.State.Terminated != nil {
					terminated := containerStatus.State.Terminated
					errorMsg := fmt.Sprintf("Pod: %s, Container: %s, ExitCode: %d, Reason: %s",
						pod.Name, containerStatus.Name, terminated.ExitCode, terminated.Reason)
					if terminated.Message != "" {
						errorMsg += fmt.Sprintf(", Message: %s", terminated.Message)
					}
					logger.Info("Retrieved profiling job error details", "error", errorMsg)
					return errorMsg
				}
			}

			for _, containerStatus := range pod.Status.ContainerStatuses {
				if containerStatus.Name == ContainerNameProfiler && containerStatus.State.Waiting != nil {
					waiting := containerStatus.State.Waiting
					errorMsg := fmt.Sprintf("Pod: %s, Container: %s, Waiting - Reason: %s, Message: %s",
						pod.Name, containerStatus.Name, waiting.Reason, waiting.Message)
					logger.Info("Retrieved profiling job waiting details", "error", errorMsg)
					return errorMsg
				}
			}
		}
	}

	return ""
}

// generateDGDSpec generates DGD spec from profiling results (online or offline/AIC)
func (r *DynamoGraphDeploymentRequestReconciler) generateDGDSpec(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest) error {
	logger := log.FromContext(ctx)
	logger.Info("Generating DGD spec from profiling results", "name", dgdr.Name, "backend", string(dgdr.Spec.Backend))

	outputConfigMapName := getOutputConfigMapName(dgdr)
	cm := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{
		Name:      outputConfigMapName,
		Namespace: dgdr.Namespace,
	}, cm)

	if err != nil {
		if apierrors.IsNotFound(err) {
			return fmt.Errorf("output ConfigMap %s not found - profiling may not have completed yet", outputConfigMapName)
		}
		return fmt.Errorf("failed to get output ConfigMap: %w", err)
	}

	// Select the right config file based on mocker flag
	var outputFile string
	if isMockerEnabled(dgdr) {
		outputFile = ProfilingOutputFileMocker
		logger.Info("Using mocker deployment config")
	} else {
		outputFile = ProfilingOutputFile
	}

	yamlContent, exists := cm.Data[outputFile]
	if !exists {
		return fmt.Errorf("key %s not found in ConfigMap %s", outputFile, outputConfigMapName)
	}

	logger.Info("Found profiling output in ConfigMap", "configMap", outputConfigMapName, "outputFile", outputFile, "size", len(yamlContent))

	dgd, additionalResources, err := r.extractResourcesFromYAML([]byte(yamlContent))
	if err != nil {
		return fmt.Errorf("failed to extract DGD from %s: %w", outputFile, err)
	}

	logger.Info("Parsed profiling output", "dgdName", dgd.Name, "additionalResources", len(additionalResources))

	// Store additional resources (ConfigMaps) in annotations first
	if len(additionalResources) > 0 {
		if err := r.storeAdditionalResources(ctx, dgdr, additionalResources); err != nil {
			logger.Error(err, "Failed to store additional resources")
			return err
		}
		// Refetch the DGDR after updating annotations
		if err := r.Get(ctx, types.NamespacedName{Name: dgdr.Name, Namespace: dgdr.Namespace}, dgdr); err != nil {
			return fmt.Errorf("failed to refetch DGDR after storing annotations: %w", err)
		}
	}

	// Store the generated DGD in status.profilingResults.selectedConfig
	if dgdr.Status.ProfilingResults == nil {
		dgdr.Status.ProfilingResults = &nvidiacomv1beta1.ProfilingResultsStatus{}
	}
	dgdr.Status.ProfilingResults.SelectedConfig = &runtime.RawExtension{
		Object: dgd,
	}

	return r.Status().Update(ctx, dgdr)
}

// storeAdditionalResources marshals additional resources to YAML and stores them in DGDR annotations.
func (r *DynamoGraphDeploymentRequestReconciler) storeAdditionalResources(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, resources []*unstructured.Unstructured) error {
	if len(resources) == 0 {
		return nil
	}

	var resourcesYAML []byte

	for i, res := range resources {
		resYAML, err := sigsyaml.Marshal(res.Object)
		if err != nil {
			return fmt.Errorf("failed to marshal resource %s/%s: %w", res.GetKind(), res.GetName(), err)
		}
		if i > 0 {
			resourcesYAML = append(resourcesYAML, []byte("\n---\n")...)
		}
		resourcesYAML = append(resourcesYAML, resYAML...)
	}

	if len(resourcesYAML) > MaxAnnotationSize {
		return fmt.Errorf("additional resources YAML size (%d bytes) exceeds maximum annotation size (%d bytes); "+
			"consider reducing the number of resources or storing them separately",
			len(resourcesYAML), MaxAnnotationSize)
	}

	if dgdr.Annotations == nil {
		dgdr.Annotations = make(map[string]string)
	}
	dgdr.Annotations[AnnotationAdditionalResources] = string(resourcesYAML)

	return r.Update(ctx, dgdr)
}

// extractResourcesFromYAML parses multi-document YAML from profiling output,
// extracting the DynamoGraphDeployment and any ConfigMaps that should be deployed with it.
func (r *DynamoGraphDeploymentRequestReconciler) extractResourcesFromYAML(yamlContent []byte) (*nvidiacomv1alpha1.DynamoGraphDeployment, []*unstructured.Unstructured, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(yamlContent), 4096)

	var dgd *nvidiacomv1alpha1.DynamoGraphDeployment
	var additionalResources []*unstructured.Unstructured

	for {
		obj := &unstructured.Unstructured{}
		if err := decoder.Decode(obj); err != nil {
			if err == io.EOF {
				break
			}
			continue
		}

		if obj.GetKind() == "" {
			continue
		}

		if obj.GetKind() == "DynamoGraphDeployment" {
			dgd = &nvidiacomv1alpha1.DynamoGraphDeployment{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, dgd); err != nil {
				return nil, nil, fmt.Errorf("failed to convert to DynamoGraphDeployment: %w", err)
			}
		} else {
			additionalResources = append(additionalResources, obj)
		}
	}

	if dgd == nil {
		return nil, nil, fmt.Errorf("no DynamoGraphDeployment found in YAML content")
	}

	return dgd, additionalResources, nil
}

// extractDGDFromYAML is a convenience wrapper that extracts only the DGD (used by tests)
func (r *DynamoGraphDeploymentRequestReconciler) extractDGDFromYAML(yamlContent []byte) (*nvidiacomv1alpha1.DynamoGraphDeployment, error) {
	dgd, _, err := r.extractResourcesFromYAML(yamlContent)
	return dgd, err
}

// updatePhaseAndRequeue updates the DGDR phase and requeues
func (r *DynamoGraphDeploymentRequestReconciler) updatePhaseAndRequeue(ctx context.Context, dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, phase nvidiacomv1beta1.DGDRPhase, _ string) (ctrl.Result, error) {
	dgdr.SetPhase(phase)
	if err := r.Status().Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

// updatePhaseWithCondition updates phase and adds/updates a condition
func (r *DynamoGraphDeploymentRequestReconciler) updatePhaseWithCondition(
	ctx context.Context,
	dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	phase nvidiacomv1beta1.DGDRPhase,
	conditionType string,
	status metav1.ConditionStatus,
	reason string,
	message string,
) (ctrl.Result, error) {
	dgdr.SetPhase(phase)

	condition := metav1.Condition{
		Type:               conditionType,
		Status:             status,
		ObservedGeneration: dgdr.Generation,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}

	dgdr.AddStatusCondition(condition)

	if err := r.Status().Update(ctx, dgdr); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{Requeue: true}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *DynamoGraphDeploymentRequestReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&nvidiacomv1beta1.DynamoGraphDeploymentRequest{}).
		Named(consts.ResourceTypeDynamoGraphDeploymentRequest).
		Owns(&batchv1.Job{}, builder.WithPredicates(predicate.Funcs{
			// ignore creation cause we don't want to be called again after we create the job
			CreateFunc:  func(ce event.CreateEvent) bool { return false },
			DeleteFunc:  func(de event.DeleteEvent) bool { return true },
			UpdateFunc:  func(de event.UpdateEvent) bool { return true },
			GenericFunc: func(ge event.GenericEvent) bool { return true },
		})). // Watch Jobs created by this controller (via ownerReference)
		Watches(
			&nvidiacomv1alpha1.DynamoGraphDeployment{},
			handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []ctrl.Request {
				// Find DGDR by label instead of owner reference
				dgd := obj.(*nvidiacomv1alpha1.DynamoGraphDeployment)
				dgdrName, hasName := dgd.Labels[LabelDGDRName]
				dgdrNamespace, hasNamespace := dgd.Labels[LabelDGDRNamespace]
				if !hasName || !hasNamespace {
					return nil
				}
				return []ctrl.Request{{
					NamespacedName: types.NamespacedName{
						Name:      dgdrName,
						Namespace: dgdrNamespace,
					},
				}}
			}),
			builder.WithPredicates(predicate.Funcs{
				// ignore creation cause we don't want to be called again after we create the DGD
				CreateFunc:  func(ce event.CreateEvent) bool { return false },
				DeleteFunc:  func(de event.DeleteEvent) bool { return true },
				UpdateFunc:  func(ue event.UpdateEvent) bool { return true },
				GenericFunc: func(ge event.GenericEvent) bool { return true },
			}),
		).                                                                          // Watch DGDs created by this controller (via label)
		WithEventFilter(commonController.EphemeralDeploymentEventFilter(r.Config)). // set the event filter to ignore resources handled by other controllers in namespace-restricted mode
		Complete(observability.NewObservedReconciler(r, consts.ResourceTypeDynamoGraphDeploymentRequest))
}
