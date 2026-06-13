{{/* Common name + label helpers. */}}
{{- define "rlhf-ppo.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rlhf-ppo.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "rlhf-ppo.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "rlhf-ppo.labels" -}}
app.kubernetes.io/name: {{ include "rlhf-ppo.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "rlhf-ppo.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "rlhf-ppo.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
