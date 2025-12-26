{{/*
Expand the name of the chart.
*/}}
{{- define "cube3d.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "cube3d.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cube3d.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cube3d.labels" -}}
helm.sh/chart: {{ include "cube3d.chart" . }}
{{ include "cube3d.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cube3d.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cube3d.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "cube3d.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "cube3d.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Triton component labels
*/}}
{{- define "cube3d.triton.labels" -}}
{{ include "cube3d.labels" . }}
app.kubernetes.io/component: triton
{{- end }}

{{/*
Worker component labels
*/}}
{{- define "cube3d.worker.labels" -}}
{{ include "cube3d.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
API component labels
*/}}
{{- define "cube3d.api.labels" -}}
{{ include "cube3d.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Temporal host
*/}}
{{- define "cube3d.temporalHost" -}}
{{- if .Values.temporal.external.enabled }}
{{- printf "%s:%v" .Values.temporal.external.host .Values.temporal.external.port }}
{{- else }}
{{- .Values.workers.temporalHost }}
{{- end }}
{{- end }}
