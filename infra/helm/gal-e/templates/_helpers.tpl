{{/* Namespace for a logical area: ingest | ml | app | data | obs */}}
{{- define "gale.ns" -}}
{{- $root := index . 0 -}}{{- $area := index . 1 -}}
{{- index $root.Values.global.namespaces $area -}}
{{- end -}}

{{/* Kubernetes-safe name for a service key (ingest_gateway → gale-ingest-gateway) */}}
{{- define "gale.svcName" -}}
{{- printf "gale-%s" (replace "_" "-" .) -}}
{{- end -}}

{{- define "gale.labels" -}}
{{- $root := index . 0 -}}{{- $component := index . 1 -}}
app.kubernetes.io/name: {{ $component }}
app.kubernetes.io/part-of: gal-e
app.kubernetes.io/instance: {{ $root.Release.Name }}
app.kubernetes.io/version: {{ $root.Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ $root.Release.Service }}
helm.sh/chart: {{ printf "%s-%s" $root.Chart.Name $root.Chart.Version }}
{{- end -}}

{{- define "gale.selector" -}}
app.kubernetes.io/name: {{ . }}
app.kubernetes.io/part-of: gal-e
{{- end -}}

{{- define "gale.tag" -}}
{{- default .Chart.AppVersion .Values.global.imageTag -}}
{{- end -}}

{{/* Image for a GAL-E component: <registry>/gale-<service>:<tag> */}}
{{- define "gale.image" -}}
{{- $root := index . 0 -}}{{- $svc := index . 1 -}}
{{- printf "%s/gale-%s:%s" $root.Values.global.imageRegistry (replace "_" "-" $svc) (include "gale.tag" $root) -}}
{{- end -}}

{{/* Env shared by every Python service: config map + secrets */}}
{{- define "gale.envFrom" -}}
- configMapRef:
    name: gale-env
- secretRef:
    name: {{ .Values.secrets.bus }}
- secretRef:
    name: {{ .Values.secrets.db }}
    optional: true
{{- if eq .Values.objectStore.mode "s3" }}
- secretRef:
    name: {{ .Values.secrets.objectStore }}
{{- end }}
{{- end -}}

{{- define "gale.podSecurity" -}}
runAsNonRoot: true
runAsUser: {{ . }}
runAsGroup: {{ . }}
fsGroup: {{ . }}
seccompProfile:
  type: RuntimeDefault
{{- end -}}

{{- define "gale.containerSecurity" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
{{- end -}}

{{/* Streams a worker consumes, for KEDA (audio is sharded over 8 streams) */}}
{{- define "gale.kedaStreams" -}}
{{- if .shardedAudio -}}
{{- range $i := until 8 }}audio.frames.{{ $i }},{{ end -}}
{{- else -}}
{{- range .streams }}{{ . }},{{ end -}}
{{- end -}}
{{- end -}}
