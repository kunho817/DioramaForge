import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8008";
const FALLBACK_PIPELINE_DEFAULTS = {
  seed: -1,
  steps: 12,
  guidance: 4.5,
  strength: 0.68,
  max_resolution: 512,
  demo_time_budget_seconds: 240,
  stage35_upscale_scale: 2,
  stage35_refinement_strength: 0.22,
  stage35_max_side: 1536,
  stage4_mesh_resolution: 96,
  stage4_max_parts: 12,
  stage5_width_mm: 120,
  stage5_relief_height_mm: 18,
  stage5_base_thickness_mm: 3,
  stage5_mesh_resolution: 96
};

const STAGES = [
  {
    key: "stage3",
    shortLabel: "이미지",
    label: "분석 및 스타일 변환",
    detail: "원본, 깊이맵, 마스크, 스타일 결과를 만듭니다."
  },
  {
    key: "stage35",
    shortLabel: "보정",
    label: "구조 보정",
    detail: "원본 구도를 유지하도록 결과를 정리합니다."
  },
  {
    key: "stage4",
    shortLabel: "3D",
    label: "3D 변환 준비",
    detail: "분할된 영역을 기반으로 3D 변환 패키지를 만듭니다."
  },
  {
    key: "stage5",
    shortLabel: "출력",
    label: "출력 정리",
    detail: "미리보기와 모델 출력 묶음을 정리합니다."
  }
];

function App() {
  const [presets, setPresets] = useState(["Fantasy Diorama"]);
  const [runtime, setRuntime] = useState("");
  const [models, setModels] = useState("");
  const [styleEngine, setStyleEngine] = useState(null);
  const [meshyStatus, setMeshyStatus] = useState(null);
  const [executionPolicy, setExecutionPolicy] = useState(null);
  const [demoReadiness, setDemoReadiness] = useState(null);
  const [pipelinePreflight, setPipelinePreflight] = useState(null);
  const [comfyWorkflows, setComfyWorkflows] = useState(null);
  const [comfyModelChoices, setComfyModelChoices] = useState(null);
  const [pipelineDefaults, setPipelineDefaults] = useState(FALLBACK_PIPELINE_DEFAULTS);
  const [image, setImage] = useState(null);
  const [workflowFile, setWorkflowFile] = useState(null);
  const [workflowInstallStatus, setWorkflowInstallStatus] = useState("");
  const [workflowInspection, setWorkflowInspection] = useState(null);
  const [workflowInspectStatus, setWorkflowInspectStatus] = useState("");
  const [preset, setPreset] = useState("Fantasy Diorama");
  const [prompt, setPrompt] = useState("");
  const [runDir, setRunDir] = useState("");
  const [stage3, setStage3] = useState(null);
  const [stage35, setStage35] = useState(null);
  const [stage4, setStage4] = useState(null);
  const [stage5, setStage5] = useState(null);
  const [pipelineInfo, setPipelineInfo] = useState(null);
  const [validation, setValidation] = useState(null);
  const [currentJob, setCurrentJob] = useState(null);
  const [recentRuns, setRecentRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState(false);
  const [setupOpen, setSetupOpen] = useState(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const internalBackend = styleEngine?.backend_mode ?? "auto";
  const localHeavyBlocked =
    executionPolicy &&
    !executionPolicy.allow_local_heavy_models &&
    ["auto", "real"].includes(internalBackend);
  const generatePreflightBlocked = pipelinePreflight?.ok === false && Array.isArray(pipelinePreflight?.checks);
  const generateDisabled = busy || !image || localHeavyBlocked || generatePreflightBlocked;
  const selectedImageName = image?.name ?? "선택된 사진 없음";
  const activeRunId = runDir ? runDir.split(/[\\/]/).pop() : "";
  const runLoaded = Boolean(stage3);
  const profileText = `${pipelineDefaults.max_resolution}px · ${pipelineDefaults.steps}단계`;
  const jobProgress = currentJob?.progress ?? {};
  const activeStage = jobProgress.current_stage ?? "";
  const activeStageLabel = jobProgress.current_label ?? "";
  const visiblePipeline = mergePipelineProgress(pipelineInfo, jobProgress);
  const userLog = useMemo(() => buildUserLog(currentJob, log), [currentJob, log]);
  const workflowSummary = useMemo(() => summarizeStage3Workflow(comfyWorkflows), [comfyWorkflows]);

  useEffect(() => {
    refreshStatus();
    refreshRuns();
    fetch(`${API_BASE}/api/presets`)
      .then((response) => response.json())
      .then((data) => {
        const nextPresets = data.presets ?? ["Fantasy Diorama"];
        setPresets(nextPresets);
        setPreset(nextPresets[0]);
      })
      .catch(() => {});
  }, []);

  const imageTiles = useMemo(() => {
    if (!stage3?.images) return [];
    return [
      ["원본", stage3.images.original],
      ["깊이맵", stage3.images.depth],
      ["마스크", stage3.images.mask_overlay],
      ["영역 계획", stage3.images.region_overlay],
      ["스타일 제어", stage3.images.style_control ?? stage3.images.flux_control],
      ["스타일 변환 결과", stage3.images.style_result ?? stage3.images.flux_result]
    ];
  }, [stage3, styleEngine]);

  const stage3Summary = useMemo(() => {
    if (!stage3) return null;
    return {
      options: stage3.options ?? {},
      styleEngine: stage3.style_engine ?? {}
    };
  }, [stage3]);

  async function refreshStatus() {
    const [runtimeData, modelsData, engineData, meshyData, policy, pipelineData, preflightData, demoData, workflowData, modelChoicesData] = await Promise.all([
      fetchJsonQuiet("/api/runtime"),
      fetchJsonQuiet("/api/models"),
      fetchJsonQuiet("/api/style-engine"),
      fetchJsonQuiet("/api/meshy/status"),
      fetchJsonQuiet("/api/execution/policy"),
      fetchJsonQuiet("/api/pipeline/defaults"),
      fetchJsonQuiet("/api/pipeline/preflight"),
      fetchJsonQuiet("/api/demo/readiness"),
      fetchJsonQuiet("/api/comfy/workflows"),
      fetchJsonQuiet("/api/comfy/model-choices")
    ]);
    setRuntime(runtimeData?.markdown ?? "실행 환경 상태를 확인할 수 없습니다.");
    setModels(modelsData?.markdown ?? "모델 상태를 확인할 수 없습니다.");
    setStyleEngine(engineData);
    setMeshyStatus(meshyData);
    setExecutionPolicy(policy);
    setDemoReadiness(demoData);
    setPipelinePreflight(preflightData);
    setComfyWorkflows(workflowData);
    setComfyModelChoices(modelChoicesData);
    setPipelineDefaults({ ...FALLBACK_PIPELINE_DEFAULTS, ...(pipelineData?.defaults ?? {}) });
  }

  async function refreshRuns() {
    try {
      const response = await fetch(`${API_BASE}/api/runs?limit=30`);
      const data = await parseResponse(response);
      setRecentRuns(data.runs ?? []);
    } catch {
      setRecentRuns([]);
    }
  }

  async function loadSelectedRun() {
    if (!selectedRunId) return;
    setBusy(true);
    setLog(["선택한 결과를 불러오고 있습니다."]);
    try {
      const response = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(selectedRunId)}`);
      const data = await parseResponse(response);
      applyRunDetail(data);
    } catch (error) {
      setLog([friendlyError(error)]);
    } finally {
      setBusy(false);
    }
  }

  async function installStage3Workflow() {
    if (!workflowFile) {
      setWorkflowInstallStatus("먼저 ComfyUI API 워크플로 JSON을 선택해 주세요.");
      return;
    }
    const form = new FormData();
    form.append("workflow", workflowFile);
    setBusy(true);
    setWorkflowInstallStatus("워크플로를 확인하고 있습니다.");
    setLog(["스타일 변환 워크플로를 설치하고 있습니다."]);
    try {
      const response = await fetch(`${API_BASE}/api/comfy/workflows/stage3/install`, {
        method: "POST",
        body: form
      });
      const data = await parseResponse(response);
      setComfyWorkflows(data.workflows ?? null);
      setWorkflowInstallStatus("스타일 변환 워크플로 설치가 완료되었습니다.");
      await refreshStatus();
    } catch (error) {
      setWorkflowInstallStatus(friendlyError(error));
      setLog([friendlyError(error)]);
    } finally {
      setBusy(false);
    }
  }

  async function inspectStage3Workflow() {
    if (!workflowFile) {
      setWorkflowInspectStatus("먼저 ComfyUI 워크플로 JSON을 선택해 주세요.");
      return;
    }
    const form = new FormData();
    form.append("workflow", workflowFile);
    setBusy(true);
    setWorkflowInspectStatus("워크플로를 검사하고 있습니다.");
    try {
      const response = await fetch(`${API_BASE}/api/comfy/workflows/stage3/inspect`, {
        method: "POST",
        body: form
      });
      const data = await parseResponse(response);
      setWorkflowInspection(data);
      setWorkflowInspectStatus(data.format === "api" ? "검사가 완료되었습니다." : "ComfyUI에서 Save (API Format)으로 다시 저장해 주세요.");
    } catch (error) {
      setWorkflowInspection(null);
      setWorkflowInspectStatus(friendlyError(error));
    } finally {
      setBusy(false);
    }
  }

  async function prepareInstallStage3Workflow() {
    if (!workflowFile) {
      setWorkflowInstallStatus("먼저 ComfyUI 워크플로 JSON을 선택해 주세요.");
      return;
    }
    const form = new FormData();
    form.append("workflow", workflowFile);
    setBusy(true);
    setWorkflowInstallStatus("워크플로를 준비하고 확인하고 있습니다.");
    setLog(["스타일 변환 워크플로를 준비하고 있습니다."]);
    try {
      const response = await fetch(`${API_BASE}/api/comfy/workflows/stage3/prepare-install`, {
        method: "POST",
        body: form
      });
      const data = await parseResponse(response);
      setWorkflowInspection(data.prepare?.inspection ?? null);
      setComfyWorkflows(data.workflows ?? null);
      const changes = data.prepare?.changes?.length ?? 0;
      setWorkflowInstallStatus(`워크플로 설치가 완료되었습니다. 자동 보정 ${changes}건이 적용되었습니다.`);
      await refreshStatus();
    } catch (error) {
      setWorkflowInstallStatus(friendlyError(error));
      setLog([friendlyError(error)]);
    } finally {
      setBusy(false);
    }
  }

  async function installExampleStage3Workflow() {
    setBusy(true);
    setWorkflowInstallStatus("기본 예제 워크플로를 설치하고 있습니다.");
    setLog(["기본 스타일 변환 워크플로를 설치하고 있습니다."]);
    try {
      const response = await fetch(`${API_BASE}/api/comfy/workflows/stage3/install-example`, {
        method: "POST"
      });
      const data = await parseResponse(response);
      setComfyWorkflows(data.workflows ?? null);
      setComfyModelChoices(data.model_choices ?? null);
      setWorkflowInspection(data.validation ? { validation: data.validation } : null);
      const notes = data.notes ?? [];
      setWorkflowInstallStatus(notes[0] ?? "기본 예제 워크플로 설치가 완료되었습니다.");
      await refreshStatus();
    } catch (error) {
      setWorkflowInstallStatus(friendlyError(error));
      setLog([friendlyError(error)]);
    } finally {
      setBusy(false);
    }
  }

  function applyRunDetail(data, options = {}) {
    setRunDir(data.run_dir ?? "");
    setStage3(data.stage3 ?? null);
    setStage35(data.stage35 ?? null);
    setStage4(data.stage4 ?? null);
    setStage5(data.stage5 ?? null);
    setPipelineInfo(data.pipeline ?? null);
    setValidation(data.validation ?? null);
    if (!options.keepJob) {
      setCurrentJob(null);
    }
    if (!options.keepUserLog) {
      setLog(toUserMessages(data.log ?? []));
    }
  }

  async function runFullPipeline() {
    if (!image) {
      setLog(["먼저 입력 사진을 선택해 주세요."]);
      return;
    }
    if (!ensureLocalHeavyPolicyAllowsRun()) return;
    setBusy(true);
    setLog(["생성 준비 상태를 확인하고 있습니다."]);
    try {
      const preflightResponse = await fetch(`${API_BASE}/api/pipeline/preflight`);
      const preflight = await parseResponse(preflightResponse);
      setPipelinePreflight(preflight);
      if (!preflight.ok) {
        setLog([
          "생성을 시작할 수 없습니다.",
          preflight.next_action ?? "준비 상태에서 막힌 항목을 먼저 해결해 주세요.",
          ...(preflight.errors ?? []).slice(0, 3)
        ]);
        return;
      }
      const form = new FormData();
      appendStage3Fields(form);
      clearRunPreview();
      setLog(["생성을 시작했습니다. 단계가 끝날 때마다 결과가 표시됩니다."]);
      const response = await fetch(`${API_BASE}/api/pipeline/jobs`, { method: "POST", body: form });
      const job = await parseResponse(response);
      const data = await waitForJob(job);
      applyRunDetail(data);
      setSelectedRunId((data.run_dir ?? "").split(/[\\/]/).pop() ?? "");
      refreshRuns();
    } catch (error) {
      setLog([friendlyError(error)]);
    } finally {
      setBusy(false);
    }
  }

  function appendStage3Fields(form) {
    form.append("image", image);
    form.append("preset_name", preset);
    form.append("custom_prompt", prompt);
    form.append("seed", String(pipelineDefaults.seed));
    form.append("steps", String(pipelineDefaults.steps));
    form.append("guidance", String(pipelineDefaults.guidance));
    form.append("strength", String(pipelineDefaults.strength));
    form.append("max_resolution", String(pipelineDefaults.max_resolution));
  }

  function clearRunPreview() {
    setRunDir("");
    setStage3(null);
    setStage35(null);
    setStage4(null);
    setStage5(null);
    setPipelineInfo(null);
    setValidation(null);
    setCurrentJob(null);
  }

  function ensureLocalHeavyPolicyAllowsRun() {
    if (!localHeavyBlocked) return true;
    setLog([
      "로컬 모델 실행이 잠겨 있습니다.",
      "현재 설정된 출력 엔진은 이 PC의 GPU 실행이 필요합니다.",
      "설정 파일에서 로컬 실행 허용 값을 확인해 주세요."
    ]);
    return false;
  }

  return (
    <main className="shell">
      <header className="app-header">
        <div className="brand-block">
          <div>
            <h1>DioramaForge</h1>
            <div className="header-meta">
              <span className={`system-pill ${readinessClass(demoReadiness)}`}>{readinessLabel(demoReadiness)}</span>
              <span>{profileText}</span>
              {activeRunId && <span>{activeRunId}</span>}
            </div>
          </div>
        </div>
        <div className="header-actions">
          <button className="ghost-button" onClick={refreshStatus}>상태 새로고침</button>
        </div>
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <section className="panel-section">
            <h2>입력</h2>
            <label className="file-picker">
              <span>사진</span>
              <input type="file" accept="image/*" onChange={(event) => setImage(event.target.files?.[0] ?? null)} />
              <small>{selectedImageName}</small>
            </label>
            <label>
              스타일
              <select value={preset} onChange={(event) => setPreset(event.target.value)}>
                {presets.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              프롬프트
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} />
            </label>
          </section>
          {localHeavyBlocked && (
            <div className="mode-warning danger">
              로컬 모델 실행이 잠겨 있습니다. 현재 출력 엔진은 이 PC의 모델 실행이 필요합니다.
            </div>
          )}
          {generatePreflightBlocked && (
            <div className="mode-warning danger">
              생성 준비 상태에서 막힌 항목이 있습니다. {pipelinePreflight.next_action ?? "상태를 새로고침해 주세요."}
            </div>
          )}
          <section className="panel-section action-panel">
            <button className="primary generate-button" disabled={generateDisabled} onClick={runFullPipeline}>생성 시작</button>
            <div className="action-note">{profileText} · {workflowSummary.family}</div>
          </section>
          <section className="panel-section">
            <h2>결과 불러오기</h2>
            <label>
              현재 경로
              <input value={runDir} onChange={(event) => setRunDir(event.target.value)} placeholder="생성 결과 경로" />
            </label>
            <div className="run-loader">
              <label>
                최근 결과
                <select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)}>
                  <option value="">결과 선택</option>
                  {recentRuns.map((item) => (
                    <option key={item.id} value={item.id}>
                      {runLabel(item)}
                    </option>
                  ))}
                </select>
              </label>
              <button disabled={busy || !selectedRunId} onClick={loadSelectedRun}>불러오기</button>
            </div>
          </section>
          <section className="setup-drawer">
            <button className="drawer-toggle" type="button" onClick={() => setSetupOpen((value) => !value)}>
              시스템 설정
            </button>
            {setupOpen && (
              <WorkflowSetup
                workflows={comfyWorkflows}
                file={workflowFile}
                inspection={workflowInspection}
                onFileChange={(file) => {
                  setWorkflowFile(file);
                  setWorkflowInspection(null);
                  setWorkflowInspectStatus("");
                  setWorkflowInstallStatus("");
                }}
                onInspect={inspectStage3Workflow}
                onInstall={installStage3Workflow}
                onPrepareInstall={prepareInstallStage3Workflow}
                onInstallExample={installExampleStage3Workflow}
                status={workflowInstallStatus}
                inspectStatus={workflowInspectStatus}
                busy={busy}
              />
            )}
          </section>
        </aside>

        <section className="results">
          <div className="result-header">
            <div>
              <span className="eyebrow">결과 화면</span>
              <h2>{runLoaded ? activeRunId : busy ? "생성 진행 중" : "아직 결과 없음"}</h2>
            </div>
            <PipelineRail pipeline={visiblePipeline} currentStage={activeStage} />
          </div>
          <LiveReadinessBanner readiness={demoReadiness} />
          {currentJob && (
            <ProgressPanel job={currentJob} currentLabel={activeStageLabel} pipeline={visiblePipeline} />
          )}
          {!runLoaded && <EmptyResults busy={busy} />}
          {runLoaded && (
            <>
              <section className="result-section">
                <div className="section-title">
                  <h3>이미지 변환</h3>
                  <span>Stage 1-3</span>
                </div>
                <div className="image-grid">
                  {imageTiles.map(([label, artifact]) => (
                    <figure key={label}>
                      <figcaption>{label}</figcaption>
                      {artifact?.url && <img src={`${API_BASE}${artifact.url}`} alt={label} />}
                    </figure>
                  ))}
                </div>
              </section>
              {(stage35?.reconstruction?.url || stage35?.refined?.url || stage4?.contact_sheet?.url || stage5?.preview?.url) && (
                <section className="result-section">
                  <div className="section-title">
                    <h3>후속 처리</h3>
                    <span>Stage 3.5-5</span>
                  </div>
                  <div className="image-grid handoff-grid">
                    {stage35?.reconstruction?.url && (
                      <figure>
                        <figcaption>구조 보정 입력</figcaption>
                        <img src={`${API_BASE}${stage35.reconstruction.url}`} alt="구조 보정 입력" />
                      </figure>
                    )}
                    {stage35?.refined?.url && (
                      <figure>
                        <figcaption>구조 보정 결과</figcaption>
                        <img src={`${API_BASE}${stage35.refined.url}`} alt="구조 보정 결과" />
                      </figure>
                    )}
                    {stage4?.contact_sheet?.url && (
                      <figure>
                        <figcaption>분할 결과</figcaption>
                        <img src={`${API_BASE}${stage4.contact_sheet.url}`} alt="분할 결과" />
                      </figure>
                    )}
                    {stage5?.preview?.url && (
                      <figure>
                        <figcaption>최종 미리보기</figcaption>
                        <img src={`${API_BASE}${stage5.preview.url}`} alt="최종 미리보기" />
                      </figure>
                    )}
                  </div>
                </section>
              )}
            </>
          )}
          <section className="diagnostics-panel">
            <button className="drawer-toggle" type="button" onClick={() => setDiagnosticsOpen((value) => !value)}>
              진단 정보
            </button>
            {diagnosticsOpen && (
              <>
                <ReadinessChecks readiness={demoReadiness} />
                <div className="status-strip">
                  <StatusCard title="실행 환경" body={runtime} />
                  <StatusCard title="모델 상태" body={models} />
                  <StatusCard title="출력 엔진" body={formatStyleEngineStatus(styleEngine)} />
                  <StatusCard title="3D 백엔드" body={formatMeshyStatus(meshyStatus)} />
                  <StatusCard title="생성 전 확인" body={formatPipelinePreflight(pipelinePreflight)} />
                  <StatusCard title="ComfyUI 모델" body={formatComfyModelChoices(comfyModelChoices)} />
                  <StatusCard title="생성 프로필" body={formatExecutionPolicy(executionPolicy)} />
                </div>
                {runLoaded && (
                  <>
                    <div className="summary-grid diagnostics-summary">
                      <RunSummary summary={stage3Summary} />
                      <PipelineSummary pipeline={pipelineInfo} />
                      <ValidationSummary validation={validation} />
                      <StyleEngineSummary engine={styleEngine} />
                    </div>
                    <div className="artifact-list">
                      {stage3?.files?.metadata && <Artifact label="실행 기록" artifact={stage3.files.metadata} />}
                      {stage35?.metadata && <Artifact label="구조 보정 기록" artifact={stage35.metadata} />}
                      {stage4?.manifest && <Artifact label="3D 변환 기록" artifact={stage4.manifest} />}
                      {stage4?.obj && <Artifact label="프록시 OBJ" artifact={stage4.obj} />}
                      {stage4?.meshy?.downloads?.glb && <Artifact label="Meshy GLB" artifact={stage4.meshy.downloads.glb} />}
                      {stage4?.meshy?.downloads?.obj && <Artifact label="Meshy OBJ" artifact={stage4.meshy.downloads.obj} />}
                      {stage4?.meshy?.downloads?.stl && <Artifact label="Meshy STL" artifact={stage4.meshy.downloads.stl} />}
                      {stage5?.manifest && <Artifact label="출력 기록" artifact={stage5.manifest} />}
                      {stage5?.stl && <Artifact label="프록시 STL" artifact={stage5.stl} />}
                      {stage5?.model_files?.glb && <Artifact label="최종 GLB" artifact={stage5.model_files.glb} />}
                      {stage5?.model_files?.obj && <Artifact label="최종 OBJ" artifact={stage5.model_files.obj} />}
                      {stage5?.model_files?.stl && <Artifact label="최종 Meshy STL" artifact={stage5.model_files.stl} />}
                      {stage5?.checklist && <Artifact label="체크리스트" artifact={stage5.checklist} />}
                    </div>
                    {validation?.checks?.length > 0 && <ValidationChecks validation={validation} />}
                  </>
                )}
              </>
            )}
          </section>
          {userLog.length > 0 && <UserLog lines={userLog} />}
        </section>
      </section>
    </main>
  );

  async function waitForJob(initialJob) {
    let job = initialJob;
    setCurrentJob(job);
    setLog(buildUserLog(job));
    applyPartialJobResult(job);
    while (job.status === "queued" || job.status === "running") {
      await sleep(1000);
      const response = await fetch(`${API_BASE}/api/jobs/${job.id}`);
      job = await parseResponse(response);
      setCurrentJob(job);
      setLog(buildUserLog(job));
      applyPartialJobResult(job);
    }
    if (job.status === "failed") {
      throw new Error(job.error ?? "작업이 중단되었습니다.");
    }
    return job.result;
  }

  function applyPartialJobResult(job) {
    if (!job?.partial_result) return;
    applyRunDetail(job.partial_result, { keepJob: true, keepUserLog: true });
  }
}

function ProgressPanel({ job, currentLabel, pipeline }) {
  const statusLabel = jobStatusLabel(job?.status);
  const current = currentLabel || stageLabel(job?.progress?.current_stage) || "준비 중";
  const latest = normalizeUserMessage(job?.last_log) || "단계를 처리하고 있습니다.";
  return (
    <section className={`progress-panel ${job?.status ?? ""}`}>
      <div className="progress-main">
        <span className="eyebrow">진행 상황</span>
        <h3>{current}</h3>
        <p>{latest}</p>
      </div>
      <div className="progress-meta">
        <span>{statusLabel}</span>
        <span>{formatJobTime(job)}</span>
      </div>
      <StageTimeline pipeline={pipeline} currentStage={job?.progress?.current_stage} />
    </section>
  );
}

function StageTimeline({ pipeline, currentStage }) {
  const status = pipeline?.stage_status ?? {};
  return (
    <div className="stage-timeline">
      {STAGES.map((stage) => (
        <div key={stage.key} className={`timeline-stage ${stageState(stage.key, status, currentStage)}`}>
          <strong>{stage.label}</strong>
          <span>{stage.detail}</span>
        </div>
      ))}
    </div>
  );
}

function UserLog({ lines }) {
  return (
    <section className="user-log">
      <h2>진행 안내</h2>
      <ul>
        {lines.map((line, index) => (
          <li key={`${line}-${index}`}>{line}</li>
        ))}
      </ul>
    </section>
  );
}

function LiveReadinessBanner({ readiness }) {
  if (!readiness || readiness.ok === false && readiness.error) {
    return null;
  }
  const missing = readiness.missing_fast_path_components ?? [];
  const budget = readiness.demo_time_budget_seconds ?? readiness.timed_smoke?.max_seconds ?? 240;
  const product3dReady = readiness.product_3d_ready ?? true;
  const failedChecks = (readiness.checks ?? []).filter((check) => !check.ok).map((check) => check.label);
  if (readiness.fast_path_ready && readiness.can_generate && readiness.timed_smoke_ready && product3dReady) {
    return (
      <div className="readiness-banner ready">
        실시간 생성이 {budget}초 예산 안에서 검증되었습니다.
        <div className="readiness-detail">{readiness.next_action}</div>
      </div>
    );
  }
  if (readiness.fast_path_ready && readiness.can_generate && product3dReady && !readiness.timed_smoke_ready) {
    return null;
  }
  return (
    <div className="readiness-banner warning">
      생성 준비가 아직 끝나지 않았습니다.
      {missing.length > 0 ? ` 필요한 항목: ${missing.join(", ")}.` : ""}
      {failedChecks.length > 0 ? <div className="readiness-detail">확인 필요: {failedChecks.join(", ")}</div> : null}
      <div className="readiness-detail">{readiness.next_action}</div>
    </div>
  );
}

function ReadinessChecks({ readiness }) {
  const checks = readiness?.checks ?? [];
  if (!checks.length) {
    return null;
  }
  return (
    <div className="readiness-checks">
      {checks.map((check) => (
        <div key={check.id ?? check.label} className={`readiness-check ${check.ok ? "pass" : "blocked"}`}>
          <strong>{check.ok ? "준비됨" : "확인 필요"}</strong>
          <span>{check.label}</span>
          <small>{check.detail}</small>
        </div>
      ))}
    </div>
  );
}

function PipelineRail({ pipeline, currentStage }) {
  const status = pipeline?.stage_status ?? {};
  return (
    <div className="pipeline-rail">
      {STAGES.map((stage) => (
        <StagePill
          key={stage.key}
          label={stage.shortLabel}
          state={stageState(stage.key, status, currentStage)}
        />
      ))}
    </div>
  );
}

function StagePill({ label, state }) {
  return <span className={`stage-pill ${state}`}>{label}</span>;
}

function EmptyResults({ busy }) {
  return (
    <div className={`empty-results ${busy ? "busy" : ""}`}>
      <div className="empty-frame">
        <span>원본</span>
        <span>깊이맵</span>
        <span>마스크</span>
        <span>스타일</span>
      </div>
      <p>{busy ? "첫 단계가 끝나면 결과가 이곳에 표시됩니다." : "사진을 선택하고 생성을 시작해 주세요."}</p>
    </div>
  );
}

function WorkflowSetup({
  workflows,
  file,
  inspection,
  onFileChange,
  onInspect,
  onInstall,
  onPrepareInstall,
  onInstallExample,
  status,
  inspectStatus,
  busy
}) {
  const stage3 = workflows?.stage3;
  const validation = stage3?.validation;
  const ready = Boolean(validation?.ok);
  const errors = validation?.errors ?? [];
  const warnings = validation?.warnings ?? [];
  const inspectionErrors = inspection?.errors ?? [];
  const inspectionWarnings = inspection?.warnings ?? [];
  const mapping = inspection?.suggested_stage3_mapping ?? [];
  return (
    <section className={`workflow-setup ${ready ? "ready" : "warning"}`}>
      <h2>스타일 변환 워크플로</h2>
      <p>{ready ? "ComfyUI API 워크플로가 설치되어 있습니다." : "생성에 필요한 ComfyUI API 워크플로를 설치해 주세요."}</p>
      <div className="workflow-path">{stage3?.path ?? "워크플로 상태를 확인할 수 없습니다."}</div>
      {errors.length > 0 && <div className="workflow-message">{errors[0]}</div>}
      {warnings.length > 0 && <div className="workflow-message">{warnings[0]}</div>}
      <label>
        API 워크플로 JSON
        <input
          type="file"
          accept=".json,application/json"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
      </label>
      <div className="workflow-actions">
        <button disabled={busy || !file} onClick={onInspect}>검사</button>
        <button disabled={busy || !file} onClick={onPrepareInstall}>준비 후 설치</button>
        <button disabled={busy || !file} onClick={onInstall}>설치</button>
        <button disabled={busy} onClick={onInstallExample}>예제 설치</button>
      </div>
      {inspectStatus && <div className="workflow-message">{inspectStatus}</div>}
      {inspection && (
        <div className="workflow-inspection">
          <div>형식: {inspection.format ?? "-"}</div>
          <div>노드: {inspection.node_count ?? 0}</div>
          {inspection.validation && (
            <div>검사: {inspection.validation.ok ? "통과" : "수정 필요"}</div>
          )}
          {inspectionErrors.slice(0, 2).map((item, index) => (
            <div key={`inspection-error-${index}`} className="workflow-message">{item}</div>
          ))}
          {inspectionWarnings.slice(0, 2).map((item, index) => (
            <div key={`inspection-warning-${index}`} className="workflow-message">{item}</div>
          ))}
          {mapping.length > 0 && (
            <ul>
              {mapping.slice(0, 6).map((item) => (
                <li key={item.placeholder}>
                  <strong>{item.placeholder}</strong>: {item.use}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {status && <div className="workflow-message">{status}</div>}
    </section>
  );
}

function Artifact({ label, artifact }) {
  return (
    <div>
      <strong>{label}</strong>
      <span>{artifact.path}</span>
    </div>
  );
}

function StatusCard({ title, body }) {
  return (
    <section className="status-card">
      <h2>{title}</h2>
      <pre>{body}</pre>
    </section>
  );
}

function RunSummary({ summary }) {
  if (!summary) {
    return (
      <section className="summary-card">
        <h2>생성 설정</h2>
        <p>아직 불러온 결과가 없습니다.</p>
      </section>
    );
  }
  const options = summary.options ?? {};
  return (
    <section className="summary-card">
      <h2>생성 설정</h2>
      <dl>
        <dt>시드</dt>
        <dd>{options.seed ?? "-"}</dd>
        <dt>단계</dt>
        <dd>{options.steps ?? "-"}</dd>
        <dt>가이던스</dt>
        <dd>{options.guidance ?? "-"}</dd>
        <dt>영향도</dt>
        <dd>{options.strength ?? "-"}</dd>
        <dt>해상도</dt>
        <dd>{options.max_resolution ?? "-"}</dd>
      </dl>
    </section>
  );
}

function PipelineSummary({ pipeline }) {
  const status = pipeline?.stage_status ?? {};
  return (
    <section className="summary-card">
      <h2>진행 단계</h2>
      <dl>
        <dt>이미지</dt>
        <dd>{status.stage3 ? "완료" : "-"}</dd>
        <dt>보정</dt>
        <dd>{status.stage35 ? "완료" : "-"}</dd>
        <dt>3D</dt>
        <dd>{status.stage4 ? "완료" : "-"}</dd>
        <dt>출력</dt>
        <dd>{status.stage5 ? "완료" : "-"}</dd>
      </dl>
    </section>
  );
}

function ValidationSummary({ validation }) {
  if (!validation) {
    return (
      <section className="summary-card">
        <h2>결과 확인</h2>
        <p>결과가 완료되면 산출물 확인 상태가 표시됩니다.</p>
      </section>
    );
  }
  return (
    <section className={`summary-card ${validation.ok ? "valid" : "invalid"}`}>
      <h2>결과 확인</h2>
      <dl>
        <dt>상태</dt>
        <dd>{validation.ok ? "정상" : "확인 필요"}</dd>
        <dt>오류</dt>
        <dd>{validation.error_count ?? 0}</dd>
        <dt>주의</dt>
        <dd>{validation.warning_count ?? 0}</dd>
        <dt>확인 항목</dt>
        <dd>{validation.checks?.length ?? 0}</dd>
      </dl>
    </section>
  );
}

function ValidationChecks({ validation }) {
  const visible = (validation.checks ?? [])
    .filter((check) => check.level !== "pass")
    .slice(0, 8);
  if (!visible.length) {
    return <div className="validation-list pass">결과 확인을 통과했습니다.</div>;
  }
  return (
    <div className="validation-list">
      {visible.map((check, index) => (
        <div key={`${check.stage}-${check.code}-${index}`} className={`validation-item ${check.level}`}>
          <strong>{check.stage}</strong>
          <span>{check.message}</span>
        </div>
      ))}
    </div>
  );
}

function StyleEngineSummary({ engine }) {
  if (!engine || engine.ok === false) {
    return (
      <section className="summary-card">
        <h2>출력 엔진</h2>
        <p>{engine?.error ?? "출력 엔진 상태를 확인할 수 없습니다."}</p>
      </section>
    );
  }
  return (
    <section className="summary-card">
      <h2>출력 엔진</h2>
      <dl>
        <dt>상태</dt>
        <dd>{engine.fast_path_ready ? "준비됨" : "확인 필요"}</dd>
        <dt>속도</dt>
        <dd>{formatLiveReadiness(engine)}</dd>
        <dt>프로필</dt>
        <dd>{engine.runtime?.backend_mode ?? "-"}</dd>
      </dl>
    </section>
  );
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") ?? "";
  const data = contentType.includes("application/json") ? await response.json() : { detail: await response.text() };
  if (!response.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ?? (detail ? JSON.stringify(detail) : response.statusText);
    throw new Error(message);
  }
  return data;
}

async function fetchJsonQuiet(path) {
  try {
    const response = await fetch(`${API_BASE}${path}`);
    return parseResponse(response);
  } catch (error) {
    return { ok: false, error: String(error.message ?? error) };
  }
}

function formatStyleEngineStatus(engine) {
  if (!engine || engine.ok === false) {
    return `확인 불가\n${engine?.error ?? ""}`.trim();
  }
  const lines = [
    `상태: ${engine.fast_path_ready ? "준비됨" : "확인 필요"}`,
    `속도 검증: ${formatLiveReadiness(engine)}`,
    `다음 작업: ${engine.next_action ?? "-"}`
  ];
  return lines.join("\n");
}

function formatMeshyStatus(status) {
  if (!status || status.ok === false && status.error) {
    return `확인 불가\n${status?.error ?? ""}`.trim();
  }
  return [
    `상태: ${status.ok ? "준비됨" : "확인 필요"}`,
    `API 키: ${status.api_key_present ? "설정됨" : `${status.api_key_env ?? "MESHY_API_KEY"} 없음`}`,
    `요청 형식: ${(status.target_formats ?? []).join(", ") || "-"}`,
    `다운로드: ${status.download_outputs_ready ? "사용" : "미사용"}`,
    `모델 형식: ${(status.model_output_formats ?? []).join(", ") || "없음"}`
  ].join("\n");
}

function formatPipelinePreflight(preflight) {
  if (!preflight || preflight.ok === false && preflight.error) {
    return `확인 불가\n${preflight?.error ?? ""}`.trim();
  }
  const failed = (preflight.checks ?? [])
    .filter((check) => check.blocking && !check.ok)
    .map((check) => check.label);
  const warnings = preflight.warnings ?? [];
  return [
    `상태: ${preflight.ok ? "시작 가능" : "막힘"}`,
    `엔진: ${preflight.resolved_engine ?? "-"}`,
    `확인 필요: ${failed.join(", ") || "-"}`,
    `주의: ${warnings[0] ?? "-"}`
  ].join("\n");
}

function formatComfyModelChoices(choices) {
  if (!choices || choices.ok === false) {
    return `확인 불가\n${choices?.error ?? "ComfyUI 서버에 연결할 수 없습니다."}`.trim();
  }
  const groups = choices.choice_groups ?? [];
  const first = groups[0];
  return [
    `항목 수: ${choices.choice_group_count ?? groups.length}`,
    first ? `${first.class_type}.${first.field}: ${(first.preview ?? []).slice(0, 3).join(", ") || "-"}` : "모델 항목을 찾지 못했습니다."
  ].join("\n");
}

function summarizeStage3Workflow(workflows) {
  const fields = workflows?.stage3?.model_fields ?? [];
  const checkpoint = fields.find((field) => field.key === "checkpoint")?.value ?? "";
  const sampler = fields.find((field) => field.key === "sampler")?.value ?? "";
  const scheduler = fields.find((field) => field.key === "scheduler")?.value ?? "";
  return {
    checkpoint: formatCheckpointName(checkpoint),
    family: workflowFamilyLabel(checkpoint),
    sampler: [sampler, scheduler].filter(Boolean).join(" / ") || "샘플러 확인 중"
  };
}

function formatCheckpointName(value) {
  const text = String(value || "").trim();
  if (!text) return "체크포인트 확인 중";
  const name = text.split(/[\\/]/).pop() || text;
  return name.replace(/\.safetensors$/i, "");
}

function workflowFamilyLabel(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("illustrious") || text.includes("wai") || text.includes("noobai") || text.includes("janku")) {
    return "Illustrious SDXL";
  }
  if (text.includes("sd_xl") || text.includes("sdxl")) {
    return "SDXL";
  }
  if (text.includes("flux")) {
    return "FLUX";
  }
  return "모델 확인 중";
}

function formatLiveReadiness(engine) {
  if (!engine) return "-";
  if (engine.demo_ready) {
    const budget = engine.timed_smoke?.max_seconds ?? 240;
    return `${budget}초 안에서 검증됨`;
  }
  if (engine.fast_path_ready) {
    return "검증 대기";
  }
  return "설정 필요";
}

function readinessLabel(readiness) {
  if (!readiness || readiness.ok === false && readiness.error) return "오프라인";
  if (readiness.fast_path_ready && readiness.can_generate && readiness.product_3d_ready && readiness.timed_smoke_ready) {
    return "검증됨";
  }
  if (readiness.fast_path_ready && readiness.can_generate && readiness.product_3d_ready) {
    return "준비됨";
  }
  return "확인 필요";
}

function readinessClass(readiness) {
  const label = readinessLabel(readiness);
  if (label === "검증됨") return "ready";
  if (label === "준비됨") return "warning";
  return "blocked";
}

function formatExecutionPolicy(policy) {
  if (!policy || policy.ok === false) {
    return `확인 불가\n${policy?.error ?? ""}`.trim();
  }
  const defaults = policy.product_pipeline_defaults ?? {};
  return [
    `흐름: ${policy.user_facing_mode ?? "single_generate"}`,
    `생성: ${defaults.max_resolution ?? "-"}px / ${defaults.steps ?? "-"}단계`,
    `단계: 이미지 -> 보정 -> 3D -> 출력`
  ].join("\n");
}

function formatJobTime(job) {
  const elapsed = Number(job.elapsed_seconds ?? 0);
  return `${elapsed.toFixed(1)}초`;
}

function mergePipelineProgress(pipeline, progress) {
  const progressStatus = progress?.stage_status;
  if (!progressStatus) return pipeline;
  return {
    ...(pipeline ?? {}),
    stage_status: {
      ...(pipeline?.stage_status ?? {}),
      ...progressStatus
    }
  };
}

function stageState(key, status, currentStage) {
  if (status?.[key]) return "done";
  if (currentStage === key) return "current";
  if (currentStage === "done") return "done";
  return "pending";
}

function stageLabel(key) {
  return STAGES.find((stage) => stage.key === key)?.label ?? "";
}

function jobStatusLabel(status) {
  if (status === "queued") return "대기 중";
  if (status === "running") return "진행 중";
  if (status === "succeeded") return "완료";
  if (status === "failed") return "중단됨";
  return "준비 중";
}

function buildUserLog(job, fallback = []) {
  const source = job?.log?.length ? job.log : fallback;
  const lines = toUserMessages(source);
  if (job?.status === "queued" && !lines.length) {
    return ["작업 순서를 기다리고 있습니다."];
  }
  if (job?.status === "running" && !lines.length) {
    return ["작업을 진행하고 있습니다."];
  }
  return lines;
}

function toUserMessages(messages) {
  const seen = new Set();
  const lines = [];
  for (const message of messages ?? []) {
    const normalized = normalizeUserMessage(message);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    lines.push(normalized);
  }
  return lines.slice(-8);
}

function normalizeUserMessage(message) {
  const text = String(message ?? "").trim();
  if (!text) return "";
  if (/[�]/.test(text) || /\?[가-힣]/.test(text)) return "";
  const lower = text.toLowerCase();
  if (lower.includes("traceback") || lower.includes("stack")) return "";
  if (lower.includes("full pipeline")) return text.includes("완료") ? "전체 작업이 완료되었습니다." : "전체 작업을 시작했습니다.";
  if (text.includes("Stage 3.5")) return text.includes("완료") ? "구조 보정이 완료되었습니다." : "구조 보정을 시작했습니다.";
  if (text.includes("Stage 4")) return text.includes("완료") ? "3D 변환 준비가 완료되었습니다." : "3D 변환 준비를 시작했습니다.";
  if (text.includes("Stage 5")) return text.includes("완료") ? "출력 정리가 완료되었습니다." : "출력 정리를 시작했습니다.";
  if (text.length > 160) return `${text.slice(0, 157)}...`;
  return text;
}

function friendlyError(error) {
  const message = String(error?.message ?? error ?? "").trim();
  if (!message) return "알 수 없는 문제가 발생했습니다.";
  if (message.toLowerCase().includes("out of memory")) {
    return "GPU 메모리가 부족합니다. 다른 GPU 작업을 종료하거나 해상도를 낮춰야 합니다.";
  }
  if (message.toLowerCase().includes("not found")) {
    return "필요한 파일이나 모델을 찾지 못했습니다. 설정과 모델 위치를 확인해 주세요.";
  }
  return normalizeUserMessage(message) || message;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function runLabel(item) {
  const stages = [
    item.has_stage35 ? "3.5" : "",
    item.has_stage4 ? "4" : "",
    item.has_stage5 ? "5" : ""
  ].filter(Boolean);
  const suffix = stages.length ? ` [${stages.join(",")}]` : "";
  const remote = item.is_remote ? " Remote" : "";
  return `${item.id}${remote}${suffix}`;
}

createRoot(document.getElementById("root")).render(<App />);
