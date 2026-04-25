const stateUrl = "/ui/state";
let currentState = null;
let currentEpisodeId = null;
let selectedScenarioFamily = null;
let selectedDifficulty = null;
let selectedSeed = 7;
let currentSessionId = window.localStorage.getItem("biomed-ui-session-id") || null;

const el = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showBanner(message, kind = "warn") {
  const banner = el("banner");
  banner.textContent = message;
  banner.className = `banner ${kind === "ok" ? "" : ""}`;
  banner.classList.remove("hidden");
}

function hideBanner() {
  el("banner").classList.add("hidden");
}

function fmt(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function formatNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(4);
}

function renderScenarioSelect(cards) {
  const select = el("scenario-select");
  const preferred = selectedScenarioFamily || select.value || null;
  select.innerHTML = "";
  cards.forEach((card) => {
    const option = document.createElement("option");
    option.value = card.scenario_family;
    option.textContent = card.title || card.scenario_family;
    option.disabled = card.available === false;
    select.appendChild(option);
  });
  if (preferred && cards.some((card) => card.scenario_family === preferred)) {
    select.value = preferred;
  } else if (cards.length) {
    select.value = cards.find((card) => card.available !== false)?.scenario_family || cards[0].scenario_family;
  }
  selectedScenarioFamily = select.value || null;
}

function renderScenarioCards(cards) {
  const root = el("scenario-cards");
  root.innerHTML = "";
  cards.forEach((card) => {
    const node = document.createElement("article");
    node.className = `card ${card.available ? "available" : "unavailable"}`;
    node.innerHTML = `
      <div class="card-title">
        <strong>${escapeHtml(card.title)}</strong>
        <span class="pill">${card.available ? "ready" : "off"}</span>
      </div>
      <div class="muted">${escapeHtml(card.subtitle || "")}</div>
      <p>${escapeHtml(card.description || "")}</p>
      <div class="muted">${escapeHtml(card.benchmark_role || "")}</div>
    `;
    root.appendChild(node);
  });
}

function renderStations(stationMap, activeStation) {
  const root = el("stations");
  root.innerHTML = "";
  stationMap.forEach((item) => {
    const station = item.station || item;
    const node = document.createElement("article");
    node.className = `station ${station === activeStation ? "active" : ""}`;
    node.innerHTML = `
      <div class="station-title">
        <strong>${escapeHtml(station)}</strong>
        <span class="pill">${station === activeStation ? "active" : "idle"}</span>
      </div>
    `;
    root.appendChild(node);
  });
}

function renderEpisodeSummary(state) {
  const root = el("episode-summary");
  const summary = state.current_episode;
  const items = summary
    ? [
        ["Episode", summary.episode_id],
        ["Scenario", summary.scenario_family],
        ["Difficulty", summary.difficulty],
        ["Stage", summary.current_stage],
        ["Steps", summary.step_count],
        ["Reward", formatNumber(summary.cumulative_reward)],
        ["Done", summary.done ? "yes" : "no"],
        ["Done reason", summary.done_reason],
        ["Station", summary.active_station],
      ]
    : [["Episode", "No active episode"], ["Status", "Ready"]];
  root.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="latest-item">
          <div class="muted">${escapeHtml(label)}</div>
          <div class="mono">${escapeHtml(fmt(value))}</div>
        </div>`
    )
    .join("");
  currentEpisodeId = summary?.episode_id || null;
}

function renderPipeline(currentSnapshot) {
  const root = el("pipeline");
  const steps = [
    ["Feedstock Intake", "inspect_feedstock"],
    ["Sample Characterization", "measure_crystallinity"],
    ["Literature Terminal", "query_literature"],
    ["Candidate Registry", "query_candidate_registry"],
    ["Stability Chamber", "run_thermostability_assay"],
    ["Assay Bench", "run_hydrolysis_assay"],
    ["Pretreatment Bench", "test_pretreatment"],
    ["Cocktail Testing", "test_cocktail"],
    ["Expert Review Table", "ask_expert"],
    ["Final Recommendation Board", "finalize_recommendation"],
  ];
  root.innerHTML = "";
  steps.forEach(([label, actionKind]) => {
    const node = document.createElement("div");
    const active = currentSnapshot?.active_station === label;
    const done = currentSnapshot?.legal_next_actions?.some((item) => item.action_kind === actionKind);
    node.className = `pipeline-step ${active ? "current" : ""} ${done ? "done" : ""}`;
    node.innerHTML = `
      <div class="bar-head">
        <strong>${escapeHtml(label)}</strong>
        <span class="pill">${active ? "current" : done ? "available" : "—"}</span>
      </div>
    `;
    root.appendChild(node);
  });
}

function renderReward(snapshot, rewardLabels) {
  const stepReward = el("step-reward");
  const cumReward = el("cum-reward");
  stepReward.textContent = formatNumber(snapshot?.reward);
  cumReward.textContent = formatNumber(snapshot?.cumulative_reward);

  const breakdown = snapshot?.reward_breakdown || {};
  const rows = breakdown && Object.keys(breakdown).length
    ? Object.entries(breakdown)
        .filter(([key]) => key !== "notes" && key !== "total")
        .map(([key, value]) => ({
          key,
          label: rewardLabels?.[key] || key.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()),
          value: Number(value) || 0,
        }))
    : [];
  const root = el("reward-breakdown");
  root.innerHTML = "";
  if (!rows.length) {
    const node = document.createElement("div");
    node.className = "bar";
    node.innerHTML = `<div class="muted">No reward data for this step.</div>`;
    root.appendChild(node);
    return;
  }
  const maxAbs = Math.max(...rows.map((row) => Math.abs(row.value)), 1);
  rows.forEach((row) => {
    const node = document.createElement("div");
    node.className = "bar";
    node.innerHTML = `
      <div class="bar-head">
        <strong>${escapeHtml(row.label)}</strong>
        <span class="mono">${formatNumber(row.value)}</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, (Math.abs(row.value) / maxAbs) * 100)}%"></div></div>
    `;
    root.appendChild(node);
  });
}

function renderLatestOutput(snapshot) {
  const root = el("latest-output");
  const output = snapshot?.latest_output;
  if (!output) {
    root.innerHTML = `<div class="muted">No latest output.</div>`;
    return;
  }
  root.innerHTML = `
    <div class="latest-item"><div class="muted">Type</div><div class="mono">${escapeHtml(output.output_type)}</div></div>
    <div class="latest-item"><div class="muted">Summary</div><div>${escapeHtml(output.summary)}</div></div>
    <div class="latest-item"><div class="muted">Success</div><div class="mono">${escapeHtml(output.success)}</div></div>
    <div class="latest-item"><div class="muted">Confidence</div><div class="mono">${escapeHtml(formatNumber(output.quality_score))}</div></div>
    <div class="latest-item"><div class="muted">Uncertainty</div><div class="mono">${escapeHtml(formatNumber(output.uncertainty))}</div></div>
  `;
}

function renderSignals(snapshot) {
  const root = el("signal-block");
  const items = [];
  if (snapshot?.why_this_mattered) items.push(["Why", snapshot.why_this_mattered]);
  if (snapshot?.warnings?.length) items.push(["Warnings", snapshot.warnings.join(" | ")]);
  if (snapshot?.violations?.hard?.length) items.push(["Hard violations", snapshot.violations.hard.join(" | ")]);
  if (snapshot?.violations?.soft?.length) items.push(["Soft violations", snapshot.violations.soft.join(" | ")]);
  if (snapshot?.uncertainty_summary) {
    items.push(["Uncertainty", JSON.stringify(snapshot.uncertainty_summary)]);
  }
  root.innerHTML = items.length
    ? items
        .map(
          ([label, value]) => `
            <div class="signal-item">
              <div class="muted">${escapeHtml(label)}</div>
              <div>${escapeHtml(value)}</div>
            </div>`
        )
        .join("")
    : `<div class="muted">No signals yet.</div>`;
}

function renderTimeline(steps) {
  const root = el("timeline");
  const template = document.getElementById("timeline-item-template");
  root.innerHTML = "";
  if (!steps || !steps.length) {
    root.innerHTML = `<div class="muted">No steps recorded.</div>`;
    return;
  }
  steps.forEach((step) => {
    const node = template.content.cloneNode(true);
    node.querySelector(".timeline-title").textContent = `${step.step_index}: ${step.action_kind || "reset"}`;
    node.querySelector(".timeline-meta").textContent = `${step.stage} · ${step.active_station}`;
    const body = node.querySelector(".timeline-body");
    body.innerHTML = `
      <div><span class="muted">Reward:</span> <span class="mono">${formatNumber(step.reward)}</span></div>
      <div><span class="muted">Done:</span> <span class="mono">${escapeHtml(step.done)}</span></div>
      <div><span class="muted">Rationale:</span> ${escapeHtml(step.action_rationale || "—")}</div>
      <div><span class="muted">Output:</span> ${escapeHtml(step.latest_output?.summary || "—")}</div>
      <div><span class="muted">Why this mattered:</span> ${escapeHtml(step.why_this_mattered || "—")}</div>
    `;
    root.appendChild(node);
  });
}

function renderActionBuilder(snapshot) {
  const select = el("action-select");
  const form = el("action-form");
  const legal = snapshot?.legal_next_actions || [];
  select.innerHTML = "";
  if (!legal.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No action";
    select.appendChild(option);
    form.innerHTML = `<div class="muted">No action available.</div>`;
    el("step-btn").disabled = true;
    return;
  }
  el("step-btn").disabled = false;
  legal.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.action_kind;
    option.textContent = item.action_kind;
    select.appendChild(option);
  });
  if (!select.value) select.value = legal[0].action_kind;
  renderActionForm(select.value, snapshot);
}

function renderActionForm(actionKind, snapshot) {
  const form = el("action-form");
  const legal = snapshot?.legal_next_actions || [];
  const spec = legal.find((item) => item.action_kind === actionKind) || { required_fields: [], optional_fields: [] };
  const fields = [];

  const commonFields = `
    <label>Reason <textarea id="action-rationale" rows="3">${escapeHtml(
      actionKind ? `I chose ${actionKind} from the visible evidence.` : ""
    )}</textarea></label>
    <label>Confidence <input id="action-confidence" type="number" min="0" max="1" step="0.05" value="0.75" /></label>
  `;

  switch (actionKind) {
    case "query_literature":
      fields.push(`<label>Query <input id="query-focus" type="text" value="PET bioremediation" /></label>`);
      break;
    case "query_candidate_registry":
      fields.push(`<label>Family hint
        <select id="family-hint">
          <option value="">auto</option>
          <option value="pretreat_then_single">pretreat_then_single</option>
          <option value="thermostable_single">thermostable_single</option>
          <option value="cocktail">cocktail</option>
          <option value="no_go">no_go</option>
        </select>
      </label>`);
      break;
    case "run_hydrolysis_assay":
      fields.push(`<label>Candidate family
        <select id="candidate-family">
          <option value="pretreat_then_single">pretreat_then_single</option>
          <option value="thermostable_single">thermostable_single</option>
          <option value="cocktail">cocktail</option>
          <option value="no_go">no_go</option>
        </select>
      </label>`);
      fields.push(`<label><input id="pretreated" type="checkbox" /> Pretreated</label>`);
      break;
    case "ask_expert":
      fields.push(`<label>Expert
        <select id="expert-id">
          <option value="wet_lab_lead">wet_lab_lead</option>
          <option value="computational_biologist">computational_biologist</option>
          <option value="process_engineer">process_engineer</option>
          <option value="cost_reviewer">cost_reviewer</option>
        </select>
      </label>`);
      fields.push(`<label>Question <input id="expert-question" type="text" placeholder="Next step" /></label>`);
      break;
    case "state_hypothesis":
      fields.push(`<label>Hypothesis <textarea id="hypothesis" rows="3">The evidence suggests a route bottleneck.</textarea></label>`);
      break;
    case "finalize_recommendation":
      fields.push(`<label>Bottleneck
        <select id="bottleneck">
          <option value="substrate_accessibility">substrate_accessibility</option>
          <option value="thermostability">thermostability</option>
          <option value="contamination_artifact">contamination_artifact</option>
          <option value="cocktail_synergy">cocktail_synergy</option>
          <option value="candidate_mismatch">candidate_mismatch</option>
          <option value="no_go">no_go</option>
        </select>
      </label>`);
      fields.push(`<label>Recommended family
        <select id="recommended-family">
          <option value="pretreat_then_single">pretreat_then_single</option>
          <option value="thermostable_single">thermostable_single</option>
          <option value="cocktail">cocktail</option>
          <option value="no_go">no_go</option>
        </select>
      </label>`);
      fields.push(`<label>Decision type
        <select id="decision-type">
          <option value="proceed">proceed</option>
          <option value="no_go">no_go</option>
        </select>
      </label>`);
      fields.push(`<label>Summary <textarea id="final-summary" rows="3">The evidence supports this route.</textarea></label>`);
      fields.push(`<label>Evidence IDs <input id="evidence-ids" type="text" value="${escapeHtml(
        (snapshot?.artifacts || []).map((item) => item.artifact_id).join(", ")
      )}" /></label>`);
      break;
    default:
      break;
  }

  form.innerHTML = `${fields.join("")}<div class="stack">${commonFields}</div>`;
}

function actionPayloadFromForm(actionKind) {
  const rationale = el("action-rationale")?.value || "";
  const confidenceValue = Number(el("action-confidence")?.value);
  const confidence = Number.isFinite(confidenceValue) ? confidenceValue : null;
  const parameters = {};

  switch (actionKind) {
    case "query_literature":
      parameters.query_focus = el("query-focus")?.value || "PET bioremediation";
      break;
    case "query_candidate_registry": {
      const hint = el("family-hint")?.value;
      if (hint) parameters.family_hint = hint;
      break;
    }
    case "run_hydrolysis_assay":
      parameters.candidate_family = el("candidate-family")?.value || "pretreat_then_single";
      parameters.pretreated = Boolean(el("pretreated")?.checked);
      break;
    case "ask_expert":
      parameters.expert_id = el("expert-id")?.value || "wet_lab_lead";
      const question = el("expert-question")?.value || "";
      if (question) parameters.question = question;
      break;
    case "state_hypothesis":
      parameters.hypothesis = el("hypothesis")?.value || "";
      break;
    case "finalize_recommendation":
      parameters.bottleneck = el("bottleneck")?.value || "substrate_accessibility";
      parameters.recommended_family = el("recommended-family")?.value || "pretreat_then_single";
      parameters.decision_type = el("decision-type")?.value || "proceed";
      parameters.summary = el("final-summary")?.value || "";
      parameters.evidence_artifact_ids = (el("evidence-ids")?.value || "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      break;
    default:
      break;
  }

  return {
    action_kind: actionKind,
    parameters,
    rationale,
    confidence,
    schema_version: "biomed_v2",
  };
}

async function fetchJson(url, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(currentSessionId ? { "x-biomed-session-id": currentSessionId } : {}),
    ...(options.headers || {}),
  };
  const response = await fetch(url, {
    headers,
    ...options,
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "string" ? body : body.detail || JSON.stringify(body);
    throw new Error(detail || response.statusText);
  }
  if (body && typeof body === "object" && typeof body.session_id === "string" && body.session_id) {
    currentSessionId = body.session_id;
    window.localStorage.setItem("biomed-ui-session-id", currentSessionId);
  }
  return body;
}

function updateExportsButtons(state) {
  const enabled = Boolean(state?.current_episode_id);
  el("export-json-btn").disabled = !enabled;
  el("export-md-btn").disabled = !enabled;
}

function renderState(state) {
  currentState = state;
  hideBanner();
  renderScenarioSelect(state.scenario_cards || []);
  renderScenarioCards(state.scenario_cards || []);
  renderEpisodeSummary(state);
  renderStations(state.station_map || [], state.current_snapshot?.active_station);
  renderPipeline(state.current_snapshot);
  renderReward(state.current_snapshot, state.reward_labels || {});
  renderLatestOutput(state.current_snapshot);
  renderSignals(state.current_snapshot);
  renderTimeline(state.current_episode_replay?.steps || []);
  renderActionBuilder(state.current_snapshot);
  updateExportsButtons(state);
  const status = el("status-pill");
  status.textContent = state.current_episode?.done ? "Done" : state.current_episode ? "Active" : "Idle";
}

async function refreshState() {
  try {
    const state = await fetchJson(stateUrl);
    if (state?.session_id) {
      currentSessionId = state.session_id;
      window.localStorage.setItem("biomed-ui-session-id", currentSessionId);
    }
    selectedScenarioFamily = selectedScenarioFamily || state?.current_episode?.scenario_family || null;
    selectedDifficulty = selectedDifficulty || state?.current_episode?.difficulty || null;
    renderState(state);
  } catch (error) {
    showBanner(error.message || String(error), "bad");
  }
}

async function doReset() {
  const body = {
    seed: Number(el("seed-input").value || selectedSeed || 0),
    scenario_family: selectedScenarioFamily || el("scenario-select").value || null,
    difficulty: selectedDifficulty || el("difficulty-select").value || null,
  };
  try {
    await fetchJson("/ui/demo/reset", { method: "POST", body: JSON.stringify(body) });
    await refreshState();
  } catch (error) {
    showBanner(error.message || String(error), "bad");
  }
}

async function doStep() {
  const actionKind = el("action-select").value;
  if (!actionKind) return;
  const body = actionPayloadFromForm(actionKind);
  try {
    await fetchJson("/ui/demo/step", { method: "POST", body: JSON.stringify(body) });
    await refreshState();
  } catch (error) {
    showBanner(error.message || String(error), "bad");
  }
}

async function doRunBaseline() {
  const body = {
    policy_name: "characterize_first",
    max_steps: 10,
    seed: Number(el("seed-input").value || selectedSeed || 0),
    scenario_family: selectedScenarioFamily || el("scenario-select").value || null,
    difficulty: selectedDifficulty || el("difficulty-select").value || null,
  };
  try {
    await fetchJson("/ui/demo/run-baseline", { method: "POST", body: JSON.stringify(body) });
    await refreshState();
  } catch (error) {
    showBanner(error.message || String(error), "bad");
  }
}

async function doJudgeMode() {
  if (!currentEpisodeId) {
    showBanner("No episode yet.", "bad");
    return;
  }
  const panel = el("judge-panel");
  try {
    const data = await fetchJson(`/ui/episodes/${encodeURIComponent(currentEpisodeId)}/debug`);
    panel.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    panel.textContent = "Hidden truth is off.";
    showBanner(error.message || String(error), "warn");
  }
}

async function exportEpisode(format) {
  if (!currentEpisodeId) return;
  const url = `/ui/export/${encodeURIComponent(currentEpisodeId)}.${format}`;
  window.open(url, "_blank", "noopener,noreferrer");
}

document.addEventListener("DOMContentLoaded", async () => {
  el("action-select").addEventListener("change", (event) => {
    renderActionForm(event.target.value, currentState?.current_snapshot);
  });
  el("scenario-select").addEventListener("change", (event) => {
    selectedScenarioFamily = event.target.value || null;
  });
  el("difficulty-select").addEventListener("change", (event) => {
    selectedDifficulty = event.target.value || null;
  });
  el("seed-input").addEventListener("change", (event) => {
    selectedSeed = Number(event.target.value || 0);
  });
  el("reset-btn").addEventListener("click", doReset);
  el("step-btn").addEventListener("click", doStep);
  el("run-demo-btn").addEventListener("click", doRunBaseline);
  el("judge-btn").addEventListener("click", doJudgeMode);
  el("export-json-btn").addEventListener("click", () => exportEpisode("json"));
  el("export-md-btn").addEventListener("click", () => exportEpisode("md"));
  selectedSeed = Number(el("seed-input").value || 7);
  selectedDifficulty = el("difficulty-select").value || null;
  await refreshState();
});
