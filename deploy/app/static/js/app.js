(function () {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const browseButton = document.getElementById("browse-button");
  const clearButton = document.getElementById("clear-button");
  const fileMeta = document.getElementById("file-meta");
  const fileName = document.getElementById("file-name");
  const sampleButton = document.getElementById("sample-button");
  const predictForm = document.getElementById("predict-form");
  const predictButton = document.getElementById("predict-button");
  const progress = document.getElementById("progress");
  const errorBox = document.getElementById("error");
  const resultEmpty = document.getElementById("result-empty");
  const resultBody = document.getElementById("result-body");
  const resultGesture = document.getElementById("result-gesture");
  const resultConfidence = document.getElementById("result-confidence");
  const resultModality = document.getElementById("result-modality");
  const resultTopk = document.getElementById("result-topk");
  const resultMetaDetails = document.getElementById("result-meta-details");
  const metaSequences = document.getElementById("meta-sequences");
  const metaLength = document.getElementById("meta-length");
  const metaInference = document.getElementById("meta-inference");
  const previewCard = document.getElementById("preview-card");
  const previewTable = document.getElementById("preview-table");

  let currentFile = null;

  function setFile(file) {
    currentFile = file;
    if (file) {
      fileName.textContent = `${file.name} (${formatBytes(file.size)})`;
      fileMeta.hidden = false;
      predictButton.disabled = false;
    } else {
      fileMeta.hidden = true;
      predictButton.disabled = true;
    }
    clearError();
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function showError(message) {
    errorBox.hidden = false;
    errorBox.textContent = message;
  }

  function setBusy(busy) {
    predictButton.disabled = busy || !currentFile;
    sampleButton.disabled = busy;
    browseButton.disabled = busy;
    clearButton.disabled = busy;
    progress.hidden = !busy;
  }

  function confidenceClass(pct) {
    if (pct >= 0.7) return "confidence--high";
    if (pct >= 0.4) return "confidence--mid";
    return "confidence--low";
  }

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderResult(data) {
    resultEmpty.hidden = true;
    resultBody.hidden = false;

    resultGesture.textContent = data.predicted_gesture;
    const pct = data.predicted_confidence * 100;
    resultConfidence.className = "result__confidence " + confidenceClass(data.predicted_confidence);
    resultConfidence.querySelector(".result__confidence-value").textContent = `${pct.toFixed(1)}%`;

    const chips = [
      { name: "IMU", present: true },
      { name: "THM", present: data.has_thm },
      { name: "ToF", present: data.has_tof },
    ];
    resultModality.innerHTML = chips
      .map((c) =>
        `<span class="modality__chip ${c.present ? "modality__chip--ok" : "modality__chip--off"}">` +
        `${escapeHtml(c.name)} ${c.present ? "✓" : "missing"}</span>`
      )
      .join("");

    resultTopk.innerHTML = data.top_k
      .map((entry) => {
        const conf = entry.confidence * 100;
        const width = Math.max(2, conf).toFixed(1);
        return (
          `<li class="topk__row">` +
          `<span class="topk__name" title="${escapeHtml(entry.gesture)}">${escapeHtml(entry.gesture)}</span>` +
          `<span class="topk__bar"><span class="topk__bar-fill" style="width: ${width}%"></span></span>` +
          `<span class="topk__pct">${conf.toFixed(1)}%</span>` +
          `</li>`
        );
      })
      .join("");

    metaSequences.textContent = String(data.n_sequences);
    metaLength.textContent = `${data.sequence_length} timesteps`;
    metaInference.textContent = `${data.inference_ms.toFixed(1)} ms`;
    resultMetaDetails.open = false;
  }

  function renderPreview(text) {
    const lines = text.split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) {
      previewCard.hidden = true;
      return;
    }
    const header = lines[0].split(",");
    const rows = lines.slice(1, 6);
    const headHtml = "<thead><tr>" +
      header.map((h) => `<th>${escapeHtml(h)}</th>`).join("") +
      "</tr></thead>";
    const bodyHtml = "<tbody>" +
      rows
        .map((line) => {
          const cells = line.split(",");
          return "<tr>" + cells.map((c) => `<td>${escapeHtml(c)}</td>`).join("") + "</tr>";
        })
        .join("") +
      "</tbody>";
    previewTable.innerHTML = headHtml + bodyHtml;
    previewCard.hidden = false;
  }

  async function submitPredict() {
    if (!currentFile) return;
    clearError();
    setBusy(true);
    try {
      const formData = new FormData();
      formData.append("file", currentFile);
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = payload && payload.message ? payload.message : `Request failed (${response.status})`;
        showError(message);
        return;
      }
      renderResult(payload);
    } catch (err) {
      showError(`Network error: ${err && err.message ? err.message : err}`);
    } finally {
      setBusy(false);
    }
  }

  async function loadSample() {
    clearError();
    setBusy(true);
    try {
      const response = await fetch("/api/sample");
      if (!response.ok) {
        showError(`Sample data not available (${response.status}).`);
        return;
      }
      const text = await response.text();
      const filename = "demo_sequence.csv";
      const file = new File([text], filename, { type: "text/csv" });
      setFile(file);
      renderPreview(text);
    } catch (err) {
      showError(`Could not load sample: ${err && err.message ? err.message : err}`);
    } finally {
      setBusy(false);
    }
  }

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setFile(file);
    file.text().then(renderPreview).catch(() => {
      previewCard.hidden = true;
    });
  });

  browseButton.addEventListener("click", () => fileInput.click());

  clearButton.addEventListener("click", () => {
    fileInput.value = "";
    setFile(null);
    previewCard.hidden = true;
    resultEmpty.hidden = false;
    resultBody.hidden = true;
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    setFile(file);
    file.text().then(renderPreview).catch(() => {
      previewCard.hidden = true;
    });
  });

  predictForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitPredict();
  });

  sampleButton.addEventListener("click", loadSample);
})();
