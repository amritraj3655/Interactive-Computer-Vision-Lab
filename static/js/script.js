const conceptSelect = document.getElementById("conceptSelect");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const processButton = document.getElementById("processButton");
const parameterSection = document.getElementById("parameterSection");
const parameterFields = document.getElementById("parameterFields");
const statusMessage = document.getElementById("statusMessage");
const originalPreview = document.getElementById("originalPreview");
const processedPreview = document.getElementById("processedPreview");
const downloadLink = document.getElementById("downloadLink");
const learnMore = document.getElementById("learnMore");
const conceptDescription = document.getElementById("conceptDescription");

let concepts = [];
let selectedFile = null;
let originalUrl = null;

async function loadConcepts() {
  try {
    const response = await fetch("/api/concepts");
    if (!response.ok) throw new Error("Could not load concepts");
    concepts = await response.json();
    conceptSelect.innerHTML = '<option value="">Choose a concept...</option>';
    concepts.forEach(concept => {
      const option = document.createElement("option");
      option.value = concept.id;
      option.textContent = concept.name;
      conceptSelect.appendChild(option);
    });
  } catch {
    conceptSelect.innerHTML = '<option value="">Could not load concepts</option>';
    statusMessage.textContent = "Could not connect to the Flask server.";
  }
}

function currentConcept() {
  return concepts.find(item => item.id === conceptSelect.value);
}

function renderParameters(concept) {
  parameterFields.innerHTML = "";
  const params = concept?.parameters || [];
  parameterSection.hidden = params.length === 0;

  params.forEach(param => {
    const wrapper = document.createElement("div");
    wrapper.className = "parameter-item";
    const label = document.createElement("label");
    label.htmlFor = `param-${param.name}`;
    label.textContent = param.label;
    let input;

    if (param.type === "select") {
      input = document.createElement("select");
      param.options.forEach(value => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        input.appendChild(option);
      });
    } else {
      input = document.createElement("input");
      input.type = "number";
      ["min", "max", "step", "value"].forEach(key => {
        if (param[key] !== undefined) input[key] = param[key];
      });
    }

    input.id = `param-${param.name}`;
    input.dataset.parameter = param.name;
    wrapper.append(label, input);
    parameterFields.appendChild(wrapper);
  });
}

function resetOutput() {
  processedPreview.className = "preview empty-preview";
  processedPreview.textContent = "Your processed result will appear here.";
  downloadLink.href = "#";
  downloadLink.classList.add("disabled");
  downloadLink.setAttribute("aria-disabled", "true");
}

conceptSelect.addEventListener("change", () => {
  const concept = currentConcept();
  conceptDescription.textContent = concept?.description || "Select a concept to get started.";
  learnMore.hidden = !concept;
  if (concept) learnMore.href = concept.learn;
  renderParameters(concept);
  resetOutput();
  updateReadyState();

  if (selectedFile && concept?.input === "video" && !selectedFile.type.startsWith("video/")) {
    statusMessage.textContent = "This concept requires a video. Please choose a video file.";
  } else {
    statusMessage.textContent = selectedFile ? "Ready to process." : "Upload a file to continue.";
  }
});

fileInput.addEventListener("change", () => {
  selectedFile = fileInput.files[0] || null;
  fileName.textContent = selectedFile
    ? `${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`
    : "No file selected.";

  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalPreview.innerHTML = "";
  resetOutput();

  if (selectedFile) {
    originalUrl = URL.createObjectURL(selectedFile);
    const element = selectedFile.type.startsWith("video/")
      ? document.createElement("video")
      : document.createElement("img");
    element.src = originalUrl;
    if (element.tagName === "VIDEO") {
      element.controls = true;
      element.muted = true;
    }
    element.alt = "Original uploaded input";
    originalPreview.appendChild(element);
  } else {
    originalPreview.textContent = "Your uploaded file will appear here.";
  }

  updateReadyState();
  statusMessage.textContent = selectedFile ? "File selected. Ready when you are." : "Select a file.";
});

function updateReadyState() {
  const concept = currentConcept();
  const validType = !concept || concept.input !== "video" ||
    (selectedFile && selectedFile.type.startsWith("video/"));
  processButton.disabled = !(concept && selectedFile && validType);
}

function collectParameters() {
  const params = {};
  parameterFields.querySelectorAll("[data-parameter]").forEach(input => {
    params[input.dataset.parameter] = input.value;
  });
  return params;
}

function showProcessedResult(data) {
  processedPreview.innerHTML = "";
  processedPreview.className = "preview";
  const element = data.output_type === "video"
    ? document.createElement("video")
    : document.createElement("img");

  element.src = `${data.result_url}?t=${Date.now()}`;
  if (element.tagName === "VIDEO") {
    element.controls = true;
    element.playsInline = true;
  }
  element.alt = "Processed output";
  processedPreview.appendChild(element);

  downloadLink.href = data.download_url;
  downloadLink.classList.remove("disabled");
  downloadLink.setAttribute("aria-disabled", "false");
}

processButton.addEventListener("click", async () => {
  const concept = currentConcept();
  if (!concept || !selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("concept", concept.id);
  Object.entries(collectParameters()).forEach(([key, value]) => {
    formData.append(key, value);
  });

  processButton.disabled = true;
  processButton.textContent = "Processing...";
  statusMessage.textContent = "Uploading and processing with OpenCV. Please wait.";
  resetOutput();

  try {
    const response = await fetch("/api/process", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Processing failed.");
    showProcessedResult(data);
    statusMessage.textContent = data.message;
  } catch (error) {
    statusMessage.textContent = error.message || "Something went wrong while processing.";
  } finally {
    processButton.textContent = "Process input →";
    updateReadyState();
  }
});

loadConcepts();
