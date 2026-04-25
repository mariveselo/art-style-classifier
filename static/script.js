const STYLE_METADATA = [
    { id: "Abstract_Expressionism", label: "Абстрактный экспрессионизм" },
    { id: "Art_Nouveau_Modern", label: "Модерн / ар-нуво" },
    { id: "Baroque", label: "Барокко" },
    { id: "Expressionism", label: "Экспрессионизм" },
    { id: "Impressionism", label: "Импрессионизм" },
    { id: "Northern_Renaissance", label: "Северное Возрождение" },
    { id: "Post_Impressionism", label: "Постимпрессионизм" },
    { id: "Realism", label: "Реализм" },
    { id: "Romanticism", label: "Романтизм" },
    { id: "Symbolism", label: "Символизм" },
];

const STYLE_BY_ID = Object.fromEntries(
    STYLE_METADATA.map((style) => [style.id, style.label])
);

const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const status = document.getElementById("status");
const supportedStyles = document.getElementById("supportedStyles");
const uploadBox = document.querySelector(".upload-box");
const themeRoot = document.documentElement;
const STORAGE_DB_NAME = "art-classifier-ui";
const STORAGE_STORE_NAME = "ui-state";
const LAST_IMAGE_KEY = "last-image";

const THEME_VARIABLES = [
    "--paper",
    "--paper-deep",
    "--wash-top",
    "--wash-bottom",
    "--grain",
    "--panel-strong",
    "--panel-surface",
    "--text-panel-surface",
    "--visual-panel-surface",
    "--overlay-top",
    "--overlay-bottom",
    "--overlay-spot",
    "--ink",
    "--muted",
    "--line",
    "--accent",
    "--accent-deep",
    "--accent-soft",
    "--success",
    "--danger",
    "--upload-border",
    "--upload-hover-border",
    "--upload-active-border",
    "--upload-bg-start",
    "--upload-bg-end",
    "--button-start",
    "--button-end",
    "--preview-bg-start",
    "--preview-bg-end",
    "--card-bg",
    "--card-bg-strong",
    "--progress-start",
    "--progress-end",
    "--shadow",
];

let currentPreviewUrl = null;
let selectedFile = null;
let themeRequestId = 0;
let storageDbPromise = null;

function formatTechnicalLabel(value) {
    return value.replaceAll("_", " ");
}

function getReadableLabel(value) {
    return STYLE_BY_ID[value] || formatTechnicalLabel(value);
}

function setStatus(message, kind = "") {
    status.textContent = message;
    status.className = kind ? `status ${kind}` : "status";
}

function syncInputFile(file) {
    try {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
    } catch (error) {
        // Некоторые браузеры запрещают напрямую назначать FileList.
    }
}

function getStorageDb() {
    if (!("indexedDB" in window)) {
        return Promise.resolve(null);
    }

    if (!storageDbPromise) {
        storageDbPromise = new Promise((resolve, reject) => {
            const request = indexedDB.open(STORAGE_DB_NAME, 1);

            request.onupgradeneeded = () => {
                const database = request.result;
                if (!database.objectStoreNames.contains(STORAGE_STORE_NAME)) {
                    database.createObjectStore(STORAGE_STORE_NAME, { keyPath: "id" });
                }
            };

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        }).catch(() => null);
    }

    return storageDbPromise;
}

async function saveLastImage(file) {
    const database = await getStorageDb();
    if (!database) {
        return;
    }

    await new Promise((resolve, reject) => {
        const transaction = database.transaction(STORAGE_STORE_NAME, "readwrite");
        const store = transaction.objectStore(STORAGE_STORE_NAME);

        store.put({
            id: LAST_IMAGE_KEY,
            blob: file,
            name: file.name,
            type: file.type,
            updatedAt: Date.now(),
        });

        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
    }).catch(() => {});
}

async function loadLastImage() {
    const database = await getStorageDb();
    if (!database) {
        return null;
    }

    return new Promise((resolve, reject) => {
        const transaction = database.transaction(STORAGE_STORE_NAME, "readonly");
        const store = transaction.objectStore(STORAGE_STORE_NAME);
        const request = store.get(LAST_IMAGE_KEY);

        request.onsuccess = () => resolve(request.result || null);
        request.onerror = () => reject(request.error);
    }).catch(() => null);
}

async function deleteLastImage() {
    const database = await getStorageDb();
    if (!database) {
        return;
    }

    await new Promise((resolve, reject) => {
        const transaction = database.transaction(STORAGE_STORE_NAME, "readwrite");
        const store = transaction.objectStore(STORAGE_STORE_NAME);

        store.delete(LAST_IMAGE_KEY);

        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
    }).catch(() => {});
}

async function restoreLastImage() {
    const stored = await loadLastImage();
    if (!stored?.blob) {
        return;
    }

    const restoredFile = new File(
        [stored.blob],
        stored.name || "last-image",
        {
            type: stored.type || stored.blob.type || "image/jpeg",
            lastModified: stored.updatedAt || Date.now(),
        }
    );

    syncInputFile(restoredFile);
    await applySelectedFile(restoredFile, { persist: false, restored: true });
}

function clampByte(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
}

function mixColor(source, target, targetWeight) {
    const sourceWeight = 1 - targetWeight;
    return {
        r: clampByte(source.r * sourceWeight + target.r * targetWeight),
        g: clampByte(source.g * sourceWeight + target.g * targetWeight),
        b: clampByte(source.b * sourceWeight + target.b * targetWeight),
    };
}

function toRgb(color) {
    return `rgb(${color.r}, ${color.g}, ${color.b})`;
}

function toRgba(color, alpha) {
    return `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
}

function resetDynamicTheme() {
    THEME_VARIABLES.forEach((name) => {
        themeRoot.style.removeProperty(name);
    });
}

function extractAverageColor(file) {
    return new Promise((resolve, reject) => {
        const objectUrl = URL.createObjectURL(file);
        const image = new Image();

        image.onload = () => {
            try {
                const canvas = document.createElement("canvas");
                const context = canvas.getContext("2d", { willReadFrequently: true });

                if (!context) {
                    throw new Error("Canvas context is unavailable.");
                }

                const sampleWidth = 24;
                const sampleHeight = Math.max(
                    1,
                    Math.round((image.naturalHeight / image.naturalWidth) * sampleWidth)
                );

                canvas.width = sampleWidth;
                canvas.height = sampleHeight;
                context.drawImage(image, 0, 0, sampleWidth, sampleHeight);

                const { data } = context.getImageData(0, 0, sampleWidth, sampleHeight);
                let red = 0;
                let green = 0;
                let blue = 0;
                let alphaTotal = 0;

                for (let index = 0; index < data.length; index += 4) {
                    const alpha = data[index + 3] / 255;
                    red += data[index] * alpha;
                    green += data[index + 1] * alpha;
                    blue += data[index + 2] * alpha;
                    alphaTotal += alpha;
                }

                if (!alphaTotal) {
                    throw new Error("Image alpha is zero.");
                }

                resolve({
                    r: clampByte(red / alphaTotal),
                    g: clampByte(green / alphaTotal),
                    b: clampByte(blue / alphaTotal),
                });
            } catch (error) {
                reject(error);
            } finally {
                URL.revokeObjectURL(objectUrl);
            }
        };

        image.onerror = () => {
            URL.revokeObjectURL(objectUrl);
            reject(new Error("Unable to load image for theme extraction."));
        };

        image.src = objectUrl;
    });
}

async function applyColorThemeFromImage(file, requestId) {
    try {
        const average = await extractAverageColor(file);
        if (requestId !== themeRequestId) {
            return;
        }

        const paper = mixColor(average, { r: 236, g: 231, b: 222 }, 0.82);
        const paperDeep = mixColor(average, { r: 216, g: 209, b: 197 }, 0.76);
        const accent = mixColor(average, { r: 62, g: 53, b: 45 }, 0.54);
        const accentDeep = mixColor(accent, { r: 24, g: 18, b: 14 }, 0.34);
        const muted = mixColor(average, { r: 100, g: 91, b: 82 }, 0.72);
        const ink = mixColor(average, { r: 30, g: 24, b: 20 }, 0.86);
        const panelSurface = mixColor(average, { r: 249, g: 246, b: 240 }, 0.88);
        const textPanelSurface = mixColor(average, { r: 246, g: 241, b: 234 }, 0.82);
        const visualPanelSurface = mixColor(average, { r: 246, g: 241, b: 234 }, 0.74);
        const cardBackground = mixColor(average, { r: 255, g: 250, b: 245 }, 0.82);
        const cardBackgroundStrong = mixColor(average, { r: 255, g: 251, b: 247 }, 0.86);
        const uploadStart = mixColor(average, { r: 255, g: 252, b: 247 }, 0.9);
        const uploadEnd = mixColor(average, { r: 235, g: 229, b: 220 }, 0.82);
        const previewStart = mixColor(average, { r: 255, g: 252, b: 248 }, 0.8);
        const previewEnd = mixColor(average, { r: 234, g: 228, b: 220 }, 0.68);
        const progressEnd = mixColor(average, { r: 133, g: 117, b: 106 }, 0.58);
        const panelStrong = mixColor(average, { r: 244, g: 240, b: 233 }, 0.86);

        themeRoot.style.setProperty("--paper", toRgb(paper));
        themeRoot.style.setProperty("--paper-deep", toRgb(paperDeep));
        themeRoot.style.setProperty("--wash-top", toRgba(mixColor(average, { r: 91, g: 76, b: 61 }, 0.45), 0.12));
        themeRoot.style.setProperty("--wash-bottom", toRgba(mixColor(average, { r: 67, g: 57, b: 49 }, 0.5), 0.1));
        themeRoot.style.setProperty("--grain", toRgba(mixColor(average, { r: 55, g: 45, b: 36 }, 0.52), 0.024));
        themeRoot.style.setProperty("--panel-strong", toRgba(panelStrong, 0.9));
        themeRoot.style.setProperty("--panel-surface", toRgba(panelSurface, 0.56));
        themeRoot.style.setProperty("--text-panel-surface", toRgba(textPanelSurface, 0.86));
        themeRoot.style.setProperty("--visual-panel-surface", toRgba(visualPanelSurface, 0.76));
        themeRoot.style.setProperty("--overlay-top", toRgba(mixColor(average, { r: 244, g: 239, b: 230 }, 0.76), 0.14));
        themeRoot.style.setProperty("--overlay-bottom", toRgba(mixColor(average, { r: 214, g: 205, b: 191 }, 0.68), 0.12));
        themeRoot.style.setProperty("--overlay-spot", "rgba(255, 255, 255, 0.05)");
        themeRoot.style.setProperty("--ink", toRgb(ink));
        themeRoot.style.setProperty("--muted", toRgb(muted));
        themeRoot.style.setProperty("--line", toRgba(accentDeep, 0.18));
        themeRoot.style.setProperty("--accent", toRgb(accent));
        themeRoot.style.setProperty("--accent-deep", toRgb(accentDeep));
        themeRoot.style.setProperty("--accent-soft", toRgba(accent, 0.1));
        themeRoot.style.setProperty("--success", toRgb(accent));
        themeRoot.style.setProperty("--danger", toRgb(mixColor(average, { r: 92, g: 54, b: 48 }, 0.5)));
        themeRoot.style.setProperty("--upload-border", toRgba(accent, 0.32));
        themeRoot.style.setProperty("--upload-hover-border", toRgba(accent, 0.62));
        themeRoot.style.setProperty("--upload-active-border", toRgba(accentDeep, 0.84));
        themeRoot.style.setProperty("--upload-bg-start", toRgba(uploadStart, 0.86));
        themeRoot.style.setProperty("--upload-bg-end", toRgba(uploadEnd, 0.94));
        themeRoot.style.setProperty("--button-start", toRgb(accent));
        themeRoot.style.setProperty("--button-end", toRgb(accentDeep));
        themeRoot.style.setProperty("--preview-bg-start", toRgba(previewStart, 0.52));
        themeRoot.style.setProperty("--preview-bg-end", toRgba(previewEnd, 0.62));
        themeRoot.style.setProperty("--card-bg", toRgba(cardBackground, 0.8));
        themeRoot.style.setProperty("--card-bg-strong", toRgba(cardBackgroundStrong, 0.88));
        themeRoot.style.setProperty("--progress-start", toRgb(accent));
        themeRoot.style.setProperty("--progress-end", toRgb(progressEnd));
        themeRoot.style.setProperty(
            "--shadow",
            `0 24px 60px ${toRgba(accentDeep, 0.16)}`
        );
    } catch (error) {
        resetDynamicTheme();
    }
}

function renderSupportedStyles() {
    supportedStyles.innerHTML = "";

    STYLE_METADATA.forEach((style) => {
        const card = document.createElement("article");
        card.className = "style-card";

        const name = document.createElement("div");
        name.className = "style-name";
        name.textContent = style.label;

        const technical = document.createElement("div");
        technical.className = "style-id";
        technical.textContent = formatTechnicalLabel(style.id);

        card.append(name, technical);
        supportedStyles.appendChild(card);
    });
}

function resetResults() {
    results.innerHTML = `
        <div class="result-empty">
            После запуска анализа страница покажет три наиболее вероятных направления.
        </div>
    `;
}

function clearPreviewUrl() {
    if (currentPreviewUrl) {
        document.body.classList.remove("has-backdrop");
        document.body.style.removeProperty("--uploaded-image");
        URL.revokeObjectURL(currentPreviewUrl);
        currentPreviewUrl = null;
    }
}

function renderPreview(file) {
    clearPreviewUrl();

    if (!file) {
        themeRequestId += 1;
        resetDynamicTheme();
        preview.innerHTML = `
            <div class="placeholder">
                Здесь будет выбранное изображение.
            </div>
        `;
        return;
    }

    currentPreviewUrl = URL.createObjectURL(file);
    preview.innerHTML = "";

    const image = document.createElement("img");
    image.src = currentPreviewUrl;
    image.alt = "Предпросмотр загруженного изображения";
    preview.appendChild(image);
    document.body.style.setProperty("--uploaded-image", `url("${currentPreviewUrl}")`);
    document.body.classList.add("has-backdrop");

    const requestId = ++themeRequestId;
    void applyColorThemeFromImage(file, requestId);
}

async function clearSelectedFile(options = {}) {
    const { clearStored = false } = options;
    selectedFile = null;
    fileInput.value = "";
    fileName.textContent = "Файл пока не выбран.";
    renderPreview(null);
    resetResults();

    if (clearStored) {
        await deleteLastImage();
    }
}

async function applySelectedFile(file, options = {}) {
    const { persist = true, restored = false } = options;
    resetResults();
    setStatus("");

    if (!file) {
        await clearSelectedFile();
        return;
    }

    if (!file.type.startsWith("image/")) {
        setStatus("Нужен файл изображения.", "error");
        selectedFile = null;
        fileName.textContent = "Файл не подходит для анализа.";
        fileInput.value = "";
        renderPreview(null);
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        setStatus("Файл оказался больше 10 МБ.", "error");
        selectedFile = null;
        fileName.textContent = "Выберите изображение меньшего размера.";
        fileInput.value = "";
        renderPreview(null);
        return;
    }

    selectedFile = file;
    fileName.textContent = restored ? `Восстановлено: ${file.name}` : `Выбрано: ${file.name}`;
    renderPreview(file);

    if (persist) {
        await saveLastImage(file);
    }

    setStatus(
        restored ? "Последнее изображение восстановлено." : "Изображение готово к анализу.",
        "success"
    );
}

function renderPredictions(payload) {
    results.innerHTML = "";

    if (!payload.predictions || payload.predictions.length === 0) {
        results.innerHTML = `
            <div class="result-empty">
                Не удалось получить результат.
            </div>
        `;
        return;
    }

    const summary = document.createElement("div");
    summary.className = "result-summary";
    summary.textContent = `Наиболее близким направлением по версии модели оказался стиль «${getReadableLabel(payload.predictions[0].label)}».`;
    results.appendChild(summary);

    payload.predictions.forEach((item, index) => {
        const percent = Math.max(0, Math.min(100, item.probability * 100));
        const card = document.createElement("article");
        card.className = "result-item";

        const topline = document.createElement("div");
        topline.className = "result-topline";

        const textBlock = document.createElement("div");

        const rank = document.createElement("div");
        rank.className = "result-rank";
        rank.textContent = `${index + 1} место`;

        const name = document.createElement("div");
        name.className = "result-name";
        name.textContent = getReadableLabel(item.label);

        const technical = document.createElement("div");
        technical.className = "result-id";
        technical.textContent = formatTechnicalLabel(item.label);

        textBlock.append(rank, name, technical);

        const percentNode = document.createElement("div");
        percentNode.className = "result-percent";
        percentNode.textContent = `${percent.toFixed(2)}%`;

        topline.append(textBlock, percentNode);

        const progress = document.createElement("div");
        progress.className = "progress";

        const fill = document.createElement("span");
        fill.style.width = `${percent.toFixed(2)}%`;
        progress.appendChild(fill);

        card.append(topline, progress);
        results.appendChild(card);
    });
}

fileInput.addEventListener("change", () => {
    void applySelectedFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
    uploadBox.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadBox.classList.add("drag-over");
    });
});

["dragleave", "dragend"].forEach((eventName) => {
    uploadBox.addEventListener(eventName, () => {
        uploadBox.classList.remove("drag-over");
    });
});

uploadBox.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadBox.classList.remove("drag-over");

    const file = event.dataTransfer?.files?.[0];
    if (!file) {
        return;
    }

    syncInputFile(file);
    void applySelectedFile(file);
});

resetButton.addEventListener("click", async () => {
    resetButton.disabled = true;

    try {
        await clearSelectedFile({ clearStored: true });
        setStatus("Изображение и сохранённая версия очищены.", "success");
    } finally {
        resetButton.disabled = false;
    }
});

predictButton.addEventListener("click", async () => {
    const file = selectedFile;

    if (!file) {
        setStatus("Сначала выберите изображение.", "error");
        return;
    }

    if (window.location.protocol === "file:") {
        setStatus(
            "Для анализа запустите сервис и откройте http://127.0.0.1:8000.",
            "error"
        );
        return;
    }

    predictButton.disabled = true;
    resetButton.disabled = true;
    resetResults();
    setStatus("Сервис сравнивает картину с десятью стилями...", "pending");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.detail || "Сервер вернул ошибку.");
        }

        renderPredictions(payload);
        setStatus("Анализ завершён.", "success");
    } catch (error) {
        setStatus(
            error.message || "Не удалось получить ответ от сервера.",
            "error"
        );
    } finally {
        predictButton.disabled = false;
        resetButton.disabled = false;
    }
});

renderSupportedStyles();
void restoreLastImage();
