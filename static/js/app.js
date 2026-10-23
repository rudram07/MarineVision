/**
 * MarineVision — Underwater Intelligence & Sonar Analytics
 * Client-Side Application Logic & Real-time Upload Interactivity
 */

// ============================================
// THEME TOGGLE (Dark/Light Mode)
// ============================================
(function initTheme() {
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('marinevision-theme', theme);
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.textContent = theme === 'dark' ? '☀️' : '🌙';
        }
    }

    const saved = localStorage.getItem('marinevision-theme') || 'light';
    applyTheme(saved);

    document.addEventListener('DOMContentLoaded', () => {
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.textContent = (document.documentElement.getAttribute('data-theme') === 'dark') ? '☀️' : '🌙';
            btn.addEventListener('click', () => {
                const current = document.documentElement.getAttribute('data-theme') || 'light';
                const next = current === 'light' ? 'dark' : 'light';
                applyTheme(next);
            });
        }

        // Reveal on scroll observer
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.02, rootMargin: '0px 0px 80px 0px' });

        document.querySelectorAll('.reveal, .reveal-slow').forEach(el => observer.observe(el));
    });
})();

document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initSampleLoaders();
    initUploadForm();
    initImageZoomModal();
});

/**
 * Drag-and-Drop and File Selection Handling
 */
function initDropzone() {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('imageFileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImg = document.getElementById('previewImg');
    const previewFilename = document.getElementById('previewFilename');
    const previewFilesize = document.getElementById('previewFilesize');
    const previewDimensions = document.getElementById('previewDimensions');
    const btnRemove = document.getElementById('btnRemoveFile');
    const submitBtn = document.getElementById('btnUploadSubmit');

    if (!dropzone || !fileInput) return;

    // Click to select
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('#previewContainer')) return;
        fileInput.click();
    });

    // Dragover / Dragleave
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    // Drop
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            handleFileSelection(files[0]);
        }
    });

    // Input Change
    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files[0]) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    // Remove file
    if (btnRemove) {
        btnRemove.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInput.value = '';
            previewContainer.style.display = 'none';
            if (submitBtn) submitBtn.disabled = true;
        });
    }

    function handleFileSelection(file) {
        // Validate extension
        const allowedExts = ['jpg', 'jpeg', 'png', 'tif', 'tiff'];
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowedExts.includes(ext)) {
            alert(`Unsupported format '.${ext}'. Please choose a JPG, PNG, or TIFF file.`);
            fileInput.value = '';
            return;
        }

        // Validate size (10 MB)
        const maxBytes = 10 * 1024 * 1024;
        if (file.size > maxBytes) {
            alert(`File size (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds 10 MB limit.`);
            fileInput.value = '';
            return;
        }

        // Display preview & inspect dimensions
        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            if (previewImg) previewImg.src = dataUrl;
            
            const tempImg = new Image();
            tempImg.onload = () => {
                if (previewDimensions) {
                    previewDimensions.textContent = `${tempImg.naturalWidth} × ${tempImg.naturalHeight} px`;
                }
                if (tempImg.naturalWidth < 64 || tempImg.naturalHeight < 64) {
                    alert('Warning: Image resolution is very small for sonar analysis.');
                }
            };
            tempImg.src = dataUrl;

            if (previewFilename) previewFilename.textContent = file.name;
            if (previewFilesize) previewFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
            if (previewContainer) previewContainer.style.display = 'block';
            if (submitBtn) submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

/**
 * 1-Click Sonar Sample Loaders
 */
function initSampleLoaders() {
    const sampleBtns = document.querySelectorAll('.btn-load-sample');
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const sampleType = btn.getAttribute('data-sample');
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner-sm"></span> Loading ${sampleType} sample...`;

            showProcessingProgress("Loading sample sonar backscatter...");

            try {
                const response = await fetch(`/api/load-sample/${sampleType}`, {
                    method: 'POST',
                    headers: { 'Accept': 'application/json' }
                });
                const data = await response.json();
                if (data.success && data.redirect_url) {
                    window.location.href = data.redirect_url;
                } else {
                    alert(data.error || 'Failed to load sample sonar image.');
                    btn.disabled = false;
                    hideProcessingProgress();
                }
            } catch (err) {
                console.error(err);
                alert('Network or server error while loading sample.');
                btn.disabled = false;
                hideProcessingProgress();
            }
        });
    });
}

/**
 * Upload Form Submission with Real Progress Steps
 */
function initUploadForm() {
    const form = document.getElementById('imageUploadForm');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        const fileInput = document.getElementById('imageFileInput');
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
            e.preventDefault();
            alert('Please select or drag a sonar image first.');
            return;
        }

        // Show progressive feedback
        showProcessingProgress("Submitting & Preprocessing Sonar Imagery...");
    });
}

function showProcessingProgress(initialText) {
    const modal = document.getElementById('processingModal');
    const msg = document.getElementById('processingModalText');
    if (modal) {
        modal.style.display = 'flex';
        if (msg && initialText) msg.textContent = initialText;

        // Sequence realistic stages
        setTimeout(() => {
            if (msg) msg.textContent = "Validating Sonar File & Checking Image Integrity...";
        }, 300);
        setTimeout(() => {
            if (msg) msg.textContent = "Applying OpenCV Preprocessing (CLAHE & Bilateral Filter)...";
        }, 700);
        setTimeout(() => {
            if (msg) msg.textContent = "Querying Marine Debris & Anomaly Model Interfaces...";
        }, 1100);
        setTimeout(() => {
            if (msg) msg.textContent = "Persisting Run Metadata to SQLite Database...";
        }, 1400);
    }
}

function hideProcessingProgress() {
    const modal = document.getElementById('processingModal');
    if (modal) modal.style.display = 'none';
}

/**
 * Image Comparison Zoom / View Toggle
 */
function initImageZoomModal() {
    const viewButtons = document.querySelectorAll('[data-view-mode]');
    const frameOriginal = document.getElementById('sonarFrameOriginal');
    const frameProcessed = document.getElementById('sonarFrameProcessed');

    viewButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            viewButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const mode = btn.getAttribute('data-view-mode');
            if (mode === 'dual') {
                if (frameOriginal) frameOriginal.style.display = 'block';
                if (frameProcessed) frameProcessed.style.display = 'block';
            } else if (mode === 'processed') {
                if (frameOriginal) frameOriginal.style.display = 'none';
                if (frameProcessed) frameProcessed.style.display = 'block';
            } else if (mode === 'original') {
                if (frameOriginal) frameOriginal.style.display = 'block';
                if (frameProcessed) frameProcessed.style.display = 'none';
            }
        });
    });
}
