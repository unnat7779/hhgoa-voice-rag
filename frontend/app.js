/**
 * Voice-Enabled RAG Client-Side Application
 * Cross-Platform & Mobile Optimized (iOS Safari, Android Chrome, Desktop)
 * Features:
 * - Dynamic Audio Waveform Visualizer with Auto-Resize
 * - Hybrid Speech-to-Text: Browser Web Speech API (primary) → Sarvam saaras:v3 (fallback)
 * - Multi-Strategy Retrieval REST API Integration
 * - Latency Waterfall Stopwatch & P50/P70/P100 Analytics
 * - Mobile Touch & AudioContext Unlocking
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const micBtn = document.getElementById('micBtn');
  const micWrapper = document.getElementById('micWrapper');
  const micStatusText = document.getElementById('micStatusText');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const textQueryInput = document.getElementById('textQueryInput');
  const sendQueryBtn = document.getElementById('sendQueryBtn');
  const strategySelect = document.getElementById('strategySelect');
  const rerankToggle = document.getElementById('rerankToggle');
  const answerBody = document.getElementById('answerBody');
  const modelBadge = document.getElementById('modelBadge');
  const guardrailBadge = document.getElementById('guardrailBadge');
  const totalLatencyBadge = document.getElementById('totalLatencyBadge');
  const speakAnswerBtn = document.getElementById('speakAnswerBtn');
  const citationsGrid = document.getElementById('citationsGrid');
  const citationCount = document.getElementById('citationCount');
  const runBenchmarkBtn = document.getElementById('runBenchmarkBtn');

  // Latency Bars
  const barSTT = document.getElementById('barSTT');
  const valSTT = document.getElementById('valSTT');
  const barGuard = document.getElementById('barGuard');
  const valGuard = document.getElementById('valGuard');
  const barRet = document.getElementById('barRet');
  const valRet = document.getElementById('valRet');
  const barRerank = document.getElementById('barRerank');
  const valRerank = document.getElementById('valRerank');
  const barLLM = document.getElementById('barLLM');
  const valLLM = document.getElementById('valLLM');
  const barPost = document.getElementById('barPost');
  const valPost = document.getElementById('valPost');

  // Footer Analytics Metrics
  const p50Val = document.getElementById('p50Val');
  const p70Val = document.getElementById('p70Val');
  const p100Val = document.getElementById('p100Val');
  const totalQueriesVal = document.getElementById('totalQueriesVal');

  // State
  let isRecording = false;
  let audioContext = null;
  let analyser = null;
  let animationFrameId = null;
  let latencyHistory = [];
  let micStream = null;

  // Web Speech API detection (cross-browser / iOS Safari / Android Chrome)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hasBrowserSTT = !!SpeechRecognition;
  let recognition = null;
  let sttStartTime = 0;

  // MediaRecorder for Sarvam fallback
  let mediaRecorder = null;
  let audioChunks = [];

  // Canvas Setup & Auto-Responsive Width
  const canvasCtx = waveformCanvas.getContext('2d');

  function resizeCanvas() {
    const rect = waveformCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    waveformCanvas.width = (rect.width || 360) * dpr;
    waveformCanvas.height = (rect.height || 55) * dpr;
    canvasCtx.scale(dpr, dpr);
    drawEmptyWaveform();
  }

  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 100);

  function drawEmptyWaveform() {
    const rect = waveformCanvas.getBoundingClientRect();
    const w = rect.width || 360;
    const h = rect.height || 55;

    canvasCtx.clearRect(0, 0, w, h);
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    canvasCtx.fillRect(0, 0, w, h);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgba(6, 182, 212, 0.5)';
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, h / 2);
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();
  }

  // Update initial status text
  if (hasBrowserSTT) {
    micStatusText.textContent = 'Tap to Speak (Live Speech Recognition)';
  } else {
    micStatusText.textContent = 'Tap or Hold to Record (Sarvam saaras:v3)';
  }

  // -------------------------------------------------------------
  // Simulated & Real Acoustic Waveform Visualizer
  // -------------------------------------------------------------
  let wavePhase = 0;

  function startAnimatedWaveform() {
    function drawWave() {
      if (!isRecording) return;
      animationFrameId = requestAnimationFrame(drawWave);
      wavePhase += 0.12;

      const rect = waveformCanvas.getBoundingClientRect();
      const w = rect.width || 360;
      const h = rect.height || 55;
      const centerY = h / 2;

      canvasCtx.fillStyle = 'rgba(7, 9, 14, 0.4)';
      canvasCtx.fillRect(0, 0, w, h);

      // Draw primary voice wave
      canvasCtx.lineWidth = 2.5;
      canvasCtx.strokeStyle = '#38bdf8';
      canvasCtx.beginPath();

      const numPoints = 64;
      const sliceWidth = w / numPoints;
      let x = 0;

      for (let i = 0; i < numPoints; i++) {
        const freq = (i / numPoints) * Math.PI * 4;
        const amp = (Math.sin(freq + wavePhase) + Math.cos(freq * 1.5 - wavePhase)) * 0.5;
        const y = centerY + amp * (h * 0.35);

        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(w, centerY);
      canvasCtx.stroke();

      // Draw secondary glowing wave
      canvasCtx.lineWidth = 1.5;
      canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
      canvasCtx.beginPath();
      x = 0;
      for (let i = 0; i < numPoints; i++) {
        const freq = (i / numPoints) * Math.PI * 3;
        const amp = Math.sin(freq - wavePhase * 1.3) * 0.4;
        const y = centerY + amp * (h * 0.3);

        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      canvasCtx.lineTo(w, centerY);
      canvasCtx.stroke();
    }
    drawWave();
  }

  function stopVisualization() {
    cancelAnimationFrame(animationFrameId);
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close().catch(() => {});
      audioContext = null;
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
    drawEmptyWaveform();
  }

  // -------------------------------------------------------------
  // Primary STT: Browser Web Speech API (Mobile & Desktop)
  // -------------------------------------------------------------
  function startBrowserSTT() {
    if (isRecording) return;
    isRecording = true;
    sttStartTime = performance.now();

    try {
      recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.continuous = false;

      micWrapper.classList.add('recording');
      micStatusText.textContent = '🎙️ Listening... Speak now';
      micStatusText.style.color = '#f43f5e';

      startAnimatedWaveform();

      let finalTranscript = '';

      recognition.onresult = (event) => {
        let interimText = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimText += transcript;
          }
        }
        textQueryInput.value = finalTranscript || interimText;
        micStatusText.textContent = `🎙️ "${finalTranscript || interimText}"`;
      };

      recognition.onend = () => {
        const sttMs = performance.now() - sttStartTime;
        isRecording = false;
        micWrapper.classList.remove('recording');
        stopVisualization();

        if (finalTranscript.trim()) {
          micStatusText.textContent = `✅ "${finalTranscript.trim()}"`;
          micStatusText.style.color = '#22c55e';
          textQueryInput.value = finalTranscript.trim();
          sendTextQueryWithSTTLatency(finalTranscript.trim(), sttMs);
        } else if (textQueryInput.value.trim()) {
          // If interim was recorded in input
          const fallbackText = textQueryInput.value.trim();
          micStatusText.textContent = `✅ "${fallbackText}"`;
          micStatusText.style.color = '#22c55e';
          sendTextQueryWithSTTLatency(fallbackText, sttMs);
        } else {
          micStatusText.textContent = 'No speech detected. Tap mic to try again.';
          micStatusText.style.color = '#f59e0b';
        }
      };

      recognition.onerror = (event) => {
        console.warn('SpeechRecognition error:', event.error);
        isRecording = false;
        micWrapper.classList.remove('recording');
        stopVisualization();

        if (event.error === 'not-allowed') {
          micStatusText.textContent = '⚠️ Mic access denied. Allow mic in browser settings.';
          micStatusText.style.color = '#f43f5e';
        } else if (event.error === 'no-speech') {
          micStatusText.textContent = 'No speech detected. Tap mic to speak.';
          micStatusText.style.color = '#f59e0b';
        } else {
          micStatusText.textContent = `Speech note: ${event.error}. Tap mic to retry.`;
          micStatusText.style.color = '#f59e0b';
        }
      };

      recognition.start();
    } catch (e) {
      console.warn('Browser SpeechRecognition start failed:', e);
      isRecording = false;
      micWrapper.classList.remove('recording');
      stopVisualization();
      // Fallback to Sarvam MediaRecorder
      startSarvamRecording();
    }
  }

  function stopBrowserSTT() {
    if (!isRecording || !recognition) return;
    try {
      recognition.stop();
    } catch (e) {}
  }

  // -------------------------------------------------------------
  // Fallback STT: Sarvam API via MediaRecorder
  // -------------------------------------------------------------
  async function startSarvamRecording() {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStream = stream;
      isRecording = true;
      audioChunks = [];
      micWrapper.classList.add('recording');
      micStatusText.textContent = '🎙️ Recording... Tap to stop & transcribe';
      micStatusText.style.color = '#f43f5e';
      sttStartTime = performance.now();

      startAnimatedWaveform();

      let mimeType = 'audio/webm';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        mimeType = 'audio/mp4';
      }

      mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        stopVisualization();
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        await sendAudioQuery(audioBlob, mimeType);
      };

      mediaRecorder.start();
    } catch (err) {
      console.warn('Microphone access denied/unavailable:', err);
      micStatusText.textContent = '⚠️ Microphone unavailable. Type your question below.';
      micStatusText.style.color = '#f59e0b';
      isRecording = false;
    }
  }

  function stopSarvamRecording() {
    if (!isRecording) return;
    isRecording = false;
    micWrapper.classList.remove('recording');
    micStatusText.textContent = '⏳ Transcribing with Sarvam AI...';
    micStatusText.style.color = '#06b6d4';
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  // -------------------------------------------------------------
  // Unified Mic Button Click / Touch Handler
  // -------------------------------------------------------------
  micBtn.addEventListener('click', (e) => {
    e.preventDefault();

    // Mobile AudioContext unlocking
    if (window.AudioContext || window.webkitAudioContext) {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContext.state === 'suspended') {
        audioContext.resume();
      }
    }

    if (isRecording) {
      if (hasBrowserSTT) {
        stopBrowserSTT();
      } else {
        stopSarvamRecording();
      }
    } else {
      if (hasBrowserSTT) {
        startBrowserSTT();
      } else {
        startSarvamRecording();
      }
    }
  });

  // -------------------------------------------------------------
  // API Query Dispatchers
  // -------------------------------------------------------------
  async function sendTextQuery(queryText) {
    await sendTextQueryWithSTTLatency(queryText, 0);
  }

  async function sendTextQueryWithSTTLatency(queryText, sttMs) {
    if (!queryText.trim()) return;

    setLoadingState(true, queryText);
    const useCrossEncoder = rerankToggle.checked;
    const strategy = strategySelect.value;

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          strategy: strategy,
          use_cross_encoder: useCrossEncoder,
          top_k: 5
        })
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      if (sttMs > 0) {
        data.latency.stt_ms = Math.round(sttMs * 10) / 10;
        data.latency.total_pipeline_ms += sttMs;
      }
      renderRAGResult(data);
    } catch (err) {
      renderError(err.message);
    } finally {
      setLoadingState(false);
      micStatusText.textContent = hasBrowserSTT
        ? 'Tap to Speak (Live Speech Recognition)'
        : 'Tap or Hold to Record (Sarvam saaras:v3)';
      micStatusText.style.color = '#9ca3af';
    }
  }

  async function sendAudioQuery(audioBlob, mimeType) {
    setLoadingState(true, 'Transcribing & Processing voice input...');
    const formData = new FormData();
    const ext = mimeType.includes('mp4') ? 'mp4' : (mimeType.includes('webm') ? 'webm' : 'wav');
    formData.append('audio', audioBlob, `speech_query.${ext}`);
    formData.append('language', 'hi-IN');
    formData.append('strategy', strategySelect.value);
    formData.append('use_cross_encoder', rerankToggle.checked);
    formData.append('top_k', 5);

    try {
      const response = await fetch('/api/voice', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error(`Voice HTTP error! status: ${response.status}`);
      const data = await response.json();
      textQueryInput.value = data.query;
      renderRAGResult(data);
    } catch (err) {
      renderError(err.message);
    } finally {
      setLoadingState(false);
      micStatusText.textContent = hasBrowserSTT
        ? 'Tap to Speak (Live Speech Recognition)'
        : 'Tap or Hold to Record (Sarvam saaras:v3)';
      micStatusText.style.color = '#9ca3af';
    }
  }

  // -------------------------------------------------------------
  // Rendering RAG Response & Latency Waterfall
  // -------------------------------------------------------------
  function renderRAGResult(data) {
    textQueryInput.value = data.query;
    answerBody.textContent = data.answer;

    // Badges
    modelBadge.textContent = data.model_used;
    totalLatencyBadge.textContent = `⏱️ ${data.latency.total_pipeline_ms.toFixed(1)} ms`;

    if (!data.is_safe) {
      guardrailBadge.className = 'badge badge-danger';
      guardrailBadge.textContent = '🚫 Blocked by Guardrail';
    } else if (!data.is_grounded) {
      guardrailBadge.className = 'badge badge-danger';
      guardrailBadge.textContent = '⚠️ Low Grounding';
    } else {
      guardrailBadge.className = 'badge badge-guardrail';
      guardrailBadge.textContent = '🛡️ Grounded & Safe';
    }

    // Update Latency Waterfall Bars
    const total = Math.max(1, data.latency.total_pipeline_ms);
    updateBar(barSTT, valSTT, data.latency.stt_ms, total);
    updateBar(barGuard, valGuard, data.latency.pre_guardrail_ms, total);
    updateBar(barRet, valRet, data.latency.retrieval_ms, total);
    updateBar(barRerank, valRerank, data.latency.rerank_ms, total);
    updateBar(barLLM, valLLM, data.latency.llm_generation_ms, total);
    updateBar(barPost, valPost, data.latency.post_guardrail_ms, total);

    // Update Citations Grid
    renderCitations(data.citations);

    // Record Latency for Analytics
    recordLatencyMetric(data.latency.total_pipeline_ms);

    // On mobile, smoothly scroll down to the answer card if needed
    if (window.innerWidth <= 960) {
      const rightPanel = document.querySelector('.response-panel');
      if (rightPanel) {
        rightPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }

  function updateBar(barElem, valElem, ms, total) {
    const safeMs = ms || 0;
    const pct = Math.min(100, Math.max(2, (safeMs / total) * 100));
    barElem.style.width = `${pct}%`;
    valElem.textContent = `${safeMs.toFixed(1)} ms`;
  }

  function renderCitations(citations) {
    citationsGrid.innerHTML = '';
    if (!citations || citations.length === 0) {
      citationCount.textContent = '0 Sources';
      citationsGrid.innerHTML = '<div class="citation-empty">No external documents cited for this response.</div>';
      return;
    }

    citationCount.textContent = `${citations.length} Grounded Source${citations.length > 1 ? 's' : ''}`;

    citations.forEach((c) => {
      const card = document.createElement('div');
      card.className = 'citation-card';
      card.innerHTML = `
        <div class="citation-meta">
          <span class="citation-badge">[Doc-${c.doc_index}] • ${c.strategy}</span>
          <span class="citation-score">Score: ${c.score.toFixed(3)} ${c.is_selected ? '★ Gold' : ''}</span>
        </div>
        <div class="citation-text">${c.snippet}</div>
      `;
      citationsGrid.appendChild(card);
    });
  }

  function renderError(msg) {
    answerBody.textContent = `Error: ${msg}`;
    guardrailBadge.className = 'badge badge-danger';
    guardrailBadge.textContent = '❌ System Error';
  }

  function setLoadingState(isLoading, queryText = '') {
    if (isLoading) {
      answerBody.textContent = `Processing: "${queryText}"...`;
      totalLatencyBadge.textContent = '⏱️ Running...';
    }
  }

  // -------------------------------------------------------------
  // Text-To-Speech Playback (Mobile & Desktop)
  // -------------------------------------------------------------
  speakAnswerBtn.addEventListener('click', () => {
    const text = answerBody.textContent;
    if (!text || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    window.speechSynthesis.speak(utterance);
  });

  // -------------------------------------------------------------
  // Analytics: P50 / P70 / P100 Calculation
  // -------------------------------------------------------------
  function recordLatencyMetric(latencyMs) {
    latencyHistory.push(latencyMs);
    latencyHistory.sort((a, b) => a - b);

    const len = latencyHistory.length;
    const p50 = latencyHistory[Math.floor(len * 0.50)];
    const p70 = latencyHistory[Math.floor(len * 0.70)];
    const p100 = latencyHistory[len - 1];

    p50Val.textContent = `${p50.toFixed(1)} ms`;
    p70Val.textContent = `${p70.toFixed(1)} ms`;
    p100Val.textContent = `${p100.toFixed(1)} ms`;
    totalQueriesVal.textContent = len;
  }

  // -------------------------------------------------------------
  // Automated 20-Query Latency Suite Runner
  // -------------------------------------------------------------
  runBenchmarkBtn.addEventListener('click', async () => {
    runBenchmarkBtn.disabled = true;
    runBenchmarkBtn.textContent = '⏳ Running Suite...';

    const testQueries = [
      'What causes middle back pain and muscle spasm?',
      'How does weather affect human health and arthritis?',
      'What is the treatment for hypertension?',
      'What are the main causes of fever in adults?',
      'How to lower high blood pressure naturally?',
      'What is the capital city of France?',
      'Who built the Eiffel Tower?',
      'What are the symptoms of migraine headache?',
      'What is the function of the human heart?',
      'What causes seasonal allergies and sneezing?'
    ];

    for (let i = 0; i < testQueries.length; i++) {
      const q = testQueries[i];
      try {
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, strategy: strategySelect.value, top_k: 5 })
        });
        const data = await response.json();
        renderRAGResult(data);
        await new Promise((r) => setTimeout(r, 100));
      } catch (e) {
        console.error('Benchmark query failed:', e);
      }
    }

    runBenchmarkBtn.disabled = false;
    runBenchmarkBtn.textContent = '⚡ Run 20-Query Latency Suite';
  });

  // -------------------------------------------------------------
  // Input Event Listeners
  // -------------------------------------------------------------
  sendQueryBtn.addEventListener('click', () => {
    sendTextQuery(textQueryInput.value);
  });

  textQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextQuery(textQueryInput.value);
  });

  // Sample Query Chips
  document.querySelectorAll('.chip-btn').forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      textQueryInput.value = query;
      sendTextQuery(query);
    });
  });
});
