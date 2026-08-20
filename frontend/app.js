/**
 * Voice-Enabled RAG Client-Side Application
 * Handles Push-To-Talk Audio, Real-time Waveforms, Multi-Strategy REST API,
 * Latency Stopwatch Visualization, and P50/P70/P100 Analytics.
 *
 * STT Strategy: Browser Web Speech API (primary) → Sarvam saaras:v3 (fallback)
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

  // Web Speech API support detection
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hasBrowserSTT = !!SpeechRecognition;
  let recognition = null;
  let sttStartTime = 0;

  // MediaRecorder for Sarvam fallback
  let mediaRecorder = null;
  let audioChunks = [];

  // Initialize Canvas
  const canvasCtx = waveformCanvas.getContext('2d');
  drawEmptyWaveform();

  // Update status text based on STT engine
  if (hasBrowserSTT) {
    micStatusText.textContent = 'Click to Speak (Browser Speech Recognition)';
  } else {
    micStatusText.textContent = 'Click or Hold to Speak (Sarvam saaras:v3 STT)';
  }

  function drawEmptyWaveform() {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#06b6d4';
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, waveformCanvas.height / 2);
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
  }

  // -------------------------------------------------------------
  // Web Audio API Oscilloscope Visualizer
  // -------------------------------------------------------------
  function visualizeWaveform(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      if (!isRecording) return;
      animationFrameId = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      canvasCtx.fillStyle = 'rgba(7, 9, 14, 0.4)';
      canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

      canvasCtx.lineWidth = 2.5;
      canvasCtx.strokeStyle = '#38bdf8';
      canvasCtx.beginPath();

      const sliceWidth = waveformCanvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * waveformCanvas.height) / 2;
        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }

      canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
      canvasCtx.stroke();
    }
    draw();
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
  // Primary STT: Browser Web Speech API
  // -------------------------------------------------------------
  function startBrowserSTT() {
    if (isRecording) return;
    isRecording = true;
    sttStartTime = performance.now();

    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    micWrapper.classList.add('recording');
    micStatusText.textContent = '🎤 Listening... Speak now';
    micStatusText.style.color = '#f43f5e';

    // Start mic for waveform visualization
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      micStream = stream;
      visualizeWaveform(stream);
    }).catch(() => {});

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
      // Show live interim transcript in input box
      textQueryInput.value = finalTranscript || interimText;
      micStatusText.textContent = `🎤 "${finalTranscript || interimText}"`;
    };

    recognition.onend = () => {
      const sttMs = performance.now() - sttStartTime;
      isRecording = false;
      micWrapper.classList.remove('recording');
      stopVisualization();

      if (finalTranscript.trim()) {
        micStatusText.textContent = `✅ Heard: "${finalTranscript.trim()}"`;
        micStatusText.style.color = '#22c55e';
        textQueryInput.value = finalTranscript.trim();
        // Automatically send query
        sendTextQueryWithSTTLatency(finalTranscript.trim(), sttMs);
      } else {
        micStatusText.textContent = 'No speech detected. Try again.';
        micStatusText.style.color = '#f59e0b';
      }
    };

    recognition.onerror = (event) => {
      console.warn('Web Speech API error:', event.error);
      isRecording = false;
      micWrapper.classList.remove('recording');
      stopVisualization();

      if (event.error === 'not-allowed') {
        micStatusText.textContent = '⚠️ Microphone permission denied.';
        micStatusText.style.color = '#f43f5e';
      } else {
        micStatusText.textContent = `Speech error: ${event.error}. Try again.`;
        micStatusText.style.color = '#f59e0b';
      }
    };

    recognition.start();
  }

  function stopBrowserSTT() {
    if (!isRecording || !recognition) return;
    recognition.stop();
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
      micStatusText.textContent = '🎤 Recording... Release to transcribe (Sarvam)';
      micStatusText.style.color = '#f43f5e';
      sttStartTime = performance.now();

      visualizeWaveform(stream);

      // Use webm/opus codec which Sarvam accepts
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

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
      micStatusText.textContent = '⚠️ Microphone unavailable. Please type your query.';
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
  // Unified Mic Button Handler (Click to toggle)
  // -------------------------------------------------------------
  micBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (isRecording) {
      // Stop whichever is running
      if (hasBrowserSTT) {
        stopBrowserSTT();
      } else {
        stopSarvamRecording();
      }
    } else {
      // Start preferred STT
      if (hasBrowserSTT) {
        startBrowserSTT();
      } else {
        startSarvamRecording();
        // Auto-stop after 5 seconds
        setTimeout(() => { if (isRecording) stopSarvamRecording(); }, 5000);
      }
    }
  });

  // Prevent double-firing from mousedown/mouseup + click
  micBtn.addEventListener('mousedown', (e) => e.preventDefault());
  micBtn.addEventListener('mouseup', (e) => e.preventDefault());
  micBtn.addEventListener('touchstart', (e) => e.preventDefault());
  micBtn.addEventListener('touchend', (e) => e.preventDefault());

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
      // Override STT latency with actual browser STT time
      if (sttMs > 0) {
        data.latency.stt_ms = Math.round(sttMs * 100) / 100;
        data.latency.total_pipeline_ms += sttMs;
      }
      renderRAGResult(data);
    } catch (err) {
      renderError(err.message);
    } finally {
      setLoadingState(false);
      if (hasBrowserSTT) {
        micStatusText.textContent = 'Click to Speak (Browser Speech Recognition)';
      } else {
        micStatusText.textContent = 'Click or Hold to Speak (Sarvam saaras:v3 STT)';
      }
      micStatusText.style.color = '#9ca3af';
    }
  }

  async function sendAudioQuery(audioBlob, mimeType) {
    setLoadingState(true, 'Transcribing & Processing voice input...');
    const formData = new FormData();
    const ext = mimeType.includes('webm') ? 'webm' : 'wav';
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
        ? 'Click to Speak (Browser Speech Recognition)'
        : 'Click or Hold to Speak (Sarvam saaras:v3 STT)';
      micStatusText.style.color = '#9ca3af';
    }
  }

  // -------------------------------------------------------------
  // Rendering RAG Response & Latency Stopwatch
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
      guardrailBadge.textContent = '⚠️ Low Grounding Score';
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
          <span class="citation-score">Cosine Score: ${c.score.toFixed(4)} ${c.is_selected ? '★ Gold' : ''}</span>
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
  // Text-To-Speech Playback
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

  // Chip sample click
  document.querySelectorAll('.chip-btn').forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      textQueryInput.value = query;
      sendTextQuery(query);
    });
  });
});
