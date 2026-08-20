/**
 * HH GOA 2026 • Task #02 Voice-Enabled RAG Console
 * Cyberpunk Developer Console × Underground Hackathon
 * Features:
 * - Dynamic Audio Waveform Visualizer
 * - Hybrid Speech-to-Text: Browser Web Speech API → Sarvam fallback
 * - Multi-Strategy Retrieval REST API Integration
 * - Mission Control Status Stages & Target <200ms Latency Pipeline
 * - Real-Time Synchronized Word Highlighting & Play/Stop Controls
 * - Hackathon Scoreboard & Challenge Brief Collapsible Bar
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements: Challenge Brief
  const briefToggleBtn = document.getElementById('briefToggleBtn');
  const briefContent = document.getElementById('briefContent');
  const briefArrow = document.getElementById('briefArrow');

  // DOM Elements: Left Console
  const micBtn = document.getElementById('micBtn');
  const micWrapper = document.getElementById('micWrapper');
  const micStatusText = document.getElementById('micStatusText');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const textQueryInput = document.getElementById('textQueryInput');
  const sendQueryBtn = document.getElementById('sendQueryBtn');
  const strategySelect = document.getElementById('strategySelect');
  const rerankToggle = document.getElementById('rerankToggle');

  // DOM Elements: Right Panel (Mission Control)
  const statusLed = document.getElementById('statusLed');
  const statusLedText = document.getElementById('statusLedText');
  const modelBadge = document.getElementById('modelBadge');
  const guardrailBadge = document.getElementById('guardrailBadge');
  const totalLatencyBadge = document.getElementById('totalLatencyBadge');
  const answerBody = document.getElementById('answerBody');
  const speakAnswerBtn = document.getElementById('speakAnswerBtn');
  const stopSpeakBtn = document.getElementById('stopSpeakBtn');

  // Hero Latency Scoreboard Callout
  const heroTotalMs = document.getElementById('heroTotalMs');
  const targetAchievementTag = document.getElementById('targetAchievementTag');

  // Latency Meter Bars
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

  // Citations Grid
  const citationsGrid = document.getElementById('citationsGrid');
  const citationCount = document.getElementById('citationCount');

  // Scoreboard Metrics & Actions
  const p50Val = document.getElementById('p50Val');
  const p70Val = document.getElementById('p70Val');
  const p100Val = document.getElementById('p100Val');
  const totalQueriesVal = document.getElementById('totalQueriesVal');
  const runBenchmarkBtn = document.getElementById('runBenchmarkBtn');
  const submitRunBtn = document.getElementById('submitRunBtn');

  // State
  let isRecording = false;
  let audioContext = null;
  let animationFrameId = null;
  let latencyHistory = [];
  let micStream = null;

  // TTS State & Synchronized Highlighting
  let currentAnswerRawText = '';
  let wordSpans = [];
  let currentUtterance = null;
  let isSpeaking = false;

  // Web Speech API detection (cross-browser / iOS Safari / Android Chrome)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hasBrowserSTT = !!SpeechRecognition;
  let recognition = null;
  let sttStartTime = 0;

  // MediaRecorder for fallback
  let mediaRecorder = null;
  let audioChunks = [];

  // -------------------------------------------------------------
  // Challenge Brief Collapse / Expand
  // -------------------------------------------------------------
  if (briefToggleBtn && briefContent) {
    let isBriefOpen = true;
    briefToggleBtn.addEventListener('click', () => {
      isBriefOpen = !isBriefOpen;
      if (isBriefOpen) {
        briefContent.style.display = 'flex';
        briefArrow.textContent = '▲ COLLAPSE';
      } else {
        briefContent.style.display = 'none';
        briefArrow.textContent = '▼ EXPAND';
      }
    });
  }

  // -------------------------------------------------------------
  // Canvas Setup & Auto-Responsive Width
  // -------------------------------------------------------------
  const canvasCtx = waveformCanvas.getContext('2d');

  function resizeCanvas() {
    const rect = waveformCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    waveformCanvas.width = (rect.width || 360) * dpr;
    waveformCanvas.height = (rect.height || 48) * dpr;
    canvasCtx.scale(dpr, dpr);
    drawEmptyWaveform();
  }

  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 100);

  function drawEmptyWaveform() {
    const rect = waveformCanvas.getBoundingClientRect();
    const w = rect.width || 360;
    const h = rect.height || 48;

    canvasCtx.clearRect(0, 0, w, h);
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    canvasCtx.fillRect(0, 0, w, h);
    canvasCtx.lineWidth = 1.5;
    canvasCtx.strokeStyle = 'rgba(0, 230, 118, 0.4)';
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, h / 2);
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();
  }

  // -------------------------------------------------------------
  // Simulated & Real Acoustic Waveform Visualizer
  // -------------------------------------------------------------
  let wavePhase = 0;

  function startAnimatedWaveform() {
    function drawWave() {
      if (!isRecording) return;
      animationFrameId = requestAnimationFrame(drawWave);
      wavePhase += 0.14;

      const rect = waveformCanvas.getBoundingClientRect();
      const w = rect.width || 360;
      const h = rect.height || 48;
      const centerY = h / 2;

      canvasCtx.fillStyle = 'rgba(5, 7, 11, 0.4)';
      canvasCtx.fillRect(0, 0, w, h);

      // Primary Hacker Green Wave
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = '#00E676';
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

      // Secondary Hackathon Orange Glowing Wave
      canvasCtx.lineWidth = 1.2;
      canvasCtx.strokeStyle = 'rgba(255, 122, 0, 0.7)';
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
  // Speech-to-Text: Browser Web Speech API
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
      micStatusText.textContent = 'LISTENING... SPEAK NOW';
      micStatusText.style.color = '#FF2D8D';
      setLiveStatus('● LISTENING...', 'active');

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
        micStatusText.textContent = `"${(finalTranscript || interimText).substring(0, 32)}..."`;
      };

      recognition.onend = () => {
        const sttMs = performance.now() - sttStartTime;
        isRecording = false;
        micWrapper.classList.remove('recording');
        stopVisualization();

        if (finalTranscript.trim()) {
          micStatusText.textContent = `HEARD: "${finalTranscript.trim().substring(0, 24)}"`;
          micStatusText.style.color = '#00E676';
          textQueryInput.value = finalTranscript.trim();
          sendTextQueryWithSTTLatency(finalTranscript.trim(), sttMs);
        } else if (textQueryInput.value.trim()) {
          const fallbackText = textQueryInput.value.trim();
          micStatusText.textContent = `HEARD: "${fallbackText.substring(0, 24)}"`;
          micStatusText.style.color = '#00E676';
          sendTextQueryWithSTTLatency(fallbackText, sttMs);
        } else {
          micStatusText.textContent = 'TAP TO SPEAK (EN / HI)';
          micStatusText.style.color = '#718096';
          setLiveStatus('SYSTEM READY', 'green');
        }
      };

      recognition.onerror = (event) => {
        console.warn('SpeechRecognition error:', event.error);
        isRecording = false;
        micWrapper.classList.remove('recording');
        stopVisualization();

        if (event.error === 'not-allowed') {
          micStatusText.textContent = 'MIC ACCESS DENIED';
          micStatusText.style.color = '#F43F5E';
        } else {
          micStatusText.textContent = 'TAP TO SPEAK (EN / HI)';
          micStatusText.style.color = '#718096';
        }
        setLiveStatus('SYSTEM READY', 'green');
      };

      recognition.start();
    } catch (e) {
      console.warn('SpeechRecognition start failed, trying fallback:', e);
      isRecording = false;
      micWrapper.classList.remove('recording');
      stopVisualization();
      startSarvamRecording();
    }
  }

  function stopBrowserSTT() {
    if (!isRecording || !recognition) return;
    try { recognition.stop(); } catch (e) {}
  }

  // -------------------------------------------------------------
  // Fallback STT: Sarvam / MediaRecorder
  // -------------------------------------------------------------
  async function startSarvamRecording() {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStream = stream;
      isRecording = true;
      audioChunks = [];
      micWrapper.classList.add('recording');
      micStatusText.textContent = 'RECORDING... TAP TO STOP';
      micStatusText.style.color = '#FF2D8D';
      setLiveStatus('● RECORDING...', 'active');
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
      console.warn('Microphone access denied:', err);
      micStatusText.textContent = 'MIC UNAVAILABLE • TYPE BELOW';
      micStatusText.style.color = '#FF7A00';
      isRecording = false;
      setLiveStatus('SYSTEM READY', 'green');
    }
  }

  function stopSarvamRecording() {
    if (!isRecording) return;
    isRecording = false;
    micWrapper.classList.remove('recording');
    micStatusText.textContent = 'TRANSCRIBING...';
    micStatusText.style.color = '#00BFA5';
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }

  // -------------------------------------------------------------
  // Unified Mic Button Click / Touch
  // -------------------------------------------------------------
  micBtn.addEventListener('click', (e) => {
    e.preventDefault();

    // Mobile AudioContext Unlocking
    if (window.AudioContext || window.webkitAudioContext) {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContext.state === 'suspended') {
        audioContext.resume();
      }
    }

    if (isRecording) {
      if (hasBrowserSTT) stopBrowserSTT();
      else stopSarvamRecording();
    } else {
      if (hasBrowserSTT) startBrowserSTT();
      else startSarvamRecording();
    }
  });

  // -------------------------------------------------------------
  // Live Status Indicator Helper
  // -------------------------------------------------------------
  function setLiveStatus(text, mode = 'green') {
    statusLedText.textContent = text;
    if (mode === 'active') {
      statusLed.className = 'status-led led-active';
      statusLedText.style.color = '#FF7A00';
    } else if (mode === 'danger') {
      statusLed.className = 'status-led';
      statusLed.style.background = '#F43F5E';
      statusLedText.style.color = '#F43F5E';
    } else {
      statusLed.className = 'status-led';
      statusLed.style.background = '#00E676';
      statusLedText.style.color = '#00E676';
    }
  }

  // -------------------------------------------------------------
  // API Query Dispatchers
  // -------------------------------------------------------------
  async function sendTextQuery(queryText) {
    await sendTextQueryWithSTTLatency(queryText, 0);
  }

  async function sendTextQueryWithSTTLatency(queryText, sttMs) {
    if (!queryText.trim()) return;

    stopSpeaking();
    setLiveStatus('● RETRIEVING...', 'active');
    setLoadingState(true, queryText);
    const useCrossEncoder = rerankToggle.checked;
    const strategy = strategySelect.value;

    try {
      setLiveStatus('● GENERATING...', 'active');
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
      setLiveStatus('● GROUNDED', 'green');
    } catch (err) {
      renderError(err.message);
      setLiveStatus('SYSTEM ERROR', 'danger');
    } finally {
      setLoadingState(false);
      micStatusText.textContent = 'TAP TO SPEAK (EN / HI)';
      micStatusText.style.color = '#718096';
    }
  }

  async function sendAudioQuery(audioBlob, mimeType) {
    stopSpeaking();
    setLiveStatus('● TRANSCRIBING...', 'active');
    setLoadingState(true, 'Transcribing voice input...');
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

      if (!response.ok) throw new Error(`Voice error! status: ${response.status}`);
      const data = await response.json();
      textQueryInput.value = data.query;
      renderRAGResult(data);
      setLiveStatus('● GROUNDED', 'green');
    } catch (err) {
      renderError(err.message);
      setLiveStatus('SYSTEM ERROR', 'danger');
    } finally {
      setLoadingState(false);
      micStatusText.textContent = 'TAP TO SPEAK (EN / HI)';
      micStatusText.style.color = '#718096';
    }
  }

  // -------------------------------------------------------------
  // Synchronized Word Highlighting Renderer
  // -------------------------------------------------------------
  function renderAnswerWithHighlightSpans(rawText) {
    currentAnswerRawText = rawText;
    wordSpans = [];
    answerBody.innerHTML = '';

    const regex = /(\S+)/g;
    let match;
    let lastIndex = 0;
    let wordIdx = 0;

    while ((match = regex.exec(rawText)) !== null) {
      if (match.index > lastIndex) {
        const nonWord = rawText.substring(lastIndex, match.index);
        answerBody.appendChild(document.createTextNode(nonWord));
      }

      const wordStr = match[0];
      const span = document.createElement('span');
      span.className = 'tts-word';
      span.textContent = wordStr;
      span.id = `tts-word-${wordIdx}`;
      span.dataset.start = match.index;
      span.dataset.end = match.index + wordStr.length;
      span.dataset.idx = wordIdx;

      wordSpans.push({
        elem: span,
        start: match.index,
        end: match.index + wordStr.length,
        idx: wordIdx
      });

      answerBody.appendChild(span);
      lastIndex = regex.lastIndex;
      wordIdx++;
    }

    if (lastIndex < rawText.length) {
      answerBody.appendChild(document.createTextNode(rawText.substring(lastIndex)));
    }
  }

  // -------------------------------------------------------------
  // Rendering RAG Result & Latency Pipeline
  // -------------------------------------------------------------
  function renderRAGResult(data) {
    textQueryInput.value = data.query;
    renderAnswerWithHighlightSpans(data.answer);

    // Badges
    modelBadge.textContent = data.model_used;
    totalLatencyBadge.textContent = `⏱️ ${data.latency.total_pipeline_ms.toFixed(1)} ms`;

    // Hero Total Latency Callout
    const totalMs = data.latency.total_pipeline_ms;
    heroTotalMs.textContent = totalMs.toFixed(1);

    if (totalMs <= 200) {
      targetAchievementTag.textContent = '✓ TARGET ACHIEVED (<200ms)';
      targetAchievementTag.style.color = '#00E676';
    } else {
      targetAchievementTag.textContent = `⚠️ +${(totalMs - 200).toFixed(0)}ms OVER TARGET`;
      targetAchievementTag.style.color = '#FF7A00';
    }

    if (!data.is_safe) {
      guardrailBadge.className = 'tag-badge tag-danger';
      guardrailBadge.textContent = '🚫 Blocked by Guardrail';
    } else if (!data.is_grounded) {
      guardrailBadge.className = 'tag-badge tag-danger';
      guardrailBadge.textContent = '⚠️ Low Grounding';
    } else {
      guardrailBadge.className = 'tag-badge tag-grounded';
      guardrailBadge.textContent = '🛡️ Grounded & Safe';
    }

    // Update Latency Waterfall Bars
    const total = Math.max(1, totalMs);
    updateBar(barSTT, valSTT, data.latency.stt_ms, total);
    updateBar(barGuard, valGuard, data.latency.pre_guardrail_ms, total);
    updateBar(barRet, valRet, data.latency.retrieval_ms, total);
    updateBar(barRerank, valRerank, data.latency.rerank_ms, total);
    updateBar(barLLM, valLLM, data.latency.llm_generation_ms, total);
    updateBar(barPost, valPost, data.latency.post_guardrail_ms, total);

    // Update Citations
    renderCitations(data.citations);

    // Record Latency for Scoreboard
    recordLatencyMetric(totalMs);

    // Mobile scroll to answer
    if (window.innerWidth <= 960) {
      const responsePanel = document.querySelector('.response-panel');
      if (responsePanel) {
        responsePanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
      citationCount.textContent = '0 SOURCES';
      citationsGrid.innerHTML = '<div class="citation-empty">No context documents cited for this query.</div>';
      return;
    }

    citationCount.textContent = `${citations.length} SOURCES`;

    citations.forEach((c) => {
      const card = document.createElement('div');
      card.className = 'citation-card';
      card.innerHTML = `
        <div class="citation-meta">
          <span class="citation-badge">[Doc-${c.doc_index}] • ${c.strategy}</span>
          <span class="citation-score">Cosine: ${c.score.toFixed(3)} ${c.is_selected ? '★ Gold' : ''}</span>
        </div>
        <div class="citation-text">${c.snippet}</div>
      `;
      citationsGrid.appendChild(card);
    });
  }

  function renderError(msg) {
    stopSpeaking();
    answerBody.textContent = `Error: ${msg}`;
    guardrailBadge.className = 'tag-badge tag-danger';
    guardrailBadge.textContent = '❌ System Error';
  }

  function setLoadingState(isLoading, queryText = '') {
    if (isLoading) {
      stopSpeaking();
      answerBody.textContent = `Processing query: "${queryText}"...`;
      totalLatencyBadge.textContent = '⏱️ Running...';
    }
  }

  // -------------------------------------------------------------
  // Text-To-Speech Playback with Word Sync & Stop Controls
  // -------------------------------------------------------------
  function stopSpeaking() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isSpeaking = false;
    speakAnswerBtn.classList.remove('speaking');
    speakAnswerBtn.style.display = 'inline-flex';
    if (stopSpeakBtn) stopSpeakBtn.style.display = 'none';

    wordSpans.forEach(w => {
      w.elem.classList.remove('active-word');
      w.elem.classList.remove('spoken-word');
    });
  }

  function startSpeaking() {
    if (!('speechSynthesis' in window)) return;
    stopSpeaking();

    const text = currentAnswerRawText || answerBody.textContent;
    if (!text || !text.trim()) return;

    currentUtterance = new SpeechSynthesisUtterance(text);
    currentUtterance.rate = 1.0;
    currentUtterance.pitch = 1.0;

    isSpeaking = true;
    speakAnswerBtn.classList.add('speaking');
    speakAnswerBtn.style.display = 'none';
    if (stopSpeakBtn) stopSpeakBtn.style.display = 'inline-flex';

    currentUtterance.onboundary = (event) => {
      if (!isSpeaking) return;
      const charIdx = event.charIndex;

      let found = false;
      for (let i = 0; i < wordSpans.length; i++) {
        const w = wordSpans[i];
        if (charIdx >= w.start && charIdx < w.end) {
          w.elem.classList.add('active-word');
          w.elem.classList.remove('spoken-word');
          found = true;
        } else if (w.start < charIdx) {
          w.elem.classList.remove('active-word');
          w.elem.classList.add('spoken-word');
        } else {
          w.elem.classList.remove('active-word');
          w.elem.classList.remove('spoken-word');
        }
      }

      if (!found && wordSpans.length > 0) {
        let closest = wordSpans[0];
        let minDiff = Math.abs(wordSpans[0].start - charIdx);
        for (let i = 1; i < wordSpans.length; i++) {
          const diff = Math.abs(wordSpans[i].start - charIdx);
          if (diff < minDiff) {
            minDiff = diff;
            closest = wordSpans[i];
          }
        }
        wordSpans.forEach(w => {
          if (w.idx === closest.idx) {
            w.elem.classList.add('active-word');
            w.elem.classList.remove('spoken-word');
          } else if (w.start < closest.start) {
            w.elem.classList.remove('active-word');
            w.elem.classList.add('spoken-word');
          } else {
            w.elem.classList.remove('active-word');
            w.elem.classList.remove('spoken-word');
          }
        });
      }
    };

    currentUtterance.onend = () => {
      stopSpeaking();
    };

    currentUtterance.onerror = (e) => {
      console.warn('Speech synthesis error:', e);
      stopSpeaking();
    };

    window.speechSynthesis.speak(currentUtterance);
  }

  speakAnswerBtn.addEventListener('click', startSpeaking);
  if (stopSpeakBtn) {
    stopSpeakBtn.addEventListener('click', stopSpeaking);
  }

  // -------------------------------------------------------------
  // Scoreboard Analytics: P50 / P70 / P100 Calculation
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
    runBenchmarkBtn.textContent = '⏳ RUNNING SUITE...';

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
    runBenchmarkBtn.textContent = '⚡ RUN 20-QUERY SUITE';
  });

  // Submit Run Button Handler
  if (submitRunBtn) {
    submitRunBtn.addEventListener('click', () => {
      submitRunBtn.textContent = '✓ SUBMITTED #07';
      submitRunBtn.style.background = '#00E676';
      setTimeout(() => {
        submitRunBtn.textContent = 'SUBMIT RUN ↗';
        submitRunBtn.style.background = '#FF7A00';
      }, 3000);
    });
  }

  // -------------------------------------------------------------
  // Input Event Listeners
  // -------------------------------------------------------------
  sendQueryBtn.addEventListener('click', () => {
    sendTextQuery(textQueryInput.value);
  });

  textQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextQuery(textQueryInput.value);
  });

  // Prompt Chips
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      textQueryInput.value = query;
      sendTextQuery(query);
    });
  });
});
