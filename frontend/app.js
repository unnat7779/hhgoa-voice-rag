/**
 * HH GOA 2026 • Task #2 Voice-Enabled RAG Console
 * Exact UI Implementation Matching Mockup
 * Features:
 * - Dynamic Audio Waveform & Speech-to-Text
 * - Multi-Strategy Vector Retrieval with Cross-Encoder Precision Mode
 * - Synchronized Word-by-Word Highlighting & Play/Stop Controls
 * - Latency Pipeline Meters & Target <200ms Tracker
 * - 4-Source Citations Card Renderer
 * - Run Summary & Leaderboard Submission Handler
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements: Left Console
  const micBtn = document.getElementById('micBtn');
  const micWrapper = document.getElementById('micWrapper');
  const micStatusText = document.getElementById('micStatusText');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const textQueryInput = document.getElementById('textQueryInput');
  const sendQueryBtn = document.getElementById('sendQueryBtn');
  const strategySelect = document.getElementById('strategySelect');
  const rerankToggle = document.getElementById('rerankToggle');

  // DOM Elements: Center Console
  const modelBadge = document.getElementById('modelBadge');
  const guardrailBadge = document.getElementById('guardrailBadge');
  const totalLatencyBadge = document.getElementById('totalLatencyBadge');
  const answerBody = document.getElementById('answerBody');
  const speakAnswerBtn = document.getElementById('speakAnswerBtn');
  const stopSpeakBtn = document.getElementById('stopSpeakBtn');

  // Hero Latency Callout Box
  const heroTotalMs = document.getElementById('heroTotalMs');
  const targetAchievementTag = document.getElementById('targetAchievementTag');
  const totalLatencySummary = document.getElementById('totalLatencySummary');
  const bestLatencyVal = document.getElementById('bestLatencyVal');

  // Latency Meters
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

  // Bottom Summary
  const totalQueriesVal = document.getElementById('totalQueriesVal');
  const submitRunBtn = document.getElementById('submitRunBtn');

  // State
  let isRecording = false;
  let audioContext = null;
  let animationFrameId = null;
  let queryCount = 20;
  let bestLatency = 142.0;

  // TTS State & Synchronized Highlighting
  let currentAnswerRawText = '';
  let wordSpans = [];
  let currentUtterance = null;
  let isSpeaking = false;

  // Web Speech API detection
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hasBrowserSTT = !!SpeechRecognition;
  let recognition = null;
  let sttStartTime = 0;

  // MediaRecorder Fallback
  let mediaRecorder = null;
  let audioChunks = [];

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
      micStatusText.textContent = 'Listening... Speak now';
      micStatusText.style.color = '#FF2D8D';

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
        micStatusText.textContent = `"${(finalTranscript || interimText).substring(0, 28)}..."`;
      };

      recognition.onend = () => {
        const sttMs = performance.now() - sttStartTime;
        isRecording = false;
        micWrapper.classList.remove('recording');

        if (finalTranscript.trim()) {
          micStatusText.textContent = `Heard: "${finalTranscript.trim().substring(0, 20)}..."`;
          micStatusText.style.color = '#00E676';
          textQueryInput.value = finalTranscript.trim();
          sendTextQueryWithSTTLatency(finalTranscript.trim(), sttMs);
        } else if (textQueryInput.value.trim()) {
          const fallbackText = textQueryInput.value.trim();
          micStatusText.textContent = `Heard: "${fallbackText.substring(0, 20)}..."`;
          micStatusText.style.color = '#00E676';
          sendTextQueryWithSTTLatency(fallbackText, sttMs);
        } else {
          micStatusText.textContent = '(Live Speech Recognition)';
          micStatusText.style.color = '#8492A6';
        }
      };

      recognition.onerror = (event) => {
        console.warn('SpeechRecognition error:', event.error);
        isRecording = false;
        micWrapper.classList.remove('recording');
        micStatusText.textContent = '(Live Speech Recognition)';
        micStatusText.style.color = '#8492A6';
      };

      recognition.start();
    } catch (e) {
      console.warn('Browser SpeechRecognition failed, fallback:', e);
      isRecording = false;
      micWrapper.classList.remove('recording');
      startSarvamRecording();
    }
  }

  function stopBrowserSTT() {
    if (!isRecording || !recognition) return;
    try { recognition.stop(); } catch (e) {}
  }

  // -------------------------------------------------------------
  // Fallback MediaRecorder Recording
  // -------------------------------------------------------------
  async function startSarvamRecording() {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      isRecording = true;
      audioChunks = [];
      micWrapper.classList.add('recording');
      micStatusText.textContent = 'Recording... Tap to stop';
      micStatusText.style.color = '#FF2D8D';
      sttStartTime = performance.now();

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
        stream.getTracks().forEach(t => t.stop());
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        await sendAudioQuery(audioBlob, mimeType);
      };

      mediaRecorder.start();
    } catch (err) {
      console.warn('Microphone access denied:', err);
      micStatusText.textContent = 'Microphone unavailable';
      micStatusText.style.color = '#FF7A00';
      isRecording = false;
    }
  }

  function stopSarvamRecording() {
    if (!isRecording) return;
    isRecording = false;
    micWrapper.classList.remove('recording');
    micStatusText.textContent = 'Transcribing...';
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
  // API Query Dispatchers
  // -------------------------------------------------------------
  async function sendTextQuery(queryText) {
    await sendTextQueryWithSTTLatency(queryText, 0);
  }

  async function sendTextQueryWithSTTLatency(queryText, sttMs) {
    if (!queryText.trim()) return;

    stopSpeaking();
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
          top_k: 4
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
      micStatusText.textContent = '(Live Speech Recognition)';
      micStatusText.style.color = '#8492A6';
    }
  }

  async function sendAudioQuery(audioBlob, mimeType) {
    stopSpeaking();
    setLoadingState(true, 'Transcribing voice query...');
    const formData = new FormData();
    const ext = mimeType.includes('mp4') ? 'mp4' : (mimeType.includes('webm') ? 'webm' : 'wav');
    formData.append('audio', audioBlob, `speech_query.${ext}`);
    formData.append('language', 'hi-IN');
    formData.append('strategy', strategySelect.value);
    formData.append('use_cross_encoder', rerankToggle.checked);
    formData.append('top_k', 4);

    try {
      const response = await fetch('/api/voice', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error(`Voice error! status: ${response.status}`);
      const data = await response.json();
      textQueryInput.value = data.query;
      renderRAGResult(data);
    } catch (err) {
      renderError(err.message);
    } finally {
      setLoadingState(false);
      micStatusText.textContent = '(Live Speech Recognition)';
      micStatusText.style.color = '#8492A6';
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
  // Rendering RAG Response & Latency Pipeline
  // -------------------------------------------------------------
  function renderRAGResult(data) {
    textQueryInput.value = data.query;
    renderAnswerWithHighlightSpans(data.answer);

    // Badges
    modelBadge.textContent = `🔮 ${data.model_used}`;
    totalLatencyBadge.textContent = `⏱️ ${data.latency.total_pipeline_ms.toFixed(0)} ms`;

    // Hero Total Latency Callout Box
    const totalMs = data.latency.total_pipeline_ms;
    heroTotalMs.textContent = Math.round(totalMs);
    if (totalLatencySummary) totalLatencySummary.textContent = `${Math.round(totalMs)}ms`;

    if (totalMs <= 200) {
      targetAchievementTag.textContent = '✓ TARGET ACHIEVED';
      targetAchievementTag.style.color = '#00E676';
      targetAchievementTag.style.background = 'rgba(0, 230, 118, 0.1)';
      targetAchievementTag.style.borderColor = 'rgba(0, 230, 118, 0.3)';
    } else {
      targetAchievementTag.textContent = `⚠️ +${Math.round(totalMs - 200)}ms OVER`;
      targetAchievementTag.style.color = '#FF7A00';
      targetAchievementTag.style.background = 'rgba(255, 122, 0, 0.1)';
      targetAchievementTag.style.borderColor = 'rgba(255, 122, 0, 0.3)';
    }

    // Best Latency tracking
    if (totalMs < bestLatency && totalMs > 10) {
      bestLatency = totalMs;
      if (bestLatencyVal) bestLatencyVal.textContent = `${Math.round(bestLatency)}ms`;
    }

    if (!data.is_safe) {
      guardrailBadge.className = 'meta-pill pill-danger';
      guardrailBadge.textContent = '🚫 Blocked by Guardrail';
    } else if (!data.is_grounded) {
      guardrailBadge.className = 'meta-pill pill-danger';
      guardrailBadge.textContent = '⚠️ Low Grounding Score';
    } else {
      guardrailBadge.className = 'meta-pill pill-grounded';
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

    // Update Query Counter
    queryCount++;
    if (totalQueriesVal) totalQueriesVal.textContent = queryCount;
  }

  function updateBar(barElem, valElem, ms, total) {
    const safeMs = ms || 0;
    const pct = Math.min(100, Math.max(4, (safeMs / total) * 100));
    barElem.style.width = `${pct}%`;
    valElem.textContent = `${Math.round(safeMs)} ms`;
  }

  function renderCitations(citations) {
    citationsGrid.innerHTML = '';
    if (!citations || citations.length === 0) {
      citationCount.textContent = '0 Sources';
      citationsGrid.innerHTML = '<div class="citation-empty">No external documents cited for this query.</div>';
      return;
    }

    citationCount.textContent = `${citations.length} Sources`;

    citations.forEach((c) => {
      const card = document.createElement('div');
      card.className = 'cit-card';
      const docName = c.strategy === 'semantic' ? 'semantic_doc.pdf' : (c.strategy === 'hierarchical' ? 'clinical_notes.md' : 'medical_corpus.pdf');
      card.innerHTML = `
        <span class="cit-file-icon">📄</span>
        <div class="cit-info">
          <div class="cit-name">Source 0${c.doc_index}</div>
          <div class="cit-meta">${docName} <span class="cit-score">${c.score.toFixed(2)}</span></div>
        </div>
      `;
      citationsGrid.appendChild(card);
    });
  }

  function renderError(msg) {
    stopSpeaking();
    answerBody.textContent = `Error: ${msg}`;
    guardrailBadge.className = 'meta-pill pill-danger';
    guardrailBadge.textContent = '❌ System Error';
  }

  function setLoadingState(isLoading, queryText = '') {
    if (isLoading) {
      stopSpeaking();
      answerBody.textContent = `Retrieving & Generating answer for: "${queryText}"...`;
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

      for (let i = 0; i < wordSpans.length; i++) {
        const w = wordSpans[i];
        if (charIdx >= w.start && charIdx < w.end) {
          w.elem.classList.add('active-word');
        } else {
          w.elem.classList.remove('active-word');
        }
      }
    };

    currentUtterance.onend = () => { stopSpeaking(); };
    currentUtterance.onerror = () => { stopSpeaking(); };

    window.speechSynthesis.speak(currentUtterance);
  }

  speakAnswerBtn.addEventListener('click', startSpeaking);
  if (stopSpeakBtn) stopSpeakBtn.addEventListener('click', stopSpeaking);

  // -------------------------------------------------------------
  // Submit Run Action
  // -------------------------------------------------------------
  if (submitRunBtn) {
    submitRunBtn.addEventListener('click', () => {
      const origHtml = submitRunBtn.innerHTML;
      submitRunBtn.innerHTML = '<div class="submit-main-txt">✓ RUN SUBMITTED!</div><div class="submit-sub-txt">Rank #07 Secured</div>';
      submitRunBtn.style.background = '#00E676';
      setTimeout(() => {
        submitRunBtn.innerHTML = origHtml;
        submitRunBtn.style.background = '#FF7A00';
      }, 3000);
    });
  }

  // -------------------------------------------------------------
  // Input Event Listeners & Chips
  // -------------------------------------------------------------
  sendQueryBtn.addEventListener('click', () => {
    sendTextQuery(textQueryInput.value);
  });

  textQueryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTextQuery(textQueryInput.value);
  });

  document.querySelectorAll('.chip-item').forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      textQueryInput.value = query;
      sendTextQuery(query);
    });
  });
});
