/**
 * HH GOA 2026 · Task #2 · Voice RAG
 * Clean modular controller
 */
document.addEventListener('DOMContentLoaded', () => {
  const $ = (s) => document.getElementById(s);

  // Elements
  const micBtn = $('micBtn');
  const micWrapper = $('micWrapper');
  const micLabel = $('micStatusText');
  const queryInput = $('textQueryInput');
  const sendBtn = $('sendQueryBtn');
  const strategySelect = $('strategySelect');
  const rerankToggle = $('rerankToggle');
  const modelBadge = $('modelBadge');
  const guardrailBadge = $('guardrailBadge');
  const latencyBadge = $('totalLatencyBadge');
  const answerBody = $('answerBody');
  const speakBtn = $('speakAnswerBtn');
  const stopBtn = $('stopSpeakBtn');
  const heroMs = $('heroTotalMs');
  const targetTag = $('targetAchievementTag');
  const latSummary = $('totalLatencySummary');
  const bestLat = $('bestLatencyVal');
  const queriesVal = $('totalQueriesVal');
  const citGrid = $('citationsGrid');
  const citCount = $('citationCount');
  const submitBtn = $('submitRunBtn');

  // Bars
  const bars = {
    stt:    { bar: $('barSTT'),    val: $('valSTT') },
    guard:  { bar: $('barGuard'),  val: $('valGuard') },
    ret:    { bar: $('barRet'),    val: $('valRet') },
    rerank: { bar: $('barRerank'), val: $('valRerank') },
    llm:    { bar: $('barLLM'),    val: $('valLLM') },
    post:   { bar: $('barPost'),   val: $('valPost') },
  };

  // State
  let isRecording = false;
  let queryCount = 0;
  let best = Infinity;
  let rawText = '';
  let wordSpans = [];

  // Speech Recognition
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let sttStart = 0;

  // ---- Mic ----
  micBtn.addEventListener('click', (e) => {
    e.preventDefault();
    if (isRecording) stopRec();
    else startRec();
  });

  function startRec() {
    if (!SR) { micLabel.textContent = 'Not supported'; return; }
    isRecording = true;
    sttStart = performance.now();
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = false;
    micWrapper.classList.add('recording');
    micLabel.textContent = 'Listening...';

    let final = '';
    recognition.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) final += e.results[i][0].transcript;
        else interim += e.results[i][0].transcript;
      }
      queryInput.value = final || interim;
    };

    recognition.onend = () => {
      const ms = performance.now() - sttStart;
      isRecording = false;
      micWrapper.classList.remove('recording');
      const text = (final || queryInput.value).trim();
      if (text) {
        micLabel.textContent = `"${text.slice(0, 24)}..."`;
        sendQuery(text, ms);
      } else {
        micLabel.textContent = 'Tap to speak';
      }
    };

    recognition.onerror = () => {
      isRecording = false;
      micWrapper.classList.remove('recording');
      micLabel.textContent = 'Tap to speak';
    };

    recognition.start();
  }

  function stopRec() {
    if (recognition) try { recognition.stop(); } catch (_) {}
  }

  // ---- Query ----
  sendBtn.addEventListener('click', () => sendQuery(queryInput.value.trim(), 0));
  queryInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendQuery(queryInput.value.trim(), 0); });

  document.querySelectorAll('.chip').forEach((c) => {
    c.addEventListener('click', () => {
      const q = c.dataset.query;
      queryInput.value = q;
      sendQuery(q, 0);
    });
  });

  async function sendQuery(q, sttMs) {
    if (!q) return;
    stopSpeaking();
    answerBody.textContent = 'Thinking...';
    latencyBadge.textContent = '...';

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, strategy: strategySelect.value, use_cross_encoder: rerankToggle.checked, top_k: 4 })
      });
      if (!res.ok) throw new Error(res.status);
      const d = await res.json();
      if (sttMs > 0) { d.latency.stt_ms = Math.round(sttMs); d.latency.total_pipeline_ms += sttMs; }
      render(d);
    } catch (err) {
      answerBody.textContent = `Error: ${err.message}`;
      guardrailBadge.textContent = 'Error';
      guardrailBadge.className = 'badge badge-red';
    } finally {
      micLabel.textContent = 'Tap to speak';
    }
  }

  // ---- Render ----
  function render(d) {
    queryInput.value = d.query;
    renderWords(d.answer);
    modelBadge.textContent = d.model_used;
    const ms = d.latency.total_pipeline_ms;
    latencyBadge.textContent = `${Math.round(ms)} ms`;
    heroMs.textContent = Math.round(ms);
    if (latSummary) latSummary.textContent = `${Math.round(ms)}ms`;

    if (ms <= 200) {
      targetTag.textContent = '✓ <200ms';
      targetTag.style.color = '#00E676';
    } else {
      targetTag.textContent = `+${Math.round(ms - 200)}ms over`;
      targetTag.style.color = '#FF7A00';
    }

    if (!d.is_safe) { guardrailBadge.textContent = 'Blocked'; guardrailBadge.className = 'badge badge-red'; }
    else if (!d.is_grounded) { guardrailBadge.textContent = 'Ungrounded'; guardrailBadge.className = 'badge badge-red'; }
    else { guardrailBadge.textContent = 'Grounded'; guardrailBadge.className = 'badge badge-green'; }

    const total = Math.max(1, ms);
    setBar(bars.stt, d.latency.stt_ms, total);
    setBar(bars.guard, d.latency.pre_guardrail_ms, total);
    setBar(bars.ret, d.latency.retrieval_ms, total);
    setBar(bars.rerank, d.latency.rerank_ms, total);
    setBar(bars.llm, d.latency.llm_generation_ms, total);
    setBar(bars.post, d.latency.post_guardrail_ms, total);

    renderCitations(d.citations);

    queryCount++;
    if (queriesVal) queriesVal.textContent = queryCount;
    if (ms < best && ms > 5) { best = ms; if (bestLat) bestLat.textContent = `${Math.round(best)}ms`; }
  }

  function setBar(b, ms, total) {
    const v = ms || 0;
    b.bar.style.width = `${Math.min(100, Math.max(4, (v / total) * 100))}%`;
    b.val.textContent = `${Math.round(v)}ms`;
  }

  function renderWords(text) {
    rawText = text;
    wordSpans = [];
    answerBody.innerHTML = '';
    const re = /(\S+)/g;
    let m, last = 0, idx = 0;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) answerBody.appendChild(document.createTextNode(text.slice(last, m.index)));
      const s = document.createElement('span');
      s.className = 'tts-word';
      s.textContent = m[0];
      s.dataset.start = m.index;
      s.dataset.end = m.index + m[0].length;
      wordSpans.push({ el: s, start: m.index, end: m.index + m[0].length });
      answerBody.appendChild(s);
      last = re.lastIndex;
      idx++;
    }
    if (last < text.length) answerBody.appendChild(document.createTextNode(text.slice(last)));
  }

  function renderCitations(cits) {
    citGrid.innerHTML = '';
    if (!cits || !cits.length) {
      citCount.textContent = '0';
      citGrid.innerHTML = '<span class="citations-empty">No sources cited.</span>';
      return;
    }
    citCount.textContent = cits.length;
    cits.forEach(c => {
      const tag = document.createElement('span');
      tag.className = 'cit-tag';
      tag.innerHTML = `Doc-${c.doc_index} · ${c.strategy} <span class="score">${c.score.toFixed(2)}</span>`;
      citGrid.appendChild(tag);
    });
  }

  // ---- TTS ----
  function stopSpeaking() {
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    speakBtn.style.display = '';
    if (stopBtn) stopBtn.style.display = 'none';
    wordSpans.forEach(w => w.el.classList.remove('active-word'));
  }

  function startSpeaking() {
    if (!('speechSynthesis' in window)) return;
    stopSpeaking();
    const text = rawText || answerBody.textContent;
    if (!text.trim()) return;
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 1;
    speakBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = '';

    utt.onboundary = (e) => {
      const ci = e.charIndex;
      wordSpans.forEach(w => {
        if (ci >= w.start && ci < w.end) w.el.classList.add('active-word');
        else w.el.classList.remove('active-word');
      });
    };
    utt.onend = stopSpeaking;
    utt.onerror = stopSpeaking;
    speechSynthesis.speak(utt);
  }

  speakBtn.addEventListener('click', startSpeaking);
  if (stopBtn) stopBtn.addEventListener('click', stopSpeaking);

  // ---- Submit ----
  if (submitBtn) {
    submitBtn.addEventListener('click', () => {
      submitBtn.textContent = '✓ Submitted';
      submitBtn.style.background = '#00E676';
      setTimeout(() => { submitBtn.textContent = 'Submit Run ↗'; submitBtn.style.background = '#FF7A00'; }, 2500);
    });
  }
});
