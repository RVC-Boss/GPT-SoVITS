/* ===== Nutq — AI voice keyboard =====
 * Speak naturally → get polished, ready-to-send text.
 * Engines: Web Speech API (free, live) or any OpenAI-compatible
 * Whisper endpoint. Polishing via OpenAI-compatible chat API.
 * Everything is stored locally on the device.
 */
"use strict";

/* ---------- i18n ---------- */
const I18N = {
  ar: {
    appName: "نُطق",
    heroTitle: "إملاء صوتي<br>بلا أخطاء",
    heroSub: "تكلّم بشكل طبيعي، ونُطق يحوّل كلامك إلى نص مصقول جاهز للإرسال — في أي تطبيق وبأكثر من 100 لغة.",
    modeMessage: "رسالة", modeEmail: "بريد إلكتروني", modeNotes: "ملاحظات", modeFormal: "رسمي", modeRaw: "نص خام",
    rawTranscript: "النص المسموع", polished: "✨ النص المصقول",
    transcriptPh: "سيظهر كلامك هنا…",
    copy: "نسخ", copied: "تم النسخ ✓", share: "مشاركة", repolish: "إعادة الصقل",
    tapToTalk: "اضغط وتكلّم", tapToStop: "اضغط للإيقاف",
    listening: "أستمع إليك…", transcribing: "جارٍ تفريغ الصوت…", polishing: "جارٍ صقل النص…",
    history: "السجل", clearAll: "مسح الكل", settings: "الإعدادات",
    uiLang: "لغة الواجهة", engine: "محرك التعرّف على الصوت",
    engineAuto: "تلقائي (الأفضل المتاح)", engineWeb: "المتصفح (مجاني، فوري)", engineWhisper: "Whisper عبر API (أدق)",
    aiSection: "إعدادات الذكاء الاصطناعي (للصقل و Whisper)",
    apiKey: "مفتاح API", baseUrl: "عنوان الخدمة (Base URL)", model: "نموذج الصقل", whisperModel: "نموذج التفريغ الصوتي",
    apiHint: "يدعم أي خدمة متوافقة مع OpenAI: ‏OpenAI، ‏Groq (مجاني)، ‏OpenRouter وغيرها. المفتاح يُحفظ على جهازك فقط.",
    dictionary: "القاموس الشخصي (أسماء ومصطلحات، افصل بينها بفواصل)",
    dictPh: "مثال: جلال، شركة لُمى، GPT-SoVITS",
    autoPolish: "صقل تلقائي بعد التوقف",
    theme: "المظهر", themeAuto: "تلقائي", themeLight: "فاتح", themeDark: "داكن",
    save: "حفظ", saved: "تم الحفظ ✓",
    searchLang: "ابحث عن لغة…", autoDetect: "اكتشاف تلقائي",
    f1t: "يعمل مع كل التطبيقات", f1d: "أملِ النص ثم انسخه إلى واتساب، البريد، أو أي تطبيق آخر.",
    f2t: "صقل بالذكاء الاصطناعي", f2d: "بلا كلمات حشو ولا أخطاء — نص بأسلوبك، جاهز للإرسال.",
    f3t: "أكثر من 100 لغة", f3d: "العربية، الإنجليزية، الفرنسية وغيرها — مع اكتشاف تلقائي.",
    f4t: "قاموسك الشخصي", f4d: "أضف أسماءً ومصطلحات خاصة ليتعلم نُطق أسلوبك.",
    errNoMic: "تعذّر الوصول إلى الميكروفون. تحقّق من الأذونات.",
    errNoSpeech: "المتصفح لا يدعم التعرّف على الصوت. أضف مفتاح API في الإعدادات لاستخدام Whisper.",
    errNoKey: "أضف مفتاح API في الإعدادات أولاً لاستخدام هذه الميزة.",
    errApi: "خطأ من الخدمة: ", errNet: "تعذّر الاتصال بالخدمة. تحقّق من الإنترنت.",
    noHistory: "لا يوجد شيء في السجل بعد.", confirmClear: "هل تريد مسح كل السجل؟",
    deleted: "تم الحذف", noPolishKey: "أُبقي النص كما هو — أضف مفتاح API في الإعدادات لتفعيل الصقل الذكي.",
    installHint: "لتثبيت التطبيق: شارك ← أضف إلى الشاشة الرئيسية",
  },
  en: {
    appName: "Nutq",
    heroTitle: "Dictation<br>without the mistakes",
    heroSub: "Talk naturally — Nutq turns your speech into polished, ready-to-send text, in any app and 100+ languages.",
    modeMessage: "Message", modeEmail: "Email", modeNotes: "Notes", modeFormal: "Formal", modeRaw: "Raw",
    rawTranscript: "Raw transcript", polished: "✨ Polished text",
    transcriptPh: "Your speech will appear here…",
    copy: "Copy", copied: "Copied ✓", share: "Share", repolish: "Re-polish",
    tapToTalk: "Tap & talk", tapToStop: "Tap to stop",
    listening: "Listening…", transcribing: "Transcribing…", polishing: "Polishing…",
    history: "History", clearAll: "Clear all", settings: "Settings",
    uiLang: "Interface language", engine: "Speech engine",
    engineAuto: "Auto (best available)", engineWeb: "Browser (free, live)", engineWhisper: "Whisper via API (accurate)",
    aiSection: "AI settings (polish & Whisper)",
    apiKey: "API key", baseUrl: "Base URL", model: "Polish model", whisperModel: "Transcription model",
    apiHint: "Works with any OpenAI-compatible service: OpenAI, Groq (free), OpenRouter… Your key never leaves this device.",
    dictionary: "Personal dictionary (names & terms, comma-separated)",
    dictPh: "e.g. Jalal, Luma Inc, GPT-SoVITS",
    autoPolish: "Auto-polish after stopping",
    theme: "Theme", themeAuto: "Auto", themeLight: "Light", themeDark: "Dark",
    save: "Save", saved: "Saved ✓",
    searchLang: "Search languages…", autoDetect: "Auto-detect",
    f1t: "Works with every app", f1d: "Dictate, then paste into WhatsApp, email, or anywhere else.",
    f2t: "AI-polished output", f2d: "No filler words, no typos — your voice, cleaned up and ready to send.",
    f3t: "100+ languages", f3d: "Arabic, English, French and more — with auto-detect.",
    f4t: "Personal dictionary", f4d: "Add custom names and terms so Nutq learns your lingo.",
    errNoMic: "Microphone access failed. Check permissions.",
    errNoSpeech: "This browser has no speech recognition. Add an API key in Settings to use Whisper.",
    errNoKey: "Add an API key in Settings first.",
    errApi: "Service error: ", errNet: "Could not reach the service. Check your connection.",
    noHistory: "Nothing here yet.", confirmClear: "Clear all history?",
    deleted: "Deleted", noPolishKey: "Kept raw text — add an API key in Settings to enable AI polish.",
    installHint: "To install: Share → Add to Home Screen",
  },
};

/* ---------- Languages (speech recognition targets) ---------- */
const LANGS = [
  ["auto", "", ""],
  ["ar-SA", "🇸🇦", "العربية"], ["ar-EG", "🇪🇬", "العربية (مصر)"], ["ar-MA", "🇲🇦", "العربية (المغرب)"],
  ["en-US", "🇺🇸", "English (US)"], ["en-GB", "🇬🇧", "English (UK)"], ["en-AU", "🇦🇺", "English (AU)"], ["en-IN", "🇮🇳", "English (India)"],
  ["fr-FR", "🇫🇷", "Français"], ["fr-CA", "🇨🇦", "Français (Canada)"],
  ["es-ES", "🇪🇸", "Español"], ["es-MX", "🇲🇽", "Español (México)"], ["es-AR", "🇦🇷", "Español (Argentina)"],
  ["de-DE", "🇩🇪", "Deutsch"], ["it-IT", "🇮🇹", "Italiano"],
  ["pt-BR", "🇧🇷", "Português (Brasil)"], ["pt-PT", "🇵🇹", "Português"],
  ["ru-RU", "🇷🇺", "Русский"], ["uk-UA", "🇺🇦", "Українська"],
  ["tr-TR", "🇹🇷", "Türkçe"], ["fa-IR", "🇮🇷", "فارسی"], ["ur-PK", "🇵🇰", "اردو"],
  ["hi-IN", "🇮🇳", "हिन्दी"], ["bn-BD", "🇧🇩", "বাংলা"], ["ta-IN", "🇮🇳", "தமிழ்"], ["te-IN", "🇮🇳", "తెలుగు"],
  ["ja-JP", "🇯🇵", "日本語"], ["ko-KR", "🇰🇷", "한국어"],
  ["zh-CN", "🇨🇳", "中文 (简体)"], ["zh-TW", "🇹🇼", "中文 (繁體)"], ["zh-HK", "🇭🇰", "粵語 (香港)"],
  ["id-ID", "🇮🇩", "Bahasa Indonesia"], ["ms-MY", "🇲🇾", "Bahasa Melayu"],
  ["th-TH", "🇹🇭", "ไทย"], ["vi-VN", "🇻🇳", "Tiếng Việt"], ["fil-PH", "🇵🇭", "Filipino"],
  ["nl-NL", "🇳🇱", "Nederlands"], ["sv-SE", "🇸🇪", "Svenska"], ["nb-NO", "🇳🇴", "Norsk"],
  ["da-DK", "🇩🇰", "Dansk"], ["fi-FI", "🇫🇮", "Suomi"], ["is-IS", "🇮🇸", "Íslenska"],
  ["pl-PL", "🇵🇱", "Polski"], ["cs-CZ", "🇨🇿", "Čeština"], ["sk-SK", "🇸🇰", "Slovenčina"],
  ["hu-HU", "🇭🇺", "Magyar"], ["ro-RO", "🇷🇴", "Română"], ["bg-BG", "🇧🇬", "Български"],
  ["el-GR", "🇬🇷", "Ελληνικά"], ["sr-RS", "🇷🇸", "Српски"], ["hr-HR", "🇭🇷", "Hrvatski"],
  ["sl-SI", "🇸🇮", "Slovenščina"], ["lt-LT", "🇱🇹", "Lietuvių"], ["lv-LV", "🇱🇻", "Latviešu"], ["et-EE", "🇪🇪", "Eesti"],
  ["he-IL", "🇮🇱", "עברית"], ["am-ET", "🇪🇹", "አማርኛ"], ["sw-KE", "🇰🇪", "Kiswahili"],
  ["af-ZA", "🇿🇦", "Afrikaans"], ["zu-ZA", "🇿🇦", "isiZulu"],
  ["ka-GE", "🇬🇪", "ქართული"], ["hy-AM", "🇦🇲", "Հայերեն"], ["az-AZ", "🇦🇿", "Azərbaycanca"],
  ["kk-KZ", "🇰🇿", "Қазақша"], ["uz-UZ", "🇺🇿", "Oʻzbekcha"],
  ["ne-NP", "🇳🇵", "नेपाली"], ["si-LK", "🇱🇰", "සිංහල"], ["km-KH", "🇰🇭", "ខ្មែរ"], ["lo-LA", "🇱🇦", "ລາວ"],
  ["my-MM", "🇲🇲", "မြန်မာ"], ["mn-MN", "🇲🇳", "Монгол"],
  ["ca-ES", "🏴", "Català"], ["eu-ES", "🏴", "Euskara"], ["gl-ES", "🏴", "Galego"],
  ["mr-IN", "🇮🇳", "मराठी"], ["gu-IN", "🇮🇳", "ગુજરાતી"], ["kn-IN", "🇮🇳", "ಕನ್ನಡ"], ["ml-IN", "🇮🇳", "മലയാളം"], ["pa-IN", "🇮🇳", "ਪੰਜਾਬੀ"],
];

/* ---------- Polish prompts ---------- */
const MODE_PROMPTS = {
  message: "Rewrite the dictated text as a natural, friendly chat message. Keep it short and casual, as if sent on WhatsApp or iMessage.",
  email: "Rewrite the dictated text as a clear, well-structured email body. Add a suitable greeting and sign-off placeholder if appropriate. Use short paragraphs and bullet points where they help.",
  notes: "Rewrite the dictated text as concise personal notes: short bullet points capturing every idea, no fluff.",
  formal: "Rewrite the dictated text in polished, formal, professional language suitable for official correspondence.",
  raw: null,
};

/* ---------- State ---------- */
const DEFAULTS = {
  uiLang: (navigator.language || "ar").startsWith("ar") ? "ar" : "en",
  lang: "auto",
  mode: "message",
  engine: "auto",
  apiKey: "",
  baseUrl: "https://api.openai.com/v1",
  model: "gpt-4o-mini",
  whisperModel: "whisper-1",
  dictionary: "",
  autoPolish: true,
  theme: "auto",
};
let S = loadSettings();
let history = loadJSON("nutq.history", []);
let recState = "idle"; // idle | listening | transcribing | polishing
let recognition = null, mediaRecorder = null, mediaStream = null, chunks = [];
let finalText = "", timerInt = null, timerStart = 0;

function loadSettings() {
  return Object.assign({}, DEFAULTS, loadJSON("nutq.settings", {}));
}
function loadJSON(k, fb) {
  try { return JSON.parse(localStorage.getItem(k)) ?? fb; } catch { return fb; }
}
function saveJSON(k, v) { localStorage.setItem(k, JSON.stringify(v)); }

/* ---------- DOM helpers ---------- */
const $ = (id) => document.getElementById(id);
const t = (key) => (I18N[S.uiLang] || I18N.ar)[key] || key;

function applyI18n() {
  document.documentElement.lang = S.uiLang;
  document.documentElement.dir = S.uiLang === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (key === "heroTitle") el.innerHTML = t(key);
    else el.textContent = t(key);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const v = t(el.dataset.i18nPlaceholder);
    if ("placeholder" in el) el.placeholder = v;
    else el.dataset.placeholder = v;
  });
  $("transcript").dataset.placeholder = t("transcriptPh");
  updateLangLabel();
}

function applyTheme() {
  const root = document.documentElement;
  if (S.theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", S.theme);
}

function toast(msg, ms = 2200) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.hidden = true), ms);
}

function setStatus(msg, isErr = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.classList.toggle("error", isErr);
}

function updateLangLabel() {
  const item = LANGS.find(([code]) => code === S.lang);
  $("langLabel").textContent =
    !item || item[0] === "auto" ? t("autoDetect") : `${item[1]} ${item[2]}`;
}

/* ---------- Recording UI ---------- */
function setRecUI(on) {
  $("btnMic").classList.toggle("rec", on);
  $("btnMic").querySelector(".mic-ico").hidden = on;
  $("btnMic").querySelector(".stop-ico").hidden = !on;
  $("waveform").hidden = !on;
  $("timer").hidden = !on;
  $("liveDot").hidden = !on;
  $("micHint").textContent = on ? t("tapToStop") : t("tapToTalk");
  $("hero").classList.toggle("compact", on || !!finalText);
  $("features").hidden = on || !!finalText || !!$("result").textContent;
  if (on) {
    $("transcriptCard").hidden = false;
    timerStart = Date.now();
    timerInt = setInterval(() => {
      const s = Math.floor((Date.now() - timerStart) / 1000);
      $("timer").textContent =
        String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
    }, 500);
  } else {
    clearInterval(timerInt);
    $("timer").textContent = "00:00";
  }
}

function renderTranscript(interim = "") {
  const el = $("transcript");
  el.innerHTML = "";
  el.appendChild(document.createTextNode(finalText));
  if (interim) {
    const span = document.createElement("span");
    span.className = "interim";
    span.textContent = (finalText ? " " : "") + interim;
    el.appendChild(span);
  }
}

/* ---------- Engine selection ---------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
function pickEngine() {
  if (S.engine === "webspeech") return SR ? "webspeech" : null;
  if (S.engine === "whisper") return S.apiKey ? "whisper" : null;
  // auto: prefer live browser engine, fall back to whisper
  if (SR) return "webspeech";
  if (S.apiKey) return "whisper";
  return null;
}

/* ---------- Web Speech engine ---------- */
function startWebSpeech() {
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  if (S.lang !== "auto") recognition.lang = S.lang;
  else if (S.uiLang === "ar") recognition.lang = "ar-SA";

  recognition.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const txt = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText = (finalText + " " + txt).trim();
      else interim += txt;
    }
    renderTranscript(interim);
  };
  recognition.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      stopDictation(false);
      setStatus(t("errNoMic"), true);
    }
    // "no-speech" and "aborted" are benign; onend handles restart
  };
  recognition.onend = () => {
    // Browsers stop after silence — keep listening until user taps stop.
    if (recState === "listening") {
      try { recognition.start(); } catch { /* already started */ }
    }
  };
  recognition.start();
}

/* ---------- Whisper engine ---------- */
async function startWhisperRecording() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  chunks = [];
  const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]
    .find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";
  mediaRecorder = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : undefined);
  mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRecorder.start(1000);
}

async function transcribeWithWhisper() {
  const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
  const ext = (mediaRecorder.mimeType || "audio/webm").includes("mp4") ? "mp4"
    : (mediaRecorder.mimeType || "").includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("file", blob, "audio." + ext);
  form.append("model", S.whisperModel || "whisper-1");
  if (S.lang !== "auto") form.append("language", S.lang.split("-")[0]);
  if (S.dictionary.trim()) form.append("prompt", S.dictionary.trim());

  const res = await fetch(S.baseUrl.replace(/\/$/, "") + "/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: "Bearer " + S.apiKey },
    body: form,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(t("errApi") + res.status + " " + body.slice(0, 200));
  }
  const data = await res.json();
  return (data.text || "").trim();
}

/* ---------- AI polish ---------- */
async function polish(text) {
  const sysMode = MODE_PROMPTS[S.mode];
  if (!sysMode) return text; // raw mode
  if (!S.apiKey) {
    toast(t("noPolishKey"), 3200);
    return text;
  }
  const dict = S.dictionary.trim()
    ? `\nPersonal dictionary (respect these exact spellings): ${S.dictionary.trim()}`
    : "";
  const sys =
    "You are Nutq, an AI dictation cleanup engine. The user dictated text by voice. " +
    "Fix transcription artifacts, remove filler words, add correct punctuation and formatting, " +
    "and keep the user's own voice and meaning. Always answer in the SAME language as the dictated text. " +
    "Output ONLY the rewritten text, no explanations." +
    "\nStyle: " + sysMode + dict;

  const res = await fetch(S.baseUrl.replace(/\/$/, "") + "/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer " + S.apiKey,
    },
    body: JSON.stringify({
      model: S.model || "gpt-4o-mini",
      temperature: 0.3,
      messages: [
        { role: "system", content: sys },
        { role: "user", content: text },
      ],
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(t("errApi") + res.status + " " + body.slice(0, 200));
  }
  const data = await res.json();
  return (data.choices?.[0]?.message?.content || "").trim() || text;
}

/* ---------- Dictation flow ---------- */
async function startDictation() {
  finalText = "";
  $("result").textContent = "";
  $("resultCard").hidden = true;
  renderTranscript();
  const engine = pickEngine();
  if (!engine) {
    setStatus(SR ? t("errNoKey") : t("errNoSpeech"), true);
    openSheet("settingsSheet");
    return;
  }
  try {
    if (engine === "webspeech") startWebSpeech();
    else await startWhisperRecording();
    recState = "listening";
    setRecUI(true);
    setStatus(t("listening"));
  } catch (err) {
    console.error(err);
    setStatus(t("errNoMic"), true);
  }
}

async function stopDictation(process = true) {
  const wasEngine = recognition ? "webspeech" : "whisper";
  recState = "idle";
  if (recognition) { try { recognition.stop(); } catch {} recognition = null; }
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    await new Promise((r) => { mediaRecorder.onstop = r; mediaRecorder.stop(); });
  }
  if (mediaStream) { mediaStream.getTracks().forEach((tr) => tr.stop()); mediaStream = null; }
  setRecUI(false);
  if (!process) return;

  try {
    if (wasEngine === "whisper" && chunks.length) {
      recState = "transcribing";
      setStatus(t("transcribing"));
      finalText = await transcribeWithWhisper();
      renderTranscript();
    }
    finalText = $("transcript").textContent.trim();
    if (!finalText) { setStatus(""); return; }
    if (S.autoPolish) await runPolish();
    else setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(err.message.includes("fetch") ? t("errNet") : err.message, true);
  } finally {
    recState = "idle";
  }
}

async function runPolish() {
  const text = $("transcript").textContent.trim();
  if (!text) return;
  recState = "polishing";
  setStatus(t("polishing"));
  try {
    const out = await polish(text);
    $("result").textContent = out;
    $("resultCard").hidden = false;
    $("resultMode").textContent = t("mode" + S.mode[0].toUpperCase() + S.mode.slice(1));
    setStatus("");
    addHistory({ raw: text, polished: out, mode: S.mode, ts: Date.now() });
    $("result").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) {
    console.error(err);
    setStatus(err.message.includes("fetch") ? t("errNet") : err.message, true);
  } finally {
    recState = "idle";
  }
}

/* ---------- History ---------- */
function addHistory(item) {
  history.unshift(item);
  history = history.slice(0, 200);
  saveJSON("nutq.history", history);
}
function renderHistory() {
  const list = $("historyList");
  list.innerHTML = "";
  if (!history.length) {
    list.innerHTML = `<div class="empty">${t("noHistory")}</div>`;
    return;
  }
  history.forEach((item, i) => {
    const div = document.createElement("div");
    div.className = "history-item";
    const date = new Date(item.ts).toLocaleString(S.uiLang === "ar" ? "ar" : "en", {
      dateStyle: "medium", timeStyle: "short",
    });
    div.innerHTML = `
      <div class="h-meta"><span>${date}</span><span class="badge">${t("mode" + item.mode[0].toUpperCase() + item.mode.slice(1))}</span></div>
      <div class="h-text"></div>
      <div class="h-actions">
        <button class="btn small primary h-copy">${t("copy")}</button>
        <button class="btn small danger h-del">✕</button>
      </div>`;
    div.querySelector(".h-text").textContent = item.polished || item.raw;
    div.querySelector(".h-copy").onclick = () => copyText(item.polished || item.raw);
    div.querySelector(".h-del").onclick = () => {
      history.splice(i, 1);
      saveJSON("nutq.history", history);
      renderHistory();
      toast(t("deleted"));
    };
    list.appendChild(div);
  });
}

/* ---------- Clipboard / share ---------- */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  toast(t("copied"));
}

/* ---------- Sheets ---------- */
function openSheet(id) { $(id).hidden = false; document.body.style.overflow = "hidden"; }
function closeSheet(id) { $(id).hidden = true; document.body.style.overflow = ""; }
["langSheet", "historySheet", "settingsSheet"].forEach((id) => {
  $(id).addEventListener("click", (e) => { if (e.target === $(id)) closeSheet(id); });
});

/* ---------- Language list ---------- */
function renderLangs(filter = "") {
  const list = $("langList");
  list.innerHTML = "";
  const q = filter.trim().toLowerCase();
  LANGS.filter(([code, , name]) =>
    !q || name.toLowerCase().includes(q) || code.toLowerCase().includes(q)
  ).forEach(([code, flag, name]) => {
    const btn = document.createElement("button");
    btn.className = "lang-item" + (S.lang === code ? " selected" : "");
    btn.innerHTML = code === "auto"
      ? `<span><span class="flag">🌐</span>${t("autoDetect")}</span>`
      : `<span><span class="flag">${flag}</span>${name}</span>`;
    btn.onclick = () => {
      S.lang = code;
      saveJSON("nutq.settings", S);
      updateLangLabel();
      closeSheet("langSheet");
    };
    list.appendChild(btn);
  });
}

/* ---------- Settings ---------- */
function fillSettingsForm() {
  $("setUiLang").value = S.uiLang;
  $("setEngine").value = S.engine;
  $("setApiKey").value = S.apiKey;
  $("setBaseUrl").value = S.baseUrl;
  $("setModel").value = S.model;
  $("setWhisperModel").value = S.whisperModel;
  $("setDictionary").value = S.dictionary;
  $("setAutoPolish").checked = S.autoPolish;
  $("setTheme").value = S.theme;
}
function saveSettingsForm() {
  S.uiLang = $("setUiLang").value;
  S.engine = $("setEngine").value;
  S.apiKey = $("setApiKey").value.trim();
  S.baseUrl = $("setBaseUrl").value.trim() || DEFAULTS.baseUrl;
  S.model = $("setModel").value.trim() || DEFAULTS.model;
  S.whisperModel = $("setWhisperModel").value.trim() || DEFAULTS.whisperModel;
  S.dictionary = $("setDictionary").value;
  S.autoPolish = $("setAutoPolish").checked;
  S.theme = $("setTheme").value;
  saveJSON("nutq.settings", S);
  applyI18n();
  applyTheme();
  closeSheet("settingsSheet");
  toast(t("saved"));
}

/* ---------- Wire up ---------- */
$("btnMic").onclick = () => {
  if (recState === "listening") stopDictation();
  else if (recState === "idle") startDictation();
};
$("btnCopy").onclick = () => copyText($("result").textContent);
$("btnShare").onclick = async () => {
  const text = $("result").textContent;
  if (navigator.share) { try { await navigator.share({ text }); } catch {} }
  else copyText(text);
};
$("btnRepolish").onclick = () => { if (recState === "idle") runPolish(); };
$("btnLang").onclick = () => { renderLangs(); openSheet("langSheet"); $("langSearch").value = ""; };
$("langSearch").oninput = (e) => renderLangs(e.target.value);
$("btnHistory").onclick = () => { renderHistory(); openSheet("historySheet"); };
$("btnClearHistory").onclick = () => {
  if (confirm(t("confirmClear"))) {
    history = [];
    saveJSON("nutq.history", history);
    renderHistory();
  }
};
$("btnSettings").onclick = () => { fillSettingsForm(); openSheet("settingsSheet"); };
$("btnSaveSettings").onclick = saveSettingsForm;

document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    S.mode = chip.dataset.mode;
    saveJSON("nutq.settings", S);
  };
});
// restore active mode chip
document.querySelectorAll(".chip").forEach((c) =>
  c.classList.toggle("active", c.dataset.mode === S.mode));

/* ---------- Init ---------- */
applyI18n();
applyTheme();
$("verInfo").textContent = "Nutq v1.0";

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  });
}
