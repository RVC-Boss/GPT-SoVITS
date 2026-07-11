# نُطق (Nutq) — لوحة المفاتيح الصوتية بالذكاء الاصطناعي 🎙️

**تكلّم بشكل طبيعي، ونُطق يكتب نصاً مصقولاً جاهزاً للإرسال.**

تطبيق ويب تقدمي (PWA) احترافي مستوحى من Wispr Flow — يعمل على **iPhone و Android والحاسوب (Windows / Mac / Linux)** من كود واحد، ويُثبَّت كتطبيق حقيقي بأيقونة على الشاشة الرئيسية وسطح المكتب.

![Nutq](icons/icon-192.png)

## المميزات

- 🎙️ **إملاء صوتي فوري** — النص يظهر مباشرة أثناء كلامك (محرك المتصفح المجاني)، أو عبر Whisper API لدقة أعلى.
- ✨ **صقل بالذكاء الاصطناعي** — إزالة كلمات الحشو، تصحيح علامات الترقيم، وإعادة صياغة بأسلوبك.
- 📝 **أنماط كتابة** — رسالة، بريد إلكتروني، ملاحظات، رسمي، أو نص خام بلا تعديل.
- 🌍 **أكثر من 100 لغة** — منها العربية بلهجاتها، مع اكتشاف تلقائي للغة.
- 📖 **قاموس شخصي** — أضف أسماءً ومصطلحات خاصة ليحترمها التطبيق عند الكتابة.
- 🕘 **سجل محلي** — كل إملاءاتك محفوظة على جهازك فقط (لا تُرسل لأي خادم).
- 🌗 **واجهة عربية RTL** كاملة + إنجليزية، مع مظهر فاتح وداكن.
- 📵 **يعمل دون إنترنت** — واجهة التطبيق تعمل offline بعد أول زيارة.

## التشغيل والنشر

### 1) تجربة محلية سريعة

```bash
cd nutq
python3 -m http.server 8080
# افتح http://localhost:8080
```

> ملاحظة: الميكروفون يتطلب HTTPS أو localhost.

### 2) النشر المجاني على GitHub Pages (موصى به)

المستودع يتضمن Workflow جاهزاً (`.github/workflows/deploy-nutq.yml`):

1. من إعدادات المستودع على GitHub: **Settings → Pages → Source: GitHub Actions**.
2. ادفع أي تعديل لمجلد `nutq/` وسيُنشر تلقائياً.
3. سيكون التطبيق على: `https://<اسم-المستخدم>.github.io/<اسم-المستودع>/`

يمكن أيضاً النشر على Netlify أو Vercel أو Cloudflare Pages بسحب مجلد `nutq/` فقط.

### 3) التثبيت كتطبيق

- **iPhone (Safari):** افتح الرابط → زر المشاركة → **إضافة إلى الشاشة الرئيسية**.
- **Android (Chrome):** افتح الرابط → قائمة ⋮ → **تثبيت التطبيق**.
- **الحاسوب (Chrome / Edge):** افتح الرابط → أيقونة التثبيت ⊕ في شريط العنوان → **تثبيت**. يصبح تطبيقاً مستقلاً بنافذته الخاصة.

## إعداد الذكاء الاصطناعي (اختياري لكن موصى به)

بدون مفتاح API يعمل الإملاء الفوري مجاناً عبر محرك المتصفح. لتفعيل **الصقل الذكي** و **Whisper**:

1. افتح **الإعدادات ⚙️** داخل التطبيق.
2. أدخل مفتاح API من أي خدمة متوافقة مع OpenAI:

| الخدمة | Base URL | ملاحظات |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` + `whisper-1` |
| Groq (مجاني) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` + `whisper-large-v3` |
| OpenRouter | `https://openrouter.ai/api/v1` | نماذج متعددة |

> 🔒 **الخصوصية:** المفتاح والسجل والقاموس تُحفظ في متصفحك فقط (localStorage)، ولا يمر أي شيء عبر خوادم وسيطة — الاتصال مباشر من جهازك إلى مزوّد الخدمة الذي تختاره.

## البنية التقنية

- HTML/CSS/JS خالص بدون أي اعتماديات أو خطوة بناء — سهل الصيانة والنشر في أي مكان.
- `Web Speech API` للإملاء الفوري، و`MediaRecorder` + Whisper للتفريغ عالي الدقة.
- Service Worker للعمل دون اتصال + Web App Manifest للتثبيت.

---

# Nutq — AI Voice Keyboard (English)

A professional Wispr Flow–style PWA: speak naturally and get polished, ready-to-send text. Installs as a real app on iPhone, Android, and desktop from a single codebase. Live dictation via the free browser speech engine or any OpenAI-compatible Whisper endpoint, AI polishing with writing modes (message / email / notes / formal), 100+ languages with auto-detect, a personal dictionary, local-only history, full RTL Arabic + English UI, and offline support. Serve the `nutq/` folder from any static host (a GitHub Pages workflow is included) and add your API key in Settings to enable AI features.
