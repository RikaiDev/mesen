"""
Locale catalog for user-facing mesen messages.

English is the default and the fallback. zh-TW carries the original strings.
ja entries are drafted for structure and flagged for native review before any
release that advertises Japanese support.
"""

import os

SUPPORTED_LOCALES = ("en", "zh-TW", "ja")
_JA_NEEDS_REVIEW = True

STRINGS: dict[str, dict[str, str]] = {
    "contrast_action": {
        "en": 'Text "{text}" contrast ({ratio}:1) is below the safety threshold ({threshold}:1).',
        "zh-TW": "文字「{text}」對比度 ({ratio}:1) 低於安全門檻 ({threshold}:1)。",
        "ja": "「{text}」のコントラスト比（{ratio}:1）が下限（{threshold}:1）を下回っています。",
    },
    "contrast_detail": {
        "en": "Foreground {fg} is too close to background {bg}; darken the foreground or lighten the background.",
        "zh-TW": "前景色 {fg} 與底色 {bg} 過度接近，請調深前景色或提高底色亮度。",
        "ja": "前景色 {fg} と背景色 {bg} が近似しています。前景を濃くするか背景を明るくしてください。",
    },
    "font_action": {
        "en": 'Text "{text}" physical height ({sp}sp) is below the mobile legibility floor ({min_sp}sp).',
        "zh-TW": "文字「{text}」實體高度 ({sp}sp) 低於行動端可讀下限 ({min_sp}sp)。",
        "ja": "「{text}」の実寸（{sp}sp）がモバイル可読下限（{min_sp}sp）を下回っています。",
    },
    "font_detail": {
        "en": "Hard to read on high-density mobile screens; raise the type level in the layout.",
        "zh-TW": "在行動裝置高密度螢幕上易造成閱讀困難，請在佈局中調升該文字級別。",
        "ja": "高密度画面では判読困難になります。レイアウト上で文字グレードを上げてください。",
    },
    "affordance_mirror_action": {
        "en": "The screen prompts press-and-hold interaction, but the central figure is a static illustration with no perceivable operation cues (Norman signifiers). Add a persistently visible press target: explicit bounds plus text or universal icon, at least 24x24 CSS px (WCAG 2.5.8), with a reduced-motion-safe variant (WCAG 2.3.3).",
        "zh-TW": "畫面提示使用者執行按住/互動操作，但中央角色呈現為純靜態插畫，缺乏可感知的操作暗示（Norman signifiers）。請加上持續可見的按壓目標：明確邊界＋文字或通用圖示，尺寸不小於 24x24 CSS px（WCAG 2.5.8），並提供 reduced-motion 安全版本（WCAG 2.3.3）。",
        "ja": "画面が長押し操作を促しているのに、中央のキャラクターは静止画で操作の手がかりがありません。常時表示の押下ターゲットを追加してください。明確な境界＋文字または汎用アイコン、24x24 CSS px以上（WCAG 2.5.8）、reduced-motion対応版（WCAG 2.3.3）。",
    },
    "affordance_neural_action": {
        "en": "The neural head finds no perceivable operation cues on the central target (confidence {prob}). Add a persistently visible press target: explicit bounds plus text or universal icon, at least 24x24 CSS px (WCAG 2.5.8), with a reduced-motion-safe variant (WCAG 2.3.3).",
        "zh-TW": "神經網路判定畫面中央核心目標缺乏可感知的操作暗示（可信度 {prob}）。請加上持續可見的按壓目標：明確邊界＋文字或通用圖示，尺寸不小於 24x24 CSS px（WCAG 2.5.8），並提供 reduced-motion 安全版本（WCAG 2.3.3）。",
        "ja": "中央コア対象に操作の手がかりがありません（確信度{prob}）。常時表示の押下ターゲットを追加してください。明確な境界＋文字または汎用アイコン、24x24 CSS px以上（WCAG 2.5.8）、reduced-motion対応版（WCAG 2.3.3）。",
    },
    "mirror_zone_action": {
        "en": "The central 60% of the half-mirror is the face-reflection and optical-measurement zone; text and cards are prohibited there. Move them to the corners or edges.",
        "zh-TW": "半面鏡中央 60% 為面部倒影與光學量測區，禁止放置文字或實體卡片，請移至四角或邊緣。",
        "ja": "ハーフミラー中央60%は顔反射・光学計測域のため、文字やカードの配置は禁止です。四隅または縁に移動してください。",
    },
    "desert_measured": {
        "en": "Content uses only {span}% of width ({wasted}% wasted on the sides)",
        "zh-TW": "內容僅佔橫向 {span}% 寬度（兩側荒廢 {wasted}%）",
        "ja": "横幅のうち {span}% のみ使用（両側 {wasted}% が未使用）",
    },
    "desert_threshold": {
        "en": "Effective landscape width utilization >= 65%",
        "zh-TW": "橫螢幕寬度有效利用率 >= 65%",
        "ja": "横幅の有効利用率 >= 65%",
    },
    "desert_action": {
        "en": "On a {aspect}:1 ultra-wide screen all content is squeezed into a single central {span}% column, wasting {left}% left and {right}% right. Strongly recommend a two-column split layout: subject visual left, headings, guidance, and actions right.",
        "zh-TW": "當前螢幕為 {aspect}:1 超寬橫螢幕，但所有內容被死板擠在中央 {span}% 的單一垂直細柱中，左右兩側各有 {left}% 與 {right}% 大面積空間完全被浪費。強烈建議改採左右「雙欄式排版（Split Layout）」：左欄放置角色視覺主體，右欄佈局標題、指引與操作按鈕。",
        "ja": "{aspect}:1 の超ワイド画面で、内容が中央 {span}% の細柱に押し込められています。左右に {left}% と {right}% の未使用領域があります。左右2カラム（Split Layout）への変更を強く推奨します。左にビジュアル主体、右にタイトル・案内・操作ボタンを配置してください。",
    },
    "aspect_measured": {
        "en": "Five-level vertical stack on a {aspect}:1 landscape screen",
        "zh-TW": "直式單欄 5 層垂直堆疊在 {aspect}:1 橫螢幕",
        "ja": "{aspect}:1 横画面での縦積み5層スタック",
    },
    "aspect_threshold": {
        "en": "Adaptive wide layout for landscape screens",
        "zh-TW": "橫螢幕自適應寬屏佈局",
        "ja": "横画面の自適応ワイドレイアウト",
    },
    "aspect_action": {
        "en": "A phone/tablet vertical stack applied verbatim to an ultra-wide {aspect}:1 screen compresses five layers vertically with bottom text in the danger zone. Drop the single-column constraint for horizontal card flow or a two-column structure.",
        "zh-TW": "將直式手機/平板的垂直堆疊設計直接硬套於超寬 {aspect}:1 橫螢幕上，導致 5 層元素垂直擠壓，底部文字貼齊下緣危險區。應解除單欄垂直堆疊約束，改採橫向流式卡片或左右兩欄結構。",
        "ja": "縦積みを{aspect}:1超ワイド画面にそのまま適用し、5層が垂直に圧縮され下部テキストが危険域に接しています。単一カラムの制約を外し、横フローカードまたは2カラム構成にしてください。",
    },
    "thumb_measured": {
        "en": "Core interaction target dead center (x=0.50)",
        "zh-TW": "核心互動目標位於正中央 (x=0.50)",
        "ja": "コア操作対象が正中央 (x=0.50)",
    },
    "thumb_threshold": {
        "en": "Natural two-handed thumb reach zones (x <= 0.25 or x >= 0.75)",
        "zh-TW": "兩手握持拇指自然熱區 (x <= 0.25 或 x >= 0.75)",
        "ja": "両手持ち親指の自然到達域 (x <= 0.25 または x >= 0.75)",
    },
    "thumb_action": {
        "en": "Two-handed grip makes the center a thumb stretch dead zone; forcing a center long-press destabilizes the hold. Move press-and-hold or the core CTA into the right-thumb natural reach zone.",
        "zh-TW": "橫向雙手握持手機時，中央區域屬於拇指最難觸及的拉伸死角（Stretch Zone）。強迫使用者長按螢幕正中央會破壞手持握持穩定性，建議將按住操作或核心 CTA 移至靠近右側拇指自然操作熱區。",
        "ja": "両手持ち時、中央は親指の届きにくいストレッチゾーンです。中央での長押しは把持を不安定にします。長押し操作やコアCTAを右親指の自然到達域に移してください。",
    },
}

# Action-prompt detection keywords across scripts (detection data, not prose).
ACTION_PROMPT_KEYWORDS = [
    "按住",
    "點擊",
    "点击",
    "按一下",
    "長押し",
    "押して",
    "タップ",
    "クリック",
    "hold",
    "tap",
    "press",
]


def resolve_locale(locale: str | None) -> str:
    """Resolve a requested locale to a supported one; English is the fallback."""
    if not locale:
        locale = ""
    normalized = locale.strip().replace("_", "-")
    if normalized in SUPPORTED_LOCALES:
        return normalized
    if normalized.startswith("ja"):
        return "ja"
    if normalized.startswith("zh"):
        return "zh-TW"
    env = (os.environ.get("MESEN_LOCALE") or "").strip().replace("_", "-")
    if env in SUPPORTED_LOCALES:
        return env
    return "en"


def t(key: str, locale: str | None = None, **params: object) -> str:
    """Render a catalog message; falls back to English, then to the key."""
    entry = STRINGS.get(key, {})
    resolved = resolve_locale(locale)
    template = entry.get(resolved) or entry.get("en") or key
    try:
        return template.format(**params)
    except (KeyError, IndexError):
        return template
