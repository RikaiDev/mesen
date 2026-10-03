# yana live audit (2026-10-03)

Production build `4bd7eecf` on `http://45.32.56.160`. Report-only gate,
8 verdicts: `/products` and the first live PDP at 375/768/1024/1440.
283 violations total.

## System 1: still collapsed, as expected

`visual_integrity=no` and `overall_quality` 0–1 on all 8 pages. The head
gives one answer regardless of input; nothing here is usable.

## System 2: real, differentiated signal — with two measurement defects

Verdicts: PDP 375 `conditional_pass`/2, PDP 1440 `conditional_pass`/1,
everything else `rejected`/0. Violation mix:

| rule | critical | warning | info |
|---|---|---|---|
| contrast-ratio-insufficient | 40 | 71 | 64 |
| text-reflow-overflow (font-size fallback) | 0 | 55 | 53 |

### The contrast findings are TRUE

Pixel crops of the cited boxes confirm real low-contrast grey-on-white text.
`/products` @1440 cited `#ABABAB`-family greys on `#FFFFFF` at 2.63–2.65:1;
the crop renders "商品介紹 規格說明 品名：歐卡牛牛・獨領風騷" in visibly
light grey. Mobile @375 cited `#B1B1B1` on white at 2.19:1. These are real
WCAG AA failures on a customer-facing page, reproducible by hand.

### Defect 1 — the DPI model is wrong for every capture (38% of violations)

`EvidenceEngine` hardcodes `default_dpi=440`, so `density_scale = 2.75` and
`sp = pixel_height / 2.75`. The witness captures at **dsf = 1.00** (measured:
375→375, 768→768, 1024→1024, 1440→1440). A 16 CSS px font therefore reads as
5.8sp and fails the 14sp mobile floor. Measured sp values on `/products`
@375: 4.4 / 5.8 / 7.3 / 8.7 / 10.2 / 11.6 / 13.1 — i.e. 12–36 CSS px text
reported as 4–13sp. 108 of 283 violations are this artifact.

Fix: carry `dpr` in witness `viewportFacts` and derive sp from real device
pixels; the context floor stays a context decision.

### Defect 2 — OCR confidence is not a truth signal

`ch_PP-OCRv4_rec` is a Simplified-Chinese model applied to a zh-TW storefront.
The cited string for the crop above is "商品介绍规格明品名：歐卡牛牛·獨领風";
the page actually renders "商品介紹 規格說明 品名：歐卡牛牛・獨領風騷".
Characters were simplified (紹→绍, 領→领) or dropped (騷). The CTC head is
confidently wrong, so `confidence >= 0.85` passes the reliability gate and
40 violations reached `critical` severity on strings that never existed.
44 of 283 violations (16%) quote Simplified-only forms.

## Receipts

- verdict files: `/tmp/witness-audit1/{pdp,products}/verdict-<vp>.json`
- crops: `/tmp/witness-audit1/crops/`
- state confirms `dsf=1.00` and `consoleErrors=[]` on all 8 pages

## Decision

Gate stays report-only — now with numbers instead of caution. Block on
System 2 only after the DPI fix (removes the 38% artifact class) and a
human-labeled comparison set exists. System 1 contributes nothing.
## Verified after the fixes (same 8 production viewports)

Re-captured from `http://45.32.56.160` after c97f127 and 5aaff75:

| | before | after |
|---|---|---|
| violations | 283 | 184 |
| font-size violations | 108 | 9 |
| locators that were OCR strings | 175 | 0 |
| criticals quoting a wrong string | 40 | 0 |
| icons judged by the 4.5:1 text rule | 3 | 0 |

- Font size: 16 CSS px body text now measures 16sp instead of 5.8sp. The 9
  remaining findings are genuine 12sp caption text, below the 14sp mobile
  floor.
- Contrast: 175 text findings, now routed by glyph count. Three graphic
  regions (a header emblem at 1.51:1, 1.47:1) are reported under
  `accessibility/non-text-contrast-insufficient` against the 1.4.11 3:1 floor
  instead of 1.4.3's 4.5:1, which does not govern them.
- Every locator is measured geometry. Decoded strings appear only as
  `OCR 辨識為「…」` hints.
