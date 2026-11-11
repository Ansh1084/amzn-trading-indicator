---
name: TradingView Pro Financial Trading Platform
colors:
  surface: '#131722'
  surface-dim: '#131722'
  surface-bright: '#2A2E39'
  surface-container-lowest: '#0e1118'
  surface-container-low: '#1E222D'
  surface-container: '#1E222D'
  surface-container-high: '#2A2E39'
  surface-container-highest: '#363A45'
  primary: '#2962FF'
  on-primary: '#FFFFFF'
  primary-container: '#1E53E5'
  on-surface: '#D1D4DC'
  on-surface-variant: '#787B86'
  outline: '#434651'
  outline-variant: '#2A2E39'
  error: '#F23645'
  success: '#089981'
  warning: '#FF9800'
  inverse-surface: '#dfe2f2'
  inverse-on-surface: '#2c303c'
  surface-tint: '#b6c4ff'
  on-primary-container: '#f7f5ff'
  inverse-primary: '#004ee8'
  secondary: '#66dabf'
  on-secondary: '#00382e'
  secondary-container: '#20a28a'
  on-secondary-container: '#003027'
  tertiary: '#ffb3b0'
  on-tertiary: '#680010'
  tertiary-container: '#da2237'
  on-tertiary-container: '#fff4f3'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dce1ff'
  primary-fixed-dim: '#b6c4ff'
  on-primary-fixed: '#001550'
  on-primary-fixed-variant: '#003ab3'
  secondary-fixed: '#84f7db'
  secondary-fixed-dim: '#66dabf'
  on-secondary-fixed: '#00201a'
  on-secondary-fixed-variant: '#005143'
  tertiary-fixed: '#ffdad8'
  tertiary-fixed-dim: '#ffb3b0'
  on-tertiary-fixed: '#410007'
  on-tertiary-fixed-variant: '#92001b'
  background: '#0f131e'
  on-background: '#dfe2f2'
  surface-variant: '#313441'
theme:
  color_mode: DARK
  font: PLUS_JAKARTA_SANS
  roundness: ROUND_FOUR
  preset: DARK_SLATE
  custom_color: '#2962FF'
typography:
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 13px
    fontWeight: '600'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.5rem
  gutter-tablet: 0.75rem
  gutter-desktop: 1rem
  margin: 0.75rem
  margin-desktop: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

# Design principles

- Use a compact dark trading-terminal layout with the chart as the visual centre.
- Use fine borders, restrained rounding, clear hierarchy, and no decorative market data.
- Use Plus Jakarta Sans for interface text and tabular monospace numerals for prices, timestamps, and returns.
- Teal denotes bullish/positive, red denotes bearish/negative, blue denotes selected controls, and amber denotes warnings or retests.
- Keep the screen dense and readable. On narrow screens, stack the analysis panel under the chart and allow signal-table horizontal scrolling.
- Distinguish raw model scores, heuristic decision scores, hypothetical levels, historical research, and live market-data freshness.
- Do not display broker execution controls or unprovided market feeds such as options flow, CVD, or gamma exposure.
